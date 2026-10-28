from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from alpha_roco.arena import (
    TEAMS,
    _legacy_checkpoint,
    _load_model_state,
    _vocab_signature,
    choose,
    load_model,
    play_match,
    wilson_interval,
)
from alpha_roco.config import ModelConfig, RewardConfig, TrainConfig
from alpha_roco.features import (
    ObservationEncoder,
    action_key,
    collate_inputs,
    selectable_candidates,
)
from alpha_roco.losses import (
    damage_loss,
    generalized_advantage,
    ko_loss,
    normalize_advantages,
    opponent_cross_entropy,
    ppo_loss,
)
from alpha_roco.network import AlphaRoco
from alpha_roco.report import write_table
from alpha_roco.rewards import (
    apply_exploration_bonus,
    assign_rewards,
    effect_gains,
    potential,
    state_summary,
    switch_action_penalty,
)
from alpha_roco.terminal import BattleSession

def _is_status_skill(encoder: ObservationEncoder, action: dict[str, Any]) -> bool:
    if action.get("kind") != "skill" or action.get("skill_id") is None:
        return False
    skill = encoder.skill_by_id.get(str(action["skill_id"]), {})
    return skill.get("category") == 3


def _is_exploration_skill(
    encoder: ObservationEncoder, action: dict[str, Any]
) -> bool:
    if action.get("kind") != "skill" or action.get("skill_id") is None:
        return False
    category = encoder.skill_by_id.get(str(action["skill_id"]), {}).get("category")
    return category in (2, 3)


def _lead_damage(
    encoder: ObservationEncoder, attacker: dict[str, Any], defender: dict[str, Any]
) -> int:
    skills = attacker.get("skills") or encoder._species(attacker).get(
        "skills", {}
    ).get("normal", [])
    return max(
        (
            encoder._estimated_damage(encoder._skill(skill), attacker, defender)
            for skill in skills
        ),
        default=0,
    )


def _lead_matchup_bonus(
    encoder: ObservationEncoder,
    request: dict[str, Any],
    config: RewardConfig,
) -> float:
    state = request.get("state", {})
    side = request.get("side", "A")
    enemy_side = "B" if side == "A" else "A"
    own_index = state.get("active", {}).get(side, -1)
    enemy_index = state.get("active", {}).get(enemy_side, -1)
    own = state.get("teams", {}).get(side, [])
    enemy = state.get("teams", {}).get(enemy_side, [])
    if not (0 <= own_index < len(own) and 0 <= enemy_index < len(enemy)):
        return 0.0
    lead = own[own_index]
    target = enemy[enemy_index]
    bonus = 0.0
    target_hp = encoder._current_hp(target, encoder._stats(target))
    if target_hp > 0 and _lead_damage(encoder, lead, target) >= target_hp:
        bonus += config.lead_oneshot_bonus
    lead_hp = encoder._current_hp(lead, encoder._stats(lead))
    if lead_hp > 0 and _lead_damage(encoder, target, lead) < lead_hp:
        bonus += config.lead_survive_bonus
    return bonus


class _Exploration:
    def __init__(
        self,
        skill_bonus: float,
        trait_bonus: float,
        skill_bonus_late: float | None = None,
        cap: float | None = None,
        cap_late: float = 0.25,
        status_weight: float = 2.5,
        status_weight_late: float = 1.5,
    ):
        self.skill_bonus_aggressive = skill_bonus
        self.skill_bonus_late = (
            skill_bonus if skill_bonus_late is None else skill_bonus_late
        )
        self.trait_bonus = trait_bonus
        self.cap_late = cap_late
        self.status_weight_late = status_weight_late
        self.set_phase(
            True,
            cap=cap,
            status_weight=status_weight,
        )
        self.skill_counts: dict[int, int] = {}
        self.trait_counts: dict[int, int] = {}
        self.lock = threading.Lock()

    def set_phase(
        self,
        aggressive: bool,
        cap: float | None = None,
        status_weight: float = 2.5,
    ) -> None:
        self.skill_bonus = (
            self.skill_bonus_aggressive if aggressive else self.skill_bonus_late
        )
        self.cap = cap if aggressive else self.cap_late
        self.status_weight = status_weight if aggressive else self.status_weight_late

    def take_skill(self, skill_id, is_status: bool = False) -> float:
        if skill_id is None:
            return 0.0
        key = int(skill_id)
        weight = self.status_weight if is_status else 1.0
        with self.lock:
            count = self.skill_counts.get(key, 0)
            self.skill_counts[key] = count + 1
            return weight * self.skill_bonus / (1 + count)

    def take_trait(self, trait_id) -> float:
        if trait_id is None:
            return 0.0
        key = int(trait_id)
        with self.lock:
            count = self.trait_counts.get(key, 0)
            self.trait_counts[key] = count + 1
            return self.trait_bonus / (1 + count)


def _masked_distribution(
    logits: torch.Tensor, mask: torch.Tensor
) -> torch.distributions.Categorical:
    mask = mask.bool()
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("决策请求没有有效候选")
    return torch.distributions.Categorical(
        logits=logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    )


def _forward(
    model: AlphaRoco,
    encoder: ObservationEncoder,
    item: dict[str, Any],
    memory: Any,
    device: torch.device,
):
    batch = collate_inputs([item], device)
    return batch, model(batch, memory=memory)


def _sample_action(
    model: AlphaRoco,
    encoder: ObservationEncoder,
    request: dict[str, Any],
    memory: Any,
    device: torch.device,
    reward_config: RewardConfig,
):
    item = encoder.encode(request)
    summary = state_summary(request["state"], request["side"], reward_config)
    with torch.no_grad():
        batch, output = _forward(model, encoder, item, memory, device)
        distribution = _masked_distribution(output["logits"], batch["action_mask"])
        action_index = distribution.sample()
        action = selectable_candidates(request)[int(action_index.item())]
        old_log_prob = float(distribution.log_prob(action_index).item())
        old_value = float(output["value"].reshape(-1)[0].item())
    return (
        action,
        tuple(value.detach() for value in output["memory"]),
        {
            "item": item,
            "phase": request["phase"],
            "action": int(action_index.item()),
            "old_log_prob": old_log_prob,
            "old_value": old_value,
            "reward": 0.0,
            "terminal": False,
            "opponent_target": None,
            "summary": summary,
            "potential": potential(summary, reward_config),
            "action_kind": action.get("kind", ""),
            "exploration_bonus": 0.0,
            "trait_events": [],
        },
    )


def _paired_opponent_targets(
    requests: list[dict[str, Any]],
    actions_by_side: dict[str, dict[str, Any]],
    encoder: ObservationEncoder,
    entries_by_side: dict[str, dict[str, Any]],
) -> None:
    requests_by_side = {request["side"]: request for request in requests}
    if set(requests_by_side) != {"A", "B"}:
        return
    for side, request in requests_by_side.items():
        if side not in entries_by_side:
            continue
        other_side = "B" if side == "A" else "A"
        other_request = requests_by_side[other_side]
        if (
            request["phase"] != other_request["phase"]
            or request["phase"] == "replacement"
        ):
            continue
        chosen = actions_by_side[other_side]
        if chosen.get("kind") == "skill" and chosen.get("skill_id") is None:
            continue
        chosen_key = action_key(chosen)
        candidates = encoder.opponent_candidates(request)
        matches = [
            index
            for index, candidate in enumerate(candidates)
            if action_key(candidate) == chosen_key
        ]
        if matches:
            entries_by_side[side]["opponent_target"] = matches[0]


def _collect_versus_game(
    encoder: ObservationEncoder,
    candidate: AlphaRoco,
    opponent: AlphaRoco,
    team_a: str,
    team_b: str,
    seed: int,
    device: torch.device,
    reward_config: RewardConfig = RewardConfig(),
    log_dir: Path | None = None,
    gamma: float = 1.0,
    timeout: float = TrainConfig.terminal_timeout,
    exploration: _Exploration | None = None,
) -> dict[str, Any]:
    episode: list[dict[str, Any]] = []
    memories: dict[str, Any] = {"A": None, "B": None}
    status_actions = 0
    total_actions = 0
    status_skill_ids: set[int] = set()
    used_skill_ids: set[int] = set()
    effect_keys: set[tuple[str, str, str, str]] = set()
    trait_effects = 0
    seen_trait_events = 0
    last_a_entry = None
    first_action = True
    last_action_kind = None
    with BattleSession(
        team_a=team_a,
        team_b=team_b,
        seed=seed,
        timeout=timeout,
        log_dir=log_dir,
        log_jsonl=log_dir / "battle.jsonl" if log_dir else None,
    ) as session:
        while session.winner is None:
            requests = session.requests()
            request_a = next(
                (request for request in requests if request["side"] == "A"), None
            )
            if request_a is not None:
                trait_events = request_a["state"].get("trait_events", [])
                new_trait_events = trait_events[seen_trait_events:]
                if last_a_entry is not None and new_trait_events:
                    last_a_entry["trait_events"].extend(new_trait_events)
                    trait_effects += len(new_trait_events)
                    if exploration is not None:
                        last_a_entry["exploration_bonus"] += sum(
                            exploration.take_trait(event.get("trait_id"))
                            for event in new_trait_events
                        )
                seen_trait_events = len(trait_events)
            if (
                len(episode) == 1
                and "lead_bonus" not in episode[0]
                and request_a is not None
                and request_a["phase"] == "action"
            ):
                episode[0]["lead_bonus"] = _lead_matchup_bonus(
                    encoder, request_a, reward_config
                )
            actions_by_side: dict[str, dict[str, Any]] = {}
            entries_by_side: dict[str, dict[str, Any]] = {}
            for request in requests:
                side = request["side"]
                if side == "A":
                    action, memories[side], entry = _sample_action(
                        candidate,
                        encoder,
                        request,
                        memories[side],
                        device,
                        reward_config,
                    )
                    entry["status"] = _is_status_skill(encoder, action)
                    entry["exploration_skill"] = _is_exploration_skill(
                        encoder, action
                    )
                    if action.get("skill_id") is not None:
                        used_skill_ids.add(int(action["skill_id"]))
                    if entry["status"] and action.get("skill_id") is not None:
                        status_skill_ids.add(int(action["skill_id"]))
                    entry["switch_penalty"] = switch_action_penalty(
                        entry["phase"],
                        action.get("kind", ""),
                        first_action,
                        last_action_kind,
                        reward_config,
                    )
                    if entry["phase"] == "action":
                        first_action = False
                        last_action_kind = action.get("kind", "")
                    else:
                        last_action_kind = None
                    if last_a_entry is not None:
                        for effect_side, key, _ in effect_gains(
                            last_a_entry["summary"], entry["summary"]
                        ):
                            effect_keys.add((effect_side, *key))
                    if exploration is not None and entry["exploration_skill"]:
                        entry["exploration_bonus"] += exploration.take_skill(
                            action.get("skill_id"),
                            entry["status"],
                        )
                    episode.append(entry)
                    entries_by_side[side] = entry
                else:
                    action, memories[side] = choose(
                        opponent, encoder, request, memories[side], device, sample=False
                    )
                actions_by_side[side] = action
            request_a = next(
                (request for request in requests if request["side"] == "A"), None
            )
            if request_a is not None and request_a["phase"] == "action":
                total_actions += 1
                if "A" in actions_by_side and _is_status_skill(
                    encoder, actions_by_side["A"]
                ):
                    status_actions += 1
            _paired_opponent_targets(
                requests, actions_by_side, encoder, entries_by_side
            )
            for side, action in actions_by_side.items():
                session.submit(side, action)
            last_a_entry = entries_by_side.get("A", last_a_entry)
        winner = session.winner
    if winner not in {"A", "B", "both_lose"}:
        raise RuntimeError(f"对局未得到有效终局结果：{winner!r}")
    if not episode:
        raise RuntimeError("candidate方本局没有决策记录")
    assign_rewards(episode, winner, gamma)
    apply_exploration_bonus(
        episode,
        reward_config,
        exploration.cap if exploration is not None else None,
    )
    _assign_aux_targets(episode, encoder)
    return {
        "episode": episode,
        "winner": winner,
        "status_actions": status_actions,
        "total_actions": total_actions,
        "status_skill_ids": status_skill_ids,
        "used_skill_ids": used_skill_ids,
        "effect_keys": effect_keys,
        "trait_effects": trait_effects,
    }


def _effect_vector(
    effects: dict[tuple[str, str, str], float], vocab: dict, size: int
) -> list[float]:
    vector = [0.0] * size
    for (_, effect_type, _), value in effects.items():
        index = vocab.get(effect_type, vocab.get(str(effect_type), 1))
        if index >= 3:
            vector[index] += value / 10.0
    return vector


def _mark_vector(
    marks: dict[tuple[str, int], float], vocab: dict, size: int
) -> list[float]:
    vector = [0.0] * size
    for (_, mark_id), stacks in marks.items():
        index = vocab.get(mark_id, vocab.get(str(mark_id), 1))
        if index >= 3:
            vector[index] += stacks / 10.0
    return vector


def _effect_delta(
    before: dict, after: dict, encoder: ObservationEncoder
) -> list[float]:
    effect_size = encoder.vocab_sizes["effect"]
    mark_size = encoder.vocab_sizes["mark"]
    own_effect = _effect_vector(
        after["own_effects"], encoder.effect_vocab, effect_size
    )
    enemy_effect = _effect_vector(
        after["enemy_effects"], encoder.effect_vocab, effect_size
    )
    before_own = _effect_vector(
        before["own_effects"], encoder.effect_vocab, effect_size
    )
    before_enemy = _effect_vector(
        before["enemy_effects"], encoder.effect_vocab, effect_size
    )
    own_mark = _mark_vector(after["own_marks"], encoder.mark_vocab, mark_size)
    enemy_mark = _mark_vector(after["enemy_marks"], encoder.mark_vocab, mark_size)
    before_own_mark = _mark_vector(
        before["own_marks"], encoder.mark_vocab, mark_size
    )
    before_enemy_mark = _mark_vector(
        before["enemy_marks"], encoder.mark_vocab, mark_size
    )
    return [
        *(current - previous for current, previous in zip(own_effect, before_own)),
        *(
            current - previous
            for current, previous in zip(enemy_effect, before_enemy)
        ),
        *(current - previous for current, previous in zip(own_mark, before_own_mark)),
        *(
            current - previous
            for current, previous in zip(enemy_mark, before_enemy_mark)
        ),
        (after["own_energy"] - before["own_energy"]) / 10.0,
        (after["enemy_energy"] - before["enemy_energy"]) / 10.0,
    ]


def _dimension_weights(mask: torch.Tensor, cap: float = 10.0) -> torch.Tensor:
    frequency = mask.float().mean(dim=0).clamp_min(1e-3)
    weights = (1.0 / frequency.sqrt()).clamp(1.0, cap)
    return weights / weights.mean().clamp_min(1e-6)


def _weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    return ((prediction - target).pow(2) * weights * mask).sum() / (
        weights * mask
    ).sum().clamp_min(1)


def _assign_aux_targets(episode: list[dict[str, Any]], encoder: ObservationEncoder) -> None:
    battle_entries = [entry for entry in episode if entry["phase"] != "lead"]
    for entry, following in zip(battle_entries, battle_entries[1:]):
        before, after = entry["summary"], following["summary"]
        trait_trigger = [0.0] * encoder.vocab_sizes["trait"]
        trait_presence = [0.0] * encoder.vocab_sizes["trait"]
        for prefix in ("own_pets", "enemy_pets"):
            for pet in entry["item"].get(prefix, []):
                index = pet.get("trait", 0)
                if index >= 3:
                    trait_presence[index] = 1.0
        for event in entry.get("trait_events", []):
            trait_id = event.get("trait_id")
            index = encoder.trait_vocab.get(
                trait_id, encoder.trait_vocab.get(str(trait_id), 1)
            )
            if index >= 3:
                trait_trigger[index] = 1.0
        entry["aux_target"] = {
            "damage_dealt": max(0.0, before["enemy_hp"] - after["enemy_hp"]),
            "damage_taken": max(0.0, before["own_hp"] - after["own_hp"]),
            "ko": [
                float(after["enemy_alive"] < before["enemy_alive"]),
                float(after["own_alive"] < before["own_alive"]),
            ],
            "effect_delta": _effect_delta(before, after, encoder),
            "trait_trigger": trait_trigger,
            "trait_presence": trait_presence,
        }


def _prepare_advantages(
    episodes: list[dict[str, Any]], config: TrainConfig, device: torch.device
) -> dict[str, int]:
    counts = {"lead": 0, "battle": 0}
    phase_advantages: dict[str, list[torch.Tensor]] = {"lead": [], "battle": []}
    for episode in episodes:
        lead_entries = [entry for entry in episode if entry["phase"] == "lead"]
        for entry in lead_entries:
            advantage = torch.tensor(
                entry["reward"] - entry["old_value"], device=device
            )
            entry["return"] = torch.tensor(entry["reward"], device=device)
            entry["raw_advantage"] = advantage
            phase_advantages["lead"].append(advantage)
            counts["lead"] += 1
        battle_entries = [entry for entry in episode if entry["phase"] != "lead"]
        if battle_entries:
            rewards = torch.tensor(
                [entry["reward"] for entry in battle_entries], device=device
            )
            values = torch.tensor(
                [entry["old_value"] for entry in battle_entries], device=device
            )
            terminal = torch.tensor(
                [entry["terminal"] for entry in battle_entries], device=device
            )
            advantages, returns = generalized_advantage(
                rewards, values, terminal, config.gamma, config.gae_lambda
            )
            for entry, advantage, target in zip(
                battle_entries, advantages, returns, strict=True
            ):
                entry["return"] = target
                entry["raw_advantage"] = advantage
                phase_advantages["battle"].append(advantage)
                counts["battle"] += 1
    for phase, advantages in phase_advantages.items():
        if not advantages:
            continue
        normalized = normalize_advantages(torch.stack(advantages))
        index = 0
        for episode in episodes:
            for entry in episode:
                if (entry["phase"] == "lead") == (phase == "lead"):
                    entry["advantage"] = normalized[index]
                    index += 1
    return counts


def _set_trainable(model: AlphaRoco, trainable: set[int]) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in trainable)


def _replay_episode(
    model: AlphaRoco,
    encoder: ObservationEncoder,
    episode: list[dict[str, Any]],
    device: torch.device,
):
    memory = None
    for entry in episode:
        batch, output = _forward(model, encoder, entry["item"], memory, device)
        memory = output["memory"]
        yield entry, batch, output


def _batched_episode_steps(
    model: AlphaRoco,
    episodes: list[list[dict[str, Any]]],
    device: torch.device,
):
    memories = [None] * len(episodes)
    max_steps = max((len(episode) for episode in episodes), default=0)
    for step in range(max_steps):
        active = [
            index for index, episode in enumerate(episodes) if step < len(episode)
        ]
        if not active:
            break
        batch = collate_inputs(
            [episodes[index][step]["item"] for index in active], device
        )
        memory = (
            None
            if step == 0
            else (
                torch.cat([memories[index][0] for index in active], dim=1),
                torch.cat([memories[index][1] for index in active], dim=1),
            )
        )
        output = model(batch, memory=memory)
        next_memory = output["memory"]
        for position, index in enumerate(active):
            memories[index] = (
                next_memory[0][:, position : position + 1],
                next_memory[1][:, position : position + 1],
            )
        yield step, active, batch, output


def _policy_update(
    model: AlphaRoco,
    optimizer: torch.optim.Optimizer,
    encoder: ObservationEncoder,
    episodes: list[list[dict[str, Any]]],
    config: TrainConfig,
    ppo_epochs: int,
    device: torch.device,
) -> dict[str, float]:
    counts = _prepare_advantages(episodes, config, device)
    policy_ids = {id(parameter) for parameter in model.policy_parameters()}
    opponent_params = list(model.opponent_parameters())
    _set_trainable(model, policy_ids)
    metrics = {
        "policy": 0.0,
        "value": 0.0,
        "entropy": 0.0,
        "kl": 0.0,
        "clip_fraction": 0.0,
        "damage": 0.0,
        "damage_baseline": 0.0,
        "ko": 0.0,
        "ko_baseline": 0.0,
        "state_change": 0.0,
        "state_change_baseline": 0.0,
        "effect_delta": 0.0,
        "effect_delta_baseline": 0.0,
        "trait_trigger": 0.0,
        "trait_trigger_baseline": 0.0,
        "trait_effect": 0.0,
        "trait_effect_baseline": 0.0,
        "grad_norm": 0.0,
    }
    aux_count = sum(
        1
        for episode in episodes
        for entry in episode
        if entry.get("aux_target") is not None
    )
    completed_epochs = 0
    for _ in range(ppo_epochs):
        optimizer.zero_grad(set_to_none=True)
        sums = {
            key: 0.0
            for key in (
                "policy",
                "value",
                "entropy",
                "kl",
                "clip_fraction",
                "damage",
                "damage_baseline",
                "ko",
                "ko_baseline",
                "state_change",
                "state_change_baseline",
                "effect_delta",
                "effect_delta_baseline",
                "trait_trigger",
                "trait_trigger_baseline",
                "trait_effect",
                "trait_effect_baseline",
            )
        }
        for start in range(0, len(episodes), config.update_chunk):
            chunk = episodes[start : start + config.update_chunk]
            losses: list[torch.Tensor] = []
            for step, active, batch, output in _batched_episode_steps(
                model, chunk, device
            ):
                entries = [chunk[index][step] for index in active]
                phases = [
                    "lead" if entry["phase"] == "lead" else "battle"
                    for entry in entries
                ]
                for phase in ("lead", "battle"):
                    positions = [
                        position
                        for position, value in enumerate(phases)
                        if value == phase
                    ]
                    if not positions:
                        continue
                    selected = [entries[position] for position in positions]
                    result = ppo_loss(
                        output["logits"][positions],
                        torch.tensor(
                            [entry["action"] for entry in selected], device=device
                        ),
                        torch.tensor(
                            [entry["old_log_prob"] for entry in selected],
                            device=device,
                        ),
                        torch.stack([entry["advantage"] for entry in selected]),
                        output["value"][positions].reshape(-1),
                        torch.stack([entry["return"] for entry in selected]),
                        batch["action_mask"][positions],
                        config.clip_ratio,
                        config.value_coefficient,
                        config.entropy_coefficient,
                    )
                    scale = len(positions) / max(counts[phase], 1)
                    losses.append(result["loss"] * scale)
                    for key in (
                        "policy",
                        "value",
                        "entropy",
                        "kl",
                        "clip_fraction",
                    ):
                        sums[key] += float(result[key].detach().item()) * scale
                aux_positions = [
                    position
                    for position, entry in enumerate(entries)
                    if entry.get("aux_target") is not None
                ]
                if aux_positions:
                    targets = [
                        entries[position]["aux_target"] for position in aux_positions
                    ]
                    scale = len(aux_positions) / max(aux_count, 1)
                    predictions = {
                        key: value[aux_positions]
                        for key, value in output["aux"].items()
                    }
                    base_targets = {
                        "damage_dealt": torch.tensor(
                            [target["damage_dealt"] for target in targets],
                            device=device,
                        ),
                        "damage_taken": torch.tensor(
                            [target["damage_taken"] for target in targets],
                            device=device,
                        ),
                        "ko": torch.tensor(
                            [target["ko"] for target in targets],
                            device=device,
                        ),
                    }
                    zero_predictions = {
                        key: torch.zeros_like(value)
                        for key, value in base_targets.items()
                    }
                    damage = damage_loss(predictions, base_targets)
                    damage_baseline = damage_loss(zero_predictions, base_targets)
                    ko = ko_loss(predictions, base_targets)
                    ko_baseline = ko_loss(zero_predictions, base_targets)
                    losses.append(damage * config.aux_coefficient * scale)
                    losses.append(ko * config.aux_coefficient * scale)
                    sums["damage"] += float(damage.detach().item()) * scale
                    sums["damage_baseline"] += (
                        float(damage_baseline.detach().item()) * scale
                    )
                    sums["ko"] += float(ko.detach().item()) * scale
                    sums["ko_baseline"] += (
                        float(ko_baseline.detach().item()) * scale
                    )
                    effect_targets = torch.tensor(
                        [target["effect_delta"] for target in targets],
                        dtype=torch.float32,
                        device=device,
                    )
                    trait_targets = torch.tensor(
                        [target["trait_trigger"] for target in targets],
                        dtype=torch.float32,
                        device=device,
                    )
                    trait_presence = torch.tensor(
                        [target["trait_presence"] for target in targets],
                        dtype=torch.bool,
                        device=device,
                    )
                    context = output["battle_context"][aux_positions]
                    skill_mask = torch.tensor(
                        [
                            entries[position]["action_kind"] == "skill"
                            for position in aux_positions
                        ],
                        dtype=torch.bool,
                        device=device,
                    )
                    effect_weights = torch.ones(
                        effect_targets.shape[1], device=device
                    )
                    if skill_mask.any():
                        positions = [
                            aux_positions[index]
                            for index in range(len(aux_positions))
                            if bool(skill_mask[index])
                        ]
                        actions = torch.tensor(
                            [entries[position]["action"] for position in positions],
                            device=device,
                        )
                        effect_source = torch.cat(
                            (
                                output["battle_context"][positions],
                                output["action_embeddings"][positions, actions],
                            ),
                            dim=-1,
                        )
                        effect_pred = model.aux_effect_delta(effect_source)
                        effect_target = effect_targets[skill_mask]
                        effect_mask = effect_target != 0
                        if effect_mask.any():
                            effect_weights = _dimension_weights(effect_mask)
                            effect = _weighted_mse(
                                effect_pred,
                                effect_target,
                                effect_mask,
                                effect_weights,
                            )
                            effect_baseline = _weighted_mse(
                                torch.zeros_like(effect_pred),
                                effect_target,
                                effect_mask,
                                effect_weights,
                            )
                            losses.append(
                                effect * config.effect_aux_coefficient * scale
                            )
                            sums["effect_delta"] += (
                                float(effect.detach().item()) * scale
                            )
                            sums["effect_delta_baseline"] += (
                                float(effect_baseline.detach().item()) * scale
                            )
                        change_target = effect_mask.float()
                        change_frequency = change_target.mean(dim=0).clamp_min(1e-3)
                        change_weight = (
                            (1.0 - change_frequency) / change_frequency
                        ).clamp(1.0, config.state_change_pos_weight)
                        change = F.binary_cross_entropy_with_logits(
                            model.aux_state_change(effect_source),
                            change_target,
                            pos_weight=change_weight,
                        )
                        change_baseline = F.binary_cross_entropy_with_logits(
                            torch.zeros_like(change_target),
                            change_target,
                            pos_weight=change_weight,
                        )
                        losses.append(
                            change * config.effect_aux_coefficient * scale
                        )
                        sums["state_change"] += (
                            float(change.detach().item()) * scale
                        )
                        sums["state_change_baseline"] += (
                            float(change_baseline.detach().item()) * scale
                        )
                    trait_logits = model.aux_trait_trigger(context)[trait_presence]
                    trigger_target = trait_targets[trait_presence]
                    trigger = F.binary_cross_entropy_with_logits(
                        trait_logits, trigger_target
                    )
                    trigger_baseline = F.binary_cross_entropy_with_logits(
                        torch.zeros_like(trigger_target), trigger_target
                    )
                    losses.append(
                        trigger * config.trait_aux_coefficient * scale
                    )
                    sums["trait_trigger"] += (
                        float(trigger.detach().item()) * scale
                    )
                    sums["trait_trigger_baseline"] += (
                        float(trigger_baseline.detach().item()) * scale
                    )
                    trait_mask = trait_targets.sum(dim=1) > 0
                    if trait_mask.any():
                        trait_weights = trait_targets[trait_mask]
                        trait_embedding = (
                            trait_weights @ model.trait_embedding.weight
                        ) / trait_weights.sum(dim=1, keepdim=True).clamp_min(1)
                        trait_embedding = model.aux_trait_source(trait_embedding)
                        trait_pred = model.aux_effect_delta(
                            torch.cat(
                                (context[trait_mask], trait_embedding), dim=-1
                            )
                        )
                        trait_target = effect_targets[trait_mask]
                        trait_effect_mask = trait_target != 0
                        if trait_effect_mask.any():
                            trait_effect = _weighted_mse(
                                trait_pred,
                                trait_target,
                                trait_effect_mask,
                                effect_weights,
                            )
                            trait_effect_baseline = _weighted_mse(
                                torch.zeros_like(trait_pred),
                                trait_target,
                                trait_effect_mask,
                                effect_weights,
                            )
                            losses.append(
                                trait_effect
                                * config.trait_aux_coefficient
                                * scale
                            )
                            sums["trait_effect"] += (
                                float(trait_effect.detach().item()) * scale
                            )
                            sums["trait_effect_baseline"] += (
                                float(trait_effect_baseline.detach().item()) * scale
                            )
            if losses:
                torch.stack(losses).sum().backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.policy_parameters(), config.max_grad_norm
        )
        epoch_kl = sums["kl"]
        if epoch_kl < config.target_kl:
            optimizer.step()
        else:
            optimizer.zero_grad(set_to_none=True)
        completed_epochs += 1
        for key in sums:
            metrics[key] = sums[key]
        metrics["grad_norm"] = float(norm.detach().item())
        if epoch_kl >= config.target_kl:
            break
    _set_trainable(model, {id(parameter) for parameter in opponent_params})
    metrics["epochs"] = float(completed_epochs)
    return metrics


def _opponent_update(
    model: AlphaRoco,
    optimizer: torch.optim.Optimizer,
    encoder: ObservationEncoder,
    episodes: list[list[dict[str, Any]]],
    config: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    opponent_ids = {id(parameter) for parameter in model.opponent_parameters()}
    _set_trainable(model, opponent_ids)
    target_count = sum(
        entry["opponent_target"] is not None
        for episode in episodes
        for entry in episode
    )
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    for start in range(0, len(episodes), config.update_chunk):
        chunk = episodes[start : start + config.update_chunk]
        losses = []
        for step, active, batch, output in _batched_episode_steps(model, chunk, device):
            targets = [chunk[index][step]["opponent_target"] for index in active]
            positions = [
                position
                for position, target in enumerate(targets)
                if target is not None
            ]
            if not positions:
                continue
            loss = opponent_cross_entropy(
                output["opponent_logits"][positions],
                torch.tensor(
                    [targets[position] for position in positions], device=device
                ),
                batch["opponent_mask"][positions],
            )
            if not bool(torch.isfinite(loss).all()):
                raise FloatingPointError("对手交叉熵出现非有限值")
            loss_sum += float(loss.detach().item()) * len(positions)
            losses.append(loss * len(positions) / max(target_count, 1))
        if losses:
            torch.stack(losses).sum().backward()
    norm = torch.nn.utils.clip_grad_norm_(
        model.opponent_parameters(), config.max_grad_norm
    )
    if target_count:
        optimizer.step()
    _set_trainable(model, {id(parameter) for parameter in model.policy_parameters()})
    return {
        "opponent_ce": loss_sum / max(target_count, 1),
        "opponent_grad_norm": float(norm.detach().item()),
        "opponent_samples": float(target_count),
    }


def _checkpoint(
    path: Path,
    model: AlphaRoco,
    optimizers: dict[str, torch.optim.Optimizer],
    encoder: ObservationEncoder,
    model_config: ModelConfig,
    train_config: TrainConfig,
    batch_index: int,
    games_seen: int,
    best_batch: int,
    best_elo: float,
    overwrite: bool = False,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"拒绝覆盖已有checkpoint：{path}")
    payload = {
        "format": "single",
        "config": {
            "model_config": dataclasses.asdict(model_config),
            "train_config": dataclasses.asdict(train_config),
        },
        "vocab_sizes": encoder.vocab_sizes,
        "vocabularies": encoder.vocabularies,
        "vocab_signature": hashlib.sha256(
            json.dumps(
                encoder.vocabularies, ensure_ascii=False, sort_keys=True
            ).encode()
        ).hexdigest(),
        "model": model.state_dict(),
        "optimizers": {
            name: optimizer.state_dict() for name, optimizer in optimizers.items()
        },
        "batch_index": batch_index,
        "next_batch": batch_index + 1,
        "games_seen": games_seen,
        "best_batch": best_batch,
        "best_elo": best_elo,
        "rng": {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
    }
    if overwrite:
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
    else:
        torch.save(payload, path)


def _restore(
    checkpoint: dict[str, Any],
    model: AlphaRoco,
    optimizers: dict[str, torch.optim.Optimizer],
    encoder: ObservationEncoder,
    device: torch.device,
) -> tuple[dict[str, Any], int, int]:
    if checkpoint.get("format") != "single" or "model" not in checkpoint:
        raise ValueError("checkpoint不是当前单网络格式")
    legacy = _legacy_checkpoint(checkpoint, encoder)
    signature = _vocab_signature(encoder.vocabularies)
    legacy_signature = _vocab_signature(
        {
            name: vocab
            for name, vocab in encoder.vocabularies.items()
            if name != "trait"
        }
    )
    if checkpoint["vocab_signature"] not in (signature, legacy_signature):
        raise ValueError("checkpoint词表签名与当前真实数据不一致")
    _load_model_state(model, checkpoint["model"], legacy)
    if not legacy:
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(checkpoint["optimizers"][name])
    model.to(device)
    if not legacy:
        for optimizer in optimizers.values():
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device)
    random.setstate(checkpoint["rng"]["python"])
    torch.set_rng_state(checkpoint["rng"]["torch_cpu"].cpu())
    if torch.cuda.is_available() and checkpoint["rng"]["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(checkpoint["rng"]["torch_cuda"])
    return checkpoint, checkpoint["next_batch"], checkpoint["games_seen"]


def _clone_model(
    model: AlphaRoco,
    model_config: ModelConfig,
    encoder: ObservationEncoder,
    device: torch.device,
) -> AlphaRoco:
    clone = AlphaRoco(config=model_config, vocab_sizes=encoder.vocab_sizes).to(device)
    clone.load_state_dict(model.state_dict())
    clone.eval()
    return clone


def _load_pool(
    pool_dir: Path, encoder: ObservationEncoder, device: torch.device, size: int
) -> list[tuple[str, AlphaRoco]]:
    paths = sorted(pool_dir.glob("best-batch-*.pt"))
    for stale in paths[:-size]:
        stale.unlink()
    return [(path.name, load_model(path, encoder, device)) for path in paths[-size:]]


def _sample_opponent(
    best_model: AlphaRoco,
    pool: list[tuple[str, AlphaRoco]],
    stats: dict[str, dict[str, int]],
    config: TrainConfig,
) -> tuple[str, AlphaRoco]:
    if not pool or random.random() < config.opponent_current_prob:
        return "current", best_model
    weights = []
    for name, _ in pool:
        record = stats.get(name, {})
        rate = record["wins"] / record["games"] if record.get("games") else 0.5
        weights.append(
            1.0
            if config.pool_rate_low <= rate <= config.pool_rate_high
            else config.pool_weight
        )
    return random.choices(pool, weights=weights, k=1)[0]


def _collect_batch(
    encoder: ObservationEncoder,
    candidate: AlphaRoco,
    best_model: AlphaRoco,
    pool: list[tuple[str, AlphaRoco]],
    pool_stats: dict[str, dict[str, int]],
    first_game: int,
    device: torch.device,
    config: TrainConfig,
    output: Path,
    exploration: _Exploration | None = None,
) -> list[tuple[int, str, dict[str, Any]]]:
    tasks = []
    for offset in range(config.games_per_batch):
        game_index = first_game + offset
        game_seed = random.randrange(2**32)
        opponent_name, opponent_model = _sample_opponent(
            best_model, pool, pool_stats, config
        )
        game_log_dir = (
            output / "games" / f"game-{game_index:06d}-seed-{game_seed}"
            if config.log_games
            else None
        )
        if game_log_dir is not None:
            game_log_dir.mkdir(parents=True, exist_ok=True)
        tasks.append(
            (
                game_index,
                opponent_name,
                opponent_model,
                random.choice(TEAMS),
                random.choice(TEAMS),
                game_seed,
                game_log_dir,
            )
        )

    def collect(task):
        (
            game_index,
            opponent_name,
            opponent_model,
            team_a,
            team_b,
            game_seed,
            log_dir,
        ) = task
        return (
            game_index,
            opponent_name,
            _collect_versus_game(
                encoder,
                candidate,
                opponent_model,
                team_a,
                team_b,
                game_seed,
                device,
                config.reward,
                log_dir=log_dir,
                gamma=config.gamma,
                timeout=config.terminal_timeout,
                exploration=exploration,
            ),
        )

    if config.sample_workers == 1:
        return [collect(task) for task in tasks]
    with ThreadPoolExecutor(
        max_workers=min(config.sample_workers, config.games_per_batch)
    ) as executor:
        return list(executor.map(collect, tasks))


def run(config: TrainConfig, resume: Path | None = None) -> list[Path]:
    if (
        config.batches < 1
        or config.games_per_batch < 1
        or config.ppo_epochs < 1
        or config.keep_checkpoints < 1
        or config.sample_workers < 1
        or config.update_chunk < 1
        or config.rare_skill_bonus < 0
        or config.rare_skill_bonus_late < 0
        or config.rare_skill_batches < 0
        or config.exploration_aggressive_batches < 0
        or config.status_exploration_weight < 0
        or config.status_exploration_weight_late < 0
        or config.effect_aux_coefficient < 0
        or config.trait_aux_coefficient < 0
    ):
        raise ValueError(
            "batches、games-per-batch、ppo-epochs、sample-workers、update-chunk与keep-checkpoints必须为正数，探索与辅助系数必须非负"
        )
    if config.eval_interval < 1:
        raise ValueError("eval-interval必须为正数")
    if config.eval_games < 2:
        raise ValueError("eval-games至少为2")
    if config.eval_workers < 1:
        raise ValueError("eval-workers必须为正数")
    if config.pool_size < 1:
        raise ValueError("pool-size必须为正数")
    if config.terminal_timeout <= 0:
        raise ValueError("terminal-timeout必须为正数")
    if not 0 <= config.opponent_current_prob <= 1:
        raise ValueError("opponent-current-prob必须在[0,1]内")
    if not 0 <= config.pool_rate_low <= config.pool_rate_high <= 1:
        raise ValueError("pool-rate-low/high必须在[0,1]内且low<=high")
    if config.pool_weight < 0:
        raise ValueError("pool-weight必须非负")
    if not 0 <= config.promotion_threshold <= 1:
        raise ValueError("promotion-threshold必须在[0,1]内")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device(config.device)
    encoder = ObservationEncoder()
    checkpoint = (
        torch.load(resume, map_location="cpu", weights_only=False)
        if resume is not None
        else None
    )
    model_config = (
        dataclasses.replace(
            ModelConfig(**checkpoint["config"]["model_config"]),
            dropout=ModelConfig().dropout,
        )
        if checkpoint is not None
        else ModelConfig()
    )
    if checkpoint is not None:
        saved_config = TrainConfig(**checkpoint["config"]["train_config"])
        config = dataclasses.replace(
            saved_config,
            **{
                field.name: getattr(config, field.name)
                for field in dataclasses.fields(TrainConfig)
            },
        )
    model = AlphaRoco(config=model_config, vocab_sizes=encoder.vocab_sizes).to(device)
    optimizers = {
        "policy": torch.optim.Adam(model.policy_parameters(), lr=config.learning_rate),
        "opponent": torch.optim.Adam(
            model.opponent_parameters(), lr=config.learning_rate
        ),
    }
    first_batch = 0
    games_seen = 0
    best_batch = 0
    best_elo = 0.0
    output = Path(config.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / "best.pt"
    pool_dir = output / "pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    pool_stats_path = output / "pool_stats.json"
    if resume is not None:
        checkpoint, first_batch, games_seen = _restore(
            checkpoint, model, optimizers, encoder, device
        )
    if resume is not None and best_path.exists():
        best_model = load_model(best_path, encoder, device)
        best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        best_batch = int(best_checkpoint["best_batch"])
        best_elo = float(best_checkpoint["best_elo"])
    else:
        best_model = _clone_model(model, model_config, encoder, device)
        if resume is not None:
            best_batch = first_batch
        _checkpoint(
            best_path,
            model,
            optimizers,
            encoder,
            model_config,
            config,
            first_batch - 1,
            games_seen,
            best_batch,
            best_elo,
            overwrite=True,
        )
    pool = _load_pool(pool_dir, encoder, device, config.pool_size)
    pool_stats = (
        json.loads(pool_stats_path.read_text(encoding="utf-8"))
        if pool_stats_path.exists()
        else {}
    )
    pool_stats = {
        name: record
        for name, record in pool_stats.items()
        if name == "current" or name in {pool_name for pool_name, _ in pool}
    }
    saved_paths = []
    exploration = _Exploration(
        config.rare_skill_bonus,
        config.reward.trait_novelty_bonus,
        skill_bonus_late=config.rare_skill_bonus_late,
        cap=config.reward.rare_skill_cap,
        cap_late=config.reward.rare_skill_cap_late,
        status_weight=config.status_exploration_weight,
        status_weight_late=config.status_exploration_weight_late,
    )
    for batch_offset, batch_index in enumerate(
        range(first_batch, first_batch + config.batches)
    ):
        batch_number = batch_index + 1
        exploration.set_phase(
            batch_number <= config.exploration_aggressive_batches,
            config.reward.rare_skill_cap,
            config.status_exploration_weight,
        )
        print(f"iter_num={batch_number} sampling={config.games_per_batch}", flush=True)
        episodes: list[list[dict[str, Any]]] = []
        both_lose = 0
        status_actions = 0
        total_actions = 0
        status_skill_ids: set[int] = set()
        used_skill_ids: set[int] = set()
        effect_keys: set[tuple[str, str, str, str]] = set()
        trait_effects = 0
        model.eval()
        results = _collect_batch(
            encoder,
            model,
            best_model,
            pool,
            pool_stats,
            games_seen,
            device,
            config,
            output,
            exploration
            if config.rare_skill_batches == 0
            or batch_offset < config.rare_skill_batches
            else None,
        )
        for _, opponent_name, result in results:
            episodes.append(result["episode"])
            both_lose += result["winner"] == "both_lose"
            status_actions += result["status_actions"]
            total_actions += result["total_actions"]
            status_skill_ids |= result["status_skill_ids"]
            used_skill_ids |= result["used_skill_ids"]
            effect_keys |= result["effect_keys"]
            trait_effects += result["trait_effects"]
            record = pool_stats.setdefault(opponent_name, {"wins": 0, "games": 0})
            record["games"] += 1
            record["wins"] += result["winner"] == "A"
        games_seen += config.games_per_batch
        model.train()
        policy_metrics = _policy_update(
            model,
            optimizers["policy"],
            encoder,
            episodes,
            config,
            config.ppo_epochs,
            device,
        )
        opponent_metrics = _opponent_update(
            model,
            optimizers["opponent"],
            encoder,
            episodes,
            config,
            device,
        )
        if not torch.isfinite(
            torch.tensor([*policy_metrics.values(), *opponent_metrics.values()])
        ).all():
            raise FloatingPointError("训练批次指标出现非有限值，拒绝保存")
        if policy_metrics["grad_norm"] <= 0:
            raise RuntimeError("训练批次没有产生有效策略梯度，拒绝保存")
        if opponent_metrics["opponent_samples"] <= 0:
            raise RuntimeError("训练批次没有可确认的对手行动标签，拒绝保存")
        if opponent_metrics["opponent_grad_norm"] <= 0:
            raise RuntimeError("训练批次没有产生有效对手梯度，拒绝保存")
        checkpoint_path = output / f"checkpoint-batch-{batch_number:06d}.pt"
        _checkpoint(
            checkpoint_path,
            model,
            optimizers,
            encoder,
            model_config,
            config,
            batch_index,
            games_seen,
            best_batch,
            best_elo,
        )
        saved_paths.append(checkpoint_path)
        for stale_path in sorted(output.glob("checkpoint-batch-*.pt"))[
            : -config.keep_checkpoints
        ]:
            stale_path.unlink()
        evaluated = (
            batch_number % config.eval_interval == 0
            or batch_index == first_batch + config.batches - 1
        )
        rate = None
        reference_batch = best_batch
        elo = best_elo
        promoted = False
        if evaluated:
            model.eval()
            tally = play_match(
                model,
                best_model,
                encoder,
                config.eval_games,
                config.seed + batch_number,
                device,
                sample=False,
                workers=config.eval_workers,
            )
            model.train()
            rate = tally["win"] / sum(tally.values())
            interval = wilson_interval(tally["win"], sum(tally.values()))
            elo = best_elo + config.elo_k * (rate - 0.5)
            promoted = rate > config.promotion_threshold
            if promoted:
                snapshot = pool_dir / f"best-batch-{reference_batch:06d}.pt"
                if not snapshot.exists():
                    shutil.copy2(best_path, snapshot)
                if snapshot.name not in {name for name, _ in pool}:
                    pool.append((snapshot.name, load_model(snapshot, encoder, device)))
                if "current" in pool_stats:
                    pool_stats[snapshot.name] = pool_stats.pop("current")
                while len(pool) > config.pool_size:
                    name, _ = pool.pop(0)
                    (pool_dir / name).unlink(missing_ok=True)
                    pool_stats.pop(name, None)
                best_batch = batch_number
                best_elo = elo
                _checkpoint(
                    best_path,
                    model,
                    optimizers,
                    encoder,
                    model_config,
                    config,
                    batch_index,
                    games_seen,
                    best_batch,
                    best_elo,
                    overwrite=True,
                )
                best_model.load_state_dict(model.state_dict())
                best_model.eval()
            eval_record = {
                "iter_num": batch_number,
                "games": config.eval_games,
                "reference_batch": reference_batch,
                "win": tally["win"],
                "loss": tally["loss"],
                "both_lose": tally["both_lose"],
                "win_rate": rate,
                "ci95": list(interval),
                "elo": elo,
                "promoted": promoted,
                "best_elo": best_elo,
            }
            with (output / "eval.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(eval_record, ensure_ascii=False) + "\n")
            write_table(output / "eval.jsonl", output / "eval_table.md")
        record = {
            "iter_num": batch_number,
            "eval_win_rate": rate,
            "promoted": promoted,
            "entropy": policy_metrics["entropy"],
            "kl": policy_metrics["kl"],
            "value_loss": policy_metrics["value"],
            "damage_loss": policy_metrics["damage"],
            "damage_baseline": policy_metrics["damage_baseline"],
            "ko_loss": policy_metrics["ko"],
            "ko_baseline": policy_metrics["ko_baseline"],
            "state_change_loss": policy_metrics["state_change"],
            "state_change_baseline": policy_metrics["state_change_baseline"],
            "effect_delta_loss": policy_metrics["effect_delta"],
            "effect_delta_baseline": policy_metrics["effect_delta_baseline"],
            "trait_trigger_loss": policy_metrics["trait_trigger"],
            "trait_trigger_baseline": policy_metrics["trait_trigger_baseline"],
            "trait_effect_loss": policy_metrics["trait_effect"],
            "trait_effect_baseline": policy_metrics["trait_effect_baseline"],
            "opponent_ce": opponent_metrics["opponent_ce"],
            "both_lose_rate": both_lose / config.games_per_batch,
            "status_actions": status_actions,
            "status_action_rate": status_actions / max(total_actions, 1),
            "unique_status_skills": len(status_skill_ids),
            "unique_skills": len(used_skill_ids),
            "unique_effects": len(effect_keys),
            "trait_effects": trait_effects,
        }
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        pool_stats_path.write_text(
            json.dumps(pool_stats, ensure_ascii=False), encoding="utf-8"
        )
        eval_text = "-" if rate is None else f"{rate:.3f}"
        print(
            f"iter_num={batch_number} eval_win_rate={eval_text} promoted={promoted} "
            f"entropy={record['entropy']:.4f} "
            f"kl={record['kl']:.4f} value={record['value_loss']:.4f} "
            f"damage={record['damage_loss']:.4f} "
            f"ko={record['ko_loss']:.4f} "
            f"state_change={record['state_change_loss']:.4f} "
            f"effect_delta={record['effect_delta_loss']:.4f} "
            f"trait_trigger={record['trait_trigger_loss']:.4f} "
            f"trait_effect={record['trait_effect_loss']:.4f} "
            f"opp_ce={record['opponent_ce']:.4f} "
            f"both_lose_rate={record['both_lose_rate']:.3f} "
            f"status_rate={record['status_action_rate']:.3f} "
            f"uniq_status={record['unique_status_skills']} "
            f"uniq_skill={record['unique_skills']} "
            f"uniq_effect={record['unique_effects']} "
            f"trait_fx={record['trait_effects']}",
            flush=True,
        )
    return saved_paths


def _parse_args() -> tuple[TrainConfig, Path | None]:
    parser = argparse.ArgumentParser(
        description="使用真实对局训练Alpha Roco recurrent PPO"
    )
    parser.add_argument("--batches", type=int, default=TrainConfig.batches)
    parser.add_argument(
        "--games-per-batch", type=int, default=TrainConfig.games_per_batch
    )
    parser.add_argument("--ppo-epochs", type=int, default=TrainConfig.ppo_epochs)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--output", default=TrainConfig.output)
    parser.add_argument("--threads", type=int, default=TrainConfig.threads)
    parser.add_argument("--log-games", action="store_true")
    parser.add_argument(
        "--keep-checkpoints", type=int, default=TrainConfig.keep_checkpoints
    )
    parser.add_argument("--pool-size", type=int, default=TrainConfig.pool_size)
    parser.add_argument(
        "--eval-interval", type=int, default=TrainConfig.eval_interval
    )
    parser.add_argument("--eval-games", type=int, default=TrainConfig.eval_games)
    parser.add_argument(
        "--eval-workers", type=int, default=TrainConfig.eval_workers
    )
    parser.add_argument(
        "--sample-workers", type=int, default=TrainConfig.sample_workers
    )
    parser.add_argument(
        "--update-chunk", type=int, default=TrainConfig.update_chunk
    )
    parser.add_argument(
        "--promotion-threshold",
        type=float,
        default=TrainConfig.promotion_threshold,
    )
    parser.add_argument(
        "--aux-coefficient", type=float, default=TrainConfig.aux_coefficient
    )
    parser.add_argument(
        "--effect-aux-coefficient",
        type=float,
        default=TrainConfig.effect_aux_coefficient,
    )
    parser.add_argument(
        "--trait-aux-coefficient",
        type=float,
        default=TrainConfig.trait_aux_coefficient,
    )
    parser.add_argument(
        "--rare-skill-bonus",
        type=float,
        default=TrainConfig.rare_skill_bonus,
    )
    parser.add_argument(
        "--rare-skill-batches",
        type=int,
        default=TrainConfig.rare_skill_batches,
    )
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    return (
        TrainConfig(
            batches=args.batches,
            games_per_batch=args.games_per_batch,
            ppo_epochs=args.ppo_epochs,
            device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            seed=args.seed,
            output=args.output,
            threads=args.threads,
            log_games=args.log_games,
            keep_checkpoints=args.keep_checkpoints,
            pool_size=args.pool_size,
            eval_interval=args.eval_interval,
            eval_games=args.eval_games,
            eval_workers=args.eval_workers,
            sample_workers=args.sample_workers,
            update_chunk=args.update_chunk,
            promotion_threshold=args.promotion_threshold,
            aux_coefficient=args.aux_coefficient,
            effect_aux_coefficient=args.effect_aux_coefficient,
            trait_aux_coefficient=args.trait_aux_coefficient,
            rare_skill_bonus=args.rare_skill_bonus,
            rare_skill_batches=args.rare_skill_batches,
        ),
        args.resume,
    )


def main() -> None:
    config, resume = _parse_args()
    run(config, resume)


if __name__ == "__main__":
    main()
