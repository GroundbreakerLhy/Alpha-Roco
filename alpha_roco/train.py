from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import multiprocessing
import random
import shutil
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from alpha_roco.arena import (
    LEGACY_VOCAB_KEYS,
    TEAMS,
    _legacy_checkpoint,
    _load_model_state,
    _vocab_signature,
    choose,
    evaluation_team_pairs,
    load_model,
    play_match,
    record_opponent_actions,
    wilson_interval,
)
from alpha_roco.fast_arena import fast_play_match_paths
from alpha_roco.config import (
    FEATURE_SCHEMA_VERSION,
    MODEL_SCHEMA_VERSION,
    STATE_SCHEMA_VERSION,
    ModelConfig,
    RewardConfig,
    TrainConfig,
)
from alpha_roco.features import (
    PET_STATE_ATTRIBUTE_KEYS,
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
    opponent_distribution_loss,
    ppo_loss,
)
from alpha_roco.network import PET_DELTA_DIM, AlphaRoco, HistoryState, collate_histories
from alpha_roco.report import write_table
from alpha_roco.rewards import (
    PET_STATE_DIM,
    apply_exploration_bonus,
    apply_repeat_penalty,
    assign_rewards,
    effect_gains,
    effect_value,
    potential,
    state_summary,
    switch_action_penalty,
)
from alpha_roco.inproc import InProcessSession


def _is_status_skill(encoder: ObservationEncoder, action: dict[str, Any]) -> bool:
    if action.get("kind") != "skill" or action.get("skill_id") is None:
        return False
    skill = encoder.skill_by_id.get(str(action["skill_id"]), {})
    return skill.get("category") == 3


def _is_exploration_skill(encoder: ObservationEncoder, action: dict[str, Any]) -> bool:
    if action.get("kind") != "skill" or action.get("skill_id") is None:
        return False
    category = encoder.skill_by_id.get(str(action["skill_id"]), {}).get("category")
    return category in (2, 3)


def _lead_damage(
    encoder: ObservationEncoder, attacker: dict[str, Any], defender: dict[str, Any]
) -> int:
    skills = attacker.get("skills") or encoder._species(attacker).get("skills", {}).get(
        "normal", []
    )
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
        cap_late: float | None = None,
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
        self.trait_counts: dict[tuple[str, int, int], int] = {}
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

    def take_trait(self, side, pet_index, trait_id) -> float:
        if trait_id is None or pet_index is None:
            return 0.0
        key = (str(side), int(pet_index), int(trait_id))
        with self.lock:
            count = self.trait_counts.get(key, 0)
            self.trait_counts[key] = count + 1
            return self.trait_bonus / (1 + count)


def _masked_probabilities(
    logits: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = mask.bool()
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("决策请求没有有效候选")
    logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    return torch.softmax(logits, dim=-1), torch.log_softmax(logits, dim=-1)


def _forward(
    model: AlphaRoco,
    encoder: ObservationEncoder,
    item: dict[str, Any],
    history: HistoryState | None,
    device: torch.device,
):
    batch = collate_inputs([item], device)
    return batch, model(
        batch,
        history=(
            None
            if history is None
            else history.batch(
                device,
                model.config.hidden_dim,
                item["opponent_candidates"],
            )
        ),
    )


def _sample_action(
    model: AlphaRoco,
    encoder: ObservationEncoder,
    request: dict[str, Any],
    history: HistoryState | None,
    device: torch.device,
    reward_config: RewardConfig,
    generator: torch.Generator | None = None,
):
    item = encoder.encode(request)
    summary = state_summary(request["state"], request["side"], reward_config)
    with torch.no_grad():
        batch, output = _forward(model, encoder, item, history, device)
        probabilities, log_probabilities = _masked_probabilities(
            output["logits"], batch["action_mask"]
        )
        action_index = torch.multinomial(probabilities, 1, generator=generator)
        action = selectable_candidates(request)[int(action_index.item())]
        old_log_prob = float(log_probabilities.gather(1, action_index).item())
        old_value = float(output["value"].reshape(-1)[0].item())
        if history is not None:
            history.observe(model, output, int(action_index.item()), request["phase"])
    return (
        action,
        history,
        {
            "item": item,
            "phase": request["phase"],
            "action": int(action_index.item()),
            "old_log_prob": old_log_prob,
            "old_value": old_value,
            "reward": 0.0,
            "terminal": False,
            "opponent_target": None,
            "opponent_action": None,
            "opponent_action_data": None,
            "opponent_distribution": None,
            "opponent_actions_before": [],
            "summary": summary,
            "potential": potential(summary, reward_config),
            "action_kind": action.get("kind", ""),
            "action_key": action_key(action),
            "exploration_bonus": 0.0,
            "trait_events": [],
        },
    )


OPPONENT_DISTRIBUTION_MIN_MASS = 0.5


def _map_opponent_distribution(candidates, distribution, keys):
    probabilities = dict(zip(keys, distribution))
    mapped = [
        probabilities.get(action_key(candidate), 0.0) for candidate in candidates
    ]
    total = sum(mapped)
    if total < OPPONENT_DISTRIBUTION_MIN_MASS:
        return None
    return [probability / total for probability in mapped]


def _paired_opponent_targets(
    requests: list[dict[str, Any]],
    actions_by_side: dict[str, dict[str, Any]],
    encoder: ObservationEncoder,
    entries_by_side: dict[str, dict[str, Any]],
    distributions_by_side: dict[str, tuple[list[float], list]] | None = None,
) -> None:
    requests_by_side = {request["side"]: request for request in requests}
    if set(requests_by_side) != {"A", "B"}:
        return
    for side, request in requests_by_side.items():
        if side not in entries_by_side:
            continue
        other_side = "B" if side == "A" else "A"
        other_request = requests_by_side[other_side]
        chosen = actions_by_side[other_side]
        entries_by_side[side]["opponent_action_data"] = chosen
        candidates = encoder.opponent_candidates(request)
        phase_matches = (
            request["phase"] == other_request["phase"]
            and request["phase"] != "replacement"
        )
        if (
            phase_matches
            and distributions_by_side is not None
            and other_side in distributions_by_side
        ):
            distribution, keys = distributions_by_side[other_side]
            mapped = _map_opponent_distribution(candidates, distribution, keys)
            if mapped is not None:
                entries_by_side[side]["opponent_distribution"] = mapped
        if chosen.get("kind") == "skill" and chosen.get("skill_id") is None:
            continue
        chosen_key = action_key(chosen)
        matches = [
            index
            for index, candidate in enumerate(candidates)
            if action_key(candidate) == chosen_key
        ]
        if matches:
            entries_by_side[side]["opponent_action"] = matches[0]
            if phase_matches:
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
    opponent_sample: bool = False,
    game_index: int | None = None,
) -> dict[str, Any]:
    episode: list[dict[str, Any]] = []
    histories = {"A": HistoryState(), "B": HistoryState()}
    generator = torch.Generator(device=device).manual_seed(seed)
    pending_public_actions: list[dict[str, Any]] = []
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
    with InProcessSession(
        team_a=team_a,
        team_b=team_b,
        seed=seed,
        timeout=timeout,
        game_index=game_index,
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
                            exploration.take_trait(
                                event.get("side"),
                                event.get("pet_index"),
                                event.get("trait_id"),
                            )
                            for event in new_trait_events
                            if event.get("side") == "A"
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
            requests_by_side: dict[str, dict[str, Any]] = {}
            distributions_by_side: dict[str, tuple[list[float], list]] = {}
            for request in requests:
                side = request["side"]
                requests_by_side[side] = request
                if side == "A":
                    action, histories[side], entry = _sample_action(
                        candidate,
                        encoder,
                        request,
                        histories[side],
                        device,
                        reward_config,
                        generator=generator,
                    )
                    entry["status"] = _is_status_skill(encoder, action)
                    entry["exploration_skill"] = _is_exploration_skill(encoder, action)
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
                    entry["opponent_actions_before"] = pending_public_actions
                    pending_public_actions = []
                    episode.append(entry)
                    entries_by_side[side] = entry
                else:
                    (
                        action,
                        histories[side],
                        distribution,
                        distribution_keys,
                    ) = choose(
                        opponent,
                        encoder,
                        request,
                        histories[side],
                        device,
                        sample=opponent_sample,
                        generator=generator,
                        return_distribution=True,
                    )
                    distributions_by_side[side] = (
                        distribution,
                        distribution_keys,
                    )
                actions_by_side[side] = action
            record_opponent_actions(
                histories,
                {"A": candidate, "B": opponent},
                encoder,
                requests_by_side,
                actions_by_side,
            )
            if "A" not in actions_by_side and "B" in actions_by_side:
                pending_public_actions.append(actions_by_side["B"])
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
                requests,
                actions_by_side,
                encoder,
                entries_by_side,
                distributions_by_side,
            )
            for side, action in actions_by_side.items():
                session.submit(side, action)
            last_a_entry = entries_by_side.get("A", last_a_entry)
        winner = session.winner
    if winner not in {"A", "B", "both_lose"}:
        raise RuntimeError(f"对局未得到有效终局结果：{winner!r}")
    if not episode:
        raise RuntimeError("candidate方本局没有决策记录")
    assign_rewards(episode, winner, gamma, reward_config)
    apply_repeat_penalty(episode, reward_config)
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


def _duration_vector(
    durations: dict[tuple[str, str, str], float], vocab: dict, size: int
) -> list[float]:
    vector = [0.0] * size
    for (_, effect_type, _), code in durations.items():
        index = vocab.get(effect_type, vocab.get(str(effect_type), 1))
        if index >= 3:
            vector[index] += code / 2.0
    return vector


def _weather_vector(weather, vocab: dict, size: int) -> list[float]:
    vector = [0.0] * size
    index = 2 if weather is None else vocab.get(weather, vocab.get(str(weather), 1))
    vector[index] = 1.0
    return vector


def _effect_delta(
    before: dict, after: dict, encoder: ObservationEncoder
) -> list[float]:
    effect_size = encoder.vocab_sizes["effect"]
    mark_size = encoder.vocab_sizes["mark"]
    weather_size = encoder.vocab_sizes["weather"]
    own_effect = _effect_vector(after["own_effects"], encoder.effect_vocab, effect_size)
    enemy_effect = _effect_vector(
        after["enemy_effects"], encoder.effect_vocab, effect_size
    )
    before_own = _effect_vector(
        before["own_effects"], encoder.effect_vocab, effect_size
    )
    before_enemy = _effect_vector(
        before["enemy_effects"], encoder.effect_vocab, effect_size
    )
    own_duration = _duration_vector(
        after["own_effect_durations"], encoder.effect_vocab, effect_size
    )
    enemy_duration = _duration_vector(
        after["enemy_effect_durations"], encoder.effect_vocab, effect_size
    )
    before_own_duration = _duration_vector(
        before["own_effect_durations"], encoder.effect_vocab, effect_size
    )
    before_enemy_duration = _duration_vector(
        before["enemy_effect_durations"], encoder.effect_vocab, effect_size
    )
    own_mark = _mark_vector(after["own_marks"], encoder.mark_vocab, mark_size)
    enemy_mark = _mark_vector(after["enemy_marks"], encoder.mark_vocab, mark_size)
    before_own_mark = _mark_vector(before["own_marks"], encoder.mark_vocab, mark_size)
    before_enemy_mark = _mark_vector(
        before["enemy_marks"], encoder.mark_vocab, mark_size
    )
    weather = _weather_vector(after["weather"], encoder.weather_vocab, weather_size)
    before_weather = _weather_vector(
        before["weather"], encoder.weather_vocab, weather_size
    )
    return [
        *(current - previous for current, previous in zip(own_effect, before_own)),
        *(current - previous for current, previous in zip(enemy_effect, before_enemy)),
        *(
            current - previous
            for current, previous in zip(own_duration, before_own_duration)
        ),
        *(
            current - previous
            for current, previous in zip(enemy_duration, before_enemy_duration)
        ),
        *(current - previous for current, previous in zip(own_mark, before_own_mark)),
        *(
            current - previous
            for current, previous in zip(enemy_mark, before_enemy_mark)
        ),
        (after["own_energy"] - before["own_energy"]) / 10.0,
        (after["enemy_energy"] - before["enemy_energy"]) / 10.0,
        *(current - previous for current, previous in zip(weather, before_weather)),
    ]


def _applied_effect_targets(
    before: dict, after: dict, encoder: ObservationEncoder
) -> tuple[list[list[float]], list[float], bool]:
    kinds = [[0.0] * encoder.vocab_sizes["effect"] for _ in range(2)]
    targets = [0.0] * 12
    present = False
    for side, (stacks_key, names_key, offset) in enumerate(
        (
            ("own_effect_stacks", "own_pet_names", 0),
            ("enemy_effect_stacks", "enemy_pet_names", 6),
        )
    ):
        previous = before.get(stacks_key, {})
        names = before.get(names_key, ())
        for key, value in after.get(stacks_key, {}).items():
            name, effect_type, source_kind, _, source_side = key
            if source_kind or source_side != "A":
                continue
            if value <= previous.get(key, 0.0):
                continue
            index = encoder.effect_vocab.get(
                effect_type, encoder.effect_vocab.get(str(effect_type), 1)
            )
            if index >= 3:
                kinds[side][index] = 1.0
            if name and names.count(name) == 1:
                pet_index = names.index(name)
                if pet_index < 6:
                    targets[offset + pet_index] = 1.0
                    present = True
    return kinds, targets, present


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


def _assign_aux_targets(
    episode: list[dict[str, Any]], encoder: ObservationEncoder
) -> None:
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
        pet_delta = []
        pet_delta_mask = []
        outcome_target = []
        outcome_mask = []
        for prefix, own in (
            ("own_pet_attributes", True),
            ("enemy_pet_attributes", False),
        ):
            before_states = before.get(prefix, [])
            after_states = after.get(prefix, [])
            for index in range(6):
                if index < len(before_states) and index < len(after_states):
                    outcome_target.extend(
                        current - previous
                        for current, previous in zip(
                            after_states[index], before_states[index]
                        )
                    )
                    outcome_mask.append(True)
                else:
                    outcome_target.extend([0.0] * len(PET_STATE_ATTRIBUTE_KEYS))
                    outcome_mask.append(False)
        for prefix, own in (("own_pet_states", True), ("enemy_pet_states", False)):
            before_states = before.get(prefix, [])
            after_states = after.get(prefix, [])
            for index in range(6):
                if index < len(before_states) and index < len(after_states):
                    pet_delta.extend(
                        current - previous
                        for current, previous in zip(
                            after_states[index], before_states[index]
                        )
                    )
                    pet_delta_mask.append(True)
                else:
                    pet_delta.extend([0.0] * PET_STATE_DIM)
                    pet_delta_mask.append(False)
        pet_trait_trigger = [0.0] * 12
        trait_target_mask = [[0.0] * 12 for _ in range(12)]
        unresolved = set()
        for event in entry.get("trait_events", []):
            side = event.get("side")
            index = event.get("pet_index")
            if side not in ("A", "B") or not isinstance(index, int):
                continue
            if not 0 <= index < 6:
                continue
            source = (0 if side == "A" else 6) + index
            pet_trait_trigger[source] = 1.0
            target_side = event.get("target_side")
            target_index = event.get("target_index")
            if target_side not in ("A", "B") or not isinstance(target_index, int):
                if event.get("event") == "buff_gain":
                    target_side = event.get("subject_side")
                    target_index = event.get("subject_index")
                else:
                    unresolved.add(source)
                    continue
            if target_side not in ("A", "B") or not isinstance(target_index, int):
                unresolved.add(source)
                continue
            if 0 <= target_index < 6:
                trait_target_mask[source][
                    (0 if target_side == "A" else 6) + target_index
                ] = 1.0
        for source_side, source_index, target_side, target_index, _ in (
            set(after.get("trait_buff_targets", ()))
            - set(before.get("trait_buff_targets", ()))
        ):
            if source_side not in ("A", "B") or target_side not in ("A", "B"):
                continue
            if not (0 <= source_index < 6 and 0 <= target_index < 6):
                continue
            trait_target_mask[(0 if source_side == "A" else 6) + source_index][
                (0 if target_side == "A" else 6) + target_index
            ] = 1.0
        for source in unresolved:
            if not any(trait_target_mask[source]):
                trait_target_mask[source][source] = 1.0
        status_target_mask = encoder.candidate_target_mask(
            entry["item"], entry.get("action", -1)
        )
        outcome_target_mask = (
            list(status_target_mask) if status_target_mask is not None else [0.0] * 12
        )
        outcome_target_present = status_target_mask is not None
        if not outcome_target_present:
            candidates = entry["item"].get("own_candidates", [])
            action = entry.get("action", -1)
            if 0 <= action < len(candidates):
                candidate = candidates[action]
                skill_id = candidate.get("skill_id")
                category = encoder.skill_by_id.get(str(skill_id), {}).get("category")
                enemy_active = entry["item"].get("enemy_active", -1)
                if category in (0, 1) and 0 <= enemy_active < 6:
                    outcome_target_mask[6 + enemy_active] = 1.0
                    outcome_target_present = True
                pet_index = int(candidate.get("pet_index", -1))
                if (
                    not outcome_target_present
                    and 0 <= pet_index < 12
                    and any(trait_target_mask[pet_index])
                ):
                    outcome_target_mask = list(trait_target_mask[pet_index])
                    outcome_target_present = True
        applied_kind, applied_target, applied_target_present = (
            _applied_effect_targets(before, after, encoder)
        )
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
            "pet_trait_trigger": pet_trait_trigger,
            "trait_target_mask": trait_target_mask,
            "status_target_mask": status_target_mask or [0.0] * 12,
            "status_target_present": status_target_mask is not None,
            "pet_delta": pet_delta,
            "pet_delta_mask": pet_delta_mask,
            "outcome_target": outcome_target,
            "outcome_mask": outcome_mask,
            "outcome_target_mask": outcome_target_mask,
            "outcome_target_present": outcome_target_present,
            "applied_effect_kind": applied_kind,
            "applied_effect_target": applied_target,
            "applied_effect_target_present": applied_target_present,
        }
    effect_values = [effect_value(entry["summary"]) for entry in battle_entries]
    for index, entry in enumerate(battle_entries):
        entry["effect_value_target"] = (
            effect_values[index + 1] if index + 1 < len(effect_values) else 0.0
        ) - effect_values[index]
    lead_entry = next(
        (entry for entry in episode if entry["phase"] == "lead"), None
    )
    if lead_entry is not None and battle_entries:
        lead_entry["lead_advantage"] = battle_entries[0]["potential"]


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
    history = HistoryState()
    for index, entry in enumerate(episode):
        if index:
            history.commit_public(
                model,
                episode[index - 1]["opponent_action"],
                episode[index - 1].get("opponent_action_data"),
            )
        for action in entry.get("opponent_actions_before", ()):
            history.commit_public(model, None, action)
        batch, output = _forward(model, encoder, entry["item"], history, device)
        history.observe(model, output, entry["action"], entry["phase"])
        yield entry, batch, output


def _batched_episode_steps(
    model: AlphaRoco,
    episodes: list[list[dict[str, Any]]],
    device: torch.device,
    teacher: AlphaRoco | None = None,
):
    histories = [HistoryState() for _ in episodes]
    teacher_histories = [HistoryState() for _ in episodes] if teacher else None
    max_steps = max((len(episode) for episode in episodes), default=0)
    for step in range(max_steps):
        active = [
            index for index, episode in enumerate(episodes) if step < len(episode)
        ]
        if not active:
            break
        for index in active:
            if step:
                histories[index].commit_public(
                    model,
                    episodes[index][step - 1]["opponent_action"],
                    episodes[index][step - 1].get("opponent_action_data"),
                )
                if teacher_histories is not None:
                    teacher_histories[index].commit_public(
                        teacher,
                        episodes[index][step - 1]["opponent_action"],
                        episodes[index][step - 1].get("opponent_action_data"),
                    )
            for action in episodes[index][step].get("opponent_actions_before", ()):
                histories[index].commit_public(model, None, action)
                if teacher_histories is not None:
                    teacher_histories[index].commit_public(teacher, None, action)
        batch = collate_inputs(
            [episodes[index][step]["item"] for index in active], device
        )
        batch["opponent_action_index"] = torch.tensor(
            [
                -1
                if episodes[index][step]["opponent_action"] is None
                else episodes[index][step]["opponent_action"]
                for index in active
            ],
            dtype=torch.long,
            device=device,
        )
        output = model(
            batch,
            history=collate_histories(
                [histories[index] for index in active],
                device,
                model.config.hidden_dim,
                [
                    episodes[index][step]["item"]["opponent_candidates"]
                    for index in active
                ],
            ),
        )
        teacher_logits = None
        if teacher_histories is not None:
            with torch.no_grad():
                teacher_output = teacher(
                    batch,
                    history=collate_histories(
                        [teacher_histories[index] for index in active],
                        device,
                        teacher.config.hidden_dim,
                        [
                            episodes[index][step]["item"]["opponent_candidates"]
                            for index in active
                        ],
                    ),
                )
                teacher_logits = teacher_output["logits"]
            for position, index in enumerate(active):
                entry = episodes[index][step]
                teacher_histories[index].observe(
                    teacher, teacher_output, entry["action"], entry["phase"], position
                )
        for position, index in enumerate(active):
            entry = episodes[index][step]
            histories[index].observe(
                model, output, entry["action"], entry["phase"], position
            )
        if teacher_logits is not None:
            output["teacher_logits"] = teacher_logits
        yield step, active, batch, output


def _policy_update(
    model: AlphaRoco,
    optimizer: torch.optim.Optimizer,
    encoder: ObservationEncoder,
    episodes: list[list[dict[str, Any]]],
    config: TrainConfig,
    ppo_epochs: int,
    device: torch.device,
    teacher: AlphaRoco | None = None,
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
        "teacher_kl": 0.0,
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
        "bloodline": 0.0,
        "bloodline_baseline": 0.0,
        "resonance": 0.0,
        "resonance_baseline": 0.0,
        "pet_delta": 0.0,
        "pet_delta_baseline": 0.0,
        "pet_trait_trigger": 0.0,
        "pet_trait_trigger_baseline": 0.0,
        "trait_target": 0.0,
        "trait_target_baseline": 0.0,
        "trait_target_delta": 0.0,
        "trait_target_delta_baseline": 0.0,
        "trait_target_own_enemy": 0.0,
        "trait_target_own_enemy_baseline": 0.0,
        "trait_target_enemy_own": 0.0,
        "trait_target_enemy_own_baseline": 0.0,
        "status_target": 0.0,
        "status_target_baseline": 0.0,
        "status_target_enemy": 0.0,
        "status_target_enemy_baseline": 0.0,
        "status_target_ally": 0.0,
        "status_target_ally_baseline": 0.0,
        "trait_pet_delta": 0.0,
        "trait_pet_delta_baseline": 0.0,
        "candidate_trait_trigger": 0.0,
        "candidate_trait_trigger_baseline": 0.0,
        "candidate_trait_target": 0.0,
        "candidate_trait_target_baseline": 0.0,
        "effect_value": 0.0,
        "effect_value_baseline": 0.0,
        "effect_kind": 0.0,
        "effect_kind_baseline": 0.0,
        "effect_target": 0.0,
        "effect_target_baseline": 0.0,
        "effect_kind_accuracy": 0.0,
        "effect_kind_accuracy_baseline": 0.0,
        "effect_kind_precision": 0.0,
        "effect_kind_precision_baseline": 0.0,
        "candidate_pet_delta": 0.0,
        "candidate_pet_delta_baseline": 0.0,
        "outcome_delta": 0.0,
        "outcome_delta_baseline": 0.0,
        "outcome_target": 0.0,
        "outcome_target_baseline": 0.0,
        "joint": 0.0,
        "joint_baseline": 0.0,
        "lead_advantage": 0.0,
        "lead_advantage_baseline": 0.0,
        "grad_norm": 0.0,
    }
    aux_count = sum(
        1
        for episode in episodes
        for entry in episode
        if entry.get("aux_target") is not None
    )
    effect_count = sum(
        1
        for episode in episodes
        for entry in episode
        if entry.get("effect_value_target") is not None
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
                "teacher_kl",
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
                "bloodline",
                "bloodline_baseline",
                "resonance",
                "resonance_baseline",
                "pet_delta",
                "pet_delta_baseline",
                "pet_trait_trigger",
                "pet_trait_trigger_baseline",
                "trait_target",
                "trait_target_baseline",
                "trait_target_delta",
                "trait_target_delta_baseline",
                "trait_target_own_enemy",
                "trait_target_own_enemy_baseline",
                "trait_target_enemy_own",
                "trait_target_enemy_own_baseline",
                "status_target",
                "status_target_baseline",
                "status_target_enemy",
                "status_target_enemy_baseline",
                "status_target_ally",
                "status_target_ally_baseline",
                "trait_pet_delta",
                "trait_pet_delta_baseline",
                "candidate_trait_trigger",
                "candidate_trait_trigger_baseline",
                "candidate_trait_target",
                "candidate_trait_target_baseline",
                "effect_value",
                "effect_value_baseline",
                "effect_kind",
                "effect_kind_baseline",
                "effect_target",
                "effect_target_baseline",
                "effect_kind_accuracy",
                "effect_kind_accuracy_baseline",
                "effect_kind_precision",
                "effect_kind_precision_baseline",
                "candidate_pet_delta",
                "candidate_pet_delta_baseline",
                "outcome_delta",
                "outcome_delta_baseline",
                "outcome_target",
                "outcome_target_baseline",
                "joint",
                "joint_baseline",
                "lead_advantage",
                "lead_advantage_baseline",
            )
        }
        for start in range(0, len(episodes), config.update_chunk):
            chunk = episodes[start : start + config.update_chunk]
            losses: list[torch.Tensor] = []
            steps_total = max((len(episode) for episode in chunk), default=1)
            for step, active, batch, output in _batched_episode_steps(
                model, chunk, device, teacher
            ):
                teacher_logits = output.get("teacher_logits")
                entries = [chunk[index][step] for index in active]
                phases = [
                    "lead" if entry["phase"] == "lead" else "battle"
                    for entry in entries
                ]
                if teacher_logits is not None:
                    mask = batch["action_mask"]
                    teacher_probs = teacher_logits.masked_fill(
                        ~mask, -torch.inf
                    ).softmax(dim=-1)
                    teacher_log = teacher_probs.clamp_min(1e-8).log()
                    student_log = output["logits"].masked_fill(
                        ~mask, -torch.inf
                    ).log_softmax(dim=-1)
                    teacher_log = teacher_log.masked_fill(~mask, 0.0)
                    student_log = student_log.masked_fill(~mask, 0.0)
                    teacher_kl = (
                        teacher_probs * (teacher_log - student_log)
                    ).sum(dim=-1).mean()
                    scale = len(active) / max(len(chunk) * steps_total, 1)
                    losses.append(
                        teacher_kl * config.teacher_coefficient * scale
                    )
                    sums["teacher_kl"] += (teacher_kl).detach() * scale
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
                        sums[key] += (result[key]).detach() * scale
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
                    sums["damage"] += (damage).detach() * scale
                    sums["damage_baseline"] += (
                        (damage_baseline).detach() * scale
                    )
                    sums["ko"] += (ko).detach() * scale
                    sums["ko_baseline"] += (ko_baseline).detach() * scale
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
                    supervised_mask = torch.tensor(
                        [
                            entries[position]["action_kind"] in ("skill", "charge")
                            for position in aux_positions
                        ],
                        dtype=torch.bool,
                        device=device,
                    )
                    effect_weights = torch.ones(effect_targets.shape[1], device=device)
                    if supervised_mask.any():
                        positions = [
                            aux_positions[index]
                            for index in range(len(aux_positions))
                            if bool(supervised_mask[index])
                        ]
                        actions = torch.tensor(
                            [entries[position]["action"] for position in positions],
                            device=device,
                        )
                        effect_source = torch.cat(
                            (
                                output["effect_context"][positions],
                                output["action_embeddings"][positions, actions],
                            ),
                            dim=-1,
                        )
                        effect_pred = model.aux_effect_delta(effect_source)
                        effect_target = effect_targets[supervised_mask]
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
                                (effect).detach() * scale
                            )
                            sums["effect_delta_baseline"] += (
                                (effect_baseline).detach() * scale
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
                        losses.append(change * config.effect_aux_coefficient * scale)
                        sums["state_change"] += (change).detach() * scale
                        sums["state_change_baseline"] += (
                            (change_baseline).detach() * scale
                        )
                    trait_logits = model.aux_trait_trigger(context)[trait_presence]
                    trigger_target = trait_targets[trait_presence]
                    trigger = F.binary_cross_entropy_with_logits(
                        trait_logits, trigger_target
                    )
                    trigger_baseline = F.binary_cross_entropy_with_logits(
                        torch.zeros_like(trigger_target), trigger_target
                    )
                    losses.append(trigger * config.trait_aux_coefficient * scale)
                    sums["trait_trigger"] += (trigger).detach() * scale
                    sums["trait_trigger_baseline"] += (
                        (trigger_baseline).detach() * scale
                    )
                    pet_trait_targets = torch.tensor(
                        [target["pet_trait_trigger"] for target in targets],
                        dtype=torch.float32,
                        device=device,
                    )
                    pet_presence = torch.cat(
                        (
                            batch["own_pet_mask"][:, :6],
                            batch["enemy_pet_mask"][:, :6],
                        ),
                        dim=1,
                    )[aux_positions]
                    pet_trait_logits = output["aux"]["pet_trait_trigger"][
                        aux_positions
                    ]
                    if pet_presence.any():
                        trigger_pets = F.binary_cross_entropy_with_logits(
                            pet_trait_logits[pet_presence],
                            pet_trait_targets[pet_presence],
                        )
                        trigger_pets_baseline = F.binary_cross_entropy_with_logits(
                            torch.zeros_like(pet_trait_logits[pet_presence]),
                            pet_trait_targets[pet_presence],
                        )
                        losses.append(
                            trigger_pets * config.trait_aux_coefficient * scale
                        )
                        sums["pet_trait_trigger"] += (
                            (trigger_pets).detach() * scale
                        )
                        sums["pet_trait_trigger_baseline"] += (
                            (trigger_pets_baseline).detach() * scale
                        )
                    trait_mask = trait_targets.sum(dim=1) > 0
                    if trait_mask.any():
                        trait_weights = trait_targets[trait_mask]
                        trait_embedding = (
                            trait_weights @ model.trait_embedding.weight
                        ) / trait_weights.sum(dim=1, keepdim=True).clamp_min(1)
                        trait_embedding = model.aux_trait_source(trait_embedding)
                        trait_pred = model.aux_effect_delta(
                            torch.cat((context[trait_mask], trait_embedding), dim=-1)
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
                                trait_effect * config.trait_aux_coefficient * scale
                            )
                            sums["trait_effect"] += (
                                (trait_effect).detach() * scale
                            )
                            sums["trait_effect_baseline"] += (
                                (trait_effect_baseline).detach() * scale
                            )
                bloodline_mask = batch["enemy_bloodline_mask"][:, :6]
                if bloodline_mask.any():
                    bloodline_targets = batch["enemy_bloodline_targets"][:, :6][
                        bloodline_mask
                    ]
                    bloodline_pred = output["opponent_bloodline_logits"][bloodline_mask]
                    bloodline = F.cross_entropy(bloodline_pred, bloodline_targets)
                    bloodline_baseline = F.cross_entropy(
                        torch.zeros_like(bloodline_pred), bloodline_targets
                    )
                    scale = len(active) / max(len(chunk) * steps_total, 1)
                    losses.append(bloodline * config.bloodline_aux_coefficient * scale)
                    sums["bloodline"] += (bloodline).detach() * scale
                    sums["bloodline_baseline"] += (
                        (bloodline_baseline).detach() * scale
                    )
                resonance_mask = batch["enemy_resonance_mask"]
                if resonance_mask.any():
                    resonance_targets = batch["enemy_resonance_target"][resonance_mask]
                    resonance_pred = output["opponent_resonance_logits"][resonance_mask]
                    resonance = F.cross_entropy(resonance_pred, resonance_targets)
                    resonance_baseline = F.cross_entropy(
                        torch.zeros_like(resonance_pred), resonance_targets
                    )
                    scale = len(active) / max(len(chunk) * steps_total, 1)
                    losses.append(resonance * config.resonance_aux_coefficient * scale)
                    sums["resonance"] += (resonance).detach() * scale
                    sums["resonance_baseline"] += (
                        (resonance_baseline).detach() * scale
                    )
                joint_index = batch["enemy_active"].clamp(
                    0, batch["enemy_bloodline_targets"].shape[1] - 1
                )
                joint_bloodline = batch["enemy_bloodline_targets"].gather(
                    1, joint_index.unsqueeze(1)
                ).squeeze(1)
                joint_mask = batch["enemy_bloodline_mask"].gather(
                    1, joint_index.unsqueeze(1)
                ).squeeze(1) & batch["enemy_resonance_mask"]
                if joint_mask.any():
                    joint_targets = (
                        joint_bloodline[joint_mask] * 3
                        + batch["enemy_resonance_target"][joint_mask]
                    )
                    joint_pred = output["opponent_joint_logits"][joint_mask]
                    joint = F.cross_entropy(joint_pred, joint_targets)
                    joint_baseline = F.cross_entropy(
                        torch.zeros_like(joint_pred), joint_targets
                    )
                    scale = len(active) / max(len(chunk) * steps_total, 1)
                    losses.append(joint * config.joint_aux_coefficient * scale)
                    sums["joint"] += (joint).detach() * scale
                    sums["joint_baseline"] += (
                        (joint_baseline).detach() * scale
                    )
                if aux_positions:
                    pet_target = torch.tensor(
                        [target["pet_delta"] for target in targets],
                        dtype=torch.float32,
                        device=device,
                    ).reshape(len(targets), 12, PET_DELTA_DIM)
                    pet_mask = torch.tensor(
                        [target["pet_delta_mask"] for target in targets],
                        dtype=torch.bool,
                        device=device,
                    ).reshape(len(targets), 12)
                    pet_pred = torch.cat(
                        (
                            output["aux"]["pet_delta_own"][aux_positions],
                            output["aux"]["pet_delta_enemy"][aux_positions],
                        ),
                        dim=1,
                    )[:, :12]
                    pet_error = (
                        (pet_pred - pet_target).pow(2)
                        * pet_mask.unsqueeze(-1)
                    )
                    pet_delta = pet_error.sum() / (
                        pet_mask.unsqueeze(-1).sum().clamp_min(1)
                    )
                    pet_delta_baseline = (
                        pet_target.pow(2) * pet_mask.unsqueeze(-1)
                    ).sum() / (pet_mask.unsqueeze(-1).sum().clamp_min(1))
                    scale = len(aux_positions) / max(aux_count, 1)
                    losses.append(
                        pet_delta * config.pet_delta_aux_coefficient * scale
                    )
                    sums["pet_delta"] += (
                        (pet_delta).detach() * scale
                    )
                    sums["pet_delta_baseline"] += (
                        (pet_delta_baseline).detach() * scale
                    )
                    trait_target_mask = torch.tensor(
                        [target["trait_target_mask"] for target in targets],
                        dtype=torch.bool,
                        device=device,
                    ).reshape(len(targets), 12, 12)
                    actor_mask = pet_trait_targets.bool() & pet_presence
                    if actor_mask.any():
                        target_logits = output["aux"]["trait_target_logits"][
                            aux_positions
                        ]
                        target_select = F.binary_cross_entropy_with_logits(
                            target_logits[actor_mask],
                            trait_target_mask[actor_mask].float(),
                        )
                        target_select_baseline = F.binary_cross_entropy_with_logits(
                            torch.zeros_like(target_logits[actor_mask]),
                            trait_target_mask[actor_mask].float(),
                        )
                        losses.append(
                            target_select * config.trait_aux_coefficient * scale
                        )
                        sums["trait_target"] += (
                            (target_select).detach() * scale
                        )
                        sums["trait_target_baseline"] += (
                            (target_select_baseline).detach() * scale
                        )
                        pair_pred = output["aux"]["trait_target_delta"][aux_positions]
                        pair_target = pet_target[:, None, :, :].expand(
                            -1, 12, -1, -1
                        )
                        pair_mask = (
                            trait_target_mask
                            & actor_mask.unsqueeze(-1)
                            & pet_presence[:, None, :]
                        )
                        if pair_mask.any():
                            pair_error = (
                                (pair_pred - pair_target).pow(2)
                                * pair_mask.unsqueeze(-1)
                            )
                            target_delta = pair_error.sum() / (
                                pair_mask.unsqueeze(-1).sum().clamp_min(1)
                            )
                            target_delta_baseline = (
                                pair_target.pow(2) * pair_mask.unsqueeze(-1)
                            ).sum() / (
                                pair_mask.unsqueeze(-1).sum().clamp_min(1)
                            )
                            losses.append(
                                target_delta
                                * config.trait_aux_coefficient
                                * scale
                            )
                            sums["trait_target_delta"] += (
                                (target_delta).detach() * scale
                            )
                            sums["trait_target_delta_baseline"] += (
                                (target_delta_baseline).detach() * scale
                            )
                            source_own = torch.arange(12, device=device) < 6
                            target_own = source_own
                            for key, direction in (
                                (
                                    "trait_target_own_enemy",
                                    source_own[:, None] & ~target_own[None, :],
                                ),
                                (
                                    "trait_target_enemy_own",
                                    ~source_own[:, None] & target_own[None, :],
                                ),
                            ):
                                directed = pair_mask & direction
                                if not directed.any():
                                    continue
                                denominator = directed.unsqueeze(-1).sum().clamp_min(1)
                                value = (
                                    (pair_pred - pair_target).pow(2)
                                    * directed.unsqueeze(-1)
                                ).sum() / denominator
                                baseline = (
                                    pair_target.pow(2) * directed.unsqueeze(-1)
                                ).sum() / denominator
                                sums[key] += (value).detach() * scale
                                sums[f"{key}_baseline"] += (
                                    (baseline).detach() * scale
                                )
                    selected = torch.tensor(
                        [entries[position]["action"] for position in aux_positions],
                        device=device,
                    )
                    applied_kind_target = torch.tensor(
                        [target["applied_effect_kind"] for target in targets],
                        dtype=torch.float32,
                        device=device,
                    ).reshape(len(targets), 2, -1)
                    applied_kind_pred = output["aux"]["effect_kind_logits"][
                        aux_positions
                    ][
                        torch.arange(len(aux_positions), device=device), selected
                    ]
                    kind_frequency = applied_kind_target.float().mean(
                        dim=0
                    ).clamp_min(1e-3)
                    kind_weight = (
                        (1.0 - kind_frequency) / kind_frequency
                    ).clamp(1.0, config.effect_kind_pos_weight)
                    effect_kind = F.binary_cross_entropy_with_logits(
                        applied_kind_pred,
                        applied_kind_target,
                        pos_weight=kind_weight,
                    )
                    effect_kind_baseline = F.binary_cross_entropy_with_logits(
                        torch.zeros_like(applied_kind_pred),
                        applied_kind_target,
                        pos_weight=kind_weight,
                    )
                    losses.append(
                        effect_kind * config.effect_aux_coefficient * scale
                    )
                    sums["effect_kind"] += (effect_kind).detach() * scale
                    sums["effect_kind_baseline"] += (
                        (effect_kind_baseline).detach() * scale
                    )
                    with torch.no_grad():
                        positive = applied_kind_pred[..., 3:] > 0
                        target_bool = applied_kind_target[..., 3:].bool()
                        accuracy = (positive == target_bool).float().mean()
                        precision = (positive & target_bool).sum().float() / (
                            positive.sum().clamp_min(1).float()
                        )
                        accuracy_baseline = (~target_bool).float().mean()
                    sums["effect_kind_accuracy"] += accuracy * scale
                    sums["effect_kind_precision"] += precision * scale
                    sums["effect_kind_accuracy_baseline"] += (
                        accuracy_baseline * scale
                    )
                    applied_target_target = torch.tensor(
                        [target["applied_effect_target"] for target in targets],
                        dtype=torch.float32,
                        device=device,
                    )
                    applied_target_present = torch.tensor(
                        [
                            target["applied_effect_target_present"]
                            for target in targets
                        ],
                        dtype=torch.bool,
                        device=device,
                    )
                    applied_target_pred = output["aux"]["effect_target_logits"][
                        aux_positions
                    ][
                        torch.arange(len(aux_positions), device=device), selected
                    ]
                    if applied_target_present.any():
                        applied_target_loss = F.binary_cross_entropy_with_logits(
                            applied_target_pred[applied_target_present],
                            applied_target_target[applied_target_present],
                        )
                        applied_target_baseline = F.binary_cross_entropy_with_logits(
                            torch.zeros_like(
                                applied_target_pred[applied_target_present]
                            ),
                            applied_target_target[applied_target_present],
                        )
                        losses.append(
                            applied_target_loss
                            * config.effect_aux_coefficient
                            * scale
                        )
                        sums["effect_target"] += (
                            (applied_target_loss).detach() * scale
                        )
                        sums["effect_target_baseline"] += (
                            (applied_target_baseline).detach() * scale
                        )
                    candidate_trigger_pred = output["aux"][
                        "candidate_trait_trigger"
                    ][aux_positions][
                        torch.arange(len(aux_positions), device=device), selected
                    ]
                    if pet_presence.any():
                        candidate_trigger = F.binary_cross_entropy_with_logits(
                            candidate_trigger_pred[pet_presence],
                            pet_trait_targets[pet_presence],
                        )
                        candidate_trigger_baseline = (
                            F.binary_cross_entropy_with_logits(
                                torch.zeros_like(candidate_trigger_pred[pet_presence]),
                                pet_trait_targets[pet_presence],
                            )
                        )
                        losses.append(
                            candidate_trigger * config.trait_aux_coefficient * scale
                        )
                        sums["candidate_trait_trigger"] += (
                            (candidate_trigger).detach() * scale
                        )
                        sums["candidate_trait_trigger_baseline"] += (
                            (candidate_trigger_baseline).detach() * scale
                        )
                    triggered = pet_trait_targets.bool() & pet_presence
                    triggered_rows = triggered.any(dim=1)
                    if triggered_rows.any():
                        target_distribution = (
                            trait_target_mask.float()
                            * triggered.float().unsqueeze(-1)
                        ).amax(dim=1)
                        target_distribution = target_distribution / (
                            target_distribution.sum(dim=-1, keepdim=True).clamp_min(
                                1.0
                            )
                        )
                        candidate_target_pred = output["aux"][
                            "candidate_trait_target"
                        ][aux_positions][
                            torch.arange(len(aux_positions), device=device), selected
                        ]
                        candidate_target = F.binary_cross_entropy_with_logits(
                            candidate_target_pred[triggered_rows],
                            target_distribution[triggered_rows],
                        )
                        candidate_target_baseline = (
                            F.binary_cross_entropy_with_logits(
                                torch.zeros_like(candidate_target_pred[triggered_rows]),
                                target_distribution[triggered_rows],
                            )
                        )
                        losses.append(
                            candidate_target * config.trait_aux_coefficient * scale
                        )
                        sums["candidate_trait_target"] += (
                            (candidate_target).detach() * scale
                        )
                        sums["candidate_trait_target_baseline"] += (
                            (candidate_target_baseline).detach() * scale
                        )
                    outcome_target = torch.tensor(
                        [target["outcome_target"] for target in targets],
                        dtype=torch.float32,
                        device=device,
                    ).reshape(len(targets), 12, len(PET_STATE_ATTRIBUTE_KEYS))
                    outcome_mask = torch.tensor(
                        [target["outcome_mask"] for target in targets],
                        dtype=torch.bool,
                        device=device,
                    ).reshape(len(targets), 12)
                    outcome_pred = output["aux"]["outcome_delta"][aux_positions][
                        torch.arange(len(aux_positions), device=device), selected
                    ]
                    outcome_error = (
                        (outcome_pred - outcome_target).pow(2)
                        * outcome_mask.unsqueeze(-1)
                    )
                    outcome_delta = outcome_error.sum() / (
                        outcome_mask.unsqueeze(-1).sum().clamp_min(1)
                    )
                    outcome_baseline = (
                        outcome_target.pow(2) * outcome_mask.unsqueeze(-1)
                    ).sum() / (outcome_mask.unsqueeze(-1).sum().clamp_min(1))
                    losses.append(
                        outcome_delta * config.outcome_aux_coefficient * scale
                    )
                    sums["outcome_delta"] += (
                        (outcome_delta).detach() * scale
                    )
                    sums["outcome_delta_baseline"] += (
                        (outcome_baseline).detach() * scale
                    )
                    outcome_target_mask = torch.tensor(
                        [target["outcome_target_mask"] for target in targets],
                        dtype=torch.float32,
                        device=device,
                    )
                    outcome_target_present = torch.tensor(
                        [target["outcome_target_present"] for target in targets],
                        dtype=torch.bool,
                        device=device,
                    )
                    outcome_target_logits = output["aux"]["outcome_target_logits"][
                        aux_positions
                    ][torch.arange(len(aux_positions), device=device), selected]
                    if outcome_target_present.any():
                        target_logits = outcome_target_logits[outcome_target_present]
                        target_values = outcome_target_mask[outcome_target_present]
                        outcome_target_loss = F.binary_cross_entropy_with_logits(
                            target_logits, target_values
                        )
                        outcome_target_baseline = F.binary_cross_entropy_with_logits(
                            torch.zeros_like(target_logits), target_values
                        )
                        losses.append(
                            outcome_target_loss
                            * config.outcome_aux_coefficient
                            * scale
                        )
                        sums["outcome_target"] += (
                            (outcome_target_loss).detach() * scale
                        )
                        sums["outcome_target_baseline"] += (
                            (outcome_target_baseline).detach() * scale
                        )
                    status_target_mask = torch.tensor(
                        [target["status_target_mask"] for target in targets],
                        dtype=torch.float32,
                        device=device,
                    )
                    status_present = torch.tensor(
                        [target["status_target_present"] for target in targets],
                        dtype=torch.bool,
                        device=device,
                    )
                    status_logits = output["aux"]["candidate_target_logits"][
                        aux_positions
                    ][torch.arange(len(aux_positions), device=device), selected]
                    if status_present.any():
                        status_pred = status_logits[status_present]
                        status_target = status_target_mask[status_present]
                        status_loss = F.binary_cross_entropy_with_logits(
                            status_pred, status_target
                        )
                        status_baseline = F.binary_cross_entropy_with_logits(
                            torch.zeros_like(status_pred), status_target
                        )
                        losses.append(
                            status_loss
                            * config.status_target_aux_coefficient
                            * scale
                        )
                        sums["status_target"] += (
                            (status_loss).detach() * scale
                        )
                        sums["status_target_baseline"] += (
                            (status_baseline).detach() * scale
                        )
                        for key, columns in (
                            ("status_target_ally", slice(0, 6)),
                            ("status_target_enemy", slice(6, 12)),
                        ):
                            values = status_target[:, columns]
                            if not values.any():
                                continue
                            value = F.binary_cross_entropy_with_logits(
                                status_pred[:, columns], values
                            )
                            baseline = F.binary_cross_entropy_with_logits(
                                torch.zeros_like(status_pred[:, columns]), values
                            )
                            sums[key] += (value).detach() * scale
                            sums[f"{key}_baseline"] += (
                                (baseline).detach() * scale
                            )
                    candidate_own = output["aux"]["candidate_pet_delta_own"][
                        aux_positions
                    ]
                    candidate_enemy = output["aux"]["candidate_pet_delta_enemy"][
                        aux_positions
                    ]
                    candidate_own = candidate_own[
                        torch.arange(len(aux_positions), device=device), selected
                    ]
                    candidate_enemy = candidate_enemy[
                        torch.arange(len(aux_positions), device=device), selected
                    ]
                    candidate_pred = torch.cat(
                        (candidate_own, candidate_enemy), dim=1
                    )
                    candidate_error = (
                        (candidate_pred - pet_target).pow(2)
                        * pet_mask.unsqueeze(-1)
                    )
                    candidate_delta = candidate_error.sum() / (
                        pet_mask.unsqueeze(-1).sum().clamp_min(1)
                    )
                    candidate_delta_baseline = (
                        pet_target.pow(2) * pet_mask.unsqueeze(-1)
                    ).sum() / (pet_mask.unsqueeze(-1).sum().clamp_min(1))
                    losses.append(
                        candidate_delta
                        * config.candidate_delta_aux_coefficient
                        * scale
                    )
                    sums["candidate_pet_delta"] += (
                        (candidate_delta).detach() * scale
                    )
                    sums["candidate_pet_delta_baseline"] += (
                        (candidate_delta_baseline).detach() * scale
                    )
                    trait_pet_mask = pet_mask & pet_trait_targets.bool()
                    if trait_pet_mask.any():
                        trait_pet_pred = output["aux"]["trait_pet_delta"][aux_positions]
                        trait_pet_error = (
                            (trait_pet_pred - pet_target).pow(2)
                            * trait_pet_mask.unsqueeze(-1)
                        )
                        trait_pet_delta = trait_pet_error.sum() / (
                            trait_pet_mask.unsqueeze(-1).sum().clamp_min(1)
                        )
                        trait_pet_baseline = (
                            pet_target.pow(2) * trait_pet_mask.unsqueeze(-1)
                        ).sum() / (
                            trait_pet_mask.unsqueeze(-1).sum().clamp_min(1)
                        )
                        losses.append(
                            trait_pet_delta
                            * config.trait_aux_coefficient
                            * scale
                        )
                        sums["trait_pet_delta"] += (
                            (trait_pet_delta).detach() * scale
                        )
                        sums["trait_pet_delta_baseline"] += (
                            (trait_pet_baseline).detach() * scale
                        )
                effect_positions = [
                    position
                    for position, entry in enumerate(entries)
                    if entry.get("effect_value_target") is not None
                ]
                if effect_positions:
                    effect_target = torch.tensor(
                        [
                            entries[position]["effect_value_target"]
                            for position in effect_positions
                        ],
                        dtype=torch.float32,
                        device=device,
                    )
                    effect_selected = torch.tensor(
                        [entries[position]["action"] for position in effect_positions],
                        device=device,
                    )
                    effect_pred = output["aux"]["effect_value"][effect_positions][
                        torch.arange(len(effect_positions), device=device),
                        effect_selected,
                    ]
                    effect_loss = F.mse_loss(effect_pred, effect_target)
                    effect_baseline = F.mse_loss(
                        torch.zeros_like(effect_pred), effect_target
                    )
                    effect_scale = len(effect_positions) / max(effect_count, 1)
                    losses.append(
                        effect_loss * config.effect_aux_coefficient * effect_scale
                    )
                    sums["effect_value"] += (effect_loss).detach() * effect_scale
                    sums["effect_value_baseline"] += (
                        (effect_baseline).detach() * effect_scale
                    )
                lead_positions = [
                    position
                    for position, entry in enumerate(entries)
                    if entry.get("lead_advantage") is not None
                ]
                if lead_positions:
                    lead_target = torch.tensor(
                        [entries[position]["lead_advantage"] for position in lead_positions],
                        dtype=torch.float32,
                        device=device,
                    )
                    lead_pred = output["aux"]["lead_advantage"][lead_positions]
                    lead_advantage = F.mse_loss(lead_pred, lead_target)
                    lead_baseline = F.mse_loss(
                        torch.zeros_like(lead_pred), lead_target
                    )
                    scale = len(lead_positions) / max(len(episodes), 1)
                    losses.append(
                        lead_advantage * config.lead_aux_coefficient * scale
                    )
                    sums["lead_advantage"] += (
                        (lead_advantage).detach() * scale
                    )
                    sums["lead_advantage_baseline"] += (
                        (lead_baseline).detach() * scale
                    )
            if losses:
                torch.stack(losses).sum().backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.policy_parameters(), config.max_grad_norm
        )
        epoch_kl = float(sums["kl"])
        if epoch_kl < config.target_kl:
            optimizer.step()
        else:
            optimizer.zero_grad(set_to_none=True)
        completed_epochs += 1
        for key in sums:
            metrics[key] = float(sums[key])
        metrics["grad_norm"] = (norm).detach()
        if epoch_kl >= config.target_kl:
            break
    _set_trainable(model, {id(parameter) for parameter in opponent_params})
    for keys, coefficient in (
        (("damage", "ko"), config.aux_coefficient),
        (
            (
                "state_change",
                "effect_delta",
                "effect_value",
                "effect_kind",
                "effect_target",
                "effect_kind_accuracy",
                "effect_kind_precision",
            ),
            config.effect_aux_coefficient,
        ),
        (
            (
                "trait_trigger",
                "trait_effect",
                "trait_target",
                "trait_target_delta",
                "trait_target_own_enemy",
                "trait_target_enemy_own",
                "trait_pet_delta",
                "pet_trait_trigger",
                "candidate_trait_trigger",
                "candidate_trait_target",
            ),
            config.trait_aux_coefficient,
        ),
        (
            ("status_target", "status_target_enemy", "status_target_ally"),
            config.status_target_aux_coefficient,
        ),
        (("pet_delta",), config.pet_delta_aux_coefficient),
        (("candidate_pet_delta",), config.candidate_delta_aux_coefficient),
    ):
        if coefficient:
            continue
        for key in keys:
            metrics[key] = 0.0
            metrics[f"{key}_baseline"] = 0.0
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
    label_count = sum(
        entry.get("opponent_distribution") is not None
        or entry["opponent_target"] is not None
        for episode in episodes
        for entry in episode
    )
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    seen = 0
    soft_seen = 0
    hard_seen = 0
    for start in range(0, len(episodes), config.update_chunk):
        chunk = episodes[start : start + config.update_chunk]
        losses = []
        for step, active, batch, output in _batched_episode_steps(model, chunk, device):
            entries = [chunk[index][step] for index in active]
            targets = [entry["opponent_target"] for entry in entries]
            distributions = [
                entry.get("opponent_distribution") for entry in entries
            ]
            soft_positions = [
                position
                for position, distribution in enumerate(distributions)
                if distribution is not None
            ]
            valid_soft = []
            if soft_positions:
                width = batch["opponent_candidates"].shape[1]
                probabilities = torch.tensor(
                    [
                        distributions[position]
                        + [0.0] * (width - len(distributions[position]))
                        for position in soft_positions
                    ],
                    dtype=output["opponent_logits"].dtype,
                    device=device,
                )
                mask = batch["opponent_mask"][soft_positions].bool()
                probabilities = probabilities * mask
                totals = probabilities.sum(dim=-1)
                valid = (totals > 0).tolist()
                valid_soft = [
                    position
                    for position, is_valid in zip(soft_positions, valid, strict=True)
                    if is_valid
                ]
                if valid_soft:
                    probabilities = probabilities[valid] / totals[valid].unsqueeze(-1)
                    loss = opponent_distribution_loss(
                        output["opponent_logits"][valid_soft],
                        probabilities,
                        batch["opponent_mask"][valid_soft],
                        reduction="none",
                    ).sum()
                    if not bool(torch.isfinite(loss).all()):
                        raise FloatingPointError("对手软交叉熵出现非有限值")
                    loss_sum += (loss).detach()
                    seen += len(valid_soft)
                    soft_seen += len(valid_soft)
                    losses.append(loss / max(label_count, 1))
            invalid_soft = set(soft_positions) - set(valid_soft)
            hard_positions = [
                position
                for position, target in enumerate(targets)
                if target is not None
                and (
                    distributions[position] is None
                    or position in invalid_soft
                )
            ]
            if hard_positions:
                loss = opponent_cross_entropy(
                    output["opponent_logits"][hard_positions],
                    torch.tensor(
                        [targets[position] for position in hard_positions],
                        device=device,
                    ),
                    batch["opponent_mask"][hard_positions],
                )
                if not bool(torch.isfinite(loss).all()):
                    raise FloatingPointError("对手交叉熵出现非有限值")
                total = loss * len(hard_positions)
                loss_sum += (total).detach()
                seen += len(hard_positions)
                hard_seen += len(hard_positions)
                losses.append(total / max(label_count, 1))
        if losses:
            torch.stack(losses).sum().backward()
    norm = torch.nn.utils.clip_grad_norm_(
        model.opponent_parameters(), config.max_grad_norm
    )
    if seen:
        optimizer.step()
    _set_trainable(model, {id(parameter) for parameter in model.policy_parameters()})
    return {
        "opponent_ce": float(loss_sum / max(seen, 1)),
        "opponent_grad_norm": (norm).detach(),
        "opponent_samples": float(seen),
        "opponent_soft_samples": float(soft_seen),
        "opponent_hard_samples": float(hard_seen),
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
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "state_schema_version": STATE_SCHEMA_VERSION,
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
    legacy_signatures = {
        _vocab_signature(
            {
                name: vocab
                for name, vocab in encoder.vocabularies.items()
                if name not in keys
            }
        )
        for keys in LEGACY_VOCAB_KEYS
    }
    if checkpoint["vocab_signature"] not in (signature, *legacy_signatures):
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


_COLLECT_CONTEXT: dict[str, Any] = {}


def _collector_init(
    weights_path: str, model_config: dict[str, Any], device: str, seed: int
) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    encoder = ObservationEncoder()
    torch_device = torch.device(device)
    model = AlphaRoco(ModelConfig(**model_config), encoder.vocab_sizes)
    model.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=False)
    )
    model.to(torch_device).eval()
    _COLLECT_CONTEXT.update(
        encoder=encoder,
        device=torch_device,
        model=model,
        weights_path=Path(weights_path),
        weights_mtime=Path(weights_path).stat().st_mtime_ns,
        opponents={},
        exploration=None,
        exploration_spec=None,
    )


def _collector_opponent(path: str) -> AlphaRoco:
    model = _COLLECT_CONTEXT["opponents"].get(path)
    if model is None:
        model = load_model(
            Path(path), _COLLECT_CONTEXT["encoder"], _COLLECT_CONTEXT["device"]
        )
        _COLLECT_CONTEXT["opponents"][path] = model
    return model


def _collector_task(task: dict[str, Any]) -> tuple[int, str, dict[str, Any]]:
    weights_path = _COLLECT_CONTEXT["weights_path"]
    mtime = weights_path.stat().st_mtime_ns
    if mtime != _COLLECT_CONTEXT["weights_mtime"]:
        _COLLECT_CONTEXT["model"].load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=False)
        )
        _COLLECT_CONTEXT["weights_mtime"] = mtime
    spec = task["exploration"]
    if spec is not None and spec != _COLLECT_CONTEXT["exploration_spec"]:
        _COLLECT_CONTEXT["exploration"] = _Exploration(**spec)
        _COLLECT_CONTEXT["exploration_spec"] = spec
    return (
        task["game_index"],
        task["opponent_name"],
        _collect_versus_game(
            _COLLECT_CONTEXT["encoder"],
            _COLLECT_CONTEXT["model"],
            _collector_opponent(task["opponent_path"]),
            task["team_a"],
            task["team_b"],
            task["seed"],
            _COLLECT_CONTEXT["device"],
            task["reward"],
            log_dir=task["log_dir"],
            gamma=task["gamma"],
            timeout=task["timeout"],
            exploration=_COLLECT_CONTEXT["exploration"],
            opponent_sample=task["opponent_sample"],
            game_index=task["game_index"],
        ),
    )


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
    collector: ProcessPoolExecutor | None = None,
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
                random.random() < config.opponent_sample_rate,
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
            opponent_sample,
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
                opponent_sample=opponent_sample,
                game_index=game_index,
            ),
        )

    if collector is not None:
        spec = (
            None
            if exploration is None
            else {
                "skill_bonus": exploration.skill_bonus,
                "trait_bonus": exploration.trait_bonus,
                "cap": exploration.cap,
                "status_weight": exploration.status_weight,
            }
        )
        return list(
            collector.map(
                _collector_task,
                [
                    {
                        "game_index": game_index,
                        "opponent_name": opponent_name,
                        "opponent_path": str(
                            output / "best.pt"
                            if opponent_name == "current"
                            else output / "pool" / opponent_name
                        ),
                        "team_a": team_a,
                        "team_b": team_b,
                        "seed": game_seed,
                        "log_dir": game_log_dir,
                        "opponent_sample": opponent_sample,
                        "reward": config.reward,
                        "gamma": config.gamma,
                        "timeout": config.terminal_timeout,
                        "exploration": spec,
                    }
                    for (
                        game_index,
                        opponent_name,
                        _,
                        team_a,
                        team_b,
                        game_seed,
                        game_log_dir,
                        opponent_sample,
                    ) in tasks
                ],
            )
        )
    if config.sample_workers == 1:
        return [collect(task) for task in tasks]
    with ThreadPoolExecutor(
        max_workers=min(config.sample_workers, config.games_per_batch)
    ) as executor:
        return list(executor.map(collect, tasks))


def run(
    config: TrainConfig,
    resume: Path | None = None,
    warm_start: Path | None = None,
    teacher: Path | None = None,
    champion: Path | None = None,
) -> list[Path]:
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
        or config.status_target_aux_coefficient < 0
        or config.bloodline_aux_coefficient < 0
        or config.resonance_aux_coefficient < 0
        or config.pet_delta_aux_coefficient < 0
        or config.candidate_delta_aux_coefficient < 0
        or config.outcome_aux_coefficient < 0
        or config.teacher_coefficient < 0
        or config.joint_aux_coefficient < 0
        or config.lead_aux_coefficient < 0
    ):
        raise ValueError(
            "batches、games-per-batch、ppo-epochs、sample-workers、update-chunk与keep-checkpoints必须为正数，探索与辅助系数必须非负"
        )
    if resume is not None and warm_start is not None:
        raise ValueError("--resume与--warm-start不能同时使用")
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
    if not 0 <= config.opponent_sample_rate <= 1:
        raise ValueError("opponent-sample-rate必须在[0,1]内")
    if not 0 <= config.pool_rate_low <= config.pool_rate_high <= 1:
        raise ValueError("pool-rate-low/high必须在[0,1]内且low<=high")
    if config.pool_weight < 0:
        raise ValueError("pool-weight必须非负")
    if not 0 <= config.promotion_threshold <= 1:
        raise ValueError("promotion-threshold必须在[0,1]内")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(config.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device(config.device)
    encoder = ObservationEncoder()
    teacher_model = load_model(teacher, encoder, device) if teacher is not None else None
    if teacher_model is not None:
        teacher_model.requires_grad_(False)
    checkpoint = (
        torch.load(resume, map_location="cpu", weights_only=False)
        if resume is not None
        else None
    )
    warm_checkpoint = (
        torch.load(warm_start, map_location="cpu", weights_only=False)
        if warm_start is not None
        else None
    )
    model_source = checkpoint if checkpoint is not None else warm_checkpoint
    model_config = (
        dataclasses.replace(
            ModelConfig(**model_source["config"]["model_config"]),
            dropout=ModelConfig().dropout,
        )
        if model_source is not None
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
    model = AlphaRoco(config=model_config, vocab_sizes=encoder.vocab_sizes)
    if warm_checkpoint is not None:
        _load_model_state(
            model,
            warm_checkpoint["model"],
            _legacy_checkpoint(warm_checkpoint, encoder),
        )
        target = model.state_dict()
        loaded = sum(
            value.numel()
            for key, value in warm_checkpoint["model"].items()
            if key in target and target[key].shape == value.shape
        )
        total = sum(value.numel() for value in target.values())
        print(
            f"warm_start loaded={loaded} total={total} skipped={total - loaded}",
            flush=True,
        )
    model.to(device)
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
    if champion is not None and best_path.exists():
        raise FileExistsError(f"champion输出目录已有best.pt：{best_path}")
    teacher_active = teacher_model is not None
    pool_dir = output / "pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    pool_stats_path = output / "pool_stats.json"
    if resume is not None:
        checkpoint, first_batch, games_seen = _restore(
            checkpoint, model, optimizers, encoder, device
        )
    if best_path.exists():
        best_model = load_model(best_path, encoder, device)
        best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        best_batch = int(best_checkpoint["best_batch"])
        best_elo = float(best_checkpoint["best_elo"])
    elif champion is not None:
        best_model = load_model(champion, encoder, device)
        _checkpoint(
            best_path,
            best_model,
            optimizers,
            encoder,
            model_config,
            config,
            first_batch - 1,
            games_seen,
            0,
            0.0,
            overwrite=True,
        )
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
        results = []
        if device.type == "cuda":
            torch.cuda.empty_cache()
        collector = None
        if device.type == "cuda" and config.sample_workers > 1:
            torch.save(model.state_dict(), output / "collector.pt")
            collector = ProcessPoolExecutor(
                max_workers=config.sample_workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_collector_init,
                initargs=(
                    str(output / "collector.pt"),
                    dataclasses.asdict(model_config),
                    str(device),
                    config.seed,
                ),
            )
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
            (
                exploration
                if config.rare_skill_batches == 0
                or batch_offset < config.rare_skill_batches
                else None
            ),
            collector,
        )
        if collector is not None:
            collector.shutdown()
            torch.cuda.empty_cache()
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
            teacher_model if teacher_active else None,
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
            results = []
            episodes = []
            if device.type == "cuda":
                torch.cuda.empty_cache()
            tally = fast_play_match_paths(
                checkpoint_path,
                best_path,
                config.eval_games,
                config.eval_seed,
                device,
                sample=False,
                workers=config.eval_workers,
                team_pairs=evaluation_team_pairs(config.eval_games),
            )
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
                teacher_active = False
            else:
                snapshot = pool_dir / f"self-batch-{batch_number:06d}.pt"
                _checkpoint(
                    snapshot,
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
                pool.append((snapshot.name, load_model(snapshot, encoder, device)))
                pool_stats.setdefault(snapshot.name, {"wins": 0, "games": 0})
                while len(pool) > config.pool_size:
                    name, _ = pool.pop(0)
                    (pool_dir / name).unlink(missing_ok=True)
                    pool_stats.pop(name, None)
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
            "teacher_kl": policy_metrics["teacher_kl"],
            "value_loss": policy_metrics["value"],
            "outcome_delta_loss": policy_metrics["outcome_delta"],
            "outcome_delta_baseline": policy_metrics["outcome_delta_baseline"],
            "outcome_target_loss": policy_metrics["outcome_target"],
            "outcome_target_baseline": policy_metrics["outcome_target_baseline"],
            "candidate_trait_trigger_loss": policy_metrics["candidate_trait_trigger"],
            "candidate_trait_trigger_baseline": policy_metrics[
                "candidate_trait_trigger_baseline"
            ],
            "candidate_trait_target_loss": policy_metrics["candidate_trait_target"],
            "candidate_trait_target_baseline": policy_metrics[
                "candidate_trait_target_baseline"
            ],
            "effect_value_loss": policy_metrics["effect_value"],
            "effect_value_baseline": policy_metrics["effect_value_baseline"],
            "opponent_ce": opponent_metrics["opponent_ce"],
            "opponent_soft_samples": opponent_metrics["opponent_soft_samples"],
            "opponent_hard_samples": opponent_metrics["opponent_hard_samples"],
            "both_lose_rate": both_lose / config.games_per_batch,
            "status_actions": status_actions,
            "status_action_rate": status_actions / max(total_actions, 1),
            "unique_status_skills": len(status_skill_ids),
            "unique_skills": len(used_skill_ids),
            "unique_effects": len(effect_keys),
            "trait_effects": trait_effects,
        }
        record.update(
            {
                key: float(value) if isinstance(value, torch.Tensor) else value
                for key, value in policy_metrics.items()
            }
        )
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        pool_stats_path.write_text(
            json.dumps(pool_stats, ensure_ascii=False), encoding="utf-8"
        )
        eval_text = "-" if rate is None else f"{rate:.3f}"
        print(
            f"iter_num={batch_number} eval_win_rate={eval_text} promoted={promoted} "
            f"entropy={record['entropy']:.4f} "
            f"kl={record['kl']:.4f} teacher_kl={record['teacher_kl']:.4f} "
            f"value={record['value_loss']:.4f} "
            f"outcome_delta={record['outcome_delta_loss']:.4f} "
            f"outcome_target={record['outcome_target_loss']:.4f} "
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


def _parse_args() -> (
    tuple[TrainConfig, Path | None, Path | None, Path | None, Path | None]
):
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
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument(
        "--entropy-coefficient", type=float, default=TrainConfig.entropy_coefficient
    )
    parser.add_argument("--output", default=TrainConfig.output)
    parser.add_argument("--threads", type=int, default=TrainConfig.threads)
    parser.add_argument("--log-games", action="store_true")
    parser.add_argument(
        "--keep-checkpoints", type=int, default=TrainConfig.keep_checkpoints
    )
    parser.add_argument("--pool-size", type=int, default=TrainConfig.pool_size)
    parser.add_argument("--eval-interval", type=int, default=TrainConfig.eval_interval)
    parser.add_argument("--eval-games", type=int, default=TrainConfig.eval_games)
    parser.add_argument("--eval-seed", type=int, default=TrainConfig.eval_seed)
    parser.add_argument("--eval-workers", type=int, default=TrainConfig.eval_workers)
    parser.add_argument(
        "--sample-workers", type=int, default=TrainConfig.sample_workers
    )
    parser.add_argument("--update-chunk", type=int, default=TrainConfig.update_chunk)
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
        "--outcome-aux-coefficient",
        type=float,
        default=TrainConfig.outcome_aux_coefficient,
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
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--champion", type=Path)
    parser.add_argument(
        "--teacher-coefficient",
        type=float,
        default=TrainConfig.teacher_coefficient,
    )
    args = parser.parse_args()
    return (
        TrainConfig(
            batches=args.batches,
            games_per_batch=args.games_per_batch,
            ppo_epochs=args.ppo_epochs,
            device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            seed=args.seed,
            learning_rate=args.learning_rate,
            entropy_coefficient=args.entropy_coefficient,
            output=args.output,
            threads=args.threads,
            log_games=args.log_games,
            keep_checkpoints=args.keep_checkpoints,
            pool_size=args.pool_size,
            eval_interval=args.eval_interval,
            eval_games=args.eval_games,
            eval_seed=args.eval_seed,
            eval_workers=args.eval_workers,
            sample_workers=args.sample_workers,
            update_chunk=args.update_chunk,
            promotion_threshold=args.promotion_threshold,
            aux_coefficient=args.aux_coefficient,
            effect_aux_coefficient=args.effect_aux_coefficient,
            trait_aux_coefficient=args.trait_aux_coefficient,
            outcome_aux_coefficient=args.outcome_aux_coefficient,
            teacher_coefficient=args.teacher_coefficient,
            rare_skill_bonus=args.rare_skill_bonus,
            rare_skill_batches=args.rare_skill_batches,
        ),
        args.resume,
        args.warm_start,
        args.teacher,
        args.champion,
    )


def main() -> None:
    config, resume, warm_start, teacher, champion = _parse_args()
    run(config, resume, warm_start, teacher, champion)


if __name__ == "__main__":
    main()
