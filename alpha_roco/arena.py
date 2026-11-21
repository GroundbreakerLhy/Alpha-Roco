from __future__ import annotations

import hashlib
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from .config import EvaluationConfig, ModelConfig, SearchConfig
from .features import (
    DESCRIPTOR_VOCAB_KEYS,
    EVENT_FEATURES,
    PET_STATE_ATTRIBUTE_KEYS,
    ObservationEncoder,
    action_key,
    collate_inputs,
    selectable_candidates,
)
from .network import EFFECT_FEATURE_DIM, AlphaRoco, HistoryState
from .terminal import BattleSession

TEAMS = tuple(f"team{index}" for index in range(24))


def evaluation_team_pairs(games: int) -> list[tuple[str, str]]:
    size = len(TEAMS)
    return [
        (TEAMS[index % size], TEAMS[(index + 1 + index // size) % size])
        for index in range(games)
    ]


EVALUATION = EvaluationConfig()
SEARCH = SearchConfig()
EVENT_VOCAB_KEYS = frozenset(
    {"event_kind", "event_action", "event_effect", "event_phase", "event_entity"}
)
LEGACY_VOCAB_KEYS = tuple(
    keys | event_keys
    for keys in (
        frozenset(),
        frozenset({"trait"}),
        frozenset({"bloodline"}),
        frozenset({"trait", "bloodline"}),
        DESCRIPTOR_VOCAB_KEYS,
        DESCRIPTOR_VOCAB_KEYS | frozenset({"trait"}),
        DESCRIPTOR_VOCAB_KEYS | frozenset({"bloodline"}),
        DESCRIPTOR_VOCAB_KEYS | frozenset({"trait", "bloodline"}),
    )
    for event_keys in (frozenset(), EVENT_VOCAB_KEYS)
)
EFFECT_PARAM_PREFIXES = (
    "effect_trait_projection.",
    "effect_projection.",
    "effect_base_score.",
    "effect_interaction_score.",
)
FREQUENCY_PARAM_PREFIX = "opponent.frequency_projection."
NEW_PARAM_PREFIXES = (
    "trait_embedding.",
    "trait_projection.",
    "opponent.trait_embedding.",
    "opponent.trait_projection.",
    FREQUENCY_PARAM_PREFIX,
    *EFFECT_PARAM_PREFIXES,
    "aux_effect_delta.",
    "aux_state_change.",
    "aux_trait_trigger.",
    "aux_trait_source.",
    "self_history.",
    "opponent.public_history.",
    "position_embedding.",
    "history_key_norm.",
    "history_value_norm.",
    "bloodline_embedding.",
    "bloodline_projection.",
    "resonance_encoder.",
    "opponent.resonance_encoder.",
    "aux_opponent_bloodline.",
    "aux_opponent_resonance.",
    "resonance_belief_embedding.",
    "resonance_belief_projection.",
    "effect_encoder.",
    "team_context_projection.",
    "aux_pet_delta_own.",
    "aux_pet_delta_enemy.",
    "aux_pet_trait_trigger.",
    "aux_trait_pet_delta.",
    "aux_trait_target_logits.",
    "aux_trait_target_delta.",
    "candidate_trait_trigger_action.",
    "candidate_trait_trigger_pet.",
    "candidate_trait_trigger_context.",
    "candidate_trait_target_action.",
    "candidate_trait_target_pet.",
    "candidate_trait_target_context.",
    "candidate_trait_score.",
    "aux_effect_value.",
    "effect_value_score.",
    "applied_effect_kind_action.",
    "applied_effect_kind_context.",
    "applied_effect_kind_pet.",
    "applied_effect_kind_own_type.",
    "applied_effect_kind_enemy_type.",
    "applied_effect_target_action.",
    "applied_effect_target_context.",
    "applied_effect_target_pet.",
    "candidate_pet_delta_own.",
    "candidate_pet_delta_enemy.",
    "candidate_pet_delta_score.",
    "candidate_team_attention.",
    "candidate_team_projection.",
    "candidate_target_projection.",
    "aux_candidate_target_logits.",
    "aux_opponent_joint.",
    "joint_belief_embedding.",
    "joint_belief_projection.",
    "lead_opponent_projection.",
    "aux_lead_advantage.",
    "pet_encoder.",
    "candidate_encoder.",
    "opponent.candidate_encoder.",
    "candidate_magic_embedding.",
    "candidate_magic_projection.",
    "opponent.public_event_",
    "event_encoder.",
    "event_attention.",
    "event_chunk_attention.",
    "event_chunk_projection.",
    "event_pet_cross.",
    "event_summary_projection.",
    "opponent.structured_event_encoder.",
    "opponent.public_event_chunk_attention.",
    "opponent.public_event_chunk_projection.",
    "skill_descriptor.",
    "opponent_skill_descriptor.",
    "skill_state_descriptor.",
    "pet_state_descriptor.",
    "skill_history_encoder.",
    "opponent.public_skill_history_encoder.",
    "outcome_head.",
)
OBSOLETE_PARAM_PREFIXES = (
    "aux_state_delta.",
    "aux_trait_delta.",
    "history_norm.",
    "self_gru.",
    "opponent.public_gru.",
    "self_event_kind.",
    "self_event_phase.",
    "self_event_action_kind.",
    "self_event_skill.",
    "self_event_encoder.",
    "opponent.event_kind.",
    "opponent.event_phase.",
    "opponent.event_action_kind.",
    "opponent.event_skill.",
    "opponent.event_encoder.",
    "opponent.public_event_kind.",
    "opponent.public_event_action.",
    "opponent.public_event_encoder.",
    "opponent.habit_projection.",
)
HISTORY_PARAM_PREFIXES = (
    "self_history.",
    "opponent.public_history.",
    "position_embedding.",
    "history_key_norm.",
    "history_value_norm.",
)


def _vocab_signature(vocabularies: dict) -> str:
    return hashlib.sha256(
        json.dumps(vocabularies, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def _aux_effect_dim(vocab_sizes: dict) -> int:
    return (
        vocab_sizes["effect"] * 4 + vocab_sizes["mark"] * 2 + 2 + vocab_sizes["weather"]
    )


def _event_parameter_shapes(
    prefix: str, vocab_sizes: dict, width: int, category_dim: int, identity_dim: int
) -> dict[str, tuple[int, ...]]:
    return {
        f"{prefix}.kind.weight": (vocab_sizes["event_kind"], category_dim),
        f"{prefix}.phase.weight": (vocab_sizes["event_phase"], category_dim),
        f"{prefix}.action.weight": (vocab_sizes["event_action"], category_dim),
        f"{prefix}.effect.weight": (vocab_sizes["event_effect"], category_dim),
        f"{prefix}.source_kind.weight": (
            vocab_sizes["event_entity"],
            category_dim,
        ),
        f"{prefix}.target_kind.weight": (
            vocab_sizes["event_entity"],
            category_dim,
        ),
        f"{prefix}.trait.weight": (vocab_sizes["trait"], category_dim),
        f"{prefix}.project.0.weight": (
            width,
            EVENT_FEATURES + category_dim * 9 + identity_dim * 3,
        ),
        f"{prefix}.project.0.bias": (width,),
        f"{prefix}.project.2.weight": (width, width),
        f"{prefix}.project.2.bias": (width,),
        f"{prefix}.project.3.weight": (width,),
        f"{prefix}.project.3.bias": (width,),
        f"{prefix}.source_projection.weight": (width, width),
        f"{prefix}.source_projection.bias": (width,),
        f"{prefix}.target_projection.weight": (width, width),
        f"{prefix}.target_projection.bias": (width,),
    }


def _attention_shapes(
    prefix: str, width: int, ffn_multiplier: int
) -> dict[str, tuple[int, ...]]:
    layers = {
        f"{prefix}.blocks.0.norm1.weight": (width,),
        f"{prefix}.blocks.0.norm1.bias": (width,),
        f"{prefix}.blocks.0.attention.in_proj_weight": (width * 3, width),
        f"{prefix}.blocks.0.attention.in_proj_bias": (width * 3,),
        f"{prefix}.blocks.0.attention.out_proj.weight": (width, width),
        f"{prefix}.blocks.0.attention.out_proj.bias": (width,),
        f"{prefix}.blocks.0.norm2.weight": (width,),
        f"{prefix}.blocks.0.norm2.bias": (width,),
        f"{prefix}.blocks.0.ffn.0.weight": (width * ffn_multiplier, width),
        f"{prefix}.blocks.0.ffn.0.bias": (width * ffn_multiplier,),
        f"{prefix}.blocks.0.ffn.2.weight": (width, width * ffn_multiplier),
        f"{prefix}.blocks.0.ffn.2.bias": (width,),
        f"{prefix}.norm.weight": (width,),
        f"{prefix}.norm.bias": (width,),
    }
    return layers


def _new_parameter_shapes(
    vocab_sizes: dict,
    width: int,
    category_dim: int,
    identity_dim: int = 32,
    ffn_multiplier: int = 4,
) -> dict[str, tuple[int, ...]]:
    effect_dim = _aux_effect_dim(vocab_sizes)
    shapes = {
        "effect_trait_projection.weight": (width, category_dim),
        "effect_trait_projection.bias": (width,),
        "effect_projection.weight": (EFFECT_FEATURE_DIM, effect_dim),
        "effect_projection.bias": (EFFECT_FEATURE_DIM,),
        "effect_base_score.weight": (1, EFFECT_FEATURE_DIM),
        "effect_base_score.bias": (1,),
        "effect_interaction_score.weight": (1, EFFECT_FEATURE_DIM + width),
        "effect_interaction_score.bias": (1,),
        "aux_effect_value.0.weight": (128, width * 2),
        "aux_effect_value.0.bias": (128,),
        "aux_effect_value.2.weight": (1, 128),
        "aux_effect_value.2.bias": (1,),
        "effect_value_score.weight": (1, 1),
        "effect_value_score.bias": (1,),
        "candidate_trait_trigger_action.weight": (128, width),
        "candidate_trait_trigger_action.bias": (128,),
        "candidate_trait_trigger_pet.weight": (128, width),
        "candidate_trait_trigger_pet.bias": (128,),
        "candidate_trait_trigger_context.weight": (128, width),
        "candidate_trait_trigger_context.bias": (128,),
        "candidate_trait_target_action.weight": (128, width),
        "candidate_trait_target_action.bias": (128,),
        "candidate_trait_target_pet.weight": (128, width),
        "candidate_trait_target_pet.bias": (128,),
        "candidate_trait_target_context.weight": (128, width),
        "candidate_trait_target_context.bias": (128,),
        "candidate_trait_score.weight": (1, 12),
        "candidate_trait_score.bias": (1,),
        "applied_effect_kind_action.weight": (128, width),
        "applied_effect_kind_action.bias": (128,),
        "applied_effect_kind_context.weight": (128, width),
        "applied_effect_kind_context.bias": (128,),
        "applied_effect_kind_pet.weight": (128, width),
        "applied_effect_kind_pet.bias": (128,),
        "applied_effect_kind_own_type.weight": (vocab_sizes["effect"], 128),
        "applied_effect_kind_enemy_type.weight": (vocab_sizes["effect"], 128),
        "applied_effect_target_action.weight": (128, width),
        "applied_effect_target_action.bias": (128,),
        "applied_effect_target_context.weight": (128, width),
        "applied_effect_target_context.bias": (128,),
        "applied_effect_target_pet.weight": (128, width),
        "applied_effect_target_pet.bias": (128,),
        "event_summary_projection.weight": (width, width),
        "event_summary_projection.bias": (width,),
        "event_chunk_projection.weight": (width, width),
        "event_chunk_projection.bias": (width,),
        "opponent.public_event_chunk_projection.weight": (width, width),
        "opponent.public_event_chunk_projection.bias": (width,),
        "event_pet_cross.in_proj_weight": (width * 3, width),
        "event_pet_cross.in_proj_bias": (width * 3,),
        "event_pet_cross.out_proj.weight": (width, width),
        "event_pet_cross.out_proj.bias": (width,),
        FREQUENCY_PARAM_PREFIX + "weight": (width, 1),
        FREQUENCY_PARAM_PREFIX + "bias": (width,),
    }
    shapes.update(
        _event_parameter_shapes(
            "event_encoder", vocab_sizes, width, category_dim, identity_dim
        )
    )
    shapes.update(_attention_shapes("event_attention", width, ffn_multiplier))
    shapes.update(
        _attention_shapes("event_chunk_attention", width, ffn_multiplier)
    )
    shapes.update(
        _event_parameter_shapes(
            "opponent.structured_event_encoder",
            vocab_sizes,
            width,
            category_dim,
            identity_dim,
        )
    )
    shapes.update(
        _attention_shapes("opponent.public_event_attention", width, ffn_multiplier)
    )
    shapes.update(
        _attention_shapes(
            "opponent.public_event_chunk_attention", width, ffn_multiplier
        )
    )
    return shapes


def _legacy_checkpoint(checkpoint: dict, encoder: ObservationEncoder) -> bool:
    if checkpoint.get("vocab_sizes") == encoder.vocab_sizes:
        state = checkpoint.get("model", {})
        expected = _aux_effect_dim(encoder.vocab_sizes)
        model_config = checkpoint.get("config", {}).get("model_config", {})
        width = model_config.get("hidden_dim")
        if width is None:
            width = state["battle_value.0.weight"].shape[1]
        category_dim = model_config.get(
            "category_embedding_dim", ModelConfig().category_embedding_dim
        )
        identity_dim = model_config.get(
            "identity_embedding_dim", ModelConfig().identity_embedding_dim
        )
        ffn_multiplier = model_config.get(
            "ffn_multiplier", ModelConfig().ffn_multiplier
        )
        expected_new = _new_parameter_shapes(
            encoder.vocab_sizes,
            width,
            category_dim,
            identity_dim,
            ffn_multiplier,
        )
        expected_new.update(
            {
                "effect_encoder.0.weight": (
                    width,
                    category_dim * 2 + identity_dim + 6,
                ),
                "effect_encoder.0.bias": (width,),
                "team_context_projection.weight": (width, width),
                "team_context_projection.bias": (width,),
                "aux_pet_delta_own.0.weight": (128, width * 2),
                "aux_pet_delta_own.0.bias": (128,),
                "aux_pet_delta_own.2.weight": (6, 128),
                "aux_pet_delta_own.2.bias": (6,),
                "aux_pet_delta_enemy.0.weight": (128, width * 2),
                "aux_pet_delta_enemy.0.bias": (128,),
                "aux_pet_delta_enemy.2.weight": (6, 128),
                "aux_pet_delta_enemy.2.bias": (6,),
                "aux_trait_target_logits.0.weight": (
                    128,
                    width * 3 + category_dim + 4,
                ),
                "aux_trait_target_logits.0.bias": (128,),
                "aux_trait_target_logits.2.weight": (1, 128),
                "aux_trait_target_logits.2.bias": (1,),
                "aux_trait_target_delta.0.weight": (
                    128,
                    width * 3 + category_dim + 4,
                ),
                "aux_trait_target_delta.0.bias": (128,),
                "aux_trait_target_delta.2.weight": (6, 128),
                "aux_trait_target_delta.2.bias": (6,),
                "candidate_target_projection.weight": (width, 6),
                "candidate_target_projection.bias": (width,),
                "aux_candidate_target_logits.0.weight": (128, width * 3 + 8),
                "aux_candidate_target_logits.0.bias": (128,),
                "aux_candidate_target_logits.2.weight": (1, 128),
                "aux_candidate_target_logits.2.bias": (1,),
                "lead_opponent_projection.weight": (width, width),
                "lead_opponent_projection.bias": (width,),
                "aux_lead_advantage.0.weight": (128, width),
                "aux_lead_advantage.0.bias": (128,),
                "aux_lead_advantage.2.weight": (1, 128),
                "aux_lead_advantage.2.bias": (1,),
                "skill_descriptor.combine.weight": (width, width * 2),
                "opponent_skill_descriptor.combine.weight": (width, width * 2),
                "skill_state_descriptor.project.weight": (width, width),
                "pet_state_descriptor.project.weight": (width, width),
                "candidate_magic_projection.weight": (width, category_dim),
                "opponent.public_event_summary_projection.weight": (width, width),
                "skill_history_encoder.project.weight": (width, width),
                "opponent.public_skill_history_encoder.project.weight": (
                    width,
                    width,
                ),
                "outcome_head.score.weight": (
                    1,
                    12 * len(PET_STATE_ATTRIBUTE_KEYS),
                ),
            }
        )
        return (
            any(key.startswith(OBSOLETE_PARAM_PREFIXES) for key in state)
            or not any(key.startswith(HISTORY_PARAM_PREFIXES) for key in state)
            or any(
                key not in state or tuple(state[key].shape) != shape
                for key, shape in expected_new.items()
            )
            or any(
                key in state and state[key].shape[0] != expected
                for key in (
                    "aux_effect_delta.2.weight",
                    "aux_state_change.2.weight",
                )
            )
        )
    for keys in LEGACY_VOCAB_KEYS:
        legacy_sizes = {
            name: size for name, size in encoder.vocab_sizes.items() if name not in keys
        }
        if checkpoint.get("vocab_sizes") != legacy_sizes:
            continue
        legacy_vocab = {
            name: vocab
            for name, vocab in encoder.vocabularies.items()
            if name not in keys
        }
        if checkpoint.get("vocab_signature") in (
            None,
            _vocab_signature(legacy_vocab),
        ):
            return True
    raise ValueError("checkpoint词表大小或签名不匹配")


def _load_model_state(model: AlphaRoco, state: dict, legacy: bool) -> None:
    if not legacy:
        model.load_state_dict(state, strict=True)
        return
    target = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in target and target[key].shape == value.shape
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    allowed = tuple(key for key in missing if key.startswith(NEW_PARAM_PREFIXES))
    obsolete = tuple(
        key for key in unexpected if key.startswith(OBSOLETE_PARAM_PREFIXES)
    )
    if len(allowed) != len(missing) or len(obsolete) != len(unexpected):
        raise ValueError(f"checkpoint新增参数不匹配：{missing!r} {unexpected!r}")


def wilson_interval(wins: int, games: int) -> tuple[float, float]:
    if games == 0:
        return 0.0, 1.0
    z = 1.959963984540054
    rate = wins / games
    denominator = 1 + z * z / games
    center = (rate + z * z / (2 * games)) / denominator
    margin = (
        z
        * math.sqrt(rate * (1 - rate) / games + z * z / (4 * games * games))
        / denominator
    )
    return center - margin, center + margin


def load_model(
    path: Path, encoder: ObservationEncoder, device: torch.device
) -> AlphaRoco:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "single" or "model" not in checkpoint:
        raise ValueError(f"checkpoint不是当前单网络格式：{path}")
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
    if checkpoint.get("vocab_signature") not in (
        None,
        signature,
        *legacy_signatures,
    ):
        raise ValueError(f"checkpoint词表签名不匹配：{path}")
    model = AlphaRoco(
        config=ModelConfig(**checkpoint["config"]["model_config"]),
        vocab_sizes=encoder.vocab_sizes,
    ).to(device)
    _load_model_state(model, checkpoint["model"], legacy)
    model.eval()
    return model


def _search_logits(output, batch, top_k, risk):
    q = torch.softmax(output["opponent_logits"], dim=-1)
    interactions = output["interactions"]
    expected = (interactions * q[:, None, :]).sum(dim=-1)
    future = interactions.masked_fill(~batch["opponent_mask"][:, None, :], torch.inf)
    indices = q.topk(min(top_k, q.shape[-1]), dim=-1).indices
    worst = (
        future.gather(2, indices.unsqueeze(1).expand(-1, future.shape[1], -1))
        .min(dim=-1)
        .values
    )
    return output["base_logits"] + (1 - risk) * expected + risk * worst


def commit_public_action(history, model, candidates, action):
    key = action_key(action)
    index = (
        next(
            (
                index
                for index, candidate in enumerate(candidates)
                if action_key(candidate) == key
            ),
            None,
        )
        if candidates is not None
        else None
    )
    with torch.no_grad():
        history.commit_public(model, index, action)


def record_opponent_actions(histories, models, encoder, requests, actions):
    for side, history in histories.items():
        other = "B" if side == "A" else "A"
        if other not in actions:
            continue
        request = requests.get(side)
        commit_public_action(
            history,
            models[side],
            None if request is None else encoder.opponent_candidates(request),
            actions[other],
        )


def choose(
    model,
    encoder,
    request,
    history,
    device,
    sample,
    search=False,
    search_top_k=SEARCH.top_k,
    search_risk=SEARCH.risk,
    generator=None,
    return_distribution=False,
):
    candidates = selectable_candidates(request)
    item = encoder.encode(request)
    batch = collate_inputs([item], device)
    with torch.no_grad():
        output = model(
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
        logits = output["logits"]
        if search and request["phase"] == "action":
            logits = _search_logits(output, batch, search_top_k, search_risk)
        logits = logits[0].masked_fill(
            ~batch["action_mask"][0], torch.finfo(output["logits"].dtype).min
        )
        if sample:
            probabilities = torch.softmax(logits, dim=-1)
            index = int(
                torch.multinomial(
                    probabilities, 1, generator=generator
                ).item()
            )
            distribution = probabilities.tolist()
        else:
            index = int(logits.argmax().item())
            distribution = [
                1.0 if position == index else 0.0
                for position in range(len(candidates))
            ]
        if history is not None:
            history.observe(model, output, index, request["phase"])
    action = candidates[index]
    if return_distribution:
        return action, history, distribution, [
            action_key(candidate) for candidate in candidates
        ]
    return action, history


def _play_game(
    candidate,
    reference,
    encoder,
    game_index,
    seed,
    device,
    sample,
    timeout,
    search,
    search_top_k,
    search_risk,
    team_pairs=None,
):
    models = {"A": candidate, "B": reference}
    histories = {"A": HistoryState(), "B": HistoryState()}
    rng = random.Random(seed + game_index * 0x9E3779B1)
    team_a, team_b = (
        team_pairs[game_index]
        if team_pairs is not None
        else (rng.choice(TEAMS), rng.choice(TEAMS))
    )
    with BattleSession(
        team_a=team_a,
        team_b=team_b,
        seed=rng.randrange(2**32),
        timeout=timeout,
    ) as session:
        while session.winner is None:
            requests = session.requests()
            actions = {}
            request_by_side = {}
            for request in requests:
                side = request["side"]
                action, histories[side] = choose(
                    models[side],
                    encoder,
                    request,
                    histories[side],
                    device,
                    sample,
                    search=search and side == "A",
                    search_top_k=search_top_k,
                    search_risk=search_risk,
                )
                actions[side] = action
                request_by_side[side] = request
                session.submit(side, action)
            record_opponent_actions(
                histories, models, encoder, request_by_side, actions
            )
    if session.winner == "both_lose":
        return "both_lose"
    return "win" if session.winner == "A" else "loss"


def _play_games(
    candidate,
    reference,
    encoder,
    game_indices,
    seed,
    device,
    sample,
    timeout,
    search,
    search_top_k,
    search_risk,
    team_pairs=None,
):
    tally = {"win": 0, "loss": 0, "both_lose": 0}
    for game_index in game_indices:
        tally[
            _play_game(
                candidate,
                reference,
                encoder,
                game_index,
                seed,
                device,
                sample,
                timeout,
                search,
                search_top_k,
                search_risk,
                team_pairs,
            )
        ] += 1
    return tally


def play_match(
    candidate: AlphaRoco,
    reference: AlphaRoco,
    encoder: ObservationEncoder,
    games: int,
    seed: int,
    device: torch.device,
    sample: bool,
    timeout: float = EVALUATION.timeout,
    workers: int = 1,
    search: bool = False,
    search_top_k: int = SEARCH.top_k,
    search_risk: float = SEARCH.risk,
    team_pairs: list[tuple[str, str]] | None = None,
) -> dict[str, int]:
    if games < 1:
        raise ValueError("games必须为正数")
    if workers < 1:
        raise ValueError("workers必须为正数")
    if search_top_k < 1:
        raise ValueError("search-top-k必须为正数")
    if not 0 <= search_risk <= 1:
        raise ValueError("search-risk必须在[0,1]内")
    if team_pairs is not None and len(team_pairs) < games:
        raise ValueError("team_pairs长度不得小于games")
    worker_count = min(workers, games)
    chunks = [range(index, games, worker_count) for index in range(worker_count)]
    arguments = (
        candidate,
        reference,
        encoder,
        seed,
        device,
        sample,
        timeout,
        search,
        search_top_k,
        search_risk,
        team_pairs,
    )
    if worker_count == 1:
        return _play_games(*arguments[:3], chunks[0], *arguments[3:])
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(
            executor.map(
                lambda chunk: _play_games(*arguments[:3], chunk, *arguments[3:]),
                chunks,
            )
        )
    tally = {"win": 0, "loss": 0, "both_lose": 0}
    for result in results:
        for outcome in tally:
            tally[outcome] += result[outcome]
    return tally
