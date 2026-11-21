from __future__ import annotations

from collections import Counter

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .features import (
    EVENT_FEATURE_KEYS,
    EVENT_FEATURES,
    PET_STATE_ATTRIBUTE_KEYS,
    action_key,
)

EFFECT_FEATURE_DIM = 16
PET_DELTA_DIM = 6
EVENT_CHUNK_SIZE = 64
CHARGE_ACTION = 3
SKILL_ACTION = 6


class HistoryAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_dim
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width,
            config.history_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * config.ffn_multiplier),
            nn.GELU(),
            nn.Linear(width * config.ffn_multiplier, width),
        )

    def forward(self, query, keys, values, key_mask, causal_mask):
        normalized = self.norm1(query)
        valid = key_mask.any(dim=1)
        safe_mask = key_mask.clone()
        safe_mask[~valid, 0] = True
        attended = self.attention(
            normalized,
            keys,
            values,
            attn_mask=~causal_mask,
            key_padding_mask=~safe_mask,
            need_weights=False,
        )[0]
        residual = query + self.dropout(attended * valid[:, None, None])
        return residual + self.dropout(self.ffn(self.norm2(residual)))


class HistoryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList(
            HistoryAttentionLayer(config) for _ in range(config.history_layers)
        )
        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, query, keys=None, values=None, mask=None):
        squeeze = query.dim() == 2
        if squeeze:
            query = query.unsqueeze(1)
        if keys is None or keys.shape[1] == 0:
            empty = torch.zeros_like(query)
            return empty.squeeze(1) if squeeze else empty
        if mask is None:
            mask = torch.ones(keys.shape[:2], dtype=torch.bool, device=keys.device)
        positions = torch.arange(keys.shape[1], device=keys.device)
        limits = (
            keys.shape[1]
            - query.shape[1]
            + torch.arange(query.shape[1], device=keys.device)
        ).clamp_min(0)
        causal = positions.unsqueeze(0) <= limits.unsqueeze(1)
        for block in self.blocks:
            query = block(query, keys, values, mask, causal)
        query = self.norm(query) * mask.any(dim=1)[:, None, None]
        return query.squeeze(1) if squeeze else query


def collate_histories(histories, device, width, candidates=None):
    own_length = max((len(history.own_keys) for history in histories), default=0)
    public_length = max((len(history.public_keys) for history in histories), default=0)
    if (
        own_length == 0
        and public_length == 0
        and not any(history.public_actions for history in histories)
    ):
        return None

    def pad(sequences, length):
        values = torch.zeros(len(histories), length, width, device=device)
        masks = torch.zeros(len(histories), length, dtype=torch.bool, device=device)
        for index, sequence in enumerate(sequences):
            if sequence:
                size = len(sequence)
                values[index, :size] = torch.stack(sequence).to(device)
                masks[index, :size] = True
        return values, masks

    own_keys, own_mask = pad([history.own_keys for history in histories], own_length)
    own_values, _ = pad([history.own_values for history in histories], own_length)
    public_keys, public_mask = pad(
        [history.public_keys for history in histories], public_length
    )
    public_values, _ = pad(
        [history.public_values for history in histories], public_length
    )
    result = {
        "own_keys": own_keys,
        "own_values": own_values,
        "own_mask": own_mask,
        "public_keys": public_keys,
        "public_values": public_values,
        "public_mask": public_mask,
        "position": torch.tensor(
            [history.position for history in histories],
            dtype=torch.long,
            device=device,
        ),
    }
    if candidates is not None:
        frequency = torch.zeros(
            len(histories),
            max((len(item) for item in candidates), default=0),
            device=device,
        )
        for index, (history, actions) in enumerate(
            zip(histories, candidates, strict=True)
        ):
            counts = Counter(action_key(action) for action in history.public_actions)
            total = max(len(history.public_actions), 1)
            for position, candidate in enumerate(actions):
                frequency[index, position] = counts.get(candidate["key"], 0) / total
        result["public_frequency"] = frequency
    return result


class HistoryState:
    def __init__(self):
        self.own_keys: list[Tensor] = []
        self.own_values: list[Tensor] = []
        self.public_keys: list[Tensor] = []
        self.public_values: list[Tensor] = []
        self.public_actions: list[dict] = []
        self.position = 0
        self.pending: tuple[Tensor, Tensor, int] | None = None

    def batch(self, device, width, candidates=None):
        return collate_histories(
            [self],
            device,
            width,
            None if candidates is None else [candidates],
        )

    def observe(self, model, output, action_index, phase, batch_index=0):
        state = output["state"][batch_index]
        action = (
            output["lead_actions"] if phase == "lead" else output["action_embeddings"]
        )[batch_index, action_index]
        self.own_keys.append(model.history_key(state, action, self.position))
        self.own_values.append(model.history_value(state, action))
        self.pending = (
            output["public_state"][batch_index],
            output["opponent_action_encodings"][batch_index],
            self.position,
        )
        self.position += 1

    def commit_public(self, model, index, action=None):
        if action is not None:
            self.public_actions.append(action)
        if self.pending is None:
            return
        state, actions, position = self.pending
        if index is not None:
            action_encoding = actions[index]
            self.public_keys.append(model.history_key(state, action_encoding, position))
            self.public_values.append(model.history_value(state, action_encoding))
        self.pending = None


class AttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_dim
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width, config.num_heads, batch_first=True
        )
        self.dropout = nn.Dropout(config.dropout)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * config.ffn_multiplier),
            nn.GELU(),
            nn.Linear(width * config.ffn_multiplier, width),
        )

    def forward(self, x, mask):
        if not mask.any(dim=1).all():
            mask = mask.clone()
            mask[~mask.any(dim=1), 0] = True
        normalized = self.norm1(x)
        attended = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~mask,
            need_weights=False,
        )[0]
        residual = x + self.dropout(attended)
        return residual + self.dropout(self.ffn(self.norm2(residual)))


class AttentionStack(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList(
            AttentionBlock(config) for _ in range(config.num_layers)
        )
        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, x, mask):
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)


def chunked_event_attention(tokens, mask, intra, inter, projection):
    """块内注意力 + 块间注意力；块间上下文经零初始化投影残差加入事件 token。"""
    length = tokens.shape[1]
    if length <= EVENT_CHUNK_SIZE:
        return intra(tokens, mask)
    batch, _, width = tokens.shape
    chunks = (length + EVENT_CHUNK_SIZE - 1) // EVENT_CHUNK_SIZE
    padded = chunks * EVENT_CHUNK_SIZE
    if padded > length:
        pad = padded - length
        tokens = torch.cat(
            (tokens, tokens.new_zeros(batch, pad, width)), dim=1
        )
        mask = torch.cat(
            (mask, torch.zeros(batch, pad, dtype=torch.bool, device=mask.device)),
            dim=1,
        )
    flat = intra(
        tokens.reshape(batch * chunks, EVENT_CHUNK_SIZE, width),
        mask.reshape(batch * chunks, EVENT_CHUNK_SIZE),
    )
    flat = flat * mask.reshape(batch * chunks, EVENT_CHUNK_SIZE).unsqueeze(-1)
    flat_mask = mask.reshape(batch * chunks, EVENT_CHUNK_SIZE)
    summaries = (flat * flat_mask.unsqueeze(-1)).sum(dim=1) / flat_mask.sum(
        dim=1, keepdim=True
    ).clamp_min(1)
    summaries = summaries.reshape(batch, chunks, width)
    chunk_mask = flat_mask.reshape(batch, chunks, EVENT_CHUNK_SIZE).any(dim=2)
    chunk_context = inter(summaries, chunk_mask) * chunk_mask.unsqueeze(-1).to(
        summaries.dtype
    )
    delta = (
        projection(chunk_context)
        .unsqueeze(2)
        .expand(batch, chunks, EVENT_CHUNK_SIZE, width)
        .reshape(batch, padded, width)
    )
    merged = (flat.reshape(batch, padded, width) + delta) * mask.unsqueeze(-1)
    return merged[:, :length]


class CandidateEncoder(nn.Module):
    def __init__(self, config, vocab_sizes):
        super().__init__()
        self.kind = nn.Embedding(
            vocab_sizes["action"], config.category_embedding_dim, padding_idx=0
        )
        self.skill = nn.Embedding(
            vocab_sizes["skill"], config.identity_embedding_dim, padding_idx=0
        )
        self.numeric = nn.Sequential(nn.Linear(2, 32), nn.GELU())
        self.derived = nn.Sequential(nn.Linear(27, 64), nn.GELU())
        size = (
            config.category_embedding_dim + config.identity_embedding_dim + 32 + 4 + 64
        )
        self.project = nn.Sequential(
            nn.Linear(size, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )

    def forward(self, actions):
        kind = actions[..., 0].long().clamp(0, self.kind.num_embeddings - 1)
        skill = actions[..., 2].long().clamp(0, self.skill.num_embeddings - 1)
        numeric = self.numeric(actions[..., 6:8])
        misc = torch.stack(
            (
                actions[..., 1] / 6,
                actions[..., 3] / 3,
                actions[..., 4] * 0.5 + actions[..., 5] * 0.5,
                actions[..., 8] / 4,
            ),
            dim=-1,
        )
        return self.project(
            torch.cat(
                (
                    self.kind(kind),
                    self.skill(skill),
                    numeric,
                    misc,
                    self.derived(actions[..., 9:36]),
                ),
                dim=-1,
            )
        )


class AttributeSetEncoder(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        category = config.category_embedding_dim
        width = config.hidden_dim
        self.key = nn.Embedding(vocab_size, category, padding_idx=0)
        self.value = nn.Linear(1, category)
        self.encode = nn.Sequential(
            nn.Linear(category * 2, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )
        self.project = nn.Linear(width, width)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, ids, values, mask):
        encoded = self.encode(torch.cat((self.key(ids), self.value(values.unsqueeze(-1))), dim=-1))
        pooled = (encoded * mask.unsqueeze(-1)).sum(dim=-2) / mask.sum(
            dim=-1, keepdim=True
        ).clamp_min(1)
        return self.project(pooled)


class SkillDescriptorEncoder(nn.Module):
    def __init__(self, config, vocab_sizes):
        super().__init__()
        category = config.category_embedding_dim
        width = config.hidden_dim
        self.attribute = AttributeSetEncoder(
            config, vocab_sizes["skill_attribute"]
        )
        self.effect_key = nn.Embedding(
            vocab_sizes["skill_effect_kind"], category, padding_idx=0
        )
        self.effect_target = nn.Embedding(
            vocab_sizes["skill_effect_target"], category, padding_idx=0
        )
        self.effect_stage = nn.Embedding(
            vocab_sizes["skill_effect_stage"], category, padding_idx=0
        )
        self.effect_value = nn.Linear(1, category)
        self.effect_duration = nn.Linear(1, category)
        self.effect_encode = nn.Sequential(
            nn.Linear(category * 4, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )
        self.attention = nn.MultiheadAttention(
            width,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.query = nn.Parameter(torch.zeros(width))
        self.combine = nn.Linear(width * 2, width)
        nn.init.zeros_(self.combine.weight)
        nn.init.zeros_(self.combine.bias)

    def forward(
        self,
        attribute_ids,
        attribute_values,
        attribute_mask,
        effect_kind,
        effect_target,
        effect_stage,
        effect_value,
        effect_duration,
        effect_mask,
    ):
        *leading, _ = attribute_values.shape
        attributes = self.attribute(
            attribute_ids, attribute_values, attribute_mask
        )
        effect = self.effect_encode(
            torch.cat(
                (
                    self.effect_key(effect_kind),
                    self.effect_target(effect_target),
                    self.effect_stage(effect_stage),
                    self.effect_value(effect_value.unsqueeze(-1))
                    + self.effect_duration(effect_duration.unsqueeze(-1)),
                ),
                dim=-1,
            )
        )
        flat_effect = effect.flatten(0, effect.dim() - 3)
        flat_mask = effect_mask.flatten(0, effect_mask.dim() - 2)
        valid = flat_mask.any(dim=1)
        safe_mask = flat_mask.clone()
        safe_mask[~valid, 0] = True
        attended = self.attention(
            self.query.expand(flat_effect.shape[0], 1, -1),
            flat_effect,
            flat_effect,
            key_padding_mask=~safe_mask,
            need_weights=False,
        )[0].squeeze(1)
        attended = (attended * valid.unsqueeze(-1)).view(*leading, -1)
        return self.combine(torch.cat((attributes, attended), dim=-1))


class OutcomeEmbeddingHead(nn.Module):
    def __init__(self, config, pet_count=12):
        super().__init__()
        width = config.hidden_dim
        self.pet_count = pet_count
        self.opponent_projection = nn.Linear(width, width)
        nn.init.zeros_(self.opponent_projection.weight)
        nn.init.zeros_(self.opponent_projection.bias)
        self.trunk = nn.Sequential(
            nn.Linear(width * 3, 256),
            nn.GELU(),
            nn.Linear(256, width),
            nn.LayerNorm(width),
        )
        self.attribute = nn.Embedding(len(PET_STATE_ATTRIBUTE_KEYS), width)
        nn.init.zeros_(self.attribute.weight)
        self.bias = nn.Parameter(torch.zeros(len(PET_STATE_ATTRIBUTE_KEYS)))
        self.target = nn.Linear(width, 1)
        nn.init.zeros_(self.target.weight)
        nn.init.zeros_(self.target.bias)
        self.score = nn.Linear(len(PET_STATE_ATTRIBUTE_KEYS) * pet_count, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, actions, pets, context, opponent_context):
        _, candidates, _ = actions.shape
        if pets.shape[1] < self.pet_count:
            pets = torch.cat(
                (
                    pets,
                    pets.new_zeros(
                        pets.shape[0],
                        self.pet_count - pets.shape[1],
                        pets.shape[-1],
                    ),
                ),
                dim=1,
            )
        pets = pets[:, : self.pet_count]
        context = context + self.opponent_projection(opponent_context)
        hidden = self.trunk(
            torch.cat(
                (
                    actions[:, :, None, :].expand(
                        -1, -1, self.pet_count, -1
                    ),
                    pets[:, None, :, :].expand(
                        -1, candidates, -1, -1
                    ),
                    context[:, None, None, :].expand(
                        -1, candidates, self.pet_count, -1
                    ),
                ),
                dim=-1,
            )
        )
        delta = hidden @ self.attribute.weight.transpose(0, 1) + self.bias
        target_logits = self.target(hidden).squeeze(-1)
        return delta, target_logits, self.score(delta.flatten(2)).squeeze(-1)


class SkillUseHistoryEncoder(nn.Module):
    def __init__(self, config, vocab_sizes):
        super().__init__()
        category = config.category_embedding_dim
        width = config.hidden_dim
        self.skill = nn.Embedding(
            vocab_sizes["skill"], config.identity_embedding_dim, padding_idx=0
        )
        self.turn = nn.Linear(1, category)
        self.replay = nn.Linear(1, category)
        self.encode = nn.Sequential(
            nn.Linear(config.identity_embedding_dim + category * 2, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )
        self.project = nn.Linear(width, width)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, turns, skill_ids, replay, mask):
        tokens = self.encode(
            torch.cat(
                (
                    self.skill(skill_ids),
                    self.turn(turns.unsqueeze(-1)),
                    self.replay(replay.float().unsqueeze(-1)),
                ),
                dim=-1,
            )
        )
        pooled = (tokens * mask.unsqueeze(-1)).sum(dim=-2) / mask.sum(
            dim=-1, keepdim=True
        ).clamp_min(1)
        return self.project(pooled)


class StructuredEventEncoder(nn.Module):
    """结构化事件编码：数值 + 词表 ID + source/target 实体表示。"""

    def __init__(self, config, vocab_sizes):
        super().__init__()
        category = config.category_embedding_dim
        identity = config.identity_embedding_dim
        width = config.hidden_dim
        self.kind = nn.Embedding(
            vocab_sizes["event_kind"], category, padding_idx=0
        )
        self.phase = nn.Embedding(
            vocab_sizes["event_phase"], category, padding_idx=0
        )
        self.action = nn.Embedding(
            vocab_sizes["event_action"], category, padding_idx=0
        )
        self.effect = nn.Embedding(
            vocab_sizes["event_effect"], category, padding_idx=0
        )
        self.source_kind = nn.Embedding(
            vocab_sizes["event_entity"], category, padding_idx=0
        )
        self.target_kind = nn.Embedding(
            vocab_sizes["event_entity"], category, padding_idx=0
        )
        self.trait = nn.Embedding(vocab_sizes["trait"], category, padding_idx=0)
        self.project = nn.Sequential(
            nn.Linear(
                EVENT_FEATURES + category * 9 + identity * 3,
                width,
            ),
            nn.GELU(),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )
        self.source_projection = nn.Linear(width, width)
        self.target_projection = nn.Linear(width, width)
        for projection in (self.source_projection, self.target_projection):
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    @staticmethod
    def _embed(embedding, ids):
        return embedding(ids.clamp(0, embedding.num_embeddings - 1))

    def _entity(self, batch, pets, prefix):
        side = batch["events"][..., EVENT_FEATURE_KEYS.index(f"{prefix}_side")]
        offset = (side < 0).long() * (pets.shape[1] // 2)
        index = batch[f"event_{prefix}_indices"] + offset
        index = index.clamp(0, max(pets.shape[1] - 1, 0))
        pet = pets.gather(
            1,
            index.unsqueeze(-1).expand(-1, -1, pets.shape[-1]),
        )
        present = batch[f"event_{prefix}_present"].unsqueeze(-1).to(pet.dtype)
        return pet * present

    def forward(self, batch, pets, spirit_embedding, trait_embedding, skill_embedding):
        source = self._entity(batch, pets, "source")
        target = self._entity(batch, pets, "target")
        tokens = self.project(
            torch.cat(
                (
                    batch["events"],
                    self._embed(self.kind, batch["event_kind_ids"]),
                    self._embed(self.phase, batch["event_phase_ids"]),
                    self._embed(self.action, batch["event_action_ids"]),
                    self._embed(self.effect, batch["event_effect_ids"]),
                    self._embed(self.trait, batch["event_trait_ids"]),
                    self._embed(self.source_kind, batch["event_source_kind_ids"]),
                    self._embed(self.target_kind, batch["event_target_kind_ids"]),
                    self._embed(
                        trait_embedding, batch["event_source_trait_ids"]
                    ),
                    self._embed(
                        trait_embedding, batch["event_target_trait_ids"]
                    ),
                    self._embed(skill_embedding, batch["event_skill_ids"]),
                    self._embed(
                        spirit_embedding, batch["event_source_spirit_ids"]
                    ),
                    self._embed(
                        spirit_embedding, batch["event_target_spirit_ids"]
                    ),
                ),
                dim=-1,
            )
        )
        return (
            tokens
            + self.source_projection(source)
            + self.target_projection(target)
        )


class OpponentModel(nn.Module):
    def __init__(self, config, vocab_sizes):
        super().__init__()
        width = config.hidden_dim
        self.spirit = nn.Embedding(
            vocab_sizes["spirit"], config.identity_embedding_dim, padding_idx=0
        )
        self.element = nn.Embedding(
            vocab_sizes["element"], config.category_embedding_dim, padding_idx=0
        )
        self.trait_embedding = nn.Embedding(
            vocab_sizes["trait"], config.category_embedding_dim, padding_idx=0
        )
        self.trait_projection = nn.Linear(config.category_embedding_dim, width)
        nn.init.zeros_(self.trait_projection.weight)
        nn.init.zeros_(self.trait_projection.bias)
        self.public_pet = nn.Sequential(
            nn.Linear(
                config.identity_embedding_dim + config.category_embedding_dim * 2 + 42,
                width,
            ),
            nn.GELU(),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )
        self.public_skill_history_encoder = SkillUseHistoryEncoder(
            config, vocab_sizes
        )
        self.public_attention = AttentionStack(config)
        self.public_event_skill = nn.Embedding(
            vocab_sizes["skill"], config.identity_embedding_dim, padding_idx=0
        )
        self.structured_event_encoder = StructuredEventEncoder(config, vocab_sizes)
        self.public_event_attention = AttentionStack(config)
        self.public_event_chunk_attention = AttentionStack(config)
        self.public_event_chunk_projection = nn.Linear(width, width)
        nn.init.zeros_(self.public_event_chunk_projection.weight)
        nn.init.zeros_(self.public_event_chunk_projection.bias)
        self.public_event_cross = nn.MultiheadAttention(
            width,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.public_event_summary_projection = nn.Linear(width, width)
        nn.init.zeros_(self.public_event_summary_projection.weight)
        nn.init.zeros_(self.public_event_summary_projection.bias)
        self.frequency_projection = nn.Linear(1, width)
        nn.init.zeros_(self.frequency_projection.weight)
        nn.init.zeros_(self.frequency_projection.bias)
        self.decision = nn.Sequential(
            nn.Linear(width * 3, 256),
            nn.GELU(),
            nn.Linear(256, width),
            nn.LayerNorm(width),
        )
        self.field_encoder = nn.Sequential(
            nn.Linear(5, width), nn.GELU(), nn.Linear(width, width), nn.LayerNorm(width)
        )
        self.resonance_encoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.GELU(),
            nn.Linear(32, width),
        )
        self.public_history = HistoryAttention(config)
        self.candidate_encoder = CandidateEncoder(config, vocab_sizes)
        self.candidate_score = nn.Sequential(
            nn.Linear(width * 3, 128), nn.GELU(), nn.Linear(128, 1)
        )

    def forward(self, batch, history=None, position_embedding=None):
        pets = self.public_pet(
            torch.cat(
                (
                    self.spirit(batch["public_pet_ids"]),
                    self.element(batch["public_pet_elements"]),
                    self.element(batch["public_pet_sub_elements"]),
                    batch["public_pets"],
                ),
                dim=-1,
            )
        ) + self.trait_projection(self.trait_embedding(batch["public_pet_traits"]))
        pets = pets + self.public_skill_history_encoder(
            batch["public_used_skill_turns"],
            batch["public_used_skill_ids"],
            batch["public_used_skill_replay"],
            batch["public_used_skill_mask"],
        )
        fields = self.field_encoder(batch["fields"])
        resonance = self.resonance_encoder(
            torch.stack((batch["own_resonance"], batch["enemy_resonance"]), dim=1)
        )
        event_mask = batch["event_mask"] & batch["event_public_mask"]
        event_tokens = self.structured_event_encoder(
            batch,
            pets,
            self.spirit,
            self.trait_embedding,
            self.public_event_skill,
        )
        event_tokens = chunked_event_attention(
            event_tokens,
            event_mask,
            self.public_event_attention,
            self.public_event_chunk_attention,
            self.public_event_chunk_projection,
        )
        event_valid = event_mask.any(dim=1)
        safe_event_mask = event_mask.clone()
        safe_event_mask[~event_valid, 0] = True
        event_cross = self.public_event_cross(
            pets,
            event_tokens,
            event_tokens,
            key_padding_mask=~safe_event_mask,
            need_weights=False,
        )[0]
        pets = pets + event_cross * event_valid[:, None, None].to(pets.dtype)
        event_summary = (event_tokens * event_mask.unsqueeze(-1)).sum(
            dim=1
        ) / event_mask.sum(dim=1, keepdim=True).clamp_min(1)
        tokens = torch.cat((pets, fields, resonance), dim=1)
        mask = torch.cat(
            (
                batch["public_pet_mask"],
                batch["field_mask"],
                torch.ones(
                    batch["public_pet_mask"].shape[0],
                    2,
                    dtype=torch.bool,
                    device=batch["public_pet_mask"].device,
                ),
            ),
            dim=1,
        )
        state = self.public_attention(tokens, mask)
        state = (state * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        state = state + self.public_event_summary_projection(event_summary)
        query_state = state
        if history is None:
            public_history = torch.zeros_like(state)
        else:
            public_query = (
                query_state
                if position_embedding is None
                else query_state + position_embedding
            )
            public_history = self.public_history(
                public_query,
                history["public_keys"],
                history["public_values"],
                history["public_mask"],
            )
        candidates = self.candidate_encoder(batch["opponent_candidates"])
        frequency = None if history is None else history.get("public_frequency")
        if frequency is not None:
            candidates = candidates + self.frequency_projection(frequency.unsqueeze(-1))
        context = self.decision(
            torch.cat(
                (query_state, public_history, query_state * public_history), dim=-1
            )
        )
        logits = self.candidate_score(
            torch.cat(
                (
                    context.unsqueeze(1).expand_as(candidates),
                    candidates,
                    context.unsqueeze(1) * candidates,
                ),
                dim=-1,
            )
        ).squeeze(-1)
        if not batch["opponent_mask"].any(dim=1).all():
            raise ValueError("each sample requires at least one opponent candidate")
        return (
            logits.masked_fill(~batch["opponent_mask"], -torch.inf),
            state,
            candidates,
            public_history,
        )


class AlphaRoco(nn.Module):
    def __init__(
        self, config: ModelConfig | None = None, vocab_sizes: dict | None = None
    ):
        super().__init__()
        self.config = config or ModelConfig()
        vocab_sizes = vocab_sizes or {
            "spirit": 640,
            "skill": 600,
            "effect": 128,
            "trait": 256,
            "bloodline": 24,
            "element": 24,
            "nature": 34,
            "category": 8,
            "weather": 8,
            "mark": 64,
            "action": 16,
            "phase": 8,
            "event": 16,
            "event_kind": 64,
            "event_action": 64,
            "event_effect": 64,
            "event_phase": 16,
            "event_entity": 16,
            "side": 8,
            "skill_attribute": 32,
            "skill_effect_kind": 32,
            "skill_effect_target": 32,
            "skill_effect_stage": 16,
            "skill_state_attribute": 32,
            "pet_state_attribute": 64,
        }
        self.spirit_embedding = nn.Embedding(
            vocab_sizes["spirit"], self.config.identity_embedding_dim, padding_idx=0
        )
        self.element_embedding = nn.Embedding(
            vocab_sizes["element"], self.config.category_embedding_dim, padding_idx=0
        )
        self.nature_embedding = nn.Embedding(
            vocab_sizes["nature"], self.config.category_embedding_dim, padding_idx=0
        )
        self.skill_embedding = nn.Embedding(
            vocab_sizes["skill"], self.config.identity_embedding_dim, padding_idx=0
        )
        self.skill_category = nn.Embedding(
            vocab_sizes["category"], self.config.category_embedding_dim, padding_idx=0
        )
        self.effect_embedding = nn.Embedding(
            vocab_sizes["effect"], self.config.category_embedding_dim, padding_idx=0
        )
        width = self.config.hidden_dim
        self.trait_embedding = nn.Embedding(
            vocab_sizes["trait"], self.config.category_embedding_dim, padding_idx=0
        )
        self.trait_projection = nn.Linear(self.config.category_embedding_dim, width)
        nn.init.zeros_(self.trait_projection.weight)
        nn.init.zeros_(self.trait_projection.bias)
        self.bloodline_embedding = nn.Embedding(
            vocab_sizes["bloodline"], self.config.category_embedding_dim, padding_idx=0
        )
        self.bloodline_projection = nn.Linear(self.config.category_embedding_dim, width)
        nn.init.zeros_(self.bloodline_projection.weight)
        nn.init.zeros_(self.bloodline_projection.bias)
        self.resonance_encoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.GELU(),
            nn.Linear(32, width),
        )
        self.aux_opponent_bloodline = nn.Linear(width, vocab_sizes["bloodline"])
        self.aux_opponent_resonance = nn.Linear(width, 3)
        self.resonance_belief_embedding = nn.Embedding(
            3, self.config.category_embedding_dim
        )
        self.resonance_belief_projection = nn.Linear(
            self.config.category_embedding_dim, width
        )
        nn.init.zeros_(self.resonance_belief_projection.weight)
        nn.init.zeros_(self.resonance_belief_projection.bias)
        self.skill_encoder = nn.Sequential(
            nn.Linear(
                self.config.identity_embedding_dim
                + self.config.category_embedding_dim * 2
                + 14,
                width,
            ),
            nn.GELU(),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )
        self.skill_descriptor = SkillDescriptorEncoder(self.config, vocab_sizes)
        self.opponent_skill_descriptor = SkillDescriptorEncoder(
            self.config, vocab_sizes
        )
        self.skill_history_encoder = SkillUseHistoryEncoder(
            self.config, vocab_sizes
        )
        self.event_encoder = StructuredEventEncoder(self.config, vocab_sizes)
        self.event_attention = AttentionStack(self.config)
        self.event_chunk_attention = AttentionStack(self.config)
        self.event_chunk_projection = nn.Linear(width, width)
        nn.init.zeros_(self.event_chunk_projection.weight)
        nn.init.zeros_(self.event_chunk_projection.bias)
        self.event_pet_cross = nn.MultiheadAttention(
            width,
            self.config.num_heads,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.event_summary_projection = nn.Linear(width, width)
        nn.init.zeros_(self.event_summary_projection.weight)
        nn.init.zeros_(self.event_summary_projection.bias)
        self.skill_state_descriptor = AttributeSetEncoder(
            self.config, vocab_sizes["skill_state_attribute"]
        )
        self.pet_state_descriptor = AttributeSetEncoder(
            self.config, vocab_sizes["pet_state_attribute"]
        )
        self.effect_encoder = nn.Sequential(
            nn.Linear(
                self.config.category_embedding_dim * 2
                + self.config.identity_embedding_dim
                + 6,
                width,
            ),
            nn.GELU(),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )
        self.pet_encoder = nn.Sequential(
            nn.Linear(
                self.config.identity_embedding_dim
                + self.config.category_embedding_dim * 4
                + 42
                + width * 2,
                256,
            ),
            nn.GELU(),
            nn.Linear(256, width),
            nn.LayerNorm(width),
        )
        self.field_encoder = nn.Sequential(
            nn.Linear(5, width), nn.GELU(), nn.Linear(width, width), nn.LayerNorm(width)
        )
        self.decision = nn.Sequential(
            nn.Embedding(vocab_sizes["phase"], width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )
        self.lead_attention = AttentionStack(self.config)
        self.battle_attention = AttentionStack(self.config)
        self.self_history = HistoryAttention(self.config)
        self.position_embedding = nn.Embedding(
            self.config.position_embedding_limit, width
        )
        self.history_key_norm = nn.LayerNorm(width)
        self.history_value_norm = nn.LayerNorm(width)
        self.opponent = OpponentModel(self.config, vocab_sizes)
        self.context = nn.Sequential(
            nn.Linear(width * 3, 256),
            nn.GELU(),
            nn.Linear(256, width),
            nn.LayerNorm(width),
        )
        self.candidate_encoder = CandidateEncoder(self.config, vocab_sizes)
        self.candidate_magic_embedding = nn.Embedding(
            4, self.config.category_embedding_dim, padding_idx=0
        )
        self.candidate_magic_projection = nn.Linear(
            self.config.category_embedding_dim, width
        )
        nn.init.zeros_(self.candidate_magic_projection.weight)
        nn.init.zeros_(self.candidate_magic_projection.bias)
        self.outcome_head = OutcomeEmbeddingHead(self.config, pet_count=12)
        self.action_skill = nn.Sequential(
            nn.Linear(width * 2, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.team_context_projection = nn.Linear(width, width)
        nn.init.zeros_(self.team_context_projection.weight)
        nn.init.zeros_(self.team_context_projection.bias)
        self.candidate_team_attention = nn.MultiheadAttention(
            width,
            self.config.num_heads,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.candidate_team_projection = nn.Linear(width, width)
        nn.init.zeros_(self.candidate_team_projection.weight)
        nn.init.zeros_(self.candidate_team_projection.bias)
        self.candidate_target_projection = nn.Linear(6, width)
        nn.init.zeros_(self.candidate_target_projection.weight)
        nn.init.zeros_(self.candidate_target_projection.bias)
        self.aux_candidate_target_logits = nn.Sequential(
            nn.Linear(width * 3 + 8, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        nn.init.zeros_(self.aux_candidate_target_logits[-1].weight)
        nn.init.zeros_(self.aux_candidate_target_logits[-1].bias)
        self.base_score = nn.Sequential(
            nn.Linear(width * 2, 128), nn.GELU(), nn.Linear(128, 1)
        )
        self.interaction_score = nn.Sequential(
            nn.Linear(width * 3, 128), nn.GELU(), nn.Linear(128, 1)
        )
        self.lead_score = nn.Sequential(
            nn.Linear(width * 2, 128), nn.GELU(), nn.Linear(128, 1)
        )
        self.lead_value = nn.Sequential(
            nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1)
        )
        self.lead_opponent_projection = nn.Linear(width, width)
        nn.init.zeros_(self.lead_opponent_projection.weight)
        nn.init.zeros_(self.lead_opponent_projection.bias)
        self.aux_lead_advantage = nn.Sequential(
            nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1)
        )
        self.battle_value = nn.Sequential(
            nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1)
        )
        self.aux_damage_dealt = nn.Sequential(
            nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1)
        )
        self.aux_damage_taken = nn.Sequential(
            nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1)
        )
        self.aux_ko = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 2))
        self.aux_pet_delta_own = nn.Sequential(
            nn.Linear(width * 2, 128),
            nn.GELU(),
            nn.Linear(128, PET_DELTA_DIM),
        )
        self.aux_pet_delta_enemy = nn.Sequential(
            nn.Linear(width * 2, 128),
            nn.GELU(),
            nn.Linear(128, PET_DELTA_DIM),
        )
        self.aux_pet_trait_trigger = nn.Sequential(
            nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1)
        )
        self.aux_trait_pet_delta = nn.Sequential(
            nn.Linear(width * 2 + self.config.category_embedding_dim, 128),
            nn.GELU(),
            nn.Linear(128, PET_DELTA_DIM),
        )
        trait_pair_dim = width * 3 + self.config.category_embedding_dim + 4
        self.aux_trait_target_logits = nn.Sequential(
            nn.Linear(trait_pair_dim, 128), nn.GELU(), nn.Linear(128, 1)
        )
        self.aux_trait_target_delta = nn.Sequential(
            nn.Linear(trait_pair_dim, 128),
            nn.GELU(),
            nn.Linear(128, PET_DELTA_DIM),
        )
        for layer in (
            self.aux_trait_target_logits[-1],
            self.aux_trait_target_delta[-1],
        ):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.candidate_trait_trigger_action = nn.Linear(width, 128)
        self.candidate_trait_trigger_pet = nn.Linear(width, 128)
        self.candidate_trait_trigger_context = nn.Linear(width, 128)
        self.candidate_trait_target_action = nn.Linear(width, 128)
        self.candidate_trait_target_pet = nn.Linear(width, 128)
        self.candidate_trait_target_context = nn.Linear(width, 128)
        self.candidate_trait_score = nn.Linear(12, 1)
        nn.init.zeros_(self.candidate_trait_score.weight)
        nn.init.zeros_(self.candidate_trait_score.bias)
        self.candidate_pet_delta_own = nn.Sequential(
            nn.Linear(width * 3 + 2, 128),
            nn.GELU(),
            nn.Linear(128, PET_DELTA_DIM),
        )
        self.candidate_pet_delta_enemy = nn.Sequential(
            nn.Linear(width * 3 + 2, 128),
            nn.GELU(),
            nn.Linear(128, PET_DELTA_DIM),
        )
        self.candidate_pet_delta_score = nn.Sequential(
            nn.Linear(PET_DELTA_DIM * 12, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.candidate_pet_delta_score[-1].weight)
        nn.init.zeros_(self.candidate_pet_delta_score[-1].bias)
        self.aux_opponent_joint = nn.Linear(
            width, vocab_sizes["bloodline"] * 3
        )
        self.joint_belief_embedding = nn.Embedding(
            vocab_sizes["bloodline"] * 3, self.config.category_embedding_dim
        )
        self.joint_belief_projection = nn.Linear(
            self.config.category_embedding_dim, width
        )
        nn.init.zeros_(self.joint_belief_projection.weight)
        nn.init.zeros_(self.joint_belief_projection.bias)
        effect_delta_dim = (
            vocab_sizes["effect"] * 4
            + vocab_sizes["mark"] * 2
            + 2
            + vocab_sizes["weather"]
        )
        self.aux_effect_delta = nn.Sequential(
            nn.Linear(width * 2, 128), nn.GELU(), nn.Linear(128, effect_delta_dim)
        )
        self.aux_state_change = nn.Sequential(
            nn.Linear(width * 2, 128), nn.GELU(), nn.Linear(128, effect_delta_dim)
        )
        self.aux_trait_trigger = nn.Sequential(
            nn.Linear(width, 128), nn.GELU(), nn.Linear(128, vocab_sizes["trait"])
        )
        self.aux_trait_source = nn.Linear(self.config.category_embedding_dim, width)
        self.effect_trait_projection = nn.Linear(
            self.config.category_embedding_dim, width
        )
        nn.init.zeros_(self.effect_trait_projection.weight)
        nn.init.zeros_(self.effect_trait_projection.bias)
        self.effect_projection = nn.Linear(effect_delta_dim, EFFECT_FEATURE_DIM)
        self.effect_base_score = nn.Linear(EFFECT_FEATURE_DIM, 1)
        self.effect_interaction_score = nn.Linear(EFFECT_FEATURE_DIM + width, 1)
        self.aux_effect_value = nn.Sequential(
            nn.Linear(width * 2, 128), nn.GELU(), nn.Linear(128, 1)
        )
        self.effect_value_score = nn.Linear(1, 1)
        for layer in (
            self.effect_base_score,
            self.effect_interaction_score,
            self.aux_effect_value[-1],
            self.effect_value_score,
        ):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.applied_effect_kind_action = nn.Linear(width, 128)
        self.applied_effect_kind_context = nn.Linear(width, 128)
        self.applied_effect_kind_pet = nn.Linear(width, 128)
        self.applied_effect_kind_own_type = nn.Embedding(
            vocab_sizes["effect"], 128
        )
        self.applied_effect_kind_enemy_type = nn.Embedding(
            vocab_sizes["effect"], 128
        )
        self.applied_effect_target_action = nn.Linear(width, 128)
        self.applied_effect_target_context = nn.Linear(width, 128)
        self.applied_effect_target_pet = nn.Linear(width, 128)

    def _pets(self, batch, prefix):
        skills = self.skill_encoder(
            torch.cat(
                (
                    self.skill_embedding(batch[f"{prefix}_skill_ids"]),
                    self.element_embedding(batch[f"{prefix}_skill_elements"]),
                    self.skill_category(batch[f"{prefix}_skill_categories"]),
                    batch[f"{prefix}_skills"],
                ),
                dim=-1,
            )
        )
        skills = (
            skills
            + self.skill_descriptor(
                batch[f"{prefix}_skill_attribute_ids"],
                batch[f"{prefix}_skill_attribute_values"],
                batch[f"{prefix}_skill_attribute_mask"],
                batch[f"{prefix}_skill_effect_kind"],
                batch[f"{prefix}_skill_effect_target"],
                batch[f"{prefix}_skill_effect_stage"],
                batch[f"{prefix}_skill_effect_value"],
                batch[f"{prefix}_skill_effect_duration"],
                batch[f"{prefix}_skill_effect_mask"],
            )
            + self.skill_state_descriptor(
                batch[f"{prefix}_skill_state_ids"],
                batch[f"{prefix}_skill_state_values"],
                batch[f"{prefix}_skill_state_mask"],
            )
        )
        skill_mask = batch[f"{prefix}_skill_mask"]
        skill_mean = (skills * skill_mask.unsqueeze(-1)).sum(dim=2) / skill_mask.sum(
            dim=2, keepdim=True
        ).clamp_min(1)
        effects = self.effect_encoder(
            torch.cat(
                (
                    self.effect_embedding(batch[f"{prefix}_effect_ids"]),
                    self.trait_embedding(batch[f"{prefix}_effect_source_traits"]),
                    self.spirit_embedding(batch[f"{prefix}_effect_source_ids"]),
                    batch[f"{prefix}_effects"],
                ),
                dim=-1,
            )
        )
        effect_mask = batch[f"{prefix}_effect_mask"]
        effect_mean = (effects * effect_mask.unsqueeze(-1)).sum(
            dim=2
        ) / effect_mask.sum(dim=2, keepdim=True).clamp_min(1)
        pet = (
            self.pet_encoder(
                torch.cat(
                    (
                        self.spirit_embedding(batch[f"{prefix}_pet_ids"]),
                        self.element_embedding(batch[f"{prefix}_pet_elements"]),
                        self.element_embedding(batch[f"{prefix}_pet_sub_elements"]),
                        self.nature_embedding(batch[f"{prefix}_pet_natures"]),
                        self.bloodline_embedding(batch[f"{prefix}_pet_bloodlines"]),
                        batch[f"{prefix}_pets"],
                        skill_mean,
                        effect_mean,
                    ),
                    dim=-1,
                )
            )
            + self.trait_projection(self.trait_embedding(batch[f"{prefix}_pet_traits"]))
            + self.pet_state_descriptor(
                batch[f"{prefix}_pet_state_ids"],
                batch[f"{prefix}_pet_state_values"],
                batch[f"{prefix}_pet_state_mask"],
            )
            + self.skill_history_encoder(
                batch[f"{prefix}_used_skill_turns"],
                batch[f"{prefix}_used_skill_ids"],
                batch[f"{prefix}_used_skill_replay"],
                batch[f"{prefix}_used_skill_mask"],
            )
        )
        return pet, skills

    def _action_embedding(self, actions, base, skills):
        pet_index = actions[..., 1].long().clamp(0, base.shape[1] - 1)
        own_pet = base.gather(1, pet_index.unsqueeze(-1).expand(-1, -1, base.shape[-1]))
        skill_index = actions[..., 8].long()
        valid_skill = (skill_index >= 0) & (skill_index < skills.shape[2])
        skill_index = skill_index.clamp(0, skills.shape[2] - 1)
        pet_skills = skills.gather(
            1,
            pet_index.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, skills.shape[2], skills.shape[-1]),
        )
        selected_skill = pet_skills.gather(
            2,
            skill_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, skills.shape[-1]),
        ).squeeze(2)
        is_skill = (actions[..., 0].long() == 6) & valid_skill
        selected_skill = selected_skill * is_skill.unsqueeze(-1)
        return own_pet, selected_skill

    def _candidate_pet_delta(
        self,
        head,
        pets,
        actions,
        context,
        target_index,
        active_index,
        target_is_own,
    ):
        batch, count, width = actions.shape
        pet_count = pets.shape[1]
        expanded = pets[:, None].expand(batch, count, pet_count, width)
        action = actions[:, :, None].expand(batch, count, pet_count, width)
        current = context[:, None, None].expand(batch, count, pet_count, width)
        positions = torch.arange(pet_count, device=pets.device)
        target = (
            positions[None, None] == target_index[:, :, None]
            if target_is_own
            else torch.zeros(
                batch, count, pet_count, dtype=torch.bool, device=pets.device
            )
        )
        active = (
            positions[None, None] == active_index[:, None, None]
        ).expand(batch, count, pet_count)
        roles = torch.stack((target.float(), active.float()), dim=-1)
        return head(torch.cat((expanded, action, current, roles), dim=-1))

    def history_key(self, state, action, position):
        positions = torch.as_tensor(
            position, dtype=torch.long, device=state.device
        ).clamp(0, self.position_embedding.num_embeddings - 1)
        return self.history_key_norm(
            state + action + self.position_embedding(positions)
        )

    def history_value(self, state, action):
        return self.history_value_norm(state + action)

    def forward(self, batch: dict, history: dict | None = None) -> dict:
        batch_size = batch["phase"].shape[0]
        device = batch["phase"].device
        is_lead = batch["phase_is_lead"]
        positions = (
            torch.zeros(batch_size, dtype=torch.long, device=device)
            if history is None
            else history["position"]
        )
        position_embedding = self.position_embedding(
            positions.clamp(0, self.position_embedding.num_embeddings - 1)
        )
        (
            opponent_logits,
            public_state,
            opponent_action_encodings,
            public_history,
        ) = self.opponent(batch, history, position_embedding)
        own, own_skills = self._pets(batch, "own")
        enemy, _ = self._pets(batch, "enemy")
        own = own[:, :6]
        enemy = enemy[:, :6]
        pet_tokens = torch.cat((own, enemy), dim=1)
        event_mask = batch["event_mask"]
        event_tokens = self.event_encoder(
            batch,
            pet_tokens,
            self.spirit_embedding,
            self.trait_embedding,
            self.skill_embedding,
        )
        event_tokens = chunked_event_attention(
            event_tokens,
            event_mask,
            self.event_attention,
            self.event_chunk_attention,
            self.event_chunk_projection,
        )
        event_valid = event_mask.any(dim=1)
        safe_event_mask = event_mask.clone()
        safe_event_mask[~event_valid, 0] = True
        event_cross = self.event_pet_cross(
            pet_tokens,
            event_tokens,
            event_tokens,
            key_padding_mask=~safe_event_mask,
            need_weights=False,
        )[0]
        event_cross = event_cross * event_valid[:, None, None].to(event_cross.dtype)
        own = own + event_cross[:, :6]
        enemy = enemy + event_cross[:, 6:]
        pet_tokens = torch.cat((own, enemy), dim=1)
        event_summary = (event_tokens * event_mask.unsqueeze(-1)).sum(
            dim=1
        ) / event_mask.sum(dim=1, keepdim=True).clamp_min(1)
        event_delta = self.event_summary_projection(event_summary)
        own_mask = batch["own_pet_mask"][:, :6]
        enemy_mask = batch["enemy_pet_mask"][:, :6]
        bloodline_logits = self.aux_opponent_bloodline(enemy)
        allowed = batch["enemy_bloodline_allowed"][:, : enemy.shape[1], :]
        allowed = allowed | (~allowed.any(dim=-1, keepdim=True))
        bloodline_logits = bloodline_logits.masked_fill(~allowed, -torch.inf)
        bloodline_belief = torch.softmax(bloodline_logits, dim=-1)
        enemy = enemy + self.bloodline_projection(
            bloodline_belief @ self.bloodline_embedding.weight
        )
        fields = self.field_encoder(batch["fields"])
        resonance = self.resonance_encoder(
            torch.stack((batch["own_resonance"], batch["enemy_resonance"]), dim=1)
        )
        fields = torch.cat((fields, resonance), dim=1)
        field_mask = torch.cat(
            (
                batch["field_mask"],
                torch.ones(batch_size, 2, dtype=torch.bool, device=device),
            ),
            dim=1,
        )
        decision = self.decision[0](batch["phase"]).unsqueeze(1)
        tokens = torch.cat((decision, own, enemy, fields), dim=1)
        token_mask = torch.cat(
            (
                torch.ones(batch_size, 1, dtype=torch.bool, device=device),
                own_mask,
                enemy_mask,
                field_mask,
            ),
            dim=1,
        )
        lead_state = self.lead_attention(tokens, token_mask)[:, 0] + event_delta
        battle_state = self.battle_attention(tokens, token_mask)[:, 0] + event_delta
        state = torch.where(is_lead[:, None], lead_state, battle_state)
        own_history = self.self_history(
            state + position_embedding,
            None if history is None else history["own_keys"],
            None if history is None else history["own_values"],
            None if history is None else history["own_mask"],
        )
        battle_context = self.context(
            torch.cat((battle_state, own_history, public_history.detach()), dim=-1)
        )
        resonance_logits = self.aux_opponent_resonance(public_state.detach())
        resonance_belief = torch.softmax(resonance_logits, dim=-1)
        battle_context = battle_context + self.resonance_belief_projection(
            resonance_belief @ self.resonance_belief_embedding.weight
        )
        joint_logits = self.aux_opponent_joint(public_state.detach())
        joint_belief = torch.softmax(joint_logits, dim=-1)
        battle_context = battle_context + self.joint_belief_projection(
            joint_belief @ self.joint_belief_embedding.weight
        )
        candidates = batch["own_candidates"]
        candidate_base = self.candidate_encoder(candidates)
        candidate_base = candidate_base + self.candidate_magic_projection(
            self.candidate_magic_embedding(
                (candidates[..., 3].long() + 1).clamp(0, 3)
            )
        )
        own_action_pet, selected_skill = self._action_embedding(
            candidates, own, own_skills
        )
        team_mask = own_mask.clone()
        team_mask[~team_mask.any(dim=1), 0] = True
        team_context = self.candidate_team_attention(
            candidate_base,
            own,
            own,
            key_padding_mask=~team_mask,
            need_weights=False,
        )[0]
        action_embeddings = (
            self.action_skill(torch.cat((candidate_base, selected_skill), dim=-1))
            + self.team_context_projection(own.mean(dim=1, keepdim=True))
            + self.candidate_team_projection(team_context)
            + self.candidate_target_projection(batch["own_candidate_targets"])
        )
        kind = candidates[..., 0].long()
        skill_index = candidates[..., 8].long()
        is_effect = (kind == CHARGE_ACTION) | (
            (kind == SKILL_ACTION)
            & (skill_index >= 0)
            & (skill_index < own_skills.shape[2])
        )
        own_active_trait = self.trait_embedding(
            batch["own_pet_traits"]
            .gather(1, batch["own_active"].clamp(0, own.shape[1] - 1).unsqueeze(1))
            .squeeze(1)
        )
        enemy_active_trait = self.trait_embedding(
            batch["enemy_pet_traits"]
            .gather(
                1,
                batch["enemy_active"].clamp(0, enemy.shape[1] - 1).unsqueeze(1),
            )
            .squeeze(1)
        )
        effect_context = (
            battle_context
            + self.effect_trait_projection(own_active_trait)
            + self.effect_trait_projection(enemy_active_trait)
        )
        effect_feature = self.effect_projection(
            self.aux_effect_delta(
                torch.cat(
                    (
                        effect_context.unsqueeze(1).expand_as(action_embeddings),
                        action_embeddings,
                    ),
                    dim=-1,
                )
            )
        ) * is_effect.unsqueeze(-1)
        candidate_pet_delta_own = self._candidate_pet_delta(
            self.candidate_pet_delta_own,
            own,
            action_embeddings,
            battle_context,
            candidates[..., 1].long(),
            batch["own_active"],
            True,
        )
        candidate_pet_delta_enemy = self._candidate_pet_delta(
            self.candidate_pet_delta_enemy,
            enemy,
            action_embeddings,
            battle_context,
            candidates[..., 1].long(),
            batch["enemy_active"],
            False,
        )
        candidate_pet_delta_score = self.candidate_pet_delta_score(
            torch.cat(
                (
                    candidate_pet_delta_own.detach().flatten(2),
                    candidate_pet_delta_enemy.detach().flatten(2),
                ),
                dim=-1,
            )
        ).squeeze(-1)
        trait_pet_trigger = self.candidate_trait_trigger_pet(pet_tokens)
        trait_action_trigger = self.candidate_trait_trigger_action(
            action_embeddings
        ) + self.candidate_trait_trigger_context(battle_context).unsqueeze(1)
        candidate_trait_trigger = (
            trait_action_trigger @ trait_pet_trigger.transpose(1, 2)
        ) * (128 ** -0.5)
        trait_pet_target = self.candidate_trait_target_pet(pet_tokens)
        trait_action_target = self.candidate_trait_target_action(
            action_embeddings
        ) + self.candidate_trait_target_context(battle_context).unsqueeze(1)
        candidate_trait_target = (
            trait_action_target @ trait_pet_target.transpose(1, 2)
        ) * (128 ** -0.5)
        effect_value = self.aux_effect_value(
            torch.cat(
                (
                    battle_context.unsqueeze(1).expand_as(action_embeddings),
                    action_embeddings,
                ),
                dim=-1,
            )
        ).squeeze(-1)
        pet_mask = torch.cat((own_mask, enemy_mask), dim=1)
        pet_pool = (pet_tokens * pet_mask.unsqueeze(-1)).sum(dim=1) / pet_mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        kind_state = (
            self.applied_effect_kind_action(action_embeddings)
            + self.applied_effect_kind_context(battle_context).unsqueeze(1)
            + self.applied_effect_kind_pet(pet_pool).unsqueeze(1)
        )
        effect_kind_logits = torch.stack(
            (
                kind_state
                @ self.applied_effect_kind_own_type.weight.transpose(0, 1),
                kind_state
                @ self.applied_effect_kind_enemy_type.weight.transpose(0, 1),
            ),
            dim=2,
        ) * (128 ** -0.5)
        applied_target_state = self.applied_effect_target_action(
            action_embeddings
        ) + self.applied_effect_target_context(battle_context).unsqueeze(1)
        effect_target_logits = (
            applied_target_state
            @ self.applied_effect_target_pet(pet_tokens).transpose(1, 2)
        ) * (128 ** -0.5)
        opponent_candidates = batch["opponent_candidates"]
        opponent_base = self.candidate_encoder(opponent_candidates)
        opponent_base = opponent_base + self.candidate_magic_projection(
            self.candidate_magic_embedding(
                (opponent_candidates[..., 3].long() + 1).clamp(0, 3)
            )
        )
        opponent_descriptor = self.opponent_skill_descriptor(
            batch["opponent_candidate_skill_attribute_ids"],
            batch["opponent_candidate_skill_attributes"],
            batch["opponent_candidate_skill_present"]
            .unsqueeze(-1)
            .expand_as(batch["opponent_candidate_skill_attribute_ids"]),
            batch["opponent_candidate_skill_effect_kind"],
            batch["opponent_candidate_skill_effect_target"],
            batch["opponent_candidate_skill_effect_stage"],
            batch["opponent_candidate_skill_effect_value"],
            batch["opponent_candidate_skill_effect_duration"],
            batch["opponent_candidate_skill_effect_mask"],
        )
        opponent_actions = self.action_skill(
            torch.cat(
                (
                    opponent_base
                    + opponent_descriptor
                    * batch["opponent_candidate_skill_present"].unsqueeze(-1),
                    torch.zeros_like(opponent_base),
                ),
                dim=-1,
            )
        )
        opponent_logits = opponent_logits.masked_fill(
            ~batch["opponent_mask"], -torch.inf
        )
        opponent_q = torch.softmax(opponent_logits, dim=-1).detach()
        opponent_lead = (opponent_q.unsqueeze(-1) * opponent_actions).sum(dim=1)
        opponent_index = batch.get("opponent_action_index")
        if opponent_index is None:
            outcome_opponent = opponent_lead
        else:
            # 训练时用实际对手候选编码；推理与评测没有实际动作，回退到分布期望。
            valid_opponent = (opponent_index >= 0) & (
                opponent_index < opponent_actions.shape[1]
            )
            selected_opponent = opponent_actions[
                torch.arange(
                    opponent_actions.shape[0], device=opponent_index.device
                ),
                opponent_index.clamp(0, opponent_actions.shape[1] - 1),
            ]
            outcome_opponent = torch.where(
                valid_opponent.unsqueeze(-1), selected_opponent, opponent_lead
            )
        outcome_delta, outcome_target_logits, outcome_score = self.outcome_head(
            action_embeddings,
            pet_tokens,
            battle_context,
            outcome_opponent,
        )
        lead_context = lead_state + self.lead_opponent_projection(opponent_lead)
        lead_logits = self.lead_score(
            torch.cat(
                (own_action_pet, lead_context.unsqueeze(1).expand_as(own_action_pet)),
                dim=-1,
            )
        ).squeeze(-1)
        base_logits = (
            self.base_score(
                torch.cat(
                    (
                        battle_context.unsqueeze(1).expand_as(action_embeddings),
                        action_embeddings,
                    ),
                    dim=-1,
                )
            ).squeeze(-1)
            + self.effect_base_score(effect_feature).squeeze(-1) * is_effect
            + self.candidate_trait_score(
                torch.sigmoid(candidate_trait_trigger)
            ).squeeze(-1)
            + self.effect_value_score(effect_value.unsqueeze(-1)).squeeze(-1)
        )
        q = opponent_q
        pair_context = battle_context[:, None, None, :].expand(
            -1, action_embeddings.shape[1], opponent_actions.shape[1], -1
        )
        own_pair = action_embeddings[:, :, None, :].expand_as(pair_context)
        opp_pair = opponent_actions[:, None, :, :].expand(
            -1, action_embeddings.shape[1], -1, -1
        )
        interactions = self.interaction_score(
            torch.cat((pair_context, own_pair, opp_pair), dim=-1)
        ).squeeze(-1)
        effect_interactions = self.effect_interaction_score(
            torch.cat(
                (
                    effect_feature[:, :, None, :].expand(
                        -1, -1, opponent_actions.shape[1], -1
                    ),
                    opponent_actions[:, None, :, :].expand(
                        -1, action_embeddings.shape[1], -1, -1
                    ),
                ),
                dim=-1,
            )
        ).squeeze(-1) * is_effect.unsqueeze(-1)
        interactions = (interactions + effect_interactions).masked_fill(
            ~batch["opponent_mask"][:, None, :], 0.0
        )
        battle_base_logits = base_logits + candidate_pet_delta_score + outcome_score
        battle_logits = battle_base_logits + (interactions * q[:, None, :]).sum(
            dim=-1
        )
        logits = torch.where(is_lead[:, None], lead_logits, battle_logits)
        if not batch["action_mask"].any(dim=1).all():
            raise ValueError("each sample requires at least one own action candidate")
        logits = logits.masked_fill(~batch["action_mask"], -torch.inf)
        value = (
            torch.where(
                is_lead,
                self.lead_value(lead_state).squeeze(-1),
                self.battle_value(battle_context).squeeze(-1),
            ).tanh()
            * 2.0
        )
        aux = {
            "damage_dealt": torch.sigmoid(
                self.aux_damage_dealt(battle_context).squeeze(-1)
            ),
            "damage_taken": torch.sigmoid(
                self.aux_damage_taken(battle_context).squeeze(-1)
            ),
            "ko": self.aux_ko(battle_context),
            "lead_advantage": self.aux_lead_advantage(lead_context).squeeze(-1),
            "pet_delta_own": self.aux_pet_delta_own(
                torch.cat(
                    (
                        own,
                        battle_context.unsqueeze(1).expand_as(own),
                    ),
                    dim=-1,
                )
            ),
            "pet_delta_enemy": self.aux_pet_delta_enemy(
                torch.cat(
                    (
                        enemy,
                        battle_context.unsqueeze(1).expand_as(enemy),
                    ),
                    dim=-1,
                )
            ),
            "candidate_pet_delta_own": candidate_pet_delta_own,
            "candidate_pet_delta_enemy": candidate_pet_delta_enemy,
            "outcome_delta": outcome_delta,
            "outcome_target_logits": outcome_target_logits,
            "outcome_score": outcome_score,
            "candidate_trait_trigger": candidate_trait_trigger,
            "candidate_trait_target": candidate_trait_target,
            "effect_value": effect_value,
            "effect_kind_logits": effect_kind_logits,
            "effect_target_logits": effect_target_logits,
        }
        pet_traits = torch.cat(
            (
                batch["own_pet_traits"][:, :6],
                batch["enemy_pet_traits"][:, :6],
            ),
            dim=1,
        )
        aux["pet_trait_trigger"] = self.aux_pet_trait_trigger(pet_tokens).squeeze(-1)
        aux["trait_pet_delta"] = self.aux_trait_pet_delta(
            torch.cat(
                (
                    pet_tokens,
                    battle_context.unsqueeze(1).expand_as(pet_tokens),
                    self.trait_embedding(pet_traits),
                ),
                dim=-1,
            )
        )
        source_index = torch.arange(12, device=device)
        source_enemy = (source_index >= 6).float()
        target_enemy = source_enemy
        relation = torch.stack(
            (
                source_enemy[:, None].expand(12, 12),
                target_enemy[None, :].expand(12, 12),
                (source_enemy[:, None] == target_enemy[None, :]).float(),
                (source_index[:, None] == source_index[None, :]).float(),
            ),
            dim=-1,
        )
        relation = relation.unsqueeze(0).expand(batch_size, -1, -1, -1)
        actor = pet_tokens[:, :, None, :].expand(
            batch_size, 12, 12, pet_tokens.shape[-1]
        )
        target = pet_tokens[:, None, :, :].expand(
            batch_size, 12, 12, pet_tokens.shape[-1]
        )
        pair_context = battle_context[:, None, None, :].expand(
            batch_size, 12, 12, battle_context.shape[-1]
        )
        pair_trait = self.trait_embedding(pet_traits)[:, :, None, :].expand(
            batch_size, 12, 12, self.config.category_embedding_dim
        )
        pair = torch.cat((actor, target, pair_context, pair_trait, relation), dim=-1)
        aux["trait_target_logits"] = self.aux_trait_target_logits(pair).squeeze(-1)
        aux["trait_target_delta"] = self.aux_trait_target_delta(pair)
        target_index = torch.arange(12, device=device)
        candidate_target = candidates[..., 1].long()
        target_flags = batch["own_candidate_targets"][:, :, None, :].expand(
            batch_size, action_embeddings.shape[1], 12, 6
        )
        candidate_pair = torch.cat(
            (
                action_embeddings[:, :, None, :].expand(
                    batch_size,
                    action_embeddings.shape[1],
                    12,
                    action_embeddings.shape[-1],
                ),
                pet_tokens[:, None, :, :].expand(
                    batch_size,
                    action_embeddings.shape[1],
                    12,
                    action_embeddings.shape[-1],
                ),
                battle_context[:, None, None, :].expand(
                    batch_size,
                    action_embeddings.shape[1],
                    12,
                    action_embeddings.shape[-1],
                ),
                target_flags,
                torch.stack(
                    (
                        (
                            target_index[None, None, :]
                            == candidate_target[:, :, None]
                        ).float(),
                        (target_index >= 6).float()[None, None, :].expand(
                            batch_size, action_embeddings.shape[1], 12
                        ),
                    ),
                    dim=-1,
                ),
            ),
            dim=-1,
        )
        aux["candidate_target_logits"] = self.aux_candidate_target_logits(
            candidate_pair
        ).squeeze(-1)
        return {
            "logits": logits,
            "base_logits": battle_base_logits,
            "interactions": interactions,
            "value": value,
            "aux": aux,
            "battle_context": battle_context,
            "effect_context": effect_context,
            "action_embeddings": action_embeddings,
            "effect_features": effect_feature,
            "lead_actions": own_action_pet,
            "opponent_logits": opponent_logits,
            "opponent_bloodline_logits": bloodline_logits,
            "opponent_bloodline_belief": bloodline_belief,
            "opponent_resonance_logits": resonance_logits,
            "opponent_resonance_belief": resonance_belief,
            "opponent_joint_logits": joint_logits,
            "opponent_joint_belief": joint_belief,
            "state": state,
            "lead_state": lead_state,
            "battle_state": battle_state,
            "own_history_context": own_history,
            "public_history_context": public_history,
            "public_state": public_state,
            "opponent_action_encodings": opponent_action_encodings,
        }

    def policy_parameters(self):
        opponent_ids = {id(parameter) for parameter in self.opponent.parameters()}
        return (
            parameter
            for parameter in self.parameters()
            if id(parameter) not in opponent_ids
        )

    def opponent_parameters(self):
        return self.opponent.parameters()
