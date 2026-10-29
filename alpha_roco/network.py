from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import ModelConfig

PUBLIC_HABIT_DIM = 11


def _public_habit(actions: list[dict]) -> list[float]:
    total = len(actions)
    recent = actions[-8:]

    def rates(items):
        counts = {"skill": 0, "switch": 0, "charge": 0, "other": 0}
        for action in items:
            kind = action.get("kind", "other")
            counts[kind if kind in counts else "other"] += 1
        size = max(len(items), 1)
        return [
            counts["skill"] / size,
            counts["switch"] / size,
            counts["charge"] / size,
            counts["other"] / size,
        ]

    skill_ids = {
        str(action["skill_id"])
        for action in actions
        if action.get("kind") == "skill" and action.get("skill_id") is not None
    }
    magic = sum(action.get("magic_id") is not None for action in actions)
    choice = sum(int(action.get("choice_branch", 0)) == 1 for action in actions)
    return [
        min(total, 50) / 50.0,
        *rates(actions),
        *rates(recent)[:3],
        len(skill_ids) / max(total, 1),
        magic / max(total, 1),
        choice / max(total, 1),
    ]


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


def collate_histories(histories, device, width):
    own_length = max((len(history.own_keys) for history in histories), default=0)
    public_length = max(
        (len(history.public_keys) for history in histories), default=0
    )
    public_habit = [_public_habit(history.public_actions) for history in histories]
    if own_length == 0 and public_length == 0 and not any(map(any, public_habit)):
        return None

    def pad(sequences, length):
        values = torch.zeros(len(histories), length, width, device=device)
        masks = torch.zeros(
            len(histories), length, dtype=torch.bool, device=device
        )
        for index, sequence in enumerate(sequences):
            if sequence:
                size = len(sequence)
                values[index, :size] = torch.stack(sequence).to(device)
                masks[index, :size] = True
        return values, masks

    own_keys, own_mask = pad(
        [history.own_keys for history in histories], own_length
    )
    own_values, _ = pad(
        [history.own_values for history in histories], own_length
    )
    public_keys, public_mask = pad(
        [history.public_keys for history in histories], public_length
    )
    public_values, _ = pad(
        [history.public_values for history in histories], public_length
    )
    return {
        "own_keys": own_keys,
        "own_values": own_values,
        "own_mask": own_mask,
        "public_keys": public_keys,
        "public_values": public_values,
        "public_mask": public_mask,
        "public_habit": torch.tensor(
            public_habit, dtype=torch.float32, device=device
        ),
        "position": torch.tensor(
            [history.position for history in histories],
            dtype=torch.long,
            device=device,
        ),
    }


class HistoryState:
    def __init__(self):
        self.own_keys: list[Tensor] = []
        self.own_values: list[Tensor] = []
        self.public_keys: list[Tensor] = []
        self.public_values: list[Tensor] = []
        self.public_actions: list[dict] = []
        self.position = 0
        self.pending: tuple[Tensor, Tensor, int] | None = None

    def batch(self, device, width):
        return collate_histories([self], device, width)

    def observe(self, model, output, action_index, phase, batch_index=0):
        state = output["state"][batch_index]
        action = (
            output["lead_actions"]
            if phase == "lead"
            else output["action_embeddings"]
        )[
            batch_index, action_index
        ]
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
            self.public_keys.append(
                model.history_key(state, action_encoding, position)
            )
            self.public_values.append(
                model.history_value(state, action_encoding)
            )
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
        self.derived = nn.Sequential(nn.Linear(24, 64), nn.GELU())
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
                    self.derived(actions[..., 9:33]),
                ),
                dim=-1,
            )
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
        self.public_attention = AttentionStack(config)
        self.habit_projection = nn.Linear(PUBLIC_HABIT_DIM, width)
        nn.init.zeros_(self.habit_projection.weight)
        nn.init.zeros_(self.habit_projection.bias)
        self.decision = nn.Sequential(
            nn.Linear(width * 3, 256),
            nn.GELU(),
            nn.Linear(256, width),
            nn.LayerNorm(width),
        )
        self.field_encoder = nn.Sequential(
            nn.Linear(5, width), nn.GELU(), nn.Linear(width, width), nn.LayerNorm(width)
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
        ) + self.trait_projection(
            self.trait_embedding(batch["public_pet_traits"])
        )
        fields = self.field_encoder(batch["fields"])
        tokens = torch.cat((pets, fields), dim=1)
        mask = torch.cat((batch["public_pet_mask"], batch["field_mask"]), dim=1)
        state = self.public_attention(tokens, mask)
        state = (state * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        habit = None if history is None else history.get("public_habit")
        query_state = state if habit is None else state + self.habit_projection(habit)
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
        context = self.decision(
            torch.cat((query_state, public_history, query_state * public_history), dim=-1)
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
            "element": 24,
            "nature": 34,
            "category": 8,
            "weather": 8,
            "mark": 64,
            "action": 16,
            "phase": 8,
            "event": 16,
            "side": 8,
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
        self.effect_encoder = nn.Sequential(
            nn.Linear(self.config.category_embedding_dim + 3, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )
        self.pet_encoder = nn.Sequential(
            nn.Linear(
                self.config.identity_embedding_dim
                + self.config.category_embedding_dim * 3
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
        self.action_skill = nn.Sequential(
            nn.Linear(width * 2, width), nn.GELU(), nn.LayerNorm(width)
        )
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
        skill_mask = batch[f"{prefix}_skill_mask"]
        skill_mean = (skills * skill_mask.unsqueeze(-1)).sum(dim=2) / skill_mask.sum(
            dim=2, keepdim=True
        ).clamp_min(1)
        effects = self.effect_encoder(
            torch.cat(
                (
                    self.effect_embedding(batch[f"{prefix}_effect_ids"]),
                    batch[f"{prefix}_effects"],
                ),
                dim=-1,
            )
        )
        effect_mask = batch[f"{prefix}_effect_mask"]
        effect_mean = (effects * effect_mask.unsqueeze(-1)).sum(
            dim=2
        ) / effect_mask.sum(dim=2, keepdim=True).clamp_min(1)
        pet = self.pet_encoder(
            torch.cat(
                (
                    self.spirit_embedding(batch[f"{prefix}_pet_ids"]),
                    self.element_embedding(batch[f"{prefix}_pet_elements"]),
                    self.element_embedding(batch[f"{prefix}_pet_sub_elements"]),
                    self.nature_embedding(batch[f"{prefix}_pet_natures"]),
                    batch[f"{prefix}_pets"],
                    skill_mean,
                    effect_mean,
                ),
                dim=-1,
            )
        ) + self.trait_projection(
            self.trait_embedding(batch[f"{prefix}_pet_traits"])
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
        opponent_logits, public_state, opponent_action_encodings, public_history = (
            self.opponent(batch, history, position_embedding)
        )
        own, own_skills = self._pets(batch, "own")
        enemy, _ = self._pets(batch, "enemy")
        own_mask = batch["own_pet_mask"]
        enemy_mask = batch["enemy_pet_mask"]
        fields = self.field_encoder(batch["fields"])
        field_mask = batch["field_mask"]
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
        lead_state = self.lead_attention(tokens, token_mask)[:, 0]
        battle_state = self.battle_attention(tokens, token_mask)[:, 0]
        state = torch.where(is_lead[:, None], lead_state, battle_state)
        own_history = self.self_history(
            state + position_embedding,
            None if history is None else history["own_keys"],
            None if history is None else history["own_values"],
            None if history is None else history["own_mask"],
        )
        battle_context = self.context(
            torch.cat(
                (battle_state, own_history, public_history.detach()), dim=-1
            )
        )
        candidates = batch["own_candidates"]
        candidate_base = self.candidate_encoder(candidates)
        own_action_pet, selected_skill = self._action_embedding(
            candidates, own, own_skills
        )
        action_embeddings = self.action_skill(
            torch.cat((candidate_base, selected_skill), dim=-1)
        )
        lead_logits = self.lead_score(
            torch.cat(
                (own_action_pet, lead_state.unsqueeze(1).expand_as(own_action_pet)),
                dim=-1,
            )
        ).squeeze(-1)
        base_logits = self.base_score(
            torch.cat(
                (
                    battle_context.unsqueeze(1).expand_as(action_embeddings),
                    action_embeddings,
                ),
                dim=-1,
            )
        ).squeeze(-1)
        opponent_candidates = batch["opponent_candidates"]
        opponent_base = self.candidate_encoder(opponent_candidates)
        opponent_actions = self.action_skill(
            torch.cat((opponent_base, torch.zeros_like(opponent_base)), dim=-1)
        )
        opponent_logits = opponent_logits.masked_fill(
            ~batch["opponent_mask"], -torch.inf
        )
        q = torch.softmax(opponent_logits, dim=-1).detach()
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
        interactions = interactions.masked_fill(
            ~batch["opponent_mask"][:, None, :], 0.0
        )
        battle_logits = base_logits + (interactions * q[:, None, :]).sum(dim=-1)
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
        }
        return {
            "logits": logits,
            "base_logits": base_logits,
            "interactions": interactions,
            "value": value,
            "aux": aux,
            "battle_context": battle_context,
            "action_embeddings": action_embeddings,
            "lead_actions": own_action_pet,
            "opponent_logits": opponent_logits,
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
