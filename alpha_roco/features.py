from __future__ import annotations

import hashlib
import itertools
import json
import math
import re

import numpy as np
import torch

from sim.data_loader import (
    DATA_DIR,
    calc_all_stats,
    calc_hp,
    calc_speed_range,
    calc_stat,
    get_nature_multipliers,
)
from sim.enums import LORD_BLOODLINE
from sim.events import (
    ENTITY_KINDS,
    EVENT_ACTIONS,
    EVENT_EFFECTS,
    EVENT_KINDS,
    EVENT_PHASES,
)
from sim.skills import impl as _skills_impl
from sim.typechart import type_multiplier

from .status import duration_code

SPECIAL = ("<PAD>", "<UNK>", "<NONE>")
EVENT_FEATURE_KEYS = (
    "turn",
    "seq",
    "order",
    "value",
    "value_present",
    "stacks",
    "duration",
    "duration_value",
    "duration_present",
    "skill_index",
    "replay",
    "choice_branch",
    "public",
    "source_side",
    "source_index",
    "source_pet_present",
    "target_side",
    "target_index",
    "target_pet_present",
    "attribute",
    "attribute_present",
)
EVENT_FEATURES = len(EVENT_FEATURE_KEYS)
SKILL_ATTRIBUTE_KEYS = (
    "category",
    "element",
    "power",
    "energy_cost",
    "hit_count_flat",
    "hit_count_percent",
    "lifesteal",
    "damage_reduction",
    "priority_delta",
    "energy_cost_delta",
    "power_percent",
    "power_flat",
    "drive",
    "swift",
    "windup",
    "choice",
    "usable",
    "counter_attack",
    "counter_defense",
    "counter_status",
    "counter_interrupt",
    "round_replace",
    "round_replace_cost_delta",
    "target_self",
    "target_enemy",
    "target_self_backline",
    "target_self_team",
    "target_enemy_team",
    "target_field",
)
SKILL_STATE_ATTRIBUTE_KEYS = (
    "drive",
    "cooldown",
    "cooldown_pending",
    "used_count",
    "windup_active",
    "temporary",
    "origin_skill",
    "growth_power",
    "growth_energy_cost",
    "growth_extra_drive",
    "growth_stamp",
    "growth_other_0",
    "growth_other_1",
    "growth_other_2",
    "growth_other_3",
)
TRAIT_STATE_KEYS = (
    "triggers",
    "uses",
    "charges",
    "count",
    "stacks",
    "entries",
    "windups",
    "kills",
    "layers",
    "active_counters",
    "active_uses",
    "restored_energy",
    "active",
    "ready",
    "sentinel_triggered",
)
PET_STATE_ATTRIBUTE_KEYS = (
    "hp_ratio",
    "energy",
    "defense_reduction",
    "skill_cooldown_count",
    "skill_cooldown_pending_count",
    "windup_active",
    "windup_skill",
    "burst_count",
    "burst_attack_flat",
    "burst_attack_percent",
    "used_skill_count",
    "last_used_skill",
    "counter_count",
    "counter_attack",
    "counter_defense",
    "counter_status",
    "has_acted_since_entry",
    "entry_turn",
    "light_heal_rounds",
    "overload_next",
    "overload_current",
    "leave_count",
    "attack_used_this_turn",
    "attack_used_last_turn",
    "last_skill_element",
    "last_skill_category",
    "prev_skill_element",
    "prev_skill_category",
    "buff_count",
    "buff_duration",
    *[f"trait_{key}" for key in TRAIT_STATE_KEYS],
    "trait_other_count",
    "trait_other_value",
)
SKILL_EFFECT_STAGES = (
    "entry_effects",
    "slot_change_effects",
    "use_start_effects",
    "hit_effects",
    "counter_effects",
    "defense_effects",
    "status_effects",
    "after_use_effects",
    "round_end_effects",
)
DESCRIPTOR_VOCAB_KEYS = frozenset(
    {
        "skill_attribute",
        "skill_effect_kind",
        "skill_effect_target",
        "skill_effect_stage",
        "skill_state_attribute",
        "pet_state_attribute",
    }
)


def _attribute_values(skill, static):
    targets = set(static["targets"])
    counter = static["counter_target"]
    return [
        _number(skill.get("category_raw"), 3),
        _number(skill.get("element_raw"), 18),
        _number(skill.get("power_raw"), 200),
        _number(skill.get("energy_raw"), 10),
        static["hit_count_flat"] / 10,
        static["hit_count_percent"] / 100,
        static["lifesteal"],
        static["damage_reduction"],
        static["priority_delta"] / 3,
        static["energy_cost_delta"] / 10,
        static["power_percent"] / 200,
        static["power_flat"] / 200,
        static["drive"] / 3,
        float(static["swift"]),
        float(static["windup"]),
        float(static["choice"]),
        float(static["usable"]),
        float(counter == "attack"),
        float(counter == "defense"),
        float(counter == "status"),
        float(static["counter_interrupt"]),
        float(static["round_replace"]),
        static["round_replace_cost_delta"] / 10,
        float("self" in targets),
        float("enemy" in targets),
        float("self_backline" in targets),
        float("self_team" in targets),
        float("enemy_team" in targets),
        float("field" in targets or "both" in targets),
    ]


def _skill_state_values(raw):
    fields = raw if isinstance(raw, dict) else {}
    growth = fields.get("growth", {})
    growth = growth if isinstance(growth, dict) else {}
    other = sorted(
        (key, value)
        for key, value in growth.items()
        if key not in ("power", "energy_cost", "extra_drive", "stamp")
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    )
    values = [
        _number(fields.get("drive", 0), 3),
        float(bool(fields.get("cooldown"))),
        float(bool(fields.get("cooldown_pending"))),
        _number(fields.get("used_count"), 10),
        float(bool(fields.get("windup_active"))),
        float(bool(fields.get("temporary"))),
        _number(fields.get("origin_skill_id"), 10_000_000),
        _number(growth.get("power"), 10),
        _number(growth.get("energy_cost"), 10),
        _number(growth.get("extra_drive"), 10),
        _number(growth.get("stamp"), 10),
        *[_number(value, 10) for _, value in other[:4]],
        *([0.0] * max(0, 4 - len(other))),
    ]
    mask = [
        "drive" in fields,
        "cooldown" in fields,
        "cooldown_pending" in fields,
        "used_count" in fields,
        "windup_active" in fields,
        "temporary" in fields,
        "origin_skill_id" in fields,
        "power" in growth,
        "energy_cost" in growth,
        "extra_drive" in growth,
        "stamp" in growth,
        *([True] * min(len(other), 4)),
        *([False] * max(0, 4 - len(other))),
    ]
    return values, mask


def _category_number(value):
    return {"attack": 0.0, "defense": 1.0, "status": 3.0}.get(value, 0.0) / 3


def _pet_state_values(raw):
    fields = raw if isinstance(raw, dict) else {}
    bursts = fields.get("bursts") or []
    counters = fields.get("counter_stats") or {}
    types = counters.get("types") or {}
    buffs = fields.get("buffs") or []
    trait_state = fields.get("trait_state") or {}
    trait_values = []
    trait_mask = []
    for key in TRAIT_STATE_KEYS:
        value = trait_state.get(key)
        known = isinstance(value, (int, float, bool))
        trait_values.append(float(value) if isinstance(value, bool) else _number(value, 10))
        trait_mask.append(known)
    other_traits = [
        value
        for key, value in trait_state.items()
        if key not in TRAIT_STATE_KEYS
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ]
    values = [
        _number(fields.get("hp_percent", 100), 100)
        if "hp_percent" in fields
        else _number(fields.get("hp"), fields.get("max_hp", 100)),
        _number(fields.get("energy"), 10),
        _number(fields.get("defense_reduction")),
        min(len(fields.get("skill_cooldowns") or ()), 10) / 10,
        min(len(fields.get("skill_cooldowns_pending") or ()), 10) / 10,
        float(fields.get("windup_skill_id") is not None),
        _number(fields.get("windup_skill_id"), 10_000_000),
        min(len(bursts), 10) / 10,
        sum(
            _number(burst.get("value"), 100)
            for burst in bursts
            if burst.get("type") == "attack_power_flat"
        ),
        sum(
            _number(burst.get("value"), 100)
            for burst in bursts
            if burst.get("type") == "attack_power_percent"
        ),
        min(len(fields.get("used_skill_counts") or {}), 20) / 20,
        _number(fields.get("last_used_skill_id"), 10_000_000),
        min(_number(counters.get("count")), 20) / 20,
        min(_number(types.get("attack")), 20) / 20,
        min(_number(types.get("defense")), 20) / 20,
        min(_number(types.get("status")), 20) / 20,
        float(bool(fields.get("has_acted_since_entry"))),
        _number(fields.get("entry_turn"), 100),
        _number(fields.get("light_heal_rounds"), 3),
        min(len(fields.get("overload_next") or {}), 10) / 10,
        min(len(fields.get("overload_current") or {}), 10) / 10,
        _number(fields.get("leave_count"), 10),
        float(bool(fields.get("attack_used_this_turn"))),
        float(bool(fields.get("attack_used_last_turn"))),
        _number(fields.get("last_skill_element"), 18),
        _category_number(fields.get("last_skill_category")),
        _number(fields.get("prev_skill_element"), 18),
        _category_number(fields.get("prev_skill_category")),
        min(len(buffs), 20) / 20,
        sum(_number(buff.get("duration"), 10) for buff in buffs),
        *trait_values,
        min(len(other_traits), 10) / 10,
        math.tanh(sum(other_traits) / 10.0),
    ]
    mask = [
        ("hp" in fields or "hp_percent" in fields),
        "energy" in fields,
        "defense_reduction" in fields,
        "skill_cooldowns" in fields,
        "skill_cooldowns_pending" in fields,
        "windup_skill_id" in fields,
        "windup_skill_id" in fields,
        "bursts" in fields,
        "bursts" in fields,
        "bursts" in fields,
        "used_skill_counts" in fields,
        "last_used_skill_id" in fields,
        "counter_stats" in fields,
        "counter_stats" in fields,
        "counter_stats" in fields,
        "counter_stats" in fields,
        "has_acted_since_entry" in fields,
        "entry_turn" in fields,
        "light_heal_rounds" in fields,
        "overload_next" in fields,
        "overload_current" in fields,
        "leave_count" in fields,
        "attack_used_this_turn" in fields,
        "attack_used_last_turn" in fields,
        "last_skill_element" in fields,
        "last_skill_category" in fields,
        "prev_skill_element" in fields,
        "prev_skill_category" in fields,
        "buffs" in fields,
        "buffs" in fields,
        *trait_mask,
        "trait_state" in fields,
        "trait_state" in fields,
    ]
    return values, mask


def _vocabulary(values):
    return {
        value: index + len(SPECIAL) for index, value in enumerate(sorted(set(values)))
    }


def _index(value, vocab):
    if isinstance(value, dict):
        value = value.get("id")
    if value is None:
        return 2
    return vocab.get(value, vocab.get(str(value), 1))


def _effect_targets(effect) -> set[str]:
    targets = set()
    target = getattr(effect, "target", None)
    if isinstance(target, str) and target:
        targets.add(target)
    for nested in getattr(effect, "effects", ()) or ():
        targets.update(_effect_targets(nested))
    return targets


def _number(value, scale=1.0):
    if isinstance(value, dict):
        value = value.get("id")
    return (
        float(value) / scale
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else 0.0
    )


def _scalar(value):
    if isinstance(value, dict):
        value = value.get("id")
    return (
        value
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )


def _log_mult(mult):
    return min(max(math.log2(max(mult, 1e-6)), -2.0), 2.0) / 2


def _load(name):
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


class _LazyPriors(dict):
    def __init__(self, species, compute):
        super().__init__()
        self._species = species
        self._compute = compute

    def get(self, key, default=None):
        try:
            identity = int(key)
        except (TypeError, ValueError):
            return default
        if identity not in self._species:
            return default
        if identity not in self:
            self[identity] = self._compute(self._species[identity])
        return self[identity]


class ObservationEncoder:
    def __init__(self):
        self.spirits = _load("spirits.json")["spirits"]
        self.skills = _load("skills.json")["skills"]
        self.buffs = _load("buff.json")["effects"]
        self.traits = _load("traits.json")["traits"]
        self.spirit_by_name = {item["name"]: item for item in self.spirits}
        self.spirit_by_id = {str(item["id"]): item for item in self.spirits}
        self.skill_by_id = {str(item["id"]): item for item in self.skills}
        self.skill_by_name = {item["name"]: item for item in self.skills}
        from sim.skills.registry import all_handlers, static_of

        self.choice_skills = {
            handler.skill_id for handler in all_handlers() if handler.choice
        }
        self.static_of = static_of
        self.skill_targets = {}
        self.skill_effect_kinds = set()
        self.skill_effect_targets = set()
        for handler in all_handlers():
            for effect in static_of(handler.skill_id)["effects"]:
                self.skill_effect_kinds.add(effect["kind"])
                if effect["target"]:
                    self.skill_effect_targets.add(effect["target"])
            if getattr(handler, "category", -1) != 3:
                continue
            targets = set()
            for effect in getattr(handler, "status_effects", ()) or ():
                targets.update(_effect_targets(effect))
            self.skill_targets[handler.skill_id] = targets
        self.typechart = _load("typechart.json")
        self.spirit_vocab = _vocabulary(item["id"] for item in self.spirits)
        self.skill_vocab = _vocabulary(item["id"] for item in self.skills)
        self.effect_vocab = _vocabulary(item["id"] for item in self.buffs)
        self.trait_vocab = _vocabulary(item["id"] for item in self.traits)
        self.bloodline_vocab = _vocabulary(range(19))
        self.bloodline_skill_map = {}
        for spirit in self.spirits:
            for entry in spirit.get("skills", {}).get("bloodline", []):
                self.bloodline_skill_map.setdefault(int(entry["skillId"]), set()).add(
                    int(entry["bloodline"])
                )
        self.element_vocab = _vocabulary(range(19))
        self.natures = _load("natures.json")["natures"]
        self.nature_vocab = _vocabulary(item["id"] for item in self.natures)
        self.nature_mults = [
            get_nature_multipliers(item["id"]) for item in self.natures
        ]
        stat_keys = ("hp", "atk", "def", "spatk", "spdef", "speed")
        self._prior_species = {
            int(spirit["id"]): spirit
            for spirit in self.spirits
            if spirit.get("stats")
            and all(
                isinstance(spirit["stats"].get(key), (int, float))
                for key in stat_keys
            )
        }
        self.prior_stats = _LazyPriors(self._prior_species, self._stat_prior)
        self._static_cache = {}
        self._stats_cache = {}
        self._speed_range_cache = {}
        self.category_vocab = _vocabulary(range(4))
        self.weather_vocab = _vocabulary(
            item["id"] for item in _load("weather.json").get("weathers", [])
        )
        self.mark_vocab = _vocabulary(
            item["id"] for item in _load("marks.json").get("marks", [])
        )
        self.action_vocab = _vocabulary(
            ("lead", "skill", "charge", "switch", "flee", "unknown_skill")
        )
        self.skill_attribute_vocab = _vocabulary(SKILL_ATTRIBUTE_KEYS)
        self.skill_effect_kind_vocab = _vocabulary(self.skill_effect_kinds)
        self.skill_effect_target_vocab = _vocabulary(self.skill_effect_targets)
        self.skill_effect_stage_vocab = _vocabulary(SKILL_EFFECT_STAGES)
        self.skill_state_attribute_vocab = _vocabulary(SKILL_STATE_ATTRIBUTE_KEYS)
        self.pet_state_attribute_vocab = _vocabulary(PET_STATE_ATTRIBUTE_KEYS)
        self.phase_vocab = _vocabulary(("lead", "action", "replacement"))
        self.event_vocab = _vocabulary(
            (
                "unknown",
                "log",
                "private",
                "skill",
                "switch",
                "charge",
                "damage",
                "status",
                "entry",
                "round",
            )
        )
        self.event_kind_vocab = _vocabulary(EVENT_KINDS)
        self.event_action_vocab = _vocabulary(EVENT_ACTIONS)
        self.event_effect_vocab = _vocabulary(EVENT_EFFECTS)
        self.event_phase_vocab = _vocabulary(EVENT_PHASES)
        self.event_entity_vocab = _vocabulary(ENTITY_KINDS)
        self.side_vocab = _vocabulary(("own", "enemy"))
        self.vocabularies = {
            "spirit": self.spirit_vocab,
            "skill": self.skill_vocab,
            "effect": self.effect_vocab,
            "trait": self.trait_vocab,
            "bloodline": self.bloodline_vocab,
            "element": self.element_vocab,
            "nature": self.nature_vocab,
            "category": self.category_vocab,
            "weather": self.weather_vocab,
            "mark": self.mark_vocab,
            "action": self.action_vocab,
            "phase": self.phase_vocab,
            "event": self.event_vocab,
            "event_kind": self.event_kind_vocab,
            "event_action": self.event_action_vocab,
            "event_effect": self.event_effect_vocab,
            "event_phase": self.event_phase_vocab,
            "event_entity": self.event_entity_vocab,
            "side": self.side_vocab,
            "skill_attribute": self.skill_attribute_vocab,
            "skill_effect_kind": self.skill_effect_kind_vocab,
            "skill_effect_target": self.skill_effect_target_vocab,
            "skill_effect_stage": self.skill_effect_stage_vocab,
            "skill_state_attribute": self.skill_state_attribute_vocab,
            "pet_state_attribute": self.pet_state_attribute_vocab,
        }
        self.vocab_sizes = {
            name: len(vocab) + len(SPECIAL) for name, vocab in self.vocabularies.items()
        }
        signature = json.dumps(
            {"schema_version": 2, "vocabularies": self.vocabularies},
            ensure_ascii=False,
            sort_keys=True,
        )
        self.vocab_signature = hashlib.sha256(signature.encode("utf-8")).hexdigest()


    def _static(self, skill_id):
        cached = self._static_cache.get(skill_id)
        if cached is None:
            cached = dict(self.static_of(skill_id))
            self._static_cache[skill_id] = cached
        return cached

    def _stats_of(self, species, raw):
        ivs = raw["ivs"]
        key = (
            species.get("id"),
            tuple(ivs[key] for key in ("hp", "atk", "def", "spatk", "spdef", "speed")),
            raw.get("nature"),
        )
        cached = self._stats_cache.get(key)
        if cached is None:
            cached = calc_all_stats(species["stats"], ivs, raw.get("nature"))
            self._stats_cache[key] = cached
        return cached

    def _speed_range_of(self, species):
        base = species.get("stats", {}).get("speed")
        if base is None:
            return [None, None]
        cached = self._speed_range_cache.get(base)
        if cached is None:
            cached = calc_speed_range(base)
            self._speed_range_cache[base] = cached
        return cached

    def _skill(self, raw):
        fields = raw if isinstance(raw, dict) else {}
        if isinstance(raw, dict):
            item = self.skill_by_id.get(str(raw.get("skill_id", raw.get("id", ""))))
            if item is None:
                item = self.skill_by_name.get(raw.get("name", ""), {})
        elif isinstance(raw, int) or (isinstance(raw, str) and raw.isdigit()):
            item = self.skill_by_id.get(str(raw), {})
        else:
            item = self.skill_by_name.get(str(raw), {})
        skill_id = item.get("id", raw if isinstance(raw, (int, str)) else None)
        static = self._static(
            skill_id
            if isinstance(skill_id, int)
            else (
                int(skill_id)
                if isinstance(skill_id, str) and skill_id.isdigit()
                else None
            )
        )
        numeric_skill_id = (
            skill_id
            if isinstance(skill_id, int)
            else int(skill_id)
            if isinstance(skill_id, str) and skill_id.isdigit()
            else None
        )
        targets = self.skill_targets.get(numeric_skill_id, set())
        power = fields.get("display_power", fields.get("power", item.get("power")))
        energy = fields.get(
            "energy_cost", fields.get("energyCost", item.get("energyCost"))
        )
        category = fields.get("category", item.get("category"))
        element = (
            static["element"]
            if static["element"] is not None
            else fields.get("element", item.get("element"))
        )
        counter = static["counter_target"]
        attributes = _attribute_values(
            {
                "category_raw": category,
                "element_raw": element,
                "power_raw": power,
                "energy_raw": energy,
            },
            static,
        )
        effects = [
            (
                self.skill_effect_kind_vocab.get(
                    effect["kind"], self.skill_effect_kind_vocab.get("<UNK>", 1)
                ),
                self.skill_effect_target_vocab.get(
                    effect["target"] or "", self.skill_effect_target_vocab.get("<NONE>", 2)
                ),
                self.skill_effect_stage_vocab.get(
                    effect["stage"], self.skill_effect_stage_vocab.get("<UNK>", 1)
                ),
                min(max((effect["value"] or 0.0) / 10, -10.0), 10.0),
                duration_code(effect["duration"]),
            )
            for effect in static["effects"]
        ]
        state_values, state_mask = _skill_state_values(fields)
        return {
            "id": _index(skill_id, self.skill_vocab),
            "category": _index(category, self.category_vocab),
            "element": _index(element, self.element_vocab),
            "attribute_ids": [
                self.skill_attribute_vocab[key] for key in SKILL_ATTRIBUTE_KEYS
            ],
            "attribute_values": attributes,
            "attribute_mask": [True] * len(SKILL_ATTRIBUTE_KEYS),
            "effects": effects,
            "state_ids": [
                self.skill_state_attribute_vocab[key]
                for key in SKILL_STATE_ATTRIBUTE_KEYS
            ],
            "state_values": state_values,
            "state_mask": state_mask,
            "numeric": [
                _number(power, 200),
                _number(energy, 10),
                float(power is not None),
                float(energy is not None),
                float(bool(item)),
                float(counter == "attack"),
                float(counter == "defense"),
                float(counter == "status"),
                float(static["counter_interrupt"]),
                static["drive"] / 3,
                float(static["swift"]),
                float(static["windup"]),
                float(static["usable"]),
                float(static["choice"]),
            ],
            "mask": True,
            "power_raw": _scalar(power),
            "energy_raw": _scalar(energy),
            "category_raw": _scalar(category),
            "element_raw": _scalar(element),
            "static": static,
            "targets": targets,
        }

    def _public_own(self, raw):
        name = raw.get("name", raw.get("spirit", ""))
        species = self.spirit_by_name.get(name, {})
        max_hp = raw.get("max_hp", 0)
        hp = raw.get("hp", max_hp)
        return {
            "name": name,
            "spirit_id": raw.get("spirit_id", species.get("id")),
            "hp_percent": raw.get(
                "hp_percent", round(hp / max_hp * 100) if max_hp else 100
            ),
            "energy": raw.get("energy"),
            "eff_speed_range": raw.get("eff_speed_range") or raw.get("speed_range"),
            "buffs": raw.get("buffs", []),
            "used_skill_counts": raw.get("used_skill_counts", {}),
            "used_skill_history": raw.get("used_skill_history", []),
            "skills": [],
        }

    def _pet(self, raw, is_own, nature=None):
        name = raw.get("name", raw.get("spirit", ""))
        nature = nature if nature is not None else raw.get("nature")
        species = self.spirit_by_id.get(
            str(raw.get("spirit_id", ""))
        ) or self.spirit_by_name.get(name, {})
        stats = raw.get("stats", {}) if is_own else {}
        if is_own and not stats and species and "ivs" in raw:
            stats = self._stats_of(species, raw)
        stat_estimate = stats or self.prior_stats.get(species.get("id"), {})
        hp = raw.get("hp", raw.get("hp_percent", 100))
        max_hp = raw.get("max_hp", 100)
        hp_ratio = (
            _number(hp, max_hp)
            if is_own and "hp_percent" not in raw
            else _number(hp, 100) if "hp_percent" in raw else 0.0
        )
        speed_range = raw.get("eff_speed_range") or raw.get("speed_range")
        if not speed_range and species:
            speed_range = self._speed_range_of(species)
        speed_range = speed_range or [None, None]
        values = [
            hp_ratio,
            _number(raw.get("energy"), 10),
            _number(raw.get("level", 60), 100) if is_own else 0.0,
            *[
                _number(stat_estimate.get(key), 500)
                for key in ("hp", "atk", "def", "spatk", "spdef", "speed")
            ],
            _number(raw.get("eff_speed"), 500) if is_own else 0.0,
            _number(speed_range[0], 500),
            _number(speed_range[-1], 500),
            _number(raw.get("bloodline"), 18) if is_own else 0.0,
        ]
        numeric_mask = [
            "hp" in raw or "hp_percent" in raw or is_own,
            "energy" in raw,
            "level" in raw and is_own,
            *([bool(stat_estimate)] * 6),
            "eff_speed" in raw and is_own,
            speed_range[0] is not None,
            speed_range[-1] is not None,
            "bloodline" in raw and is_own,
        ]
        for key in ("atk", "def", "spatk", "spdef"):
            base = species.get("stats", {}).get(key)
            if stats:
                values.extend([_number(stats.get(key), 500)] * 2)
                known = key in stats
            else:
                values.extend(
                    [calc_stat(base, 0, 0.9) / 500, calc_stat(base, 10, 1.2) / 500]
                    if base is not None
                    else [0.0, 0.0]
                )
                known = base is not None
            numeric_mask.extend([known, known])
        skills = [self._skill(skill) for skill in raw.get("skills", [])]
        effects = [
            {
                "id": _index(effect.get("type"), self.effect_vocab),
                "source_id": _index(
                    self.spirit_by_name.get(effect.get("source_pet", ""), {}).get("id"),
                    self.spirit_vocab,
                ),
                "source_trait": _index(
                    effect.get("source_trait_id"), self.trait_vocab
                ),
                "numeric": [
                    _number(effect.get("value"), 100),
                    duration_code(effect.get("duration")),
                    1.0,
                    float(
                        bool(effect.get("source_side"))
                        and effect.get("source_side") == raw.get("side")
                    ),
                    {"": 0.0, "trait": 1.0, "debuff": 2.0}.get(
                        effect.get("source_kind", ""), 0.0
                    )
                    / 2.0,
                    float(effect.get("source_pet", "") == name),
                ],
                "mask": True,
            }
            for effect in raw.get("buffs", [])
        ]
        bloodline = raw.get("bloodline") if is_own else None
        state_values, state_mask = _pet_state_values(raw)
        return {
            "state_ids": [
                self.pet_state_attribute_vocab[key] for key in PET_STATE_ATTRIBUTE_KEYS
            ],
            "state_values": state_values,
            "state_mask": state_mask,
            "id": _index(species.get("id", raw.get("spirit_id")), self.spirit_vocab),
            "element": _index(species.get("mainElement"), self.element_vocab),
            "sub_element": (
                _index(species["subElement"], self.element_vocab)
                if species.get("subElement") is not None
                else 0
            ),
            "nature": _index(nature, self.nature_vocab) if nature is not None else 0,
            "trait": _index(species.get("feature", {}).get("id"), self.trait_vocab),
            "bloodline": _index(bloodline, self.bloodline_vocab),
            "bloodline_mask": bloodline is not None,
            "numeric": values,
            "numeric_mask": numeric_mask,
            "skills": skills,
            "effects": effects,
            "used_skill_history": raw.get("used_skill_history", []),
            "used_skill_ids": [
                _index(
                    entry.get("skill_id") if isinstance(entry, dict) else entry[1],
                    self.skill_vocab,
                )
                for entry in raw.get("used_skill_history", [])
            ],
            "mask": True,
        }

    def _resonance(self, state, side, revealed_only=False):
        magic = state.get("resonance_magic", {}).get(side)
        usage = state.get("resonance_usage", {}).get(side, {})
        cooldown = state.get("resonance_cooldown", {}).get(side, {})
        revealed = any(
            usage.get(magic_id, usage.get(str(magic_id), 0)) > 0
            for magic_id in (0, 1, 2)
        )
        if revealed_only and not revealed:
            magic = None
        values = [0.0 if magic is None else _number(magic, 2), float(magic is not None)]
        for magic_id in (0, 1, 2):
            values.append(min(usage.get(magic_id, usage.get(str(magic_id), 0)), 2) / 2)
        for magic_id in (0, 1, 2):
            values.append(
                min(cooldown.get(magic_id, cooldown.get(str(magic_id), 0)), 3) / 3
            )
        return values

    def _team_nature(self, team, index, raw):
        if raw.get("nature") is not None:
            return raw["nature"]
        name = raw.get("name", raw.get("spirit", ""))
        if (
            index < len(team)
            and team[index].get("name", team[index].get("spirit", "")) == name
        ):
            return team[index].get("nature")
        return next(
            (
                pet.get("nature")
                for pet in team
                if pet.get("name", pet.get("spirit", "")) == name
            ),
            None,
        )

    def _lord_branches(self, species):
        return [
            branch
            for chain in species.get("evolution", [])
            for branch in chain.get("lordBranches", [])
        ]

    def opponent_candidates(self, request):
        state = request.get("state", {})
        side = request.get("side", "A")
        enemy_side = "B" if side == "A" else "A"
        enemies = state.get("teams", {}).get(enemy_side, [])
        if request.get("phase") == "lead":
            names = request.get("opponent_team", []) or [
                pet.get("name", "") for pet in enemies
            ]
            return [
                {"kind": "lead", "lead": index} for index, _ in enumerate(names)
            ] or [{"kind": "unknown_skill"}]
        if not enemies:
            enemies = [{"name": name} for name in request.get("opponent_team", [])]
        active_index = state.get("active", {}).get(enemy_side, -1)
        if request.get("phase") == "replacement":
            return [
                {"kind": "switch", "pet_index": index}
                for index, raw in enumerate(enemies)
                if index != active_index and raw.get("hp_percent", 100) > 0
            ] or [{"kind": "unknown_skill"}]
        active = enemies[active_index] if 0 <= active_index < len(enemies) else {}
        species = self.spirit_by_name.get(
            active.get("name", "")
        ) or self.spirit_by_id.get(str(active.get("spirit_id", "")))
        skill_ids = list(species.get("skills", {}).get("normal", [])) if species else []
        skill_ids.extend(
            skill.get(
                "skill_id", self.skill_by_name.get(skill.get("name", ""), {}).get("id")
            )
            for skill in active.get("skills", [])
            if isinstance(skill, dict)
        )
        pool = []
        for skill_id in dict.fromkeys(skill_ids):
            if skill_id is None or str(skill_id) not in self.skill_by_id:
                continue
            for choice_branch in ((0, 1) if skill_id in self.choice_skills else (0,)):
                pool.append(
                    {
                        "kind": "skill",
                        "skill_id": skill_id,
                        "pet_index": active_index,
                        "choice_branch": choice_branch,
                    }
                )
        pool.extend(
            {"kind": "switch", "pet_index": index}
            for index, raw in enumerate(enemies)
            if index != active_index and raw.get("hp_percent", 100) > 0
        )
        pool.extend(({"kind": "charge"}, {"kind": "unknown_skill"}))
        magic_id = state.get("resonance_magic", {}).get(enemy_side)
        usage = state.get("resonance_usage", {}).get(enemy_side, {})
        cooldown = state.get("resonance_cooldown", {}).get(enemy_side, {})
        if (
            magic_id is not None
            and usage.get(magic_id, usage.get(str(magic_id), 0))
            < (2 if magic_id == 0 else 1)
            and cooldown.get(magic_id, cooldown.get(str(magic_id), 0)) <= 0
        ):
            branches = (
                len(self._lord_branches(species))
                if magic_id == 1 and species
                else (1 if magic_id != 1 else 0)
            )
            for candidate in tuple(pool):
                if candidate["kind"] == "unknown_skill":
                    continue
                for branch in range(branches):
                    pool.append(
                        {**candidate, "magic_id": magic_id, "magic_branch": branch}
                    )
        unique = {action_key(candidate): candidate for candidate in pool}
        return list(unique.values()) or [{"kind": "unknown_skill"}]

    def _event_vocab_index(self, value, vocab):
        return vocab.get(value, vocab.get(str(value), vocab.get("<UNK>", 1)))

    def _event_entity(self, entity, side):
        entity = entity if isinstance(entity, dict) else {}
        index = entity.get("index", -1)
        index = index if isinstance(index, int) and index >= 0 else -1
        return {
            "kind": self._event_vocab_index(
                entity.get("kind", ""), self.event_entity_vocab
            ),
            "relative": (
                1.0
                if entity.get("side") == side
                else -1.0
                if entity.get("side") in ("A", "B")
                else 0.0
            ),
            "index": index,
            "pet": (
                entity.get("kind") == "pet"
                and index >= 0
                and entity.get("spirit_id") is not None
            ),
            "spirit": _index(entity.get("spirit_id"), self.spirit_vocab),
            "trait": _index(entity.get("trait_id"), self.trait_vocab),
        }

    def _event_numeric(self, event, source, target, order, count):
        value = event.get("value", 0)
        duration = event.get("duration", "")
        attribute = event.get("attribute", "")
        value_known = isinstance(value, (int, float)) and not isinstance(value, bool)
        duration_known = isinstance(duration, (int, float)) and not isinstance(
            duration, bool
        )
        attribute_known = isinstance(
            attribute, (int, float)
        ) and not isinstance(attribute, bool)
        return [
            min(max(_number(event.get("turn"), 100), 0.0), 1.0),
            min(max(_number(event.get("seq"), 1000), 0.0), 10.0),
            order / max(count, 1),
            min(max(_number(value, 100), -10.0), 10.0),
            float(value_known and value != 0),
            min(max(_number(event.get("stacks"), 10), -10.0), 10.0),
            duration_code(duration) / 2,
            min(max(_number(duration, 10), 0.0), 10.0),
            float(duration_known or duration in ("normal", "temporary", "permanent")),
            min(max(_number(event.get("skill_index"), 10), -1.0), 1.0),
            float(bool(event.get("replay"))),
            float(bool(event.get("choice_branch"))),
            float(bool(event.get("public", True))),
            source["relative"],
            min(source["index"], 6) / 6 if source["index"] >= 0 else 0.0,
            float(source["pet"]),
            target["relative"],
            min(target["index"], 6) / 6 if target["index"] >= 0 else 0.0,
            float(target["pet"]),
            min(max(_number(attribute, 100), -10.0), 10.0),
            float(attribute_known or bool(isinstance(attribute, str) and attribute)),
        ]

    def _structured_event(self, event, side, order, count):
        source = self._event_entity(event.get("source"), side)
        target = self._event_entity(event.get("target"), side)
        return {
            "kind": self._event_vocab_index(
                event.get("kind", "unknown"), self.event_kind_vocab
            ),
            "phase": self._event_vocab_index(
                event.get("phase", ""), self.event_phase_vocab
            ),
            "action": self._event_vocab_index(
                event.get("action", ""), self.event_action_vocab
            ),
            "effect": self._event_vocab_index(
                event.get("effect", ""), self.event_effect_vocab
            ),
            "source_kind": source["kind"],
            "target_kind": target["kind"],
            "skill": _index(event.get("skill_id"), self.skill_vocab),
            "trait": _index(event.get("trait_id"), self.trait_vocab),
            "source_spirit": source["spirit"],
            "source_trait": source["trait"],
            "target_spirit": target["spirit"],
            "target_trait": target["trait"],
            "source_index": source["index"],
            "target_index": target["index"],
            "source_present": source["pet"],
            "target_present": target["pet"],
            "numeric": self._event_numeric(event, source, target, order, count),
            "mask": True,
            "public": bool(event.get("public", True)),
        }

    def _fallback_event(
        self, event, line, side, enemy_side, state, order, count, is_private, phase
    ):
        participants = sorted(
            (
                (pet.get("name", ""), team_side, index)
                for team_side, team in state.get("teams", {}).items()
                for index, pet in enumerate(team)
                if pet.get("name") and pet["name"] in line
            ),
            key=lambda item: (item[1] != side, item[2]),
        )
        source = participants[0] if participants else ("", "", -1)
        target = next(
            (item for item in participants[1:] if item[1:] != source[1:]),
            ("", "", -1),
        )

        def entity(item):
            name, team_side, index = item
            if not team_side or index < 0:
                return {"kind": "field"}
            pet = state.get("teams", {}).get(team_side, [])[index]
            return {
                "kind": "pet",
                "side": team_side,
                "index": index,
                "name": name,
                "spirit_id": pet.get("spirit_id"),
                "trait_id": pet.get("trait_id"),
            }

        skill_id = event.get("skill_id") if isinstance(event, dict) else None
        if skill_id is None and not is_private:
            skill_id = next(
                (
                    self.skill_by_name[name]["id"]
                    for name in self.skill_by_name
                    if name in line
                ),
                None,
            )
        if skill_id is None and is_private and isinstance(event, dict):
            skill_index = event.get("skill_index", -1)
            active = state.get("active", {}).get(side, -1)
            team = state.get("teams", {}).get(side, [])
            skills = team[active].get("skills", []) if 0 <= active < len(team) else []
            if 0 <= skill_index < len(skills):
                skill = skills[skill_index]
                skill_id = (
                    skill.get("skill_id", skill.get("name"))
                    if isinstance(skill, dict)
                    else skill
                )
        matches = re.findall(
            r"(?:造成|伤害|回复|恢复|获得|失去|损失|扣除)[^0-9-]*(-?\d+(?:\.\d+)?)",
            line,
        )
        raw_value = (
            event.get("value") if isinstance(event, dict) else None
        )
        kind = event.get("kind") if isinstance(event, dict) else None
        if kind not in self.event_kind_vocab:
            kind = "private" if is_private else "log"
        synthetic = {
            "kind": kind,
            "phase": (
                event.get("phase", phase) if isinstance(event, dict) else phase
            ),
            "action": "",
            "effect": "",
            "source": entity(source),
            "target": entity(target),
            "skill_id": skill_id,
            "trait_id": event.get("trait_id") if isinstance(event, dict) else None,
            "skill_index": event.get("skill_index", -1)
            if isinstance(event, dict)
            else -1,
            "value": (
                raw_value
                if isinstance(raw_value, (int, float))
                else float(matches[0]) if matches else 0
            ),
            "duration": event.get("duration", "") if isinstance(event, dict) else "",
            "stacks": event.get("stacks", 0) if isinstance(event, dict) else 0,
            "replay": event.get("replay", False) if isinstance(event, dict) else False,
            "choice_branch": event.get("choice_branch", 0)
            if isinstance(event, dict)
            else 0,
            "public": not is_private,
        }
        return self._structured_event(synthetic, side, order, count)

    def encode(self, request):
        state = request.get("state", {})
        side = request.get("side", "A")
        enemy_side = "B" if side == "A" else "A"
        phase = request.get("phase", "action")
        own_raw = state.get("teams", {}).get(side, [])
        enemy_raw = state.get("teams", {}).get(enemy_side, [])
        if phase == "lead" and not own_raw:
            own_raw = request.get("team", [])
        team = request.get("team", [])
        own_pets = [
            self._pet(raw, True, self._team_nature(team, index, raw))
            for index, raw in enumerate(own_raw)
        ]
        enemy_pets = [self._pet(raw, False) for raw in enemy_raw]
        public_state = request.get("public_state")
        if public_state is not None:
            public_own_raw = public_state.get("teams", {}).get(side, [])
            public_enemy_raw = public_state.get("teams", {}).get(enemy_side, [])
            public_log = "\n".join(str(line) for line in public_state.get("log", []))
            public_own = []
            for raw in public_own_raw:
                public_raw = self._public_own(raw)
                public_raw["skills"] = [
                    skill
                    for skill in raw.get("skills", [])
                    if (
                        skill.get("name", "") if isinstance(skill, dict) else str(skill)
                    )
                    in public_log
                ]
                public_own.append(self._pet(public_raw, False))
        else:
            public_own = [self._pet(self._public_own(raw), False) for raw in own_raw]
            public_enemy_raw = enemy_raw or [
                {"name": name} for name in request.get("opponent_team", [])
            ]
        public_pets = public_own + [self._pet(raw, False) for raw in public_enemy_raw]
        bloodline_size = len(self.bloodline_vocab) + len(SPECIAL)
        enemy_bloodline_targets = []
        enemy_bloodline_mask = []
        for raw in enemy_raw:
            value = raw.get("bloodline")
            enemy_bloodline_targets.append(
                _index(value, self.bloodline_vocab) if value is not None else 0
            )
            enemy_bloodline_mask.append(value is not None)
        enemy_bloodline_allowed = []
        for index in range(len(enemy_raw)):
            allowed = [True] * bloodline_size
            if index < len(public_enemy_raw):
                mapped = set()
                for skill in public_enemy_raw[index].get("skills", []):
                    skill_id = (
                        skill.get("skill_id", skill.get("name"))
                        if isinstance(skill, dict)
                        else skill
                    )
                    if isinstance(skill_id, str) and skill_id.isdigit():
                        skill_id = int(skill_id)
                    mapped.update(self.bloodline_skill_map.get(skill_id, ()))
                if mapped:
                    allowed = [False] * bloodline_size
                    for value in mapped:
                        allowed[_index(value, self.bloodline_vocab)] = True
            enemy_bloodline_allowed.append(allowed)
        structured = state.get("events")
        if structured:
            events = [
                self._structured_event(event, side, index, len(structured))
                for index, event in enumerate(structured)
            ]
        else:
            public_events = state.get("log", [])[-128:] or request.get("events", [])
            events = []
            for is_private, values in (
                (False, public_events),
                (True, request.get("private_events", [])),
            ):
                for index, event in enumerate(values):
                    line = event if isinstance(event, str) else event.get("line", "")
                    events.append(
                        self._fallback_event(
                            event,
                            str(line),
                            side,
                            enemy_side,
                            state,
                            index,
                            len(values),
                            is_private,
                            phase,
                        )
                    )
        fields = []
        weather = state.get("weather")
        fields.append(
            {
                "id": 0,
                "numeric": [
                    _number(weather),
                    _number(
                        weather.get("remaining") if isinstance(weather, dict) else 0
                    ),
                ],
                "mask": weather is not None,
            }
        )
        for field_id, mark_side in enumerate((side, enemy_side), 1):
            for polarity in ("positive", "negative"):
                mark = state.get("marks", {}).get(mark_side, {}).get(polarity)
                fields.append(
                    {
                        "id": field_id + (0 if polarity == "positive" else 2),
                        "numeric": [
                            _number(mark),
                            _number(
                                mark.get("stacks") if isinstance(mark, dict) else 0
                            ),
                        ],
                        "mask": mark is not None,
                    }
                )
        for field_id, key in enumerate(("magic", "resonance_magic"), 5):
            values = state.get(key, {})
            fields.append(
                {
                    "id": field_id,
                    "numeric": [
                        _number(values.get(side)),
                        _number(values.get(enemy_side)),
                    ],
                    "mask": bool(values),
                }
            )
        fields.append(
            {
                "id": 7,
                "numeric": [_number(state.get("turn", 0), 1000), 0.0],
                "mask": True,
            }
        )
        active = state.get("active", {}).get(side, 0)
        enemy_active = state.get("active", {}).get(enemy_side, -1)
        own_candidates = selectable_candidates(request)
        if phase == "lead" and not own_candidates:
            own_candidates = [
                {"kind": "lead", "lead": index} for index in range(len(own_raw))
            ]
        opponent_candidates = self.opponent_candidates(request)
        return {
            "phase": _index(phase, self.phase_vocab),
            "phase_is_lead": phase == "lead",
            "side": _index("own", self.side_vocab),
            "turn": _number(state.get("turn", 0), 1000),
            "weather": _index(weather, self.weather_vocab),
            "own_active": active if isinstance(active, int) else 0,
            "enemy_active": enemy_active if isinstance(enemy_active, int) else -1,
            "own_pets": own_pets,
            "enemy_pets": enemy_pets,
            "public_pets": public_pets,
            "fields": fields,
            "events": events,
            "own_resonance": self._resonance(state, side),
            "enemy_resonance": self._resonance(state, enemy_side, revealed_only=True),
            "enemy_resonance_target": (
                int(enemy_resonance)
                if (enemy_resonance := state.get("resonance_magic", {}).get(enemy_side))
                in (0, 1, 2)
                else 0
            ),
            "enemy_resonance_mask": state.get("resonance_magic", {}).get(enemy_side)
            in (0, 1, 2),
            "enemy_bloodline_targets": enemy_bloodline_targets,
            "enemy_bloodline_mask": enemy_bloodline_mask,
            "enemy_bloodline_allowed": enemy_bloodline_allowed,
            "own_candidates": [
                self._candidate(
                    action, active, own_raw, enemy_active, enemy_raw, is_own=True
                )
                for action in own_candidates
            ],
            "opponent_candidates": [
                self._candidate(
                    action, enemy_active, enemy_raw, active, own_raw, is_own=False
                )
                for action in opponent_candidates
            ],
            "action_mask": [
                bool(action.get("mask", True)) for action in own_candidates
            ],
            "opponent_mask": [True] * len(opponent_candidates),
        }

    def _stat_prior(self, species):
        base = species["stats"]
        keys = ("hp", "atk", "def", "spatk", "spdef", "speed")
        totals = {key: 0.0 for key in keys}
        count = 0
        for high in itertools.combinations(keys, 3):
            ivs = {key: 10 if key in high else 0 for key in keys}
            for multipliers in self.nature_mults:
                totals["hp"] += calc_hp(
                    base["hp"], ivs["hp"], multipliers["hp"]
                )
                for key in keys[1:]:
                    totals[key] += calc_stat(
                        base[key], ivs[key], multipliers[key]
                    )
                count += 1
        return {key: totals[key] / count for key in keys}

    def _species(self, raw):
        return self.spirit_by_id.get(
            str(raw.get("spirit_id", ""))
        ) or self.spirit_by_name.get(raw.get("name", raw.get("spirit", "")), {})

    def _elements(self, raw):
        attributes = raw.get("attributes")
        if attributes:
            return list(attributes)
        species = self._species(raw)
        return [
            element
            for element in (species.get("mainElement"), species.get("subElement"))
            if element is not None
        ]

    def _stats(self, raw):
        if raw.get("stats"):
            return raw["stats"]
        species = self._species(raw)
        return self.prior_stats.get(species.get("id"), {})

    def _speed(self, raw):
        speed = raw.get("eff_speed")
        if isinstance(speed, (int, float)):
            return float(speed)
        speed_range = raw.get("eff_speed_range") or raw.get("speed_range")
        if speed_range:
            return (speed_range[0] + speed_range[-1]) / 2
        speed = (raw.get("stats") or {}).get("speed")
        return float(speed) if isinstance(speed, (int, float)) else None

    def _speed_range(self, raw):
        speed = raw.get("eff_speed")
        if isinstance(speed, (int, float)):
            return [float(speed), float(speed)]
        speed_range = raw.get("eff_speed_range") or raw.get("speed_range")
        if speed_range:
            return [speed_range[0], speed_range[-1]]
        speed = (raw.get("stats") or {}).get("speed")
        return (
            [float(speed), float(speed)]
            if isinstance(speed, (int, float))
            else [None, None]
        )

    def _initiative(self, speed, speed_range):
        low, high = speed_range
        if speed is None or low is None or high is None:
            return 0.5
        if high <= low:
            return 0.5 if speed == low else float(speed > low)
        return min(max((speed - low) / (high - low), 0.0), 1.0)

    def _current_hp(self, raw, stats):
        hp = raw.get("hp")
        if isinstance(hp, (int, float)):
            return float(hp)
        percent = raw.get("hp_percent", 100)
        return (
            float(percent) / 100 * stats.get("hp", 0)
            if isinstance(percent, (int, float))
            else 0.0
        )

    def _hp_ratio(self, raw):
        if "hp_percent" in raw:
            return _number(raw["hp_percent"], 100)
        return _number(raw.get("hp"), raw.get("max_hp", 100)) if "hp" in raw else 0.0

    def _best_mult(self, raw, defender_elements):
        if not defender_elements:
            return 1.0
        skills = raw.get("skills") or self._species(raw).get("skills", {}).get(
            "normal", []
        )
        mults = [
            type_multiplier(skill["element_raw"], defender_elements, self.typechart)
            for skill in map(self._skill, skills)
            if skill["category_raw"] in (0, 1) and skill["element_raw"] is not None
        ]
        return max(mults, default=1.0)

    def _estimated_damage(self, skill, attacker, defender):
        category = skill["category_raw"]
        power = skill["power_raw"]
        if category not in (0, 1) or not isinstance(power, (int, float)) or power <= 0:
            return 0
        attacker_stats = self._stats(attacker)
        defender_stats = self._stats(defender)
        attack = attacker_stats.get("atk" if category == 0 else "spatk")
        defense = defender_stats.get("def" if category == 0 else "spdef")
        if not isinstance(attack, (int, float)) or not isinstance(
            defense, (int, float)
        ):
            return 0
        element = skill["element_raw"]
        stab = (
            1.25 if element is not None and element in self._elements(attacker) else 1.0
        )
        mult = (
            type_multiplier(element, self._elements(defender), self.typechart)
            if element is not None
            else 1.0
        )
        level = attacker.get("level")
        level = level if isinstance(level, (int, float)) else 60
        return int(
            attack
            * power
            * stab
            * mult
            * (level * 45 / 100 + 10)
            / 41
            / max(defense, 1)
        )

    def _skill_derived(self, skill, attacker, defender):
        element = skill["element_raw"]
        attacker_elements = self._elements(attacker)
        mult = (
            type_multiplier(element, self._elements(defender), self.typechart)
            if element is not None
            else 1.0
        )
        damage = self._estimated_damage(skill, attacker, defender)
        defender_hp = self._current_hp(defender, self._stats(defender))
        static = skill["static"]
        counter = static["counter_target"]
        energy = attacker.get("energy")
        energy = float(energy) if isinstance(energy, (int, float)) else 0.0
        cost = skill["energy_raw"]
        cost = float(cost) if isinstance(cost, (int, float)) else 0.0
        return [
            _log_mult(mult),
            float(element is not None and element in attacker_elements),
            min(damage / defender_hp, 2) / 2 if defender_hp > 0 else 0.0,
            (
                min(math.ceil(defender_hp / damage), 10) / 10
                if damage > 0 and defender_hp > 0
                else 1.0
            ),
            self._initiative(self._speed(attacker), self._speed_range(defender)),
            min(max(energy - cost, 0.0), 10.0) / 10,
            float(energy >= cost),
            float(counter == "attack"),
            float(counter == "defense"),
            float(counter == "status"),
            float(static["counter_interrupt"]),
            min(max(static["drive"] / 3, 0.0), 1.0),
            float(static["swift"]),
            float(static["windup"]),
            float(static["usable"]),
            float(static["choice"]),
        ]

    def _switch_derived(self, team, pet_index, enemy_team, enemy_index):
        if not 0 <= pet_index < len(team):
            return [0.0] * 4
        entry = team[pet_index]
        defender = enemy_team[enemy_index] if 0 <= enemy_index < len(enemy_team) else {}
        return [
            _log_mult(self._best_mult(entry, self._elements(defender))),
            _log_mult(self._best_mult(defender, self._elements(entry))),
            self._hp_ratio(entry),
            self._initiative(self._speed(entry), self._speed_range(defender)),
        ]

    def _candidate(
        self,
        action,
        active_index=None,
        team=None,
        enemy_index=None,
        enemy_team=None,
        is_own=False,
    ):
        kind = action.get("kind", "unknown_skill")
        team = team or []
        enemy_team = enemy_team or []
        active_index = active_index if isinstance(active_index, int) else -1
        enemy_index = enemy_index if isinstance(enemy_index, int) else -1
        skill_id = action.get("skill_id")
        skill_index = action.get("skill_index", -1)
        pet_index = action.get(
            "pet_index", action.get("lead", active_index if active_index >= 0 else 0)
        )
        pet_index = int(pet_index if pet_index is not None else 0)
        skills = team[pet_index].get("skills", []) if 0 <= pet_index < len(team) else []
        raw_skill = skills[skill_index] if 0 <= skill_index < len(skills) else None
        if raw_skill is None and skill_id is not None:
            raw_skill = next(
                (
                    skill
                    for skill in skills
                    if isinstance(skill, dict)
                    and str(skill.get("skill_id", skill.get("id"))) == str(skill_id)
                ),
                None,
            )
        if skill_id is None and raw_skill is not None:
            skill_id = (
                raw_skill.get("skill_id", raw_skill.get("name"))
                if isinstance(raw_skill, dict)
                else raw_skill
            )
        skill_data = (
            self._skill(raw_skill if raw_skill is not None else skill_id)
            if raw_skill is not None or skill_id is not None
            else self._skill(None)
        )
        attacker = team[pet_index] if 0 <= pet_index < len(team) else {}
        defender = enemy_team[enemy_index] if 0 <= enemy_index < len(enemy_team) else {}
        derived = [0.0] * 27
        if kind == "skill":
            derived[:16] = self._skill_derived(skill_data, attacker, defender)
        elif kind == "switch":
            derived[16:20] = self._switch_derived(
                team, pet_index, enemy_team, enemy_index
            )
        derived[20] = float(kind == "skill")
        derived[21] = float(kind == "switch")
        derived[22] = float(kind == "charge")
        derived[23] = float(kind == "lead")
        magic_id = action.get("magic_id")
        bloodline = _scalar(attacker.get("bloodline")) if is_own else None
        species = self._species(attacker)
        if magic_id == 0:
            element = bloodline if bloodline is not None else species.get("mainElement")
            derived[24] = _number(element, 18)
            derived[25] = float(bloodline == LORD_BLOODLINE)
        elif magic_id == 1:
            derived[25] = float(bloodline == LORD_BLOODLINE)
            derived[26] = min(len(self._lord_branches(species)), 8) / 8
        targets = skill_data.get("targets", set())
        numeric_skill_id = (
            int(skill_id)
            if isinstance(skill_id, int)
            or (isinstance(skill_id, str) and skill_id.isdigit())
            else self.skill_by_name.get(str(skill_id), {}).get("id")
        )
        return {
            "key": action_key(action),
            "kind": _index(kind, self.action_vocab),
            "pet_index": pet_index,
            "skill": skill_data["id"],
            "skill_id": numeric_skill_id,
            "magic_id": int(
                action.get("magic_id", -1) if action.get("magic_id") is not None else -1
            ),
            "branches": [
                int(action.get("magic_branch", 0)),
                int(action.get("choice_branch", 0)),
            ],
            "numeric": skill_data["numeric"][:2],
            "targets": [
                float("self" in targets),
                float("enemy" in targets),
                float("self_backline" in targets),
                float("self_team" in targets),
                float("enemy_team" in targets),
                float("field" in targets or "both" in targets),
            ],
            "skill_index": int(skill_index),
            "derived": derived,
            "skill_present": bool(kind == "skill" and skill_id is not None),
            "skill_attribute_ids": skill_data["attribute_ids"],
            "skill_attributes": skill_data["attribute_values"],
            "skill_effects": skill_data["effects"],
            "mask": bool(action.get("mask", True)),
        }

    def candidate_target_mask(self, item, index):
        candidates = item.get("own_candidates", [])
        if not 0 <= index < len(candidates):
            return None
        candidate = candidates[index]
        skill_id = candidate.get("skill_id")
        if self.skill_by_id.get(str(skill_id), {}).get("category") != 3:
            return None
        flags = candidate.get("targets", ())
        if not any(flags):
            return None
        actor = max(0, min(int(candidate.get("pet_index", 0)), 5))
        enemy = int(item.get("enemy_active", -1))
        mask = [0.0] * 12
        if flags[0]:
            mask[actor] = 1.0
        if flags[1] and 0 <= enemy < 6:
            mask[6 + enemy] = 1.0
        if flags[2]:
            for pet in range(6):
                if pet != actor:
                    mask[pet] = 1.0
        if flags[3]:
            for pet in range(6):
                mask[pet] = 1.0
        if flags[4]:
            for pet in range(6):
                mask[6 + pet] = 1.0
        if flags[5]:
            mask[actor] = 1.0
            if 0 <= enemy < 6:
                mask[6 + enemy] = 1.0
        return mask if any(mask) else None


def selectable_candidates(request):
    return [
        candidate
        for candidate in request.get("candidates", [])
        if candidate.get("kind") != "flee"
    ]


def action_key(action):
    kind = action.get("kind", "unknown_skill")
    if kind == "lead":
        return (
            "lead",
            int(action.get("lead", action.get("pet_index", 0))),
            action.get("magic_id"),
            int(action.get("magic_branch", 0)),
        )
    if kind in ("switch", "flee"):
        identity = int(action.get("pet_index", -1)) if kind == "switch" else None
        return (
            kind,
            identity,
            action.get("magic_id"),
            int(action.get("magic_branch", 0)),
        )
    if kind == "skill":
        skill_id = action.get("skill_id")
        if skill_id is None and "skill_index" in action:
            skill_id = ("slot", int(action["skill_index"]))
        return (
            "skill",
            skill_id,
            action.get("magic_id"),
            int(action.get("magic_branch", 0)),
            int(action.get("choice_branch", 0)),
        )
    if kind == "charge":
        return ("charge", action.get("magic_id"), int(action.get("magic_branch", 0)))
    if kind == "unknown_skill":
        return ("unknown_skill", action.get("skill_id"), action.get("pet_index"))
    return (kind, action.get("skill_id"), action.get("pet_index"))


_NP_DTYPES = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.long: np.int64,
    torch.bool: np.bool_,
}


def _np_zeros(*shape, dtype=torch.float32, device=None):
    return np.zeros(shape, dtype=_NP_DTYPES[dtype])


def _np_array(values, dtype=torch.float32, device=None):
    return np.asarray(values, dtype=_NP_DTYPES[dtype])


def collate_inputs(items, device="cpu"):
    if not items:
        return {}
    target_device = device
    device = "cpu"
    if all(
        all(isinstance(value, torch.Tensor) for value in item.values())
        for item in items
    ):
        result = {}
        for key in items[0]:
            tensors = [item[key] for item in items]
            if tensors[0].ndim == 0:
                result[key] = torch.stack(tensors).to(target_device)
            else:
                shape = tuple(
                    max(tensor.shape[axis] for tensor in tensors)
                    for axis in range(tensors[0].ndim)
                )
                batch = torch.zeros(
                    (len(items), *shape), dtype=tensors[0].dtype, device=device
                )
                for index, tensor in enumerate(tensors):
                    slices = tuple(slice(0, size) for size in tensor.shape)
                    batch[(index, *slices)] = tensor.to(device)
                result[key] = batch.to(target_device)
        return result
    batch_size = len(items)
    max_pets = max(
        1,
        max(
            max(len(item[key]) for key in ("own_pets", "enemy_pets", "public_pets"))
            for item in items
        ),
    )
    max_skills = max(
        1,
        max(
            (
                len(pet["skills"])
                for item in items
                for key in ("own_pets", "enemy_pets")
                for pet in item[key]
            ),
            default=0,
        ),
    )
    max_effects = max(
        1,
        max(
            (
                len(pet["effects"])
                for item in items
                for key in ("own_pets", "enemy_pets")
                for pet in item[key]
            ),
            default=0,
        ),
    )
    max_skill_history = max(
        1,
        max(
            (
                len(pet.get("used_skill_history", ()))
                for item in items
                for key in ("own_pets", "enemy_pets", "public_pets")
                for pet in item[key]
            ),
            default=0,
        ),
    )
    max_skill_effects = max(
        1,
        max(
            (
                len(skill["effects"])
                for item in items
                for key in ("own_pets", "enemy_pets")
                for pet in item[key]
                for skill in pet["skills"]
            ),
            default=0,
        ),
    )
    max_candidate_effects = max(
        1,
        max(
            (
                len(candidate.get("skill_effects", ()))
                for item in items
                for key in ("own_candidates", "opponent_candidates")
                for candidate in item[key]
            ),
            default=0,
        ),
    )
    max_fields = max(1, max(len(item["fields"]) for item in items))
    max_events = max(1, max(len(item["events"]) for item in items))
    max_actions = max(1, max(len(item["own_candidates"]) for item in items))
    max_opponents = max(1, max(len(item["opponent_candidates"]) for item in items))
    bloodline_size = max(
        1,
        max(
            (
                len(allowed)
                for item in items
                for allowed in item["enemy_bloodline_allowed"]
            ),
            default=1,
        ),
    )
    result = {
        "phase": _np_array(
            [item["phase"] for item in items], dtype=torch.long, device=device
        ),
        "phase_is_lead": _np_array(
            [item["phase_is_lead"] for item in items], dtype=torch.bool, device=device
        ),
        "side": _np_array(
            [item["side"] for item in items], dtype=torch.long, device=device
        ),
        "turn": _np_array(
            [item["turn"] for item in items], dtype=torch.float32, device=device
        ),
        "weather": _np_array(
            [item["weather"] for item in items], dtype=torch.long, device=device
        ),
        "own_active": _np_array(
            [
                min(
                    max(int(item["own_active"]), 0),
                    max(len(item["own_pets"]) - 1, 0),
                )
                for item in items
            ],
            dtype=torch.long,
            device=device,
        ),
        "enemy_active": _np_array(
            [
                min(
                    max(int(item["enemy_active"]), 0),
                    max(len(item["enemy_pets"]) - 1, 0),
                )
                for item in items
            ],
            dtype=torch.long,
            device=device,
        ),
    }
    for prefix in ("own", "enemy", "public"):
        result[f"{prefix}_pets"] = _np_zeros(
            batch_size, max_pets, 42, dtype=torch.float32, device=device
        )
        result[f"{prefix}_pet_ids"] = _np_zeros(
            batch_size, max_pets, dtype=torch.long, device=device
        )
        result[f"{prefix}_pet_elements"] = _np_zeros(
            batch_size, max_pets, dtype=torch.long, device=device
        )
        result[f"{prefix}_pet_sub_elements"] = _np_zeros(
            batch_size, max_pets, dtype=torch.long, device=device
        )
        result[f"{prefix}_pet_natures"] = _np_zeros(
            batch_size, max_pets, dtype=torch.long, device=device
        )
        result[f"{prefix}_pet_traits"] = _np_zeros(
            batch_size, max_pets, dtype=torch.long, device=device
        )
        result[f"{prefix}_pet_bloodlines"] = _np_zeros(
            batch_size, max_pets, dtype=torch.long, device=device
        )
        result[f"{prefix}_pet_bloodline_mask"] = _np_zeros(
            batch_size, max_pets, dtype=torch.bool, device=device
        )
        result[f"{prefix}_pet_mask"] = _np_zeros(
            batch_size, max_pets, dtype=torch.bool, device=device
        )
        result[f"{prefix}_used_skill_turns"] = _np_zeros(
            batch_size, max_pets, max_skill_history, dtype=torch.float32, device=device
        )
        result[f"{prefix}_used_skill_ids"] = _np_zeros(
            batch_size, max_pets, max_skill_history, dtype=torch.long, device=device
        )
        result[f"{prefix}_used_skill_replay"] = _np_zeros(
            batch_size, max_pets, max_skill_history, dtype=torch.bool, device=device
        )
        result[f"{prefix}_used_skill_mask"] = _np_zeros(
            batch_size, max_pets, max_skill_history, dtype=torch.bool, device=device
        )
    for prefix in ("own", "enemy"):
        result[f"{prefix}_skills"] = _np_zeros(
            batch_size, max_pets, max_skills, 14, dtype=torch.float32, device=device
        )
        result[f"{prefix}_skill_ids"] = _np_zeros(
            batch_size, max_pets, max_skills, dtype=torch.long, device=device
        )
        result[f"{prefix}_skill_elements"] = _np_zeros(
            batch_size, max_pets, max_skills, dtype=torch.long, device=device
        )
        result[f"{prefix}_skill_categories"] = _np_zeros(
            batch_size, max_pets, max_skills, dtype=torch.long, device=device
        )
        result[f"{prefix}_skill_mask"] = _np_zeros(
            batch_size, max_pets, max_skills, dtype=torch.bool, device=device
        )
        result[f"{prefix}_skill_attribute_ids"] = _np_zeros(
            batch_size,
            max_pets,
            max_skills,
            len(SKILL_ATTRIBUTE_KEYS),
            dtype=torch.long,
            device=device,
        )
        result[f"{prefix}_skill_attribute_values"] = _np_zeros(
            batch_size,
            max_pets,
            max_skills,
            len(SKILL_ATTRIBUTE_KEYS),
            dtype=torch.float32,
            device=device,
        )
        result[f"{prefix}_skill_attribute_mask"] = _np_zeros(
            batch_size,
            max_pets,
            max_skills,
            len(SKILL_ATTRIBUTE_KEYS),
            dtype=torch.bool,
            device=device,
        )
        result[f"{prefix}_skill_effect_kind"] = _np_zeros(
            batch_size,
            max_pets,
            max_skills,
            max_skill_effects,
            dtype=torch.long,
            device=device,
        )
        result[f"{prefix}_skill_effect_target"] = _np_zeros(
            batch_size,
            max_pets,
            max_skills,
            max_skill_effects,
            dtype=torch.long,
            device=device,
        )
        result[f"{prefix}_skill_effect_stage"] = _np_zeros(
            batch_size,
            max_pets,
            max_skills,
            max_skill_effects,
            dtype=torch.long,
            device=device,
        )
        result[f"{prefix}_skill_effect_value"] = _np_zeros(
            batch_size,
            max_pets,
            max_skills,
            max_skill_effects,
            dtype=torch.float32,
            device=device,
        )
        result[f"{prefix}_skill_effect_duration"] = _np_zeros(
            batch_size,
            max_pets,
            max_skills,
            max_skill_effects,
            dtype=torch.float32,
            device=device,
        )
        result[f"{prefix}_skill_effect_mask"] = _np_zeros(
            batch_size,
            max_pets,
            max_skills,
            max_skill_effects,
            dtype=torch.bool,
            device=device,
        )
        result[f"{prefix}_skill_state_ids"] = _np_zeros(
            batch_size,
            max_pets,
            max_skills,
            len(SKILL_STATE_ATTRIBUTE_KEYS),
            dtype=torch.long,
            device=device,
        )
        result[f"{prefix}_skill_state_values"] = _np_zeros(
            batch_size,
            max_pets,
            max_skills,
            len(SKILL_STATE_ATTRIBUTE_KEYS),
            dtype=torch.float32,
            device=device,
        )
        result[f"{prefix}_skill_state_mask"] = _np_zeros(
            batch_size,
            max_pets,
            max_skills,
            len(SKILL_STATE_ATTRIBUTE_KEYS),
            dtype=torch.bool,
            device=device,
        )
        result[f"{prefix}_pet_state_ids"] = _np_zeros(
            batch_size,
            max_pets,
            len(PET_STATE_ATTRIBUTE_KEYS),
            dtype=torch.long,
            device=device,
        )
        result[f"{prefix}_pet_state_values"] = _np_zeros(
            batch_size,
            max_pets,
            len(PET_STATE_ATTRIBUTE_KEYS),
            dtype=torch.float32,
            device=device,
        )
        result[f"{prefix}_pet_state_mask"] = _np_zeros(
            batch_size,
            max_pets,
            len(PET_STATE_ATTRIBUTE_KEYS),
            dtype=torch.bool,
            device=device,
        )
        result[f"{prefix}_effects"] = _np_zeros(
            batch_size, max_pets, max_effects, 6, dtype=torch.float32, device=device
        )
        result[f"{prefix}_effect_ids"] = _np_zeros(
            batch_size, max_pets, max_effects, dtype=torch.long, device=device
        )
        result[f"{prefix}_effect_source_ids"] = _np_zeros(
            batch_size, max_pets, max_effects, dtype=torch.long, device=device
        )
        result[f"{prefix}_effect_source_traits"] = _np_zeros(
            batch_size, max_pets, max_effects, dtype=torch.long, device=device
        )
        result[f"{prefix}_effect_mask"] = _np_zeros(
            batch_size, max_pets, max_effects, dtype=torch.bool, device=device
        )
    result["public_pet_state_ids"] = _np_zeros(
        batch_size,
        max_pets,
        len(PET_STATE_ATTRIBUTE_KEYS),
        dtype=torch.long,
        device=device,
    )
    result["public_pet_state_values"] = _np_zeros(
        batch_size,
        max_pets,
        len(PET_STATE_ATTRIBUTE_KEYS),
        dtype=torch.float32,
        device=device,
    )
    result["public_pet_state_mask"] = _np_zeros(
        batch_size,
        max_pets,
        len(PET_STATE_ATTRIBUTE_KEYS),
        dtype=torch.bool,
        device=device,
    )
    result.update(
        {
            "fields": _np_zeros(
                batch_size, max_fields, 5, dtype=torch.float32, device=device
            ),
            "field_mask": _np_zeros(
                batch_size, max_fields, dtype=torch.bool, device=device
            ),
            "events": _np_zeros(
                batch_size,
                max_events,
                EVENT_FEATURES,
                dtype=torch.float32,
                device=device,
            ),
            "event_mask": _np_zeros(
                batch_size, max_events, dtype=torch.bool, device=device
            ),
            "event_public_mask": _np_zeros(
                batch_size, max_events, dtype=torch.bool, device=device
            ),
            "event_kind_ids": _np_zeros(
                batch_size, max_events, dtype=torch.long, device=device
            ),
            "event_phase_ids": _np_zeros(
                batch_size, max_events, dtype=torch.long, device=device
            ),
            "event_action_ids": _np_zeros(
                batch_size, max_events, dtype=torch.long, device=device
            ),
            "event_effect_ids": _np_zeros(
                batch_size, max_events, dtype=torch.long, device=device
            ),
            "event_source_kind_ids": _np_zeros(
                batch_size, max_events, dtype=torch.long, device=device
            ),
            "event_target_kind_ids": _np_zeros(
                batch_size, max_events, dtype=torch.long, device=device
            ),
            "event_skill_ids": _np_zeros(
                batch_size, max_events, dtype=torch.long, device=device
            ),
            "event_trait_ids": _np_zeros(
                batch_size, max_events, dtype=torch.long, device=device
            ),
            "event_source_spirit_ids": _np_zeros(
                batch_size, max_events, dtype=torch.long, device=device
            ),
            "event_source_trait_ids": _np_zeros(
                batch_size, max_events, dtype=torch.long, device=device
            ),
            "event_target_spirit_ids": _np_zeros(
                batch_size, max_events, dtype=torch.long, device=device
            ),
            "event_target_trait_ids": _np_zeros(
                batch_size, max_events, dtype=torch.long, device=device
            ),
            "event_source_indices": _np_zeros(
                batch_size, max_events, dtype=torch.long, device=device
            ),
            "event_target_indices": _np_zeros(
                batch_size, max_events, dtype=torch.long, device=device
            ),
            "event_source_present": _np_zeros(
                batch_size, max_events, dtype=torch.bool, device=device
            ),
            "event_target_present": _np_zeros(
                batch_size, max_events, dtype=torch.bool, device=device
            ),
            "own_resonance": _np_zeros(
                batch_size, 8, dtype=torch.float32, device=device
            ),
            "enemy_resonance": _np_zeros(
                batch_size, 8, dtype=torch.float32, device=device
            ),
            "enemy_resonance_target": _np_zeros(
                batch_size, dtype=torch.long, device=device
            ),
            "enemy_resonance_mask": _np_zeros(
                batch_size, dtype=torch.bool, device=device
            ),
            "enemy_bloodline_targets": _np_zeros(
                batch_size, max_pets, dtype=torch.long, device=device
            ),
            "enemy_bloodline_mask": _np_zeros(
                batch_size, max_pets, dtype=torch.bool, device=device
            ),
            "enemy_bloodline_allowed": _np_zeros(
                batch_size, max_pets, bloodline_size, dtype=torch.bool, device=device
            ),
            "own_candidates": _np_zeros(
                batch_size, max_actions, 36, dtype=torch.float32, device=device
            ),
            "own_candidate_targets": _np_zeros(
                batch_size, max_actions, 6, dtype=torch.float32, device=device
            ),
            "action_mask": _np_zeros(
                batch_size, max_actions, dtype=torch.bool, device=device
            ),
            "opponent_candidates": _np_zeros(
                batch_size, max_opponents, 36, dtype=torch.float32, device=device
            ),
            "opponent_mask": _np_zeros(
                batch_size, max_opponents, dtype=torch.bool, device=device
            ),
            "own_candidate_skill_present": _np_zeros(
                batch_size, max_actions, dtype=torch.bool, device=device
            ),
            "own_candidate_skill_attributes": _np_zeros(
                batch_size,
                max_actions,
                len(SKILL_ATTRIBUTE_KEYS),
                dtype=torch.float32,
                device=device,
            ),
            "own_candidate_skill_attribute_ids": _np_zeros(
                batch_size,
                max_actions,
                len(SKILL_ATTRIBUTE_KEYS),
                dtype=torch.long,
                device=device,
            ),
            "opponent_candidate_skill_present": _np_zeros(
                batch_size, max_opponents, dtype=torch.bool, device=device
            ),
            "opponent_candidate_skill_attributes": _np_zeros(
                batch_size,
                max_opponents,
                len(SKILL_ATTRIBUTE_KEYS),
                dtype=torch.float32,
                device=device,
            ),
            "opponent_candidate_skill_attribute_ids": _np_zeros(
                batch_size,
                max_opponents,
                len(SKILL_ATTRIBUTE_KEYS),
                dtype=torch.long,
                device=device,
            ),
            "own_candidate_skill_effect_kind": _np_zeros(
                batch_size,
                max_actions,
                max_candidate_effects,
                dtype=torch.long,
                device=device,
            ),
            "own_candidate_skill_effect_target": _np_zeros(
                batch_size,
                max_actions,
                max_candidate_effects,
                dtype=torch.long,
                device=device,
            ),
            "own_candidate_skill_effect_stage": _np_zeros(
                batch_size,
                max_actions,
                max_candidate_effects,
                dtype=torch.long,
                device=device,
            ),
            "own_candidate_skill_effect_value": _np_zeros(
                batch_size,
                max_actions,
                max_candidate_effects,
                dtype=torch.float32,
                device=device,
            ),
            "own_candidate_skill_effect_duration": _np_zeros(
                batch_size,
                max_actions,
                max_candidate_effects,
                dtype=torch.float32,
                device=device,
            ),
            "own_candidate_skill_effect_mask": _np_zeros(
                batch_size,
                max_actions,
                max_candidate_effects,
                dtype=torch.bool,
                device=device,
            ),
            "opponent_candidate_skill_effect_kind": _np_zeros(
                batch_size,
                max_opponents,
                max_candidate_effects,
                dtype=torch.long,
                device=device,
            ),
            "opponent_candidate_skill_effect_target": _np_zeros(
                batch_size,
                max_opponents,
                max_candidate_effects,
                dtype=torch.long,
                device=device,
            ),
            "opponent_candidate_skill_effect_stage": _np_zeros(
                batch_size,
                max_opponents,
                max_candidate_effects,
                dtype=torch.long,
                device=device,
            ),
            "opponent_candidate_skill_effect_value": _np_zeros(
                batch_size,
                max_opponents,
                max_candidate_effects,
                dtype=torch.float32,
                device=device,
            ),
            "opponent_candidate_skill_effect_duration": _np_zeros(
                batch_size,
                max_opponents,
                max_candidate_effects,
                dtype=torch.float32,
                device=device,
            ),
            "opponent_candidate_skill_effect_mask": _np_zeros(
                batch_size,
                max_opponents,
                max_candidate_effects,
                dtype=torch.bool,
                device=device,
            ),
        }
    )
    for batch_index, item in enumerate(items):
        for prefix in ("own", "enemy", "public"):
            for pet_index, pet in enumerate(item[f"{prefix}_pets"]):
                result[f"{prefix}_pets"][batch_index, pet_index] = _np_array(
                    [*pet["numeric"], *pet["numeric_mask"]],
                    dtype=torch.float32,
                    device=device,
                )
                result[f"{prefix}_pet_ids"][batch_index, pet_index] = pet["id"]
                result[f"{prefix}_pet_elements"][batch_index, pet_index] = pet[
                    "element"
                ]
                result[f"{prefix}_pet_sub_elements"][batch_index, pet_index] = pet[
                    "sub_element"
                ]
                result[f"{prefix}_pet_natures"][batch_index, pet_index] = pet["nature"]
                result[f"{prefix}_pet_traits"][batch_index, pet_index] = pet["trait"]
                result[f"{prefix}_pet_bloodlines"][batch_index, pet_index] = pet[
                    "bloodline"
                ]
                result[f"{prefix}_pet_bloodline_mask"][batch_index, pet_index] = pet[
                    "bloodline_mask"
                ]
                result[f"{prefix}_pet_mask"][batch_index, pet_index] = pet["mask"]
                for history_index, entry in enumerate(
                    pet.get("used_skill_history", ())
                ):
                    if isinstance(entry, dict):
                        turn = entry.get("turn", 0)
                        skill_id = entry.get("skill_id")
                        replay = entry.get("replay", False)
                    else:
                        turn, skill_id, replay = entry
                    result[f"{prefix}_used_skill_turns"][
                        batch_index, pet_index, history_index
                    ] = _number(turn, 100)
                    result[f"{prefix}_used_skill_ids"][
                        batch_index, pet_index, history_index
                    ] = pet["used_skill_ids"][history_index]
                    result[f"{prefix}_used_skill_replay"][
                        batch_index, pet_index, history_index
                    ] = bool(replay)
                    result[f"{prefix}_used_skill_mask"][
                        batch_index, pet_index, history_index
                    ] = True
                result[f"{prefix}_pet_state_ids"][batch_index, pet_index] = (
                    _np_array(pet["state_ids"], dtype=torch.long, device=device)
                )
                result[f"{prefix}_pet_state_values"][batch_index, pet_index] = (
                    _np_array(
                        pet["state_values"], dtype=torch.float32, device=device
                    )
                )
                result[f"{prefix}_pet_state_mask"][batch_index, pet_index] = (
                    _np_array(pet["state_mask"], dtype=torch.bool, device=device)
                )
                if prefix == "public":
                    continue
                for skill_index, skill in enumerate(pet["skills"]):
                    result[f"{prefix}_skills"][batch_index, pet_index, skill_index] = (
                        _np_array(
                            skill["numeric"], dtype=torch.float32, device=device
                        )
                    )
                    result[f"{prefix}_skill_ids"][
                        batch_index, pet_index, skill_index
                    ] = skill["id"]
                    result[f"{prefix}_skill_elements"][
                        batch_index, pet_index, skill_index
                    ] = skill["element"]
                    result[f"{prefix}_skill_categories"][
                        batch_index, pet_index, skill_index
                    ] = skill["category"]
                    result[f"{prefix}_skill_mask"][
                        batch_index, pet_index, skill_index
                    ] = skill["mask"]
                    result[f"{prefix}_skill_attribute_ids"][
                        batch_index, pet_index, skill_index
                    ] = _np_array(
                        skill["attribute_ids"], dtype=torch.long, device=device
                    )
                    result[f"{prefix}_skill_attribute_values"][
                        batch_index, pet_index, skill_index
                    ] = _np_array(
                        skill["attribute_values"], dtype=torch.float32, device=device
                    )
                    result[f"{prefix}_skill_attribute_mask"][
                        batch_index, pet_index, skill_index
                    ] = _np_array(
                        skill["attribute_mask"], dtype=torch.bool, device=device
                    )
                    result[f"{prefix}_skill_state_ids"][
                        batch_index, pet_index, skill_index
                    ] = _np_array(
                        skill["state_ids"], dtype=torch.long, device=device
                    )
                    result[f"{prefix}_skill_state_values"][
                        batch_index, pet_index, skill_index
                    ] = _np_array(
                        skill["state_values"], dtype=torch.float32, device=device
                    )
                    result[f"{prefix}_skill_state_mask"][
                        batch_index, pet_index, skill_index
                    ] = _np_array(
                        skill["state_mask"], dtype=torch.bool, device=device
                    )
                    for effect_index, effect in enumerate(skill["effects"]):
                        kind, target, stage, value, duration = effect
                        result[f"{prefix}_skill_effect_kind"][
                            batch_index, pet_index, skill_index, effect_index
                        ] = kind
                        result[f"{prefix}_skill_effect_target"][
                            batch_index, pet_index, skill_index, effect_index
                        ] = target
                        result[f"{prefix}_skill_effect_stage"][
                            batch_index, pet_index, skill_index, effect_index
                        ] = stage
                        result[f"{prefix}_skill_effect_value"][
                            batch_index, pet_index, skill_index, effect_index
                        ] = value
                        result[f"{prefix}_skill_effect_duration"][
                            batch_index, pet_index, skill_index, effect_index
                        ] = duration
                        result[f"{prefix}_skill_effect_mask"][
                            batch_index, pet_index, skill_index, effect_index
                        ] = True
                for effect_index, effect in enumerate(pet["effects"]):
                    result[f"{prefix}_effects"][
                        batch_index, pet_index, effect_index
                    ] = _np_array(
                        effect["numeric"], dtype=torch.float32, device=device
                    )
                    result[f"{prefix}_effect_ids"][
                        batch_index, pet_index, effect_index
                    ] = effect["id"]
                    result[f"{prefix}_effect_source_ids"][
                        batch_index, pet_index, effect_index
                    ] = effect["source_id"]
                    result[f"{prefix}_effect_source_traits"][
                        batch_index, pet_index, effect_index
                    ] = effect["source_trait"]
                    result[f"{prefix}_effect_mask"][
                        batch_index, pet_index, effect_index
                    ] = effect["mask"]
        result["own_resonance"][batch_index] = _np_array(
            item["own_resonance"], dtype=torch.float32, device=device
        )
        result["enemy_resonance"][batch_index] = _np_array(
            item["enemy_resonance"], dtype=torch.float32, device=device
        )
        result["enemy_resonance_target"][batch_index] = item["enemy_resonance_target"]
        result["enemy_resonance_mask"][batch_index] = item["enemy_resonance_mask"]
        for pet_index in range(min(max_pets, len(item["enemy_bloodline_targets"]))):
            result["enemy_bloodline_targets"][batch_index, pet_index] = item[
                "enemy_bloodline_targets"
            ][pet_index]
            result["enemy_bloodline_mask"][batch_index, pet_index] = item[
                "enemy_bloodline_mask"
            ][pet_index]
            result["enemy_bloodline_allowed"][batch_index, pet_index] = _np_array(
                item["enemy_bloodline_allowed"][pet_index],
                dtype=torch.bool,
                device=device,
            )
        for field_index, field in enumerate(item["fields"]):
            result["fields"][batch_index, field_index] = _np_array(
                [field["id"] / 10, *field["numeric"], float(field["mask"]), 0],
                dtype=torch.float32,
                device=device,
            )
            result["field_mask"][batch_index, field_index] = field["mask"]
        for event_index, event in enumerate(item["events"]):
            result["events"][batch_index, event_index] = _np_array(
                event["numeric"], dtype=torch.float32, device=device
            )
            for key, event_key in (
                ("event_kind_ids", "kind"),
                ("event_phase_ids", "phase"),
                ("event_action_ids", "action"),
                ("event_effect_ids", "effect"),
                ("event_source_kind_ids", "source_kind"),
                ("event_target_kind_ids", "target_kind"),
                ("event_skill_ids", "skill"),
                ("event_trait_ids", "trait"),
                ("event_source_spirit_ids", "source_spirit"),
                ("event_source_trait_ids", "source_trait"),
                ("event_target_spirit_ids", "target_spirit"),
                ("event_target_trait_ids", "target_trait"),
                ("event_source_indices", "source_index"),
                ("event_target_indices", "target_index"),
            ):
                result[key][batch_index, event_index] = event[event_key]
            result["event_mask"][batch_index, event_index] = event["mask"]
            result["event_public_mask"][batch_index, event_index] = event["public"]
            result["event_source_present"][batch_index, event_index] = event[
                "source_present"
            ]
            result["event_target_present"][batch_index, event_index] = event[
                "target_present"
            ]
        for key, mask_key in (
            ("own_candidates", "action_mask"),
            ("opponent_candidates", "opponent_mask"),
        ):
            for candidate_index, candidate in enumerate(item[key]):
                result[key][batch_index, candidate_index] = _np_array(
                    [
                        candidate["kind"],
                        candidate["pet_index"],
                        candidate["skill"],
                        candidate["magic_id"],
                        *candidate["branches"],
                        *candidate["numeric"],
                        candidate["skill_index"],
                        *candidate["derived"],
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                result[mask_key][batch_index, candidate_index] = candidate["mask"]
                prefix = "own" if key == "own_candidates" else "opponent"
                result[f"{prefix}_candidate_skill_present"][
                    batch_index, candidate_index
                ] = candidate["skill_present"]
                result[f"{prefix}_candidate_skill_attributes"][
                    batch_index, candidate_index
                ] = _np_array(
                    candidate["skill_attributes"], dtype=torch.float32, device=device
                )
                result[f"{prefix}_candidate_skill_attribute_ids"][
                    batch_index, candidate_index
                ] = _np_array(
                    candidate["skill_attribute_ids"],
                    dtype=torch.long,
                    device=device,
                )
                for effect_index, effect in enumerate(candidate["skill_effects"]):
                    kind, target, stage, value, duration = effect
                    result[f"{prefix}_candidate_skill_effect_kind"][
                        batch_index, candidate_index, effect_index
                    ] = kind
                    result[f"{prefix}_candidate_skill_effect_target"][
                        batch_index, candidate_index, effect_index
                    ] = target
                    result[f"{prefix}_candidate_skill_effect_stage"][
                        batch_index, candidate_index, effect_index
                    ] = stage
                    result[f"{prefix}_candidate_skill_effect_value"][
                        batch_index, candidate_index, effect_index
                    ] = value
                    result[f"{prefix}_candidate_skill_effect_duration"][
                        batch_index, candidate_index, effect_index
                    ] = duration
                    result[f"{prefix}_candidate_skill_effect_mask"][
                        batch_index, candidate_index, effect_index
                    ] = True
                if key == "own_candidates":
                    result["own_candidate_targets"][
                        batch_index, candidate_index
                    ] = _np_array(
                        candidate["targets"],
                        dtype=torch.float32,
                        device=device,
                    )
    return {
        key: torch.from_numpy(value).to(target_device)
        if isinstance(value, np.ndarray)
        else value
        for key, value in result.items()
    }
