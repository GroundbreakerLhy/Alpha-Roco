"""Damage calculation for the MVP battle engine."""

from __future__ import annotations

from . import buffs, traits
from .data_loader import round_half_up
from .models import BattlePet, BattleSkill
from .typechart import type_multiplier

LEVEL_COEFF_BASE = 60


def level_coefficient(level: int) -> float:
    return (level * 45 / 100 + 10) / 41


def stab_multiplier(attacker: BattlePet, skill: BattleSkill, element: int | None = None) -> float:
    # 本系加成：技能属性与精灵任一属性相同则 1.25
    eff_element = element if element is not None else skill.element
    return 1.25 if eff_element in getattr(attacker, "attributes", []) else 1.0



def calc_damage(attacker: BattlePet, defender: BattlePet, skill: BattleSkill, typechart: dict,
                damage_reduction: float = 0.0, extra_power_percent: float = 0.0,
                extra_power_flat: float = 0.0, element: int | None = None,
                state=None, is_first: bool = False) -> dict:
    if skill.power is None or skill.category not in (0, 1):
        return {"damage": 0, "display_power": 0, "type_multiplier": 1.0, "stab": 1.0}

    attack_key = "atk" if skill.category == 0 else "spatk"
    defense_key = "def" if skill.category == 0 else "spdef"
    attack_stat = attacker.stats[attack_key] * buffs.get_stat_multiplier(attacker, attack_key)
    defense_stat = defender.stats[defense_key] * buffs.get_stat_multiplier(defender, defense_key)
    dealt_mod = 1.0
    taken_mod = 1.0
    if state is not None:
        attack_stat *= 1.0 + traits.query_stat_multiplier(state, attacker, attack_key)
        defense_stat *= 1.0 + traits.query_stat_multiplier(state, defender, defense_key)
        dealt_mod = 1.0 + traits.query_damage_dealt(state, attacker, defender, skill=skill)
        taken_mod = 1.0 + traits.query_damage_taken(state, defender, attacker,
                                                    skill=skill, is_first=is_first)
    eff_element = element if element is not None else skill.element
    defender_elements = getattr(defender, "attributes", [])
    stab = stab_multiplier(attacker, skill, element)
    type_mult = type_multiplier(eff_element, defender_elements, typechart)
    power_percent, power_flat = buffs.get_skill_power_modifier(attacker)
    modified_power = (skill.power + power_flat + extra_power_flat) * (1.0 + (power_percent + extra_power_percent) / 100.0)
    effective_power = modified_power * stab * type_mult
    display_power = round_half_up(effective_power)
    coeff = level_coefficient(attacker.level)
    raw = round_half_up(attack_stat * display_power * coeff) / defense_stat
    rounded = int(raw // 1)
    damage = max(0, int(rounded * (1.0 - damage_reduction) * dealt_mod * taken_mod))
    return {
        "damage": damage,
        "display_power": display_power,
        "type_multiplier": type_mult,
        "stab": stab,
    }

