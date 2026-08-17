"""Skill inspection helpers."""

from __future__ import annotations

from . import buffs

ATTACK = "attack"
DEFENSE = "defense"
STATUS = "status"
NON_SKILL = "non_skill"


def category_of(skill) -> str:
    if skill is None:
        return NON_SKILL
    if skill.category in (0, 1):
        return ATTACK
    if skill.category == 2:
        return DEFENSE
    if skill.category == 3:
        return STATUS
    return NON_SKILL


def raw_energy_cost(skill) -> int:
    return skill.energy_cost if skill is not None else 0


def modified_energy_cost(pet, skill) -> int:
    if skill is None:
        return 0
    return max(0, skill.energy_cost + buffs.get_energy_cost_modifier(pet))


def actual_energy_cost(pet, skill, mark_energy_bonus: int = 0) -> int:
    if skill is None:
        return 0
    return max(0, skill.energy_cost + buffs.get_energy_cost_modifier(pet) + mark_energy_bonus)


def is_same_attribute(skill, pet) -> bool:
    if skill is None:
        return False
    return skill.element in pet.attributes


def attributes_equal(skill_a, skill_b) -> bool:
    if skill_a is None or skill_b is None:
        return False
    return skill_a.element == skill_b.element


def counter_target_of(skill) -> str:
    if skill is None:
        return ""
    return skill.counter_target


def has_counter_effect(skill) -> bool:
    return bool(counter_target_of(skill))
