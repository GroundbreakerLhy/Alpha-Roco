"""Burst effects: extra effects consumed on the first action after entering field.

迸发类型（burst["type"]）：
  attack_power_flat    本次攻击技能威力 +value（固定值）
  attack_power_percent 本次攻击技能威力 +value（百分比）
  skill_use_count      本次行动技能使用次数 +value
  energy_cost_flat     本次技能能耗 -value
  enemy_energy_cost    本次行动后敌方全技能能耗 +value（施加给敌方在场精灵）
"""

from __future__ import annotations


def add_burst(pet, burst_type: str, value: int) -> None:
    pet.bursts.append({"type": burst_type, "value": value})


def clear_bursts(pet) -> None:
    pet.bursts.clear()


def take_bursts(pet) -> list:
    bursts = list(pet.bursts)
    pet.bursts.clear()
    return bursts


def get_bursts(pet) -> list:
    return list(pet.bursts)
