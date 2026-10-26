"""迸发：入场后首次行动消耗的额外效果。

type 取值：attack_power_flat / attack_power_percent / skill_use_count /
energy_cost_flat / enemy_energy_cost。
"""

from __future__ import annotations


def add_burst(pet, burst_type: str, value: int, *, expires_on_turn: int | None = None) -> None:
    effect = {"type": burst_type, "value": value}
    if expires_on_turn is not None:
        effect["expires_on_turn"] = expires_on_turn
    pet.bursts.append(effect)


def expire_bursts(pet, current_turn: int) -> None:
    """移除到本回合末到期、且尚未消费的限时迸发。"""
    pet.bursts[:] = [
        effect for effect in pet.bursts
        if effect.get("expires_on_turn") is None
        or effect["expires_on_turn"] > current_turn
    ]


def clear_bursts(pet) -> None:
    pet.bursts.clear()


def take_bursts(pet) -> list:
    bursts = list(pet.bursts)
    pet.bursts.clear()
    return bursts


def get_bursts(pet) -> list:
    return list(pet.bursts)
