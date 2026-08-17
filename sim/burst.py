"""Burst effects: extra effects consumed on the first action after entering field."""

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
