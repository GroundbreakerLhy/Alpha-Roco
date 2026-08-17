"""Evolution-chain helpers, including cute degeneration."""

from __future__ import annotations

from .data_loader import calc_all_stats, get_attributes, load_spirits, round_half_up


def get_spirit_by_id(spirit_id: int):
    for spirit in load_spirits():
        if spirit["id"] == spirit_id:
            return spirit
    return None


def get_previous_stage(spirit):
    if not spirit.get("evolution"):
        return None
    for chain in spirit["evolution"]:
        stages = chain.get("chain", [])
        for i, stage in enumerate(stages):
            if stage["id"] == spirit["id"] and i > 0:
                return get_spirit_by_id(stages[i - 1]["id"])
    return None


def can_cute(pet) -> bool:
    spirit = get_spirit_by_id(pet.spirit_id)
    if spirit is None:
        return False
    return get_previous_stage(spirit) is not None


def apply_cute(pet) -> bool:
    if not can_cute(pet):
        return False
    spirit = get_spirit_by_id(pet.spirit_id)
    prev = get_previous_stage(spirit)
    if prev is None:
        return False

    pet.spirit_id = prev["id"]
    pet.name = prev["name"]
    pet.attributes = get_attributes(prev)
    new_stats = calc_all_stats(prev["stats"], ivs=pet.ivs, nature=pet.nature)
    old_max_hp = pet.max_hp
    old_hp = pet.hp
    pet.stats = new_stats
    pet.max_hp = new_stats["hp"]
    pet.hp = min(pet.max_hp, round_half_up(old_hp * pet.max_hp / old_max_hp))
    return True
