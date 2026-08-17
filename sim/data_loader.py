"""Load JSON data and build battle entities."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from .models import BattlePet, BattleSkill, SkillData

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = ROOT / "data"


def load_json(name: str):
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def load_spirits():
    return load_json("spirits.json")["spirits"]


def load_skills():
    return load_json("skills.json")["skills"]


def load_typechart():
    return load_json("typechart.json")


def load_natures():
    return load_json("natures.json")["natures"]


DEFAULT_IVS = {
    "hp": 10,
    "atk": 0,
    "def": 0,
    "spatk": 10,
    "spdef": 0,
    "speed": 10,
}

NATURE_UP_MULTIPLIER = 1.2
NATURE_DOWN_MULTIPLIER = 0.9


def get_nature_multipliers(nature):
    if nature is None or nature == -1:
        return {stat: 1.0 for stat in DEFAULT_IVS}
    for item in load_natures():
        if item["id"] == nature:
            mults = {stat: 1.0 for stat in DEFAULT_IVS}
            mults[item["up"]] = NATURE_UP_MULTIPLIER
            mults[item["down"]] = NATURE_DOWN_MULTIPLIER
            return mults
    return {stat: 1.0 for stat in DEFAULT_IVS}


def build_skill_map() -> dict:
    return {s["id"]: s for s in load_skills()}


def find_spirit(name: str, spirits=None):
    if spirits is None:
        spirits = load_spirits()
    for sp in spirits:
        if sp["name"] == name:
            return sp
    return None


def round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


def calc_stat(base_stat: int, iv: int, nature_multiplier: float = 1.0) -> int:
    return round_half_up((round_half_up(1.1 * (base_stat + 3 * iv)) + 10) * nature_multiplier) + 50


def calc_hp(base_hp: int, iv: int, nature_multiplier: float = 1.0) -> int:
    growth = 0.01 * base_hp + 0.005 * iv * 6
    return round_half_up((70 + growth * 170) * nature_multiplier) + 100


def calc_all_stats(base_stats: dict, ivs=None, nature=None) -> dict:
    if ivs is None:
        ivs = DEFAULT_IVS
    mults = get_nature_multipliers(nature)
    return {
        "hp": calc_hp(base_stats["hp"], ivs["hp"], mults["hp"]),
        "atk": calc_stat(base_stats["atk"], ivs["atk"], mults["atk"]),
        "spatk": calc_stat(base_stats["spatk"], ivs["spatk"], mults["spatk"]),
        "def": calc_stat(base_stats["def"], ivs["def"], mults["def"]),
        "spdef": calc_stat(base_stats["spdef"], ivs["spdef"], mults["spdef"]),
        "speed": calc_stat(base_stats["speed"], ivs["speed"], mults["speed"]),
    }


def make_battle_pet(spirit, side: str, level: int = 60, ivs=None, nature=None,
                   skill_names=None, skill_limit: int = 4, bloodline=None) -> BattlePet:
    skill_map = build_skill_map()
    skill_ids = [sid for sid in spirit["skills"].get("normal", [])]
    if skill_names is None:
        selected_ids = [sid for sid in skill_ids if skill_map.get(sid, {}).get("name") != "蓄能"]
        selected_ids = selected_ids[:skill_limit]
    else:
        by_name = {raw["name"]: raw for raw in load_skills()}
        selected_ids = []
        for name in skill_names:
            raw = by_name.get(name)
            if raw is not None:
                selected_ids.append(raw["id"])
    skills = []
    for sid in selected_ids:
        raw = skill_map.get(sid)
        if raw is None:
            continue
        skills.append(BattleSkill(
            skill_id=sid,
            name=raw["name"],
            element=raw["element"],
            category=raw["category"],
            power=raw.get("power"),
            energy_cost=raw.get("energyCost", 0),
            desc=raw.get("desc", ""),
        ))
    stats = calc_all_stats(spirit["stats"], ivs=ivs, nature=nature)
    speed_range = [
        calc_stat(spirit["stats"]["speed"], 0, 0.9),
        calc_stat(spirit["stats"]["speed"], 10, 1.2),
    ]
    return BattlePet(
        side=side,
        spirit_id=spirit["id"],
        name=spirit["name"],
        level=level,
        stats=stats,
        hp=stats["hp"],
        max_hp=stats["hp"],
        energy=10,
        skills=skills,
        attributes=get_attributes(spirit),
        speed_range=speed_range,
        ivs=dict(ivs) if ivs is not None else dict(DEFAULT_IVS),
        nature=nature,
        bloodline=bloodline if bloodline is not None else spirit.get("mainElement"),
    )


def get_attributes(spirit) -> list:
    elements = []
    if spirit.get("mainElement") is not None:
        elements.append(spirit["mainElement"])
    if spirit.get("subElement") is not None:
        elements.append(spirit["subElement"])
    return elements


def create_default_pets() -> dict:
    spirits = load_spirits()
    a = make_battle_pet(find_spirit("岚鸟", spirits), "A")
    b = make_battle_pet(find_spirit("奇丽花", spirits), "B")
    return {"A": a, "B": b}
