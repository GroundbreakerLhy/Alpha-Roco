"""Mark system: each side has one positive mark slot and one negative mark slot.

Marks persist when a Pokémon leaves the field; the next Pokémon inherits them.
Gaining a new mark replaces the previous mark on the same side.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import buffs
from .data_loader import load_typechart
from .typechart import type_multiplier

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MARKS_PATH = ROOT / "data" / "marks.json"

POSITIVE = "positive"
NEGATIVE = "negative"


def load_marks():
    with open(MARKS_PATH, encoding="utf-8") as f:
        return json.load(f)["marks"]


def mark_side(mark_id: int) -> str:
    for mark in load_marks():
        if mark["id"] == mark_id:
            return mark["side"]
    return POSITIVE


def add_mark(state, side: str, mark_id: int, amount: int = 1, stack: bool = False) -> None:
    if amount <= 0:
        return
    side_name = mark_side(mark_id)
    if stack:
        # 叠加模式(特性:赋予的印记不替换其他印记):同 id 合并,异 id 追加
        for m in state.marks_extra[side][side_name]:
            if m["id"] == mark_id:
                m["stacks"] = min(99, m["stacks"] + amount)
                return
        state.marks_extra[side][side_name].append({"id": mark_id, "stacks": min(99, amount)})
        return
    current = state.marks[side][side_name]
    if current is not None and current["id"] == mark_id:
        current["stacks"] = min(99, current["stacks"] + amount)
    else:
        state.marks[side][side_name] = {"id": mark_id, "stacks": min(99, amount)}


def get_extra_marks(state, side: str, slot: str) -> list:
    return state.marks_extra[side][slot]


def find_mark(state, side: str, slot: str, mark_id: int):
    """在槽位(含叠加印记)中查找指定 id 的印记。"""
    cur = state.marks[side][slot]
    if cur is not None and cur["id"] == mark_id:
        return dict(cur)
    for m in state.marks_extra[side][slot]:
        if m["id"] == mark_id:
            return dict(m)
    return None


def find_stacks(state, side: str, mark_id: int) -> int:
    slot = mark_side(mark_id)
    total = 0
    cur = state.marks[side][slot]
    if cur is not None and cur["id"] == mark_id:
        total += cur["stacks"]
    for m in state.marks_extra[side][slot]:
        if m["id"] == mark_id:
            total += m["stacks"]
    return total


def remove_mark(state, side: str, mark_id: int) -> None:
    for side_name in (POSITIVE, NEGATIVE):
        current = state.marks[side][side_name]
        if current is not None and current["id"] == mark_id:
            state.marks[side][side_name] = None
        state.marks_extra[side][side_name] = [
            m for m in state.marks_extra[side][side_name] if m["id"] != mark_id
        ]


def clear_side(state, side: str, side_name: str) -> None:
    state.marks[side][side_name] = None
    state.marks_extra[side][side_name] = []


def get_marks(state, side: str) -> dict:
    result = {}
    for side_name, value in state.marks[side].items():
        result[side_name] = dict(value) if value is not None else None
    return result


def get_mark(state, side: str, side_name: str):
    current = state.marks[side].get(side_name)
    if current is None:
        return None
    return dict(current)


def get_stacks(state, side: str, mark_id: int) -> int:
    side_name = mark_side(mark_id)
    current = state.marks[side][side_name]
    if current is not None and current["id"] == mark_id:
        return current["stacks"]
    return 0


# 印记 8：萌芽印记 —— 获得增益时额外获得一层
def amplify_buff_gain(state, side: str, buff_type: str, value: int) -> int:
    if value == 0 or not buffs.is_buff(buff_type, value):
        return value
    positive = state.marks[side]["positive"]
    if positive is not None and positive["id"] == 8:
        return value + positive["stacks"]
    return value


# 回合结束印记结算：中毒印记(4)、光合印记(10)
def on_round_end(state) -> None:
    typechart = load_typechart()
    order = ["A", "B"] if state.home_side == "A" else ["B", "A"]
    for side in order:
        active_idx = state.active[side]
        if active_idx < 0:
            continue
        pet = state.teams[side][active_idx]
        if pet.hp <= 0:
            continue

        positive = find_mark(state, side, "positive", 10)
        if positive is not None:
            gain = positive["stacks"]
            pet.energy = min(10, pet.energy + gain)
            state.log.append(f"{side} {pet.name} 光合印记回复 {gain} 能量")

        negative = find_mark(state, side, "negative", 4)
        if negative is not None:
            mult = type_multiplier(9, pet.attributes, typechart)
            damage = int(pet.max_hp * 0.03 * negative["stacks"] * mult)
            pet.hp = max(0, pet.hp - damage)
            state.log.append(f"{side} {pet.name} 受到中毒印记 {damage} 伤害")
