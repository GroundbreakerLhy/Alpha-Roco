"""应对判定与按精灵的应对统计。

counter_target 与对方行动类别匹配、且本侧强制先手时，应对成功。
"""

from __future__ import annotations

from .models import Action, BattlePet, BattleState
from .skill_utils import (
    ATTACK,
    DEFENSE,
    NON_SKILL,
    STATUS,
    category_of,
    counter_target_of,
)


def action_category(state: BattleState, side: str, action: Action) -> str:
    if action.kind == "charge":
        return STATUS
    if action.kind != "skill" or action.skill_index is None:
        return NON_SKILL
    idx = state.active[side]
    if idx < 0:
        # 在场精灵已力竭（active=-1）：本侧无有效行动，不参与应对判定。
        return NON_SKILL
    pet = state.teams[side][idx]
    if not (0 <= action.skill_index < len(pet.skills)):
        return NON_SKILL
    skill = pet.skills[action.skill_index]
    # 蓄力不是状态行动（B33）：开始蓄力段不参与应对，释放段按技能类别参与。
    if getattr(skill, "windup", False) and pet.windup_skill is not skill:
        return NON_SKILL
    return category_of(skill)


def _action_skill(state: BattleState, side: str, action: Action):
    if action.kind != "skill" or action.skill_index is None:
        return None
    idx = state.active[side]
    if idx < 0:
        return None
    pet = state.teams[side][idx]
    if not (0 <= action.skill_index < len(pet.skills)):
        return None
    return pet.skills[action.skill_index]


def forced_first(state: BattleState, action_a: Action, action_b: Action):
    cat_a = action_category(state, "A", action_a)
    cat_b = action_category(state, "B", action_b)
    skill_a = _action_skill(state, "A", action_a)
    skill_b = _action_skill(state, "B", action_b)

    if counter_target_of(skill_a) == cat_b:
        return "A"
    if counter_target_of(skill_b) == cat_a:
        return "B"
    return None


# ---- 应对统计（按精灵） ----


def record_counter(pet: BattlePet, target_category: str) -> None:
    """记录一次应对成功：次数+1 并累计类别。"""
    pet.counter_stats["count"] += 1
    if target_category:
        pet.counter_stats["types"][target_category] = (
            pet.counter_stats["types"].get(target_category, 0) + 1
        )


def get_counter_count(pet: BattlePet, target_category: str = "") -> int:
    """应对次数；target_category 为空返回总数，否则返回该类别的次数。"""
    if not target_category:
        return pet.counter_stats["count"]
    return pet.counter_stats["types"].get(target_category, 0)


def get_counter_types(pet: BattlePet) -> dict:
    """各类别应对次数。"""
    return dict(pet.counter_stats["types"])
