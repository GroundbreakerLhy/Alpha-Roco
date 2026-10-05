"""Counter/response priority rules and per-pet counter statistics.

应对的发生（forced_first 判定）依赖技能代码化后的 counter_target 数据，
当前暂不触发；本模块先打通"应对统计"：按精灵记录应对次数与种类，
供"每应对成功N次"类特性（如斗技 200148）读取。

A skill counters successfully when its counter_target matches the opponent's
action category, and the countering side is forced first.
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
        # 在场精灵已力竭（active=-1，等待换人）：本侧没有有效技能行动，不参与应对判定。
        return NON_SKILL
    pet = state.teams[side][idx]
    if not (0 <= action.skill_index < len(pet.skills)):
        return NON_SKILL
    skill = pet.skills[action.skill_index]
    # 蓄力不是状态行动（SPEC B41）：蓄力技能的首段"开始蓄力"不参与应对判定——
    # 既不会被应对，也不作为应对的触发类别；释放段按技能自身类别参与应对。
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


# ==================== 应对统计（按精灵） ====================


def record_counter(pet: BattlePet, target_category: str) -> None:
    """记录一次应对成功：累计次数+1，并统计应对的种类（应对了攻击/防御/状态）。"""
    pet.counter_stats["count"] += 1
    if target_category:
        pet.counter_stats["types"][target_category] = (
            pet.counter_stats["types"].get(target_category, 0) + 1
        )


def get_counter_count(pet: BattlePet, target_category: str = "") -> int:
    """应对次数统计。

    target_category 为空时返回累计应对总次数；
    指定 "attack"/"defense"/"status" 时返回该种类的应对次数。
    """
    if not target_category:
        return pet.counter_stats["count"]
    return pet.counter_stats["types"].get(target_category, 0)


def get_counter_types(pet: BattlePet) -> dict:
    """应对种类统计（应对了攻击/防御/状态各多少次）。"""
    return dict(pet.counter_stats["types"])
