"""Counter/response priority rules.

A skill counters successfully when its counter_target matches the opponent's
action category, and the countering side is forced first.
"""

from __future__ import annotations

from .models import Action, BattleState
from .skill_utils import ATTACK, DEFENSE, NON_SKILL, STATUS, category_of, counter_target_of


def action_category(state: BattleState, side: str, action: Action) -> str:
    if action.kind == "charge":
        return STATUS
    if action.kind != "skill" or action.skill_index is None:
        return NON_SKILL
    pet = state.teams[side][state.active[side]]
    if not (0 <= action.skill_index < len(pet.skills)):
        return NON_SKILL
    return category_of(pet.skills[action.skill_index])


def _action_skill(state: BattleState, side: str, action: Action):
    if action.kind != "skill" or action.skill_index is None:
        return None
    pet = state.teams[side][state.active[side]]
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
