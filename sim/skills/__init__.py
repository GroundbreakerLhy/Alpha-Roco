"""技能系统接口层：绑定静态属性、汇总修正、按阶段分发 handler。

技能效果全部以代码定义，不解析 desc；类别见 ``skill_utils``。
"""

from __future__ import annotations

from typing import Optional

from .. import skill_utils
from ..models import BattlePet, BattleSkill, BattleState
from .base import SkillContext, SkillHandler
from .ops import (  # noqa: F401  效果原语（按需从 sim.skills 直接引用）
    COUNTER_ATTACK,
    COUNTER_DEFENSE,
    COUNTER_STATUS,
    LEAVE_BOTH,
    LEAVE_EMERGENCY,
    LEAVE_ENEMY,
    LEAVE_SELF,
    STAGE_AFTER_USE,
    STAGE_COUNTER,
    STAGE_DEFENSE,
    STAGE_HIT,
    STAGE_ROUND_END,
    STAGE_STATUS,
    STAGE_TAKE_DAMAGE,
    STAGE_ENTRY,
    STAGE_SLOT_CHANGE,
    STAGE_USE_START,
    Always,
    Any,
    Condition,
    Effect,
    IsCounter,
    IsFirst,
    IsSecond,
    OpsSkill,
    When,
    apply_effects,
)
from .registry import (  # noqa: F401
    all_handlers,
    bind_skill,
    check,
    get_handler,
    raw_of,
    register,
    report,
    static_of,
)

__all__ = [
    "SkillContext",
    "SkillHandler",
    "OpsSkill",
    "Effect",
    "Condition",
    "When",
    "Always",
    "Any",
    "IsCounter",
    "IsFirst",
    "IsSecond",
    "apply_effects",
    "make_ctx",
    "emit",
    "query_power",
    "query_absolute_power",
    "compute_hit_count",
    "query_hit_count",
    "query_energy_cost",
    "query_energy_shortfall",
    "query_priority",
    "query_lifesteal",
    "query_damage_reduction",
    "query_element",
    "bind_skill",
    "get_handler",
    "static_of",
    "raw_of",
    "register",
    "report",
    "check",
    "all_handlers",
    "STAGE_ENTRY",
    "STAGE_SLOT_CHANGE",
    "STAGE_USE_START",
    "STAGE_HIT",
    "STAGE_COUNTER",
    "STAGE_DEFENSE",
    "STAGE_STATUS",
    "STAGE_AFTER_USE",
    "STAGE_ROUND_END",
    "STAGE_TAKE_DAMAGE",
    "emit_take_damage",
    "COUNTER_ATTACK",
    "COUNTER_DEFENSE",
    "COUNTER_STATUS",
    "LEAVE_SELF",
    "LEAVE_ENEMY",
    "LEAVE_BOTH",
    "LEAVE_EMERGENCY",
]


def _weather_id(state: BattleState) -> Optional[int]:
    if state.weather is None:
        return None
    return state.weather["id"] if isinstance(state.weather, dict) else state.weather


def _opponent(state: BattleState, pet: BattlePet) -> Optional[BattlePet]:
    opp_side = "B" if pet.side == "A" else "A"
    idx = state.active[opp_side]
    if idx < 0:
        return None
    return state.teams[opp_side][idx]


_CTX_FIELDS = set(SkillContext.__dataclass_fields__.keys()) - {"state", "actor", "skill", "extra"}


def make_ctx(state: BattleState, actor: BattlePet, skill: BattleSkill,
             **kw) -> SkillContext:
    """构建技能上下文：已知字段进 ctx，未知字段进 extra。

    weather / target / category 缺省自动补齐。
    """
    ctx_kw = {}
    extra = {}
    for k, v in kw.items():
        if k == "extra" and isinstance(v, dict):
            extra.update(v)
        elif k in _CTX_FIELDS:
            ctx_kw[k] = v
        else:
            extra[k] = v
    ctx_kw.setdefault("weather", _weather_id(state))
    ctx_kw.setdefault("target", _opponent(state, actor))
    if "category" not in ctx_kw:
        ctx_kw["category"] = skill_utils.category_of(skill)
    return SkillContext(state=state, actor=actor, skill=skill, extra=extra, **ctx_kw)


_STAGE_HOOKS = {
    STAGE_ENTRY: "on_entry",
    STAGE_SLOT_CHANGE: "on_slot_change",
    STAGE_USE_START: "on_use_start",
    STAGE_HIT: "on_hit",
    STAGE_COUNTER: "on_counter",
    STAGE_DEFENSE: "on_defense",
    STAGE_STATUS: "on_status",
    STAGE_AFTER_USE: "on_after_use",
    STAGE_ROUND_END: "on_round_end",
    STAGE_TAKE_DAMAGE: "on_take_damage",
}


def emit(state: BattleState, stage: str, actor: BattlePet, skill: BattleSkill,
         **kw) -> SkillContext:
    """在指定阶段执行技能效果，返回上下文（调用方可读 ctx.extra）。

    技能阶段必须有技能实体：skill 为必填。
    """
    ctx = make_ctx(state, actor, skill, **kw)
    hook_name = _STAGE_HOOKS.get(stage)
    if hook_name is None:
        raise ValueError(f"未知技能阶段：{stage}")
    handler = get_handler(skill.skill_id)
    if handler is not None:
        getattr(handler, hook_name)(ctx)
    return ctx


def emit_take_damage(state: BattleState, defender: BattlePet, damage: int, *,
                     attacker: Optional[BattlePet] = None,
                     skill: BattleSkill | None = None,
                     hit_count: int = 1,
                     is_first: bool = False,
                     logs: Optional[list] = None) -> None:
    """受击方技能侧反应：把"受到攻击伤害"广播给受击方携带的每个技能。

    actor 是受击方，ctx.skill 逐个取受击方技能，ctx.target 是攻击方；
    extra 带 damage_taken / incoming_skill。引擎仅在攻击伤害落位后调用。
    """
    for index, own in enumerate(defender.skills):
        emit(
            state,
            STAGE_TAKE_DAMAGE,
            defender,
            own,
            skill_index=index,
            target=attacker,
            is_first=is_first,
            hit_count=hit_count,
            logs=logs,
            extra={
                "damage_taken": damage,
                "incoming_skill": skill,
                "attacker": attacker,
            },
        )


# ==================== 数值查询（增量语义） ====================


def query_power(state: BattleState, actor: BattlePet, skill: BattleSkill,
                base_percent: float = 0.0, base_flat: float = 0.0, **kw) -> tuple:
    """技能威力增量，返回 ``(百分比, 固定值)``。"""
    handler = get_handler(skill.skill_id)
    if handler is None:
        return (base_percent, base_flat)
    percent, flat = handler.modify_power(make_ctx(state, actor, skill, **kw))
    return (base_percent + percent, base_flat + flat)


def query_absolute_power(state: BattleState, actor: BattlePet, skill: BattleSkill,
                         **kw) -> Optional[int]:
    """威力绝对值覆盖；None = 用 skills.json 的 power。"""
    handler = get_handler(skill.skill_id)
    if handler is None:
        return None
    return handler.absolute_power(make_ctx(state, actor, skill, **kw))


def query_hit_count(state: BattleState, actor: BattlePet, skill: BattleSkill, **kw) -> tuple:
    """连击增量，返回 ``(固定值, 百分比, 强制值|None)``。"""
    handler = get_handler(skill.skill_id)
    if handler is None:
        return (0, 0, None)
    return handler.modify_hit_count(make_ctx(state, actor, skill, **kw))


def compute_hit_count(state: BattleState, actor: BattlePet, skill: BattleSkill,
                      target: BattlePet | None = None,
                      skill_index: int | None = None, **kw) -> int:
    """连击数 = 技能固有连击数 + buff + 特性（含强制值）。

    额外关键字（如 is_counter）透传进 ctx，供 handler 即时判定。
    """
    from .ops import hit_count_of
    return hit_count_of(
        make_ctx(state, actor, skill, target=target, skill_index=skill_index, **kw)
    )


def query_energy_cost(state: BattleState, actor: BattlePet, skill: BattleSkill,
                      base: int = 0, **kw) -> int:
    """技能自身能耗修正 + 携带技能对它的被动能耗修正。"""
    handler = get_handler(skill.skill_id)
    total = base
    if handler is not None:
        total += handler.modify_energy_cost(make_ctx(state, actor, skill, **kw))

    target_index = kw.get("skill_index")
    if target_index is None:
        target_index = next(
            (i for i, candidate in enumerate(actor.skills) if candidate is skill), -1
        )
    if target_index < 0:
        return total

    source_kw = {k: v for k, v in kw.items() if k != "skill_index"}
    for source_index, source_skill in enumerate(actor.skills):
        if source_index == target_index:
            continue
        source_handler = get_handler(source_skill.skill_id)
        if source_handler is None:
            continue
        source_ctx = make_ctx(
            state, actor, source_skill, skill_index=source_index, **source_kw
        )
        total += source_handler.modify_other_skill_energy_cost(
            source_ctx, skill, target_index
        )
    return total


def query_energy_shortfall(state: BattleState, actor: BattlePet, skill: BattleSkill,
                           need: int, dry_run: bool = False, **kw) -> int:
    """能量不足时技能自身可补充的能量。

    dry_run=True 只探测、不支付代价。
    """
    handler = get_handler(skill.skill_id)
    if handler is None:
        return 0
    return handler.modify_energy_shortfall(make_ctx(state, actor, skill, **kw), need,
                                           dry_run=dry_run)


def query_priority(state: BattleState, actor: BattlePet, skill: BattleSkill,
                   base: int = 0, **kw) -> int:
    handler = get_handler(skill.skill_id)
    if handler is None:
        return base
    return base + handler.modify_priority(make_ctx(state, actor, skill, **kw))


def query_lifesteal(state: BattleState, actor: BattlePet, skill: BattleSkill,
                    base: float = 0.0, **kw) -> float:
    handler = get_handler(skill.skill_id)
    if handler is None:
        return base
    return base + handler.modify_lifesteal(make_ctx(state, actor, skill, **kw))


def query_damage_reduction(state: BattleState, actor: BattlePet, skill: BattleSkill,
                           base: float = 0.0, **kw) -> float:
    """防御技能减伤比例。"""
    handler = get_handler(skill.skill_id)
    if handler is None:
        return base
    return base + handler.modify_damage_reduction(make_ctx(state, actor, skill, **kw))


def query_ignore_resist(state: BattleState, actor: BattlePet, skill: BattleSkill,
                        **kw) -> bool:
    """本次伤害是否无视系别抵抗。"""
    handler = get_handler(skill.skill_id)
    if handler is None:
        return False
    return handler.ignore_resist(make_ctx(state, actor, skill, **kw))


def query_use_count(state: BattleState, actor: BattlePet, skill: BattleSkill,
                    **kw) -> int:
    """本次行动由技能自身提供的额外执行次数。"""
    handler = get_handler(skill.skill_id)
    if handler is None:
        return 0
    return handler.modify_use_count(make_ctx(state, actor, skill, **kw))


def query_element(state: BattleState, actor: BattlePet, skill: BattleSkill,
                  base: Optional[int] = None, **kw) -> int:
    """本次技能实际属性；handler 未改写时返回 base（缺省 = skill.element）。

    特性先改写，技能侧以特性结果为底再改写。
    """
    fallback = skill.element if base is None else base
    handler = get_handler(skill.skill_id)
    if handler is None:
        return fallback
    override = handler.modify_element(make_ctx(state, actor, skill, **kw))
    return fallback if override is None else override
