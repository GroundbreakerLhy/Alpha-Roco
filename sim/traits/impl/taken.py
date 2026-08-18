"""第 1 批-d：受到伤害修正（modify_damage_taken）+ 展翅的属性改写。

- 200074 偏振       受到自己携带技能系别的攻击伤害-40%。
- 200127 展翅       在场时，自己携带的普通系技能变为翼系技能；若后于对手行动，自己受到的伤害+25%。
- 200157 惊吓       能量等于0的精灵，无法对自己造成伤害。
- 200232 逐魂鸟     能耗小于等于1的攻击技能，无法对自己造成伤害。
- 200268 绝对秩序   受到非敌方系别的技能攻击时伤害-50%。
- 280010 完全偏振   抵抗自己携带技能系别的攻击伤害（"抵抗"按减半实现，语义待确认）。

语义说明：受击查询只查被查询精灵自身。ctx.skill=攻击技能，ctx.subject=攻击方，ctx.is_first=是否先手。
惊吓/逐魂鸟返回 -1.0（乘数归零=免疫）。
"""

from __future__ import annotations

from ...enums import Element as E
from ..registry import register
from ..base import TraitHandler

NORMAL = E.NORMAL    # 0 普通
WING = E.WING        # 12 翼


def _self(ctx):
    return ctx.target is ctx.actor


def _own_skill_elements(pet) -> set:
    return {s.element for s in pet.skills}


# ---------------- 200074 偏振 ----------------
class Polarization(TraitHandler):
    trait_id = 200074
    name = "偏振"
    desc = "受到自己携带技能系别的攻击伤害-40%。"
    implemented = True

    def modify_damage_taken(self, ctx):
        if not _self(ctx) or ctx.skill is None:
            return 0.0
        if ctx.skill.element in _own_skill_elements(ctx.actor):
            return -0.4
        return 0.0


# ---------------- 200127 展翅 ----------------
class SpreadWings(TraitHandler):
    trait_id = 200127
    name = "展翅"
    desc = "在场时，自己携带的普通系技能变为翼系技能；若后于对手行动，自己受到的伤害+25%。"
    implemented = True

    def modify_skill_element(self, ctx, skill):
        if not _self(ctx):
            return None
        if skill.element == NORMAL:
            return WING
        return None

    def modify_damage_taken(self, ctx):
        if not _self(ctx):
            return 0.0
        if not ctx.is_first:
            return 0.25
        return 0.0


# ---------------- 200157 惊吓 ----------------
class Frighten(TraitHandler):
    trait_id = 200157
    name = "惊吓"
    desc = "能量等于0的精灵，无法对自己造成伤害。"
    implemented = True

    def modify_damage_taken(self, ctx):
        if not _self(ctx) or ctx.subject is None:
            return 0.0
        if ctx.subject.energy == 0:
            return -1.0
        return 0.0


# ---------------- 200232 逐魂鸟 ----------------
class SoulChaser(TraitHandler):
    trait_id = 200232
    name = "逐魂鸟"
    desc = "能耗小于等于1的攻击技能，无法对自己造成伤害。"
    implemented = True

    def modify_damage_taken(self, ctx):
        if not _self(ctx) or ctx.skill is None:
            return 0.0
        if ctx.skill.category in (0, 1) and ctx.skill.energy_cost <= 1:
            return -1.0
        return 0.0


# ---------------- 200268 绝对秩序 ----------------
class AbsoluteOrder(TraitHandler):
    trait_id = 200268
    name = "绝对秩序"
    desc = "受到非敌方系别的技能攻击时伤害-50%。"
    implemented = True

    def modify_damage_taken(self, ctx):
        if not _self(ctx) or ctx.skill is None or ctx.subject is None:
            return 0.0
        if ctx.skill.element not in ctx.subject.attributes:
            return -0.5
        return 0.0


# ---------------- 280010 完全偏振 ----------------
class FullPolarization(TraitHandler):
    trait_id = 280010
    name = "完全偏振"
    desc = "抵抗自己携带技能系别的攻击伤害。"
    implemented = True

    def modify_damage_taken(self, ctx):
        if not _self(ctx) or ctx.skill is None:
            return 0.0
        if ctx.skill.element in _own_skill_elements(ctx.actor):
            return -0.5  # "抵抗"按减半实现
        return 0.0


def register_batch1_taken() -> None:
    for cls in (Polarization, SpreadWings, Frighten, SoulChaser,
                AbsoluteOrder, FullPolarization):
        register(cls())


register_batch1_taken()
