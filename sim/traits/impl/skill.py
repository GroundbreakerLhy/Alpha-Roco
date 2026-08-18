"""技能使用类特性（on_skill_end / on_attack）。

- 200191 最好的伙伴  造成克制伤害后，获得攻防速+20%，并回复2能量。
- 200107 浸润       使用水系技能后，全技能能耗-1。
- 200146 助燃       使用火系技能后，获得双攻+20%。
- 200076 氧循环     使用草系技能后，回复10%生命。

语义说明：
- 最好的伙伴："攻防速"按 物攻/物防 各 +20%（ATK/DEF 各 2 层）、速度 +20%
  （SPEED_PERCENT 2 层，百分比速度见 buffs.get_speed_value）。
- 特性带来的所有 buff 均带 source_kind="trait" 标记（不算普通"增益"，见 buffs.Buff.is_gain）。
"""

from __future__ import annotations

from ... import buffs as B
from ... import traits as T
from ...enums import Element as E
from ..base import TraitHandler
from ..registry import register

WATER = E.WATER    # 3 水
FIRE = E.FIRE      # 2 火
GRASS = E.GRASS    # 1 草


# ---------------- 200191 最好的伙伴 ----------------
class BestCompanion(TraitHandler):
    trait_id = 200191
    name = "最好的伙伴"
    desc = "造成克制伤害后，获得攻防速+20%，并回复2能量。"
    implemented = True

    def on_attack(self, ctx):
        if ctx.subject is not ctx.actor:
            return
        if ctx.extra.get("type_mult", 1.0) <= 1.0:
            return
        B.add_buff(ctx.actor, B.BuffType.ATK, 2, source_kind="trait")
        B.add_buff(ctx.actor, B.BuffType.DEF, 2, source_kind="trait")
        B.add_buff(ctx.actor, B.BuffType.SPEED_PERCENT, 2, source_kind="trait")
        T.grant_energy(ctx.state, ctx.actor, 2)


# ---------------- 200107 浸润 ----------------
class Soak(TraitHandler):
    trait_id = 200107
    name = "浸润"
    desc = "使用水系技能后，全技能能耗-1。"
    implemented = True

    def on_skill_end(self, ctx):
        if ctx.subject is not ctx.actor or ctx.skill is None:
            return
        if ctx.skill.element != WATER:
            return
        B.add_buff(ctx.actor, B.BuffType.ENERGY_COST, -1, source_kind="trait")


# ---------------- 200146 助燃 ----------------
class FireFuel(TraitHandler):
    trait_id = 200146
    name = "助燃"
    desc = "使用火系技能后，获得双攻+20%。"
    implemented = True

    def on_skill_end(self, ctx):
        if ctx.subject is not ctx.actor or ctx.skill is None:
            return
        if ctx.skill.element != FIRE:
            return
        B.add_buff(ctx.actor, B.BuffType.ATK, 2, source_kind="trait")
        B.add_buff(ctx.actor, B.BuffType.SPATK, 2, source_kind="trait")


# ---------------- 200076 氧循环 ----------------
class OxygenCycle(TraitHandler):
    trait_id = 200076
    name = "氧循环"
    desc = "使用草系技能后，回复10%生命。"
    implemented = True

    def on_skill_end(self, ctx):
        if ctx.subject is not ctx.actor or ctx.skill is None:
            return
        if ctx.skill.element != GRASS:
            return
        heal = int(ctx.actor.max_hp * 0.1)
        ctx.actor.hp = min(ctx.actor.max_hp, ctx.actor.hp + heal)


def register_skill() -> None:
    for cls in (BestCompanion, Soak, FireFuel, OxygenCycle):
        register(cls())


register_skill()
