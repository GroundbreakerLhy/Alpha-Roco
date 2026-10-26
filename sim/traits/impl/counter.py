"""应对类特性（on_counter）。

counter 事件在应对成功（本精灵强制先手）时广播，ctx.subject=强制先手的精灵。
斗技给自己加永久 SKILL_POWER_FLAT（下场不消失、可叠加）。
"""

from __future__ import annotations

from ... import buffs as B
from ... import counter
from ..base import TraitHandler
from ..registry import register


class DuelSkill(TraitHandler):
    trait_id = 200148
    name = "斗技"
    desc = "应对成功后，获得全技能威力永久+30。"
    implemented = True

    def on_counter(self, ctx):
        if ctx.subject is not ctx.actor:
            return
        B.grant_buff(
            ctx.actor,
            B.BuffType.SKILL_POWER_FLAT,
            30,
            B.DurationKind.PERMANENT,
            source_kind="trait",
        )

    def display(self, state, pet):
        n = counter.get_counter_count(pet)
        if n <= 0:
            return None
        return [{"name": "威力", "layers": n, "per": "30", "gain": True}]


class WildSense(TraitHandler):
    trait_id = 200249
    name = "野性感官"
    desc = "应对成功后，下次行动先手+1。"
    implemented = True

    def on_counter(self, ctx):
        if ctx.subject is not ctx.actor:
            return
        # 排序时生效；行动开始结算时引擎一次性消耗
        B.grant_buff(ctx.actor, B.BuffType.PRIORITY, 1, source_kind="trait")

    def display(self, state, pet):
        n = B.get_buff_value(pet, B.BuffType.PRIORITY)
        if n <= 0:
            return None
        return [{"name": "先手", "layers": n, "per": "1", "gain": True}]


def register_counter() -> None:
    register(DuelSkill())
    register(WildSense())


register_counter()
