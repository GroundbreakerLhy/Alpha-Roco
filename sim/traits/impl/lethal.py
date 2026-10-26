"""致命伤害保护类特性（on_lethal）。

on_lethal 在伤害应用前判定，返回 True 表示本次伤害被免除（引擎不扣血）。
"""

from __future__ import annotations

from ... import buffs as B
from ...evolution import can_cute
from ..base import TraitHandler
from ..registry import register

MAX_TRIGGERS = 2


class Phoenix(TraitHandler):
    trait_id = 200151
    name = "不死鸟"
    desc = "每场战斗1次，受到致命伤害时保留1血，且敌方获得15层灼烧。"
    implemented = True

    def on_lethal(self, ctx):
        if ctx.subject is not ctx.actor:
            return False
        if ctx.state_of("used", False):
            return False
        attacker = ctx.extra.get("attacker")
        if attacker is None or attacker.side == ctx.actor.side:
            return False

        ctx.actor.hp = 1
        B.grant_buff(
            attacker,
            B.BuffType.BURN,
            15,
            source_side=ctx.actor.side,
            source_pet=ctx.actor.name,
            source_kind="debuff",
        )
        ctx.set_state("used", True)
        return True


class Cocoon(TraitHandler):
    trait_id = 200195
    name = "化茧"
    desc = "受到致命伤害时，获得1层萌化，并免疫此次伤害。（最多触发2次）"
    implemented = True

    def on_lethal(self, ctx):
        if ctx.subject is not ctx.actor:
            return False
        if ctx.state_of("triggers", 0) >= MAX_TRIGGERS:
            return False
        # 必须能获得萌化（有前一阶段）才触发；否则不免疫、不消耗次数
        if not can_cute(ctx.actor):
            return False
        ctx.add_state("triggers", 1)
        had_cute = any(b.buff_type == B.BuffType.CUTE for b in ctx.actor.buffs)
        gained = B.grant_buff(ctx.actor, B.BuffType.CUTE, 1, source_kind="debuff")
        if gained == 0:
            return False
        # 广播 buff_gain 供“再获得萌化”响应类特性使用；pre_had=获得前是否已萌化
        from ... import traits as T

        T.on_buff_gain(
            ctx.state,
            ctx.actor,
            B.BuffType.CUTE,
            1,
            ctx.actor.side,
            ctx.actor.name,
            "debuff",
            pre_had=had_cute,
        )
        return True


def register_lethal() -> None:
    register(Phoenix())
    register(Cocoon())


register_lethal()
