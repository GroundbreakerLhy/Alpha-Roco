"""增益响应类特性（on_buff_gain）。

buff_gain 在引擎 add_buff 调用处广播；特性内部加的 buff 不广播（防连锁）。
"""

from __future__ import annotations

import copy

from ... import buffs as B
from ...evolution import evolve
from ..base import TraitHandler
from ..registry import register


class NutrientBubble(TraitHandler):
    trait_id = 200084
    name = "营养液泡"
    desc = "获得增益时，额外获得层数+2。"
    implemented = True

    def modify_buff_gain_layers(self, pet, buff_type, value):
        return 2


class Cheerleader(TraitHandler):
    trait_id = 200288
    name = "拉拉队长"
    desc = "若自己在萌化状态下再获得萌化会解除萌化。"
    implemented = True

    def allow_cute_reapplication(self, pet):
        return B.has_buff(pet, B.BuffType.CUTE)

    def on_buff_gain(self, ctx):
        if ctx.subject is not ctx.actor:
            return
        if ctx.extra.get("buff_type") != B.BuffType.CUTE:
            return
        # pre_had 由施加方广播时携带：本次获得前是否已处于萌化
        if not ctx.extra.get("pre_had"):
            return
        # 第二次施加不再退化：恢复既有萌化造成的退化，再清除萌化状态
        layers = B.get_buff_value(ctx.actor, B.BuffType.CUTE)
        added = ctx.extra.get("value", 0)
        for _ in range(max(0, layers - added)):
            if not evolve(ctx.actor):
                break
        B.remove_buff(ctx.actor, B.BuffType.CUTE)


class Carefree(TraitHandler):
    trait_id = 200182
    name = "无忧无虑"
    desc = "可获得的萌化层数不受限制。"
    implemented = True

    def allow_cute_beyond_limit(self, pet):
        # 退化到基础形态后仍可获得萌化；仍可退化时走正常退化流程
        return True


class Measure(TraitHandler):
    trait_id = 200221
    name = "衡量"
    desc = "入场时，复制敌方的增益。在场时，若敌方获得增益自己也会获得。"
    implemented = True

    def on_entry(self, ctx):
        # 复制敌方全部增益，保留原层数与来源标记
        if ctx.subject is not ctx.actor:
            return
        enemy = ctx.opponent()
        if enemy is None:
            return
        for buff in list(enemy.buffs):
            if buff.is_gain():
                ctx.actor.buffs.append(copy.copy(buff))

    def on_buff_gain(self, ctx):
        if ctx.subject is None or ctx.subject.side == ctx.actor.side:
            return
        if not ctx.is_active():
            return
        if ctx.extra.get("source_kind") == "trait":
            return
        buff_type = ctx.extra.get("buff_type")
        value = ctx.extra.get("value", 0)
        if not buff_type or value == 0:
            return
        if not B.is_buff(buff_type, value):
            return
        B.grant_buff(ctx.actor, buff_type, value)


def register_buff_gain() -> None:
    register(NutrientBubble())
    register(Measure())
    register(Cheerleader())
    register(Carefree())


register_buff_gain()
