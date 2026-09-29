"""增益响应类特性（on_buff_gain）。

- 200084 营养液泡  获得增益时，额外获得层数+2。
- 200221 衡量      入场时复制敌方的增益；在场时敌方获得增益自己也获得。
- 200288 拉拉队长  若自己在萌化状态下再获得萌化会解除萌化。

接入说明：buff_gain 事件在引擎 add_buff 调用处（battle.py 迸发/印记、weather.py
暴风雪/雷鸣、化茧施加萌化后）由 traits.on_buff_gain 广播；特性内部自己加的
buff 不广播（防连锁）。
营养液泡只响应自己获得"增益"（Buff.is_gain()，特性来源不算增益）。
衡量复制的增益保留原有来源标记（正常显示为该精灵的普通增益）；
拉拉队长响应自己获得萌化（CUTE）：若已有萌化（CUTE buff 数量>1，即本次之前已有）
则解除——evolve 恢复一阶 + 减少一层 CUTE buff。
"""

from __future__ import annotations

import copy

from ... import buffs as B
from ...evolution import evolve
from ..base import TraitHandler
from ..registry import register


# ---------------- 200084 营养液泡 ----------------
class NutrientBubble(TraitHandler):
    trait_id = 200084
    name = "营养液泡"
    desc = "获得增益时，额外获得层数+2。"
    implemented = True

    def on_buff_gain(self, ctx):
        # 自己获得增益（subject=获得 buff 的精灵）
        if ctx.subject is not ctx.actor:
            return
        buff_type = ctx.extra.get("buff_type")
        value = ctx.extra.get("value", 0)
        if not buff_type or value == 0:
            return
        # 只有"增益"触发（特性来源不算增益）；减益/中性不触发
        if not B.is_buff(buff_type, value):
            return
        if not ctx.is_active():
            return
        B.add_buff(ctx.actor, buff_type, 2, source_kind="trait")
        ctx.add_state("triggers", 1)

    def display(self, state, pet):
        n = pet.trait_state.get("triggers", 0)
        if n <= 0:
            return None
        return [{"name": "增益层数", "layers": n, "per": "2", "gain": True}]


# ---------------- 200288 拉拉队长 ----------------
class Cheerleader(TraitHandler):
    trait_id = 200288
    name = "拉拉队长"
    desc = "若自己在萌化状态下再获得萌化会解除萌化。"
    implemented = True

    def on_buff_gain(self, ctx):
        if ctx.subject is not ctx.actor:
            return
        if ctx.extra.get("buff_type") != B.BuffType.CUTE:
            return
        # 本次获得萌化前是否已处于萌化状态（pre_had 由施加方广播时携带）
        if not ctx.extra.get("pre_had"):
            return  # 第一次萌化，不解除
        # 已有萌化 → 解除：evolve 恢复一阶 + 减少一层萌化 buff
        if not evolve(ctx.actor):
            return
        cute_buffs = [b for b in ctx.actor.buffs if b.buff_type == B.BuffType.CUTE]
        if cute_buffs:
            b = cute_buffs[-1]
            b.value -= 1
            if b.value <= 0:
                ctx.actor.buffs.remove(b)


# ---------------- 注册 ----------------
# ---------------- 200221 衡量 ----------------
class Measure(TraitHandler):
    trait_id = 200221
    name = "衡量"
    desc = "入场时，复制敌方的增益。在场时，若敌方获得增益自己也会获得。"
    implemented = True

    def on_entry(self, ctx):
        # 入场时复制敌方在场精灵的全部增益（保留原层数与来源标记）
        if ctx.subject is not ctx.actor:
            return
        enemy = ctx.opponent()
        if enemy is None:
            return
        for buff in list(enemy.buffs):
            if buff.is_gain():
                ctx.actor.buffs.append(copy.copy(buff))

    def on_buff_gain(self, ctx):
        # 敌方获得增益时自己也获得一份
        if ctx.subject is None or ctx.subject.side == ctx.actor.side:
            return
        if not ctx.is_active():
            return
        if ctx.extra.get("source_kind") == "trait":
            return  # 特性来源不算普通增益
        buff_type = ctx.extra.get("buff_type")
        value = ctx.extra.get("value", 0)
        if not buff_type or value == 0:
            return
        if not B.is_buff(buff_type, value):
            return
        B.add_buff(ctx.actor, buff_type, value)


def register_buff_gain() -> None:
    register(NutrientBubble())
    register(Measure())
    register(Cheerleader())


register_buff_gain()
