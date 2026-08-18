"""第 1 批-c：能耗/能量修正（modify_energy_cost / modify_energy_limit / modify_energy_shortfall）。

- 200108 缩壳       携带的防御技能能耗-2。
- 200112 倾轧       携带的技能受能耗变化效果的影响翻倍。
- 200113 对流       自己的能耗增加变为能耗降低；能耗降低变为能耗增加。
- 200123 冰封       在场时，敌方全技能能耗+1（敌方光环，query 会查敌方在场精灵）。
- 200224 消波块     每携带1个水系技能进入战斗，地系技能能耗-1。
- 200252 留学生     自己全技能能耗+2（"可以学习全部攻击技能石"未建模，暂不实现）。
- 200215 多人宿舍   自己的能量可以超过能量上限（上限+999，视为无实际限制）。
- 200100 石头大餐   能量不足时，消耗5%生命，代替1能量。
- 280030 盛宴       能量不足时，消耗5%最大生命，代替1能量；生命低于50%时，获得吸血100%（modify_lifesteal）。

语义说明：能耗修正查询会额外查敌方在场精灵特性（冰封），因此 handler 用
ctx.target.side != ctx.actor.side 判断是否对敌方生效；自身类用 ctx.target is ctx.actor。
倾轧/对流基于自身当前 ENERGY_COST buff 总值翻转/翻倍（含特性来源，特性也是"能耗变化效果"）。
"""

from __future__ import annotations

from ... import buffs as B
from ...enums import Element as E
from ..registry import register
from ..base import TraitHandler

WATER = E.WATER    # 3 水
EARTH = E.EARTH    # 5 地
NO_LIMIT = 999     # 多人宿舍：能量上限视为无实际限制


def _self(ctx):
    return ctx.target is ctx.actor


def _pay_hp(ctx, need, percent_of_max: bool):
    """每1能量消耗 X% 生命（石头大餐=当前生命，盛宴=最大生命），返回可补充能量数。"""
    if not _self(ctx):
        return 0
    base = ctx.actor.max_hp if percent_of_max else ctx.actor.hp
    cost = max(1, int(base * 0.05))
    available = (ctx.actor.hp - 1) // cost
    n = min(need, available)
    if n <= 0:
        return 0
    ctx.actor.hp -= n * cost
    return n


# ---------------- 200108 缩壳 ----------------
class ShrinkShell(TraitHandler):
    trait_id = 200108
    name = "缩壳"
    desc = "携带的防御技能能耗-2。"
    implemented = True

    def modify_energy_cost(self, ctx):
        if not _self(ctx) or ctx.skill is None:
            return 0
        if ctx.skill.category == 2:
            return -2
        return 0


# ---------------- 200112 倾轧 ----------------
class Overwhelm(TraitHandler):
    trait_id = 200112
    name = "倾轧"
    desc = "携带的技能受能耗变化效果的影响翻倍。"
    implemented = True

    def modify_energy_cost(self, ctx):
        if not _self(ctx):
            return 0
        return B.get_buff_value(ctx.target, B.BuffType.ENERGY_COST)


# ---------------- 200113 对流 ----------------
class Convection(TraitHandler):
    trait_id = 200113
    name = "对流"
    desc = "自己的能耗增加变为能耗降低；能耗降低变为能耗增加。"
    implemented = True

    def modify_energy_cost(self, ctx):
        if not _self(ctx):
            return 0
        return -2 * B.get_buff_value(ctx.target, B.BuffType.ENERGY_COST)


# ---------------- 200123 冰封 ----------------
class IceSeal(TraitHandler):
    trait_id = 200123
    name = "冰封"
    desc = "在场时，敌方全技能能耗+1。"
    implemented = True

    def modify_energy_cost(self, ctx):
        if not ctx.is_active():
            return 0
        if ctx.target.side != ctx.actor.side:
            return 1
        return 0


# ---------------- 200224 消波块 ----------------
class WaveBreaker(TraitHandler):
    trait_id = 200224
    name = "消波块"
    desc = "每携带1个水系技能进入战斗，地系技能能耗-1。"
    implemented = True

    def modify_energy_cost(self, ctx):
        if not _self(ctx) or ctx.skill is None:
            return 0
        if ctx.skill.element != EARTH:
            return 0
        n_water = sum(1 for s in ctx.actor.skills if s.element == WATER)
        return -n_water


# ---------------- 200252 留学生 ----------------
class ExchangeStudent(TraitHandler):
    trait_id = 200252
    name = "留学生"
    desc = "自己全技能能耗+2，可以学习全部攻击技能石。"
    implemented = True

    def modify_energy_cost(self, ctx):
        if not _self(ctx):
            return 0
        return 2  # "攻击技能石"机制未建模，暂只实现能耗部分


# ---------------- 200215 多人宿舍 ----------------
class SharedDorm(TraitHandler):
    trait_id = 200215
    name = "多人宿舍"
    desc = "自己的能量可以超过能量上限。"
    implemented = True

    def modify_energy_limit(self, ctx):
        if not _self(ctx):
            return 0
        return NO_LIMIT


# ---------------- 200100 石头大餐 ----------------
class StoneFeast(TraitHandler):
    trait_id = 200100
    name = "石头大餐"
    desc = "能量不足时，消耗5%生命，代替1能量。"
    implemented = True

    def modify_energy_shortfall(self, ctx, need):
        return _pay_hp(ctx, need, percent_of_max=False)


# ---------------- 280030 盛宴 ----------------
class GrandFeast(TraitHandler):
    trait_id = 280030
    name = "盛宴"
    desc = "能量不足时，消耗5%最大生命，代替1能量。生命低于50%时，获得吸血100%。"
    implemented = True

    def modify_energy_shortfall(self, ctx, need):
        return _pay_hp(ctx, need, percent_of_max=True)

    def modify_lifesteal(self, ctx):
        if not _self(ctx):
            return 0.0
        if ctx.actor.hp * 2 < ctx.actor.max_hp:
            return 1.0
        return 0.0


def register_batch1_energy() -> None:
    for cls in (ShrinkShell, Overwhelm, Convection, IceSeal, WaveBreaker,
                ExchangeStudent, SharedDorm, StoneFeast, GrandFeast):
        register(cls())


register_batch1_energy()
