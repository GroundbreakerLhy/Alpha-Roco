"""回合结束类特性（on_round_end / on_dot_damage / on_defense）。

- 200080 养分重吸收  回合结束时，回复3能量。
- 200095 双向光速    在场时，双方回合结束时的效果额外触发1次。
- 200203 煤渣草      在场时，所有灼烧的衰减变为增长。
- 200087 生长        回合结束时，回复12%生命。
- 200290 安可        使用光系技能后，在本侧特性回合末阶段完整返场。
- 200189 吸积盘      回合结束时，敌方获得2层星陨印记（marks id=7）。
- 200241 陨落        在场时，双方回合结束时的效果不会触发。
- 200240 耐活王      敌方受到中毒效果伤害时，自己回复等量生命（on_dot_damage）。
"""

from __future__ import annotations

from ... import traits as T
from ...enums import Element as E
from ..base import TraitHandler
from ..registry import register

LIGHT = E.LIGHT  # 4 光


# ---------------- 200203 煤渣草 ----------------
class CinderGrass(TraitHandler):
    trait_id = 200203
    name = "煤渣草"
    desc = "在场时，所有灼烧的衰减变为增长。"
    implemented = True

    def modify_burn_growth(self, ctx):
        return ctx.is_active()


# ---------------- 200095 双向光速 ----------------
class TwoWayLightSpeed(TraitHandler):
    trait_id = 200095
    name = "双向光速"
    desc = "在场时，双方回合结束时的效果会额外触发1次。"
    implemented = True

    def modify_round_end_repeats(self, ctx):
        if not ctx.is_active():
            return 0
        return 1


# ---------------- 200241 陨落 ----------------
class Fallen(TraitHandler):
    trait_id = 200241
    name = "陨落"
    desc = "在场时，双方回合结束时的效果不会触发。"
    implemented = True

    def modify_round_end_repeats(self, ctx):
        if not ctx.is_active():
            return 0
        return -1


# ---------------- 200080 养分重吸收 ----------------
class NourishReabsorb(TraitHandler):
    trait_id = 200080
    name = "养分重吸收"
    desc = "回合结束时，回复3能量。"
    implemented = True

    def on_round_end(self, ctx):
        if not ctx.is_active():
            return
        T.grant_energy(ctx.state, ctx.actor, 3)


# ---------------- 200087 生长 ----------------
class Growth(TraitHandler):
    trait_id = 200087
    name = "生长"
    desc = "回合结束时，回复12%生命。"
    implemented = True

    def on_round_end(self, ctx):
        if not ctx.is_active():
            return
        heal = int(ctx.actor.max_hp * 0.12)
        ctx.actor.hp = min(ctx.actor.max_hp, ctx.actor.hp + heal)


# ---------------- 200290 安可 ----------------
class Encore(TraitHandler):
    trait_id = 200290
    name = "安可"
    desc = "使用光系技能后，回合结束时自己返场。"
    implemented = True

    def on_skill_end(self, ctx):
        if ctx.subject is not ctx.actor or ctx.skill is None:
            return
        if ctx.skill.element != LIGHT:
            return
        ctx.set_state("used_light", True)

    def on_round_end(self, ctx):
        if not ctx.is_active():
            return
        if not ctx.state_of("used_light", False):
            return
        # 清除触发标记后立即完成返场，入场事件也在本侧特性阶段内结算。
        ctx.set_state("used_light", False)
        from ...battle import reenter
        reenter(ctx.state, ctx.actor.side, ctx.state.log)


# ---------------- 200189 吸积盘 ----------------
class AccretionDisk(TraitHandler):
    trait_id = 200189
    name = "吸积盘"
    desc = "回合结束时，敌方获得2层星陨印记。"
    implemented = True

    def on_round_end(self, ctx):
        if not ctx.is_active():
            return
        enemy = ctx.opponent()
        if enemy is None:
            return
        from ... import marks

        marks.add_mark(ctx.state, enemy.side, 7, 2)


# ---------------- 200240 耐活王 ----------------
class HardyKing(TraitHandler):
    trait_id = 200240
    name = "耐活王"
    desc = "敌方受到中毒效果伤害时，自己回复等量生命。"
    implemented = True

    def on_dot_damage(self, ctx):
        # 敌方（subject=受毒击的精灵）受到中毒伤害；自己/己方不触发
        if ctx.subject is None or ctx.subject.side == ctx.actor.side:
            return
        if ctx.extra.get("dot_type") != "poison":
            return
        if ctx.damage <= 0:
            return
        if not ctx.is_active():
            return
        ctx.actor.hp = min(ctx.actor.max_hp, ctx.actor.hp + ctx.damage)


def register_round_end() -> None:
    register(CinderGrass())
    register(TwoWayLightSpeed())
    register(Fallen())
    register(NourishReabsorb())
    register(Growth())
    register(Encore())
    register(AccretionDisk())
    register(HardyKing())


register_round_end()
