"""回合结束类特性（on_round_end / on_dot_damage / on_defense）。

- 200080 养分重吸收  回合结束时，回复3能量。
- 200095 双向光速    在场时，双方回合结束时的效果额外触发1次。
- 200203 煤渣草      在场时，所有灼烧的衰减变为增长。
- 200087 生长        回合结束时，回复12%生命。
- 200092 警惕        回合结束时，若自己能量为0则脱离（通过 state.pending_switch
  请求，服务端发 choose_replacement 由玩家手动选择上场精灵）。
- 200105 奔波命      使用防御技能后，回合结束时脱离（同上机制）。
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


# ---------------- 200092 警惕 ----------------
class Vigilance(TraitHandler):
    trait_id = 200092
    name = "警惕"
    desc = "回合结束时，若自己能量为0则脱离。"
    implemented = True

    def on_round_end(self, ctx):
        if not ctx.is_active():
            return
        if ctx.actor.energy != 0:
            return
        # 请求换人：服务端发 choose_replacement，玩家手动选择上场精灵
        # （场下无存活精灵时不请求）
        if not any(p.hp > 0 for i, p in enumerate(ctx.state.teams[ctx.actor.side])
                   if i != ctx.state.active[ctx.actor.side]):
            return
        ctx.state.pending_switch[ctx.actor.side] = True


# ---------------- 200105 奔波命 ----------------
class BusyLife(TraitHandler):
    trait_id = 200105
    name = "奔波命"
    desc = "使用防御技能后，回合结束时脱离。"
    implemented = True

    def on_defense(self, ctx):
        if ctx.subject is not ctx.actor:
            return
        ctx.set_state("defended", True)

    def on_round_end(self, ctx):
        if not ctx.is_active():
            return
        if not ctx.state_of("defended", False):
            return
        # 本回合用过防御技能：请求换人（服务端发 choose_replacement 玩家手动选）
        if not any(p.hp > 0 for i, p in enumerate(ctx.state.teams[ctx.actor.side])
                   if i != ctx.state.active[ctx.actor.side]):
            return
        ctx.state.pending_switch[ctx.actor.side] = True
        ctx.set_state("defended", False)


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
        # 本回合用过光系技能：自己返场（离场并立即入场，引擎自动执行）
        ctx.state.pending_reenter[ctx.actor.side] = True
        ctx.set_state("used_light", False)


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
    register(Vigilance())
    register(BusyLife())
    register(Encore())
    register(AccretionDisk())
    register(HardyKing())


register_round_end()
