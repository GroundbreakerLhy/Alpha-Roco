"""击败/力竭类特性（on_kill / on_faint）。

- 200141 付给恶魔的赎价  击败敌方精灵时，敌方额外损失1点魔力；被敌方精灵击败时，自己额外损失1点魔力。
- 200081 诈死          自己力竭时，少损失1点魔力。
- 200230 不朽          力竭4回合后复活，恢复满血满能量，并保留萌化。
- 200211 飓风          其他翼系队友携带相同技能时，本精灵对应技能获得迅捷；被敌方击败时额外损失1点魔力。
- 200257 恶魔的晚宴    主动击败敌方精灵时，自己永久获得双攻+50%（PERMANENT buff，下场不消失）。
"""

from __future__ import annotations

from ... import buffs as B
from ... import traits as T
from ...enums import Element
from ..base import TraitHandler
from ..registry import register


# ---------------- 200141 付给恶魔的赎价 ----------------
class DevilsPrice(TraitHandler):
    trait_id = 200141
    name = "付给恶魔的赎价"
    desc = (
        "击败敌方精灵时，敌方额外损失1点魔力。被敌方精灵击败时，自己额外损失1点魔力。"
    )
    implemented = True

    def on_kill(self, ctx):
        if ctx.target is not ctx.actor:
            return
        ctx.state.magic[ctx.subject.side] -= 1

    def on_faint(self, ctx):
        if ctx.subject is not ctx.actor:
            return
        ctx.state.magic[ctx.actor.side] -= 1


# ---------------- 200081 诈死 ----------------
class FeignDeath(TraitHandler):
    trait_id = 200081
    name = "诈死"
    desc = "自己力竭时，少损失1点魔力。"
    implemented = True

    def on_faint(self, ctx):
        if ctx.subject is not ctx.actor:
            return
        # 引擎在力竭时已扣 1 点魔力（_apply_faint 先扣再广播 faint），
        # 这里回调 1 点 = 少损失 1 点魔力
        ctx.state.magic[ctx.actor.side] += 1


# ---------------- 200230 不朽 ----------------
class Undying(TraitHandler):
    trait_id = 200230
    name = "不朽"
    desc = "力竭4回合后复活。"
    implemented = True

    def on_battle_start(self, ctx):
        ctx.set_state("revive_in", None)
        ctx.set_state("revive_faint_turn", None)
        ctx.set_state("revive_tick_turn", None)

    def on_faint(self, ctx):
        if ctx.subject is not ctx.actor:
            return
        # 力竭事件发生在引擎扣除魔力之后；这里只设置复活计时，不返还魔力。
        if ctx.state_of("revive_in") is not None:
            return
        ctx.set_state("revive_in", 4)
        ctx.set_state("revive_faint_turn", ctx.state.turn)
        ctx.set_state("revive_tick_turn", None)

    def on_round_end(self, ctx):
        turns = ctx.state_of("revive_in")
        if turns is None or ctx.actor.hp > 0:
            return
        # 力竭发生的当回合不计入，且同一回合重复结算时只递减一次。
        if ctx.state.turn <= ctx.state_of("revive_faint_turn", -1):
            return
        if ctx.state_of("revive_tick_turn") == ctx.state.turn:
            return
        ctx.set_state("revive_tick_turn", ctx.state.turn)
        turns -= 1
        if turns > 0:
            ctx.set_state("revive_in", turns)
            return

        ctx.actor.hp = ctx.actor.max_hp
        ctx.actor.energy = T.query_energy_limit(ctx.state, ctx.actor)
        ctx.actor.buffs = [
            buff for buff in ctx.actor.buffs if buff.buff_type == B.BuffType.CUTE
        ]
        ctx.set_state("revive_in", None)
        ctx.set_state("revive_faint_turn", None)
        ctx.set_state("revive_tick_turn", None)
        T.emit(ctx.state, "revive", scope="all", side=ctx.actor.side, subject=ctx.actor)


# ---------------- 200211 飓风 ----------------
class Hurricane(TraitHandler):
    trait_id = 200211
    name = "飓风"
    desc = "对本精灵的技能，若其他翼系精灵携带相同技能，则获得迅捷。被敌方精灵击败时，自己额外损失1点魔力。"
    implemented = True

    def on_battle_start(self, ctx):
        teammates = [
            pet
            for pet in ctx.state.teams[ctx.actor.side]
            if pet is not ctx.actor and Element.WING in pet.attributes
        ]
        if not teammates:
            return
        teammate_skill_ids = {
            skill.skill_id for pet in teammates for skill in pet.skills
        }
        for skill in ctx.actor.skills:
            if skill.skill_id in teammate_skill_ids:
                skill.swift = True

    def on_kill(self, ctx):
        # kill 事件的 subject 是被击败者，target 是击杀者。
        if ctx.subject is not ctx.actor:
            return
        if ctx.target is None or ctx.target.side == ctx.actor.side:
            return
        ctx.set_state("defeated_by_enemy", True)

    def on_faint(self, ctx):
        if ctx.subject is not ctx.actor:
            return
        if not ctx.state_of("defeated_by_enemy", False):
            return
        ctx.state.magic[ctx.actor.side] -= 1
        ctx.set_state("defeated_by_enemy", False)


# ---------------- 200257 恶魔的晚宴 ----------------
class DemonFeast(TraitHandler):
    trait_id = 200257
    name = "恶魔的晚宴"
    desc = "主动击败敌方精灵时，自己永久获得双攻+50%。"
    implemented = True

    def on_kill(self, ctx):
        if ctx.target is not ctx.actor:
            return
        # 双攻+50% = 物攻/魔攻各 +5 层（每层10%），PERMANENT 下场不消失
        B.add_buff(
            ctx.actor, B.BuffType.ATK, 5, B.DurationKind.PERMANENT, source_kind="trait"
        )
        B.add_buff(
            ctx.actor,
            B.BuffType.SPATK,
            5,
            B.DurationKind.PERMANENT,
            source_kind="trait",
        )
        ctx.add_state("kills", 1)

    def display(self, state, pet):
        n = pet.trait_state.get("kills", 0)
        if n <= 0:
            return None
        return [{"name": "双攻", "layers": n, "per": "50%", "gain": True}]


def register_kill() -> None:
    register(DevilsPrice())
    register(FeignDeath())
    register(Undying())
    register(Hurricane())
    register(DemonFeast())


register_kill()
