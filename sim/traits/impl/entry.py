"""入场/离场类特性（on_entry / on_leave）。

- 200149 专注力    入场首回合，获得物攻+100%（首次行动——任意技能或聚能——后结束）。
- 200126 洁癖     离场后，自己的增益和减益会被更换入场的精灵继承。
- 200162 噼啪！   入场后首次行动，所选技能使用次数+1（迸发：skill_use_count）。

语义说明：
- 专注力：buff 为 NORMAL 时长，持续到首次行动（技能或聚能）之后才结束；下场/重新入场重新触发。
- 洁癖：离场事件在引擎清除普通 buff 之前广播，因此能拿到完整的增益/减益列表，全部复制给入场精灵。
- 特性带来的所有 buff 均带 source_kind="trait" 标记（不算普通"增益"，见 buffs.Buff.is_gain）。
"""

from __future__ import annotations

import copy

from ... import buffs as B
from ... import burst
from ..base import TraitHandler
from ..registry import register


# ---------------- 200149 专注力 ----------------
class Focus(TraitHandler):
    trait_id = 200149
    name = "专注力"
    desc = "入场首回合，获得物攻+100%。"
    implemented = True

    def on_entry(self, ctx):
        if ctx.subject is not ctx.actor:
            return
        # 重新入场/复活时清除可能残留的旧 buff 引用
        old = ctx.state_of("buff")
        if old is not None and old in ctx.actor.buffs:
            ctx.actor.buffs.remove(old)
        buff = B.Buff(buff_type=B.BuffType.ATK, value=10, duration=B.DurationKind.NORMAL,
                      source_kind="trait")
        ctx.actor.buffs.append(buff)
        ctx.set_state("buff", buff)

    def on_skill_end(self, ctx):
        # 首次使用技能后结束
        self._end(ctx)

    def on_charge(self, ctx):
        # 聚能（蓄能技能）同样视为使用技能，也消耗
        self._end(ctx)

    def on_leave(self, ctx):
        if ctx.subject is not ctx.actor:
            return
        ctx.set_state("buff", None)

    def _end(self, ctx):
        if ctx.subject is not ctx.actor:
            return
        buff = ctx.state_of("buff")
        if buff is None:
            return
        ctx.set_state("buff", None)
        if buff in ctx.actor.buffs:
            ctx.actor.buffs.remove(buff)


# ---------------- 200126 洁癖 ----------------
class Cleanliness(TraitHandler):
    trait_id = 200126
    name = "洁癖"
    desc = "离场后，自己的增益和减益会被更换入场的精灵继承。"
    implemented = True

    def on_leave(self, ctx):
        if ctx.subject is not ctx.actor:
            return
        incoming = ctx.extra.get("incoming")
        if incoming is None:
            return
        for buff in list(ctx.actor.buffs):
            incoming.buffs.append(copy.copy(buff))


# ---------------- 200162 噼啪！ ----------------
class Crackle(TraitHandler):
    trait_id = 200162
    name = "噼啪！"
    desc = "入场后首次行动，所选技能使用次数+1。"
    implemented = True

    def on_entry(self, ctx):
        if ctx.subject is not ctx.actor:
            return
        burst.add_burst(ctx.actor, "skill_use_count", 1)


def register_entry() -> None:
    for cls in (Focus, Cleanliness, Crackle):
        register(cls())


register_entry()
