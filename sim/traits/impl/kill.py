"""击败/力竭类特性（on_kill / on_faint）。

- 200141 付给恶魔的赎价  击败敌方精灵时，敌方额外损失1点魔力；被敌方精灵击败时，自己额外损失1点魔力。
"""

from __future__ import annotations

from ..base import TraitHandler
from ..registry import register


# ---------------- 200141 付给恶魔的赎价 ----------------
class DevilsPrice(TraitHandler):
    trait_id = 200141
    name = "付给恶魔的赎价"
    desc = "击败敌方精灵时，敌方额外损失1点魔力。被敌方精灵击败时，自己额外损失1点魔力。"
    implemented = True

    def on_kill(self, ctx):
        if ctx.target is not ctx.actor:
            return
        ctx.state.magic[ctx.subject.side] -= 1

    def on_faint(self, ctx):
        if ctx.subject is not ctx.actor:
            return
        ctx.state.magic[ctx.actor.side] -= 1


def register_kill() -> None:
    register(DevilsPrice())


register_kill()
