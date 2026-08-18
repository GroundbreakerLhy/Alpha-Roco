"""回合结束类特性（on_round_end）。

- 200080 养分重吸收  回合结束时，回复3能量。
"""

from __future__ import annotations

from ... import traits as T
from ..base import TraitHandler
from ..registry import register


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


def register_round_end() -> None:
    register(NourishReabsorb())


register_round_end()
