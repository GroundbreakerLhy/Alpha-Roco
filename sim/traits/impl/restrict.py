"""第 1 批-g：技能位限制（is_skill_usable）。

- 200208 正位宝剑   仅可以使用1号位技能（槽位 0）。
- 280017 宝剑王牌   仅可使用1号和3号位技能（槽位 0、2）。
"""

from __future__ import annotations

from ..registry import register
from ..base import TraitHandler


def _self(ctx):
    return ctx.target is ctx.actor


# ---------------- 200208 正位宝剑 ----------------
class UprightSword(TraitHandler):
    trait_id = 200208
    name = "正位宝剑"
    desc = "仅可以使用1号位技能。"
    implemented = True

    def is_skill_usable(self, ctx, skill):
        if not _self(ctx):
            return None
        if ctx.skill_index != 0:
            return False
        return None


# ---------------- 280017 宝剑王牌 ----------------
class SwordAce(TraitHandler):
    trait_id = 280017
    name = "宝剑王牌"
    desc = "仅可使用1号和3号位技能。"
    implemented = True

    def is_skill_usable(self, ctx, skill):
        if not _self(ctx):
            return None
        if ctx.skill_index not in (0, 2):
            return False
        return None


def register_batch1_restrict() -> None:
    for cls in (UprightSword, SwordAce):
        register(cls())


register_batch1_restrict()
