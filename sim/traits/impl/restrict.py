"""第 1 批-g：技能位限制 / 蓄力放宽（is_skill_usable / allow_any_skill_in_windup）。

- 200208 正位宝剑   仅可以使用1号位技能（槽位 0）。
- 280017 宝剑王牌   仅可使用1号和3号位技能（槽位 0、2）。
- 200174 嫉妒       蓄力状态下可以使用任一携带技能（放宽引擎的蓄力限制）。
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


# ---------------- 200174 嫉妒 ----------------
class Jealousy(TraitHandler):
    trait_id = 200174
    name = "嫉妒"
    desc = "蓄力状态下，可以使用任一携带技能。"
    implemented = True

    def allow_any_skill_in_windup(self, ctx):
        return _self(ctx)


def register_batch1_restrict() -> None:
    for cls in (UprightSword, SwordAce, Jealousy):
        register(cls())


register_batch1_restrict()
