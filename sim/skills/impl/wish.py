"""愿力冲击（特性替换版，skill_id=-2）。

共鸣魔法版是 -1 的临时虚拟技能，回合结束还原；本版由特性永久替换、不还原，
威力 80，应对状态成功时 +150%，两者互不干扰。
"""

from __future__ import annotations

from ..base import SkillHandler
from ..ops import COUNTER_STATUS
from ..registry import register

WISH_TRAIT_SKILL_ID = -2  # 特性替换版（共鸣魔法版为 -1）


class WishImpact(SkillHandler):
    """愿力冲击（特性替换版）：威力 80，应对状态成功时本次威力+150%。"""

    skill_id = WISH_TRAIT_SKILL_ID
    name = "愿力冲击"
    category = 0
    counter_target = COUNTER_STATUS
    implemented = True

    def modify_power(self, ctx):
        percent, flat = super().modify_power(ctx)
        if ctx.is_counter:
            percent += 150.0
        return (percent, flat)


register(WishImpact())
