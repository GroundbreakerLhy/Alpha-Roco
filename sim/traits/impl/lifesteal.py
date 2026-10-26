"""吸血修正（modify_lifesteal）。

盛宴（280030）的吸血部分在 energy.py（按主效果归类）。
"""

from __future__ import annotations

from ...enums import Element as E
from ..registry import register
from ..base import TraitHandler

WATER = E.WATER
RAIN = 0


def _self(ctx):
    return ctx.target is ctx.actor


class EyeDrops(TraitHandler):
    trait_id = 200144
    name = "滴眼液"
    desc = "天气为雨天时，使用水系攻击技能时吸血50%。"
    implemented = True

    def modify_lifesteal(self, ctx):
        if not _self(ctx) or ctx.skill is None:
            return 0.0
        if (
            ctx.weather == RAIN
            and ctx.skill.element == WATER
            and ctx.skill.category in (0, 1)
        ):
            return 0.5
        return 0.0


def register_batch1_lifesteal() -> None:
    register(EyeDrops())


register_batch1_lifesteal()
