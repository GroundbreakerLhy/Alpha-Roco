"""巧变类特性（morph_pool 填充 + 威力修正）。

巧变由引擎（battle._apply_morph）结算：原技能用后替换为池中随机技能（能耗-1），
临时技能用后再还原；特性只负责填充 morph_pool。池为同元素全部技能（不限类别）。
"""

from __future__ import annotations

from ...data_loader import skill_pool_for_element
from ..base import TraitHandler
from ..registry import register

POWER_UP = {
    7020890: 10,
    7020410: 20,
    7070130: 20,
    7170210: 5,
}

CHARGE_SKILL_ID = 9999999  # 蓄能：不进巧变随机池


class DiscSwap(TraitHandler):
    trait_id = 200287
    name = "换碟"
    desc = "自己携带的音波弹/音爆/金属噪音/午夜噪音威力提升，且获得巧变：同系别技能。"
    implemented = True

    def modify_power(self, ctx):
        if ctx.target is not ctx.actor or ctx.skill is None:
            return (0.0, 0.0)
        gain = POWER_UP.get(ctx.skill.skill_id, 0)
        if gain <= 0:
            return (0.0, 0.0)
        return (0.0, float(gain))

    def on_battle_start(self, ctx):
        # 池中排除技能自身与蓄能
        for skill in ctx.actor.skills:
            if skill.skill_id not in POWER_UP:
                continue
            skill.morph_pool = [
                template
                for template in skill_pool_for_element(skill.element)
                if template.skill_id != skill.skill_id
                and template.skill_id != CHARGE_SKILL_ID
            ]


def register_morph() -> None:
    register(DiscSwap())


register_morph()
