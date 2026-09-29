"""巧变类特性（morph_pool 填充 + 威力修正）。

- 200287 换碟  自己携带的音波弹/音爆/金属噪音/午夜噪音威力提升，且获得巧变：同系别技能。

巧变机制（引擎侧已实现）：BattleSkill.morph_pool = 随机池模板列表，
技能使用后由 battle._apply_morph 结算——原技能用后替换为池中随机技能（能耗-1），
临时技能用后再还原为原技能。特性只负责填充 morph_pool。

语义：
- 威力提升为固定值：音波弹+10、音爆+20、金属噪音+20、午夜噪音+5。
- 巧变池 = 全游戏中与该技能同元素（同系别）的全部技能（不限类别）。
"""

from __future__ import annotations

from ...data_loader import skill_pool_for_element
from ..base import TraitHandler
from ..registry import register

# 技能 id -> 威力固定提升值
POWER_UP = {
    7020890: 10,  # 音波弹
    7020410: 20,  # 音爆
    7070130: 20,  # 金属噪音
    7170210: 5,   # 午夜噪音
}

CHARGE_SKILL_ID = 9999999  # 蓄能：不进巧变随机池


# ---------------- 200287 换碟 ----------------
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
        # 四个技能获得巧变：同系别技能池（全游戏该元素的全部技能）
        # 池中排除技能自身与蓄能
        for skill in ctx.actor.skills:
            if skill.skill_id not in POWER_UP:
                continue
            skill.morph_pool = [
                template for template in skill_pool_for_element(skill.element)
                if template.skill_id != skill.skill_id
                and template.skill_id != CHARGE_SKILL_ID
            ]


def register_morph() -> None:
    register(DiscSwap())


register_morph()
