"""第 1 批-a：技能威力修正（modify_power）。

- 200075 目空       携带的非光系技能，威力+25%。
- 200077 勇敢       携带的能耗大于3的技能，威力+40%。
- 200088 挺起胸脯    携带的能耗为1的技能，威力+50%。
- 200106 观星       敌方每有1层星陨印记，自己的地系技能威力+20%。
- 200117 冰钻       敌方携带技能总能耗每有1点，自己攻击时威力+10%。
- 200120 冻土       每携带1个冰系技能进入战斗，地系技能威力+10%。
- 200124 顺风       若先于敌方攻击，本次技能威力+50%。（原第 0 批测试队伍，归入此类）
- 200192 绒粉星光    攻击时，若敌方血脉是非本系的系别血脉，技能威力+100%。
- 200193 月光审判    攻击时，若敌方血脉是首领血脉，技能威力+100%。
- 200243 变形活画    行动时，敌方每有1层增益，本次行动技能威力+10%，速度+5（modify_speed 也在本文件）。
- 200251 血型吸引    敌方每携带1种系别的技能，自己攻击时威力+10（固定值）。
- 200267 涂鸦       使用非本系技能时威力+50%。
- 280011 坠星       敌方每有1层星陨印记，自己的技能威力+20%。
- 280023 破空       若先于敌方攻击，本次技能威力+75%。

语义说明：除注明"敌方"的判定外，其余都是自身效果（handler 内用 ctx.target is ctx.actor 判定）。
绒粉星光"非本系的系别血脉"= 敌方血脉是元素且不在自己属性中（首领血脉 18 不算）；月光审判=敌方血脉为首领血脉（bloodline==18）。
变形活画数的是敌方"增益"层数（Buff.is_gain()，特性来源不算）。
"""

from __future__ import annotations

from ... import buffs as B
from ...enums import LORD_BLOODLINE, Element as E
from ..registry import register
from ..base import TraitHandler

LIGHT = E.LIGHT    # 4 光
EARTH = E.EARTH    # 5 地
ICE = E.ICE        # 6 冰
STAR_MARK = 7      # 星陨印记


def _self(ctx):
    return ctx.target is ctx.actor


def _enemy_star_stacks(ctx) -> int:
    """敌方星陨印记层数（敌方=ctx.subject）。"""
    enemy = ctx.subject
    if enemy is None:
        enemy = ctx.opponent()
    if enemy is None:
        return 0
    from ... import marks
    return marks.get_stacks(ctx.state, enemy.side, STAR_MARK)


def _enemy_gain_layers(pet) -> int:
    return sum(b.value for b in pet.buffs if b.is_gain())


# ---------------- 200124 顺风 ----------------
class ShunFeng(TraitHandler):
    trait_id = 200124
    name = "顺风"
    desc = "若先于敌方攻击，本次技能威力+50%。"
    implemented = True

    def modify_power(self, ctx):
        if not _self(ctx):
            return (0.0, 0.0)
        if ctx.is_first:
            return (50.0, 0.0)
        return (0.0, 0.0)


# ---------------- 200075 目空 ----------------
class MuKong(TraitHandler):
    trait_id = 200075
    name = "目空"
    desc = "携带的非光系技能，威力+25%。"
    implemented = True

    def modify_power(self, ctx):
        if not _self(ctx) or ctx.skill is None:
            return (0.0, 0.0)
        if ctx.skill.element == LIGHT:
            return (0.0, 0.0)
        return (25.0, 0.0)


# ---------------- 200077 勇敢 ----------------
class Brave(TraitHandler):
    trait_id = 200077
    name = "勇敢"
    desc = "携带的能耗大于3的技能，威力+40%。"
    implemented = True

    def modify_power(self, ctx):
        if not _self(ctx) or ctx.skill is None:
            return (0.0, 0.0)
        if ctx.skill.energy_cost > 3:
            return (40.0, 0.0)
        return (0.0, 0.0)


# ---------------- 200088 挺起胸脯 ----------------
class PuffChest(TraitHandler):
    trait_id = 200088
    name = "挺起胸脯"
    desc = "携带的能耗为1的技能，威力+50%。"
    implemented = True

    def modify_power(self, ctx):
        if not _self(ctx) or ctx.skill is None:
            return (0.0, 0.0)
        if ctx.skill.energy_cost == 1:
            return (50.0, 0.0)
        return (0.0, 0.0)


# ---------------- 200106 观星 ----------------
class StarGaze(TraitHandler):
    trait_id = 200106
    name = "观星"
    desc = "敌方每有1层星陨印记，自己的地系技能威力+20%。"
    implemented = True

    def modify_power(self, ctx):
        if not _self(ctx) or ctx.skill is None:
            return (0.0, 0.0)
        if ctx.skill.element != EARTH:
            return (0.0, 0.0)
        return (20.0 * _enemy_star_stacks(ctx), 0.0)


# ---------------- 200117 冰钻 ----------------
class IceDrill(TraitHandler):
    trait_id = 200117
    name = "冰钻"
    desc = "敌方携带技能总能耗每有1点，自己攻击时威力+10%。"
    implemented = True

    def modify_power(self, ctx):
        if not _self(ctx) or ctx.subject is None:
            return (0.0, 0.0)
        total = sum(s.energy_cost for s in ctx.subject.skills)
        return (10.0 * total, 0.0)


# ---------------- 200120 冻土 ----------------
class FrozenSoil(TraitHandler):
    trait_id = 200120
    name = "冻土"
    desc = "每携带1个冰系技能进入战斗，地系技能威力+10%。"
    implemented = True

    def modify_power(self, ctx):
        if not _self(ctx) or ctx.skill is None:
            return (0.0, 0.0)
        if ctx.skill.element != EARTH:
            return (0.0, 0.0)
        n_ice = sum(1 for s in ctx.actor.skills if s.element == ICE)
        return (10.0 * n_ice, 0.0)


# ---------------- 200192 绒粉星光 ----------------
class FluffyStarlight(TraitHandler):
    trait_id = 200192
    name = "绒粉星光"
    desc = "攻击时，若敌方血脉是非本系的系别血脉，技能威力+100%。"
    implemented = True

    def modify_power(self, ctx):
        if not _self(ctx) or ctx.subject is None:
            return (0.0, 0.0)
        # "非本系的系别血脉"：血脉是元素且不在自己属性中（首领血脉 18 不是系别血脉，不触发）
        if ctx.subject.bloodline is not None \
                and ctx.subject.bloodline != LORD_BLOODLINE \
                and ctx.subject.bloodline not in ctx.actor.attributes:
            return (100.0, 0.0)
        return (0.0, 0.0)


# ---------------- 200193 月光审判 ----------------
class MoonlightJudgment(TraitHandler):
    trait_id = 200193
    name = "月光审判"
    desc = "攻击时，若敌方血脉是首领血脉，技能威力+100%。"
    implemented = True

    def modify_power(self, ctx):
        if not _self(ctx) or ctx.subject is None:
            return (0.0, 0.0)
        if ctx.subject.bloodline == LORD_BLOODLINE:
            return (100.0, 0.0)
        return (0.0, 0.0)


# ---------------- 200243 变形活画 ----------------
class LivingCanvas(TraitHandler):
    trait_id = 200243
    name = "变形活画"
    desc = "行动时，敌方每有1层增益，本次行动技能威力+10%，速度+5。"
    implemented = True

    def modify_power(self, ctx):
        if not _self(ctx):
            return (0.0, 0.0)
        enemy = ctx.opponent()
        if enemy is None:
            return (0.0, 0.0)
        return (10.0 * _enemy_gain_layers(enemy), 0.0)

    def modify_speed(self, ctx):
        if not _self(ctx):
            return 0
        enemy = ctx.opponent()
        if enemy is None:
            return 0
        return 5 * _enemy_gain_layers(enemy)


# ---------------- 200251 血型吸引 ----------------
class BloodTypeAttraction(TraitHandler):
    trait_id = 200251
    name = "血型吸引"
    desc = "敌方每携带1种系别的技能，自己攻击时威力+10。"
    implemented = True

    def modify_power(self, ctx):
        if not _self(ctx) or ctx.subject is None:
            return (0.0, 0.0)
        n_types = len({s.element for s in ctx.subject.skills})
        return (0.0, 10.0 * n_types)


# ---------------- 200267 涂鸦 ----------------
class Graffiti(TraitHandler):
    trait_id = 200267
    name = "涂鸦"
    desc = "使用非本系技能时威力+50%。"
    implemented = True

    def modify_power(self, ctx):
        if not _self(ctx) or ctx.skill is None:
            return (0.0, 0.0)
        if ctx.skill.element not in ctx.actor.attributes:
            return (50.0, 0.0)
        return (0.0, 0.0)


# ---------------- 280011 坠星 ----------------
class FallingStar(TraitHandler):
    trait_id = 280011
    name = "坠星"
    desc = "敌方每有1层星陨印记，自己的技能威力+20%。"
    implemented = True

    def modify_power(self, ctx):
        if not _self(ctx):
            return (0.0, 0.0)
        return (20.0 * _enemy_star_stacks(ctx), 0.0)


# ---------------- 280023 破空 ----------------
class BreakSky(TraitHandler):
    trait_id = 280023
    name = "破空"
    desc = "若先于敌方攻击，本次技能威力+75%。"
    implemented = True

    def modify_power(self, ctx):
        if not _self(ctx):
            return (0.0, 0.0)
        if ctx.is_first:
            return (75.0, 0.0)
        return (0.0, 0.0)


def register_batch1_power() -> None:
    for cls in (ShunFeng, MuKong, Brave, PuffChest, StarGaze, IceDrill, FrozenSoil,
                FluffyStarlight, MoonlightJudgment, LivingCanvas, BloodTypeAttraction,
                Graffiti, FallingStar, BreakSky):
        register(cls())


register_batch1_power()
