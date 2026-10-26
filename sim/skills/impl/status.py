"""状态类技能（category 3）。

效果全部以代码定义，desc 不参与运行时；能耗/属性取自 skills.json。
"""

from __future__ import annotations

from ... import buffs, skill_utils
from ...data_loader import build_skill_map, load_spirits
from ..ops import (
    AddMark,
    ApplyBuff,
    CanCute,
    ConvertBuffs,
    COUNTER_DEFENSE,
    DispelBuffs,
    DispelMarks,
    DoubleBuffs,
    ENEMY,
    EnemySwitchedThisTurn,
    FIELD,
    GROWTH_ENERGY_COST,
    GainEnergy,
    Heal,
    LEAVE_BOTH,
    LEAVE_SELF,
    LeaveField,
    OpsSkill,
    Overload,
    SlotIn,
    per_any_mark,
    RandomStatBuffs,
    Reenter,
    Repeat,
    SELF,
    SELF_BACKLINE,
    SkillGrowth,
    When,
    apply_effects,
    hit_count_of,
    per_recorded,
    skill_by_id,
)
from ..registry import register


# ---------------- 纯 buff 状态技（无其他效果） ----------------
class BuffStatus(OpsSkill):
    """纯 buff 状态技（只有"自己/敌方获得 X"）。

    属性增益 10% = 1 层；速度每层 +10；冻结由 add_buff 强制 PERMANENT 并实时
    检查斩杀（buffs.check_freeze）；常规增益离场清除（B36）、可驱散（S18，
    含永久时长；特性自身的展示效果不算增益/减益）。
    """

    def __init__(
        self, skill_id: int, name: str, buff_type: str, value: int, target: str
    ):
        self.skill_id = skill_id
        self.name = name
        self.category = 3
        self.status_effects = (
            ApplyBuff(buff_type=buff_type, value=value, target=target),
        )
        self.implemented = True


for _handler in (
    BuffStatus(7090160, "霜降", buffs.BuffType.FREEZE, 4, ENEMY),
    BuffStatus(7120120, "毒孢子", buffs.BuffType.POISON, 5, ENEMY),
    BuffStatus(7040260, "引燃", buffs.BuffType.BURN, 10, ENEMY),
    BuffStatus(7020770, "退化", buffs.BuffType.CUTE, 1, ENEMY),
):
    register(_handler)

for _handler in (
    BuffStatus(7020620, "力量增效", buffs.BuffType.ATK, 10, SELF),
    BuffStatus(7020630, "魔法增效", buffs.BuffType.SPATK, 7, SELF),
    BuffStatus(7050220, "润泽", buffs.BuffType.SPATK, 19, SELF),
    BuffStatus(7150120, "乘风", buffs.BuffType.SPEED, 12, SELF),
    BuffStatus(7150140, "暴风眼", buffs.BuffType.HIT_COUNT_PERCENT, 100, SELF),
):
    register(_handler)


# ---------------- 纯印记状态技（无其他效果） ----------------
class MarkStatus(OpsSkill):
    """纯印记状态技（只有"自己/敌方获得 N 层 X 印记"）。

    印记以阵营为单位：marks.add_mark 按印记正/负属性写入对应槽位，单条上限
    99（B38）。
    """

    def __init__(
        self, skill_id: int, name: str, mark_id: int, stacks: int, target: str
    ):
        self.skill_id = skill_id
        self.name = name
        self.category = 3
        self.status_effects = (AddMark(mark_id=mark_id, stacks=stacks, target=target),)
        self.implemented = True


for _handler in (
    MarkStatus(7030300, "光合作用", 10, 1, SELF),
    MarkStatus(7050240, "打湿", 11, 1, SELF),
):
    register(_handler)

for _handler in (
    MarkStatus(7090330, "速冻", 3, 2, ENEMY),
    MarkStatus(7020740, "棘刺", 1, 1, ENEMY),
):
    register(_handler)


# ---------------- 7020640 休息回复 ----------------
class RestHeal(OpsSkill):
    """回复自身生命。

    回复量经 ``traits.query_heal`` 修正（如"戏耍"可把回复转为敌方扣血），
    不超过最大生命。
    """

    skill_id = 7020640
    name = "休息回复"
    category = 3
    implemented = True

    status_effects = (Heal(percent=0.30),)


register(RestHeal())


# ---------------- 7020840 借用 ----------------
class Borrow(OpsSkill):
    """每回合随机变成己方队伍中其他精灵的技能。

    替换由引擎统一处理（battle._apply_round_replacements_for_pet）：入场抽取、
    每回合末重随、离场还原。池 = 己方其他存活精灵携带的技能，按 id 从
    skills.json 重建原始状态——不继承对方的运行时成长。直接使用本技能无效果。
    """

    skill_id = 7020840
    name = "借用"
    category = 3
    implemented = True
    round_replace = True

    def round_replace_pool(self, state, pet, skill):
        skill_map = build_skill_map()
        return [
            skill_by_id(s.skill_id)
            for p in state.teams[pet.side]
            if p is not pet and p.hp > 0
            for s in p.skills
            if s.skill_id in skill_map and s.skill_id != 9999999
        ]


register(Borrow())


# ---------------- 7020850 取念 ----------------
class ThoughtSteal(OpsSkill):
    """每回合随机变成敌方任意精灵的技能，并使其能耗降低。

    池 = 敌方存活精灵携带的技能，从 skills.json 重建原始状态（不含运行时成长）；
    临时技能能耗降低（round_replace_cost_delta）。
    """

    skill_id = 7020850
    name = "取念"
    category = 3
    implemented = True
    round_replace = True
    round_replace_cost_delta = -2

    def round_replace_pool(self, state, pet, skill):
        enemy_side = "B" if pet.side == "A" else "A"
        skill_map = build_skill_map()
        return [
            skill_by_id(s.skill_id)
            for p in state.teams[enemy_side]
            if p.hp > 0
            for s in p.skills
            if s.skill_id in skill_map and s.skill_id != 9999999
        ]


register(ThoughtSteal())


# ---------------- 7020860 复写 ----------------
class Rewrite(OpsSkill):
    """每回合随机变成自己未携带的技能，并使其能耗降低。

    池 = 可习得技能（normal 池 + 血脉编号与当前血脉一致的血脉池）减去当前
    携带的技能；临时技能能耗降低。
    """

    skill_id = 7020860
    name = "复写"
    category = 3
    implemented = True
    round_replace = True
    round_replace_cost_delta = -2

    def round_replace_pool(self, state, pet, skill):
        spirit = next(
            (sp for sp in load_spirits() if sp["id"] == pet.spirit_id), None
        )
        if spirit is None:
            return []
        carried = {s.skill_id for s in pet.skills}
        skill_map = build_skill_map()
        pool_ids = [
            sid for sid in spirit["skills"]["normal"]
            if sid not in carried and sid in skill_map and sid != 9999999
        ]
        for entry in spirit["skills"].get("bloodline", []):
            sid = entry["skillId"]
            if (
                entry["bloodline"] == pet.bloodline
                and sid not in carried
                and sid in skill_map
                and sid != 9999999
            ):
                pool_ids.append(sid)
        return [skill_by_id(sid) for sid in pool_ids]


register(Rewrite())


# ---------------- 7030510 富养化 ----------------
class Eutrophication(OpsSkill):
    """为场下每个精灵回复能量。

    目标为己方场下（SELF_BACKLINE，不含使用者）；回复经 traits.grant_energy
    走能量上限修正与 energy_gain 广播。
    """

    skill_id = 7030510
    name = "富养化"
    category = 3
    implemented = True

    status_effects = (GainEnergy(value=3, target=SELF_BACKLINE),)


register(Eutrophication())


# ---------------- 7110270 加大功率 ----------------
class PowerUp(OpsSkill):
    """自己脱离，替换入场的精灵回复能量。

    脱离走 LeaveField 登记（B16：行动照常结算，行动后立即离场、暂停选人）；
    entry_energy 随暂停上下文传给续接流程，替补入场结算后经 traits.grant_energy
    回复。无存活替补 / 蓄力免疫时脱离不生效，能量也不发。
    """

    skill_id = 7110270
    name = "加大功率"
    category = 3
    implemented = True

    status_effects = (LeaveField(mode=LEAVE_SELF, entry_energy=8),)


register(PowerUp())


# ---------------- 7180190 恶意逃离 ----------------
class MaliciousEscape(OpsSkill):
    """脱离，应对防御：额外提高敌方攻击技能能耗。

    脱离走 LeaveField 登记（B16）。应对从句延迟到回合末结算（B46），离场不
    影响结算（登记先于离场）。ATTACK_ENERGY_COST 只对攻击技能生效，经
    skill_utils.energy_modifier_before_traits 计入真实能耗（付费/校验/显示同管线）。
    """

    skill_id = 7180190
    name = "恶意逃离"
    category = 3
    counter_target = COUNTER_DEFENSE
    implemented = True

    status_effects = (LeaveField(mode=LEAVE_SELF),)
    counter_effects = (
        ApplyBuff(buff_type=buffs.BuffType.ATTACK_ENERGY_COST, value=4, target=ENEMY),
    )


register(MaliciousEscape())


# ---------------- 7150260 飞羽 ----------------
class SwiftFeather(OpsSkill):
    """迅捷，驱散敌方 1 种增益。

    迅捷是静态属性（构造期由 bind_skill 写入 ``skill.swift``）：主动换人入场时
    能量足够则自动释放，不强制先手。驱散涵盖全部增益（含永久时长）；特性自身
    的展示效果不是增益（S18）。
    """

    skill_id = 7150260
    name = "飞羽"
    category = 3
    swift = True
    implemented = True

    status_effects = (DispelBuffs(kind="buff", count=1, target=ENEMY),)


register(SwiftFeather())


# ---------------- 7021100 晒太阳 ----------------
class Sunbath(OpsSkill):
    """驱散敌方所有增益。

    count=None 表示全部；常规时长与永久时长的增益都在范围内。特性自身的展示
    效果不是增益（另一个类别），不受影响（S18）。
    """

    skill_id = 7021100
    name = "晒太阳"
    category = 3
    implemented = True

    status_effects = (DispelBuffs(kind="buff", count=None, target=ENEMY),)


register(Sunbath())


# ---------------- 7030290 移花接木 ----------------
class Switcheroo(OpsSkill):
    """自己回复生命，随后脱离。

    先回复后脱离：脱离不取消本次行动——回复照常结算，随后立即离场、跳过自己的
    回合末结算，并由己方重新选人（引擎暂停、choose_replacement 续接）。
    无存活替补时脱离不生效。
    """

    skill_id = 7030290
    name = "移花接木"
    category = 3
    implemented = True

    status_effects = (
        Heal(percent=0.15),
        LeaveField(mode=LEAVE_SELF),
    )


register(Switcheroo())


# ---------------- 7030490 花炮 ----------------
class FlowerCannon(OpsSkill):
    """连击，每次连击自己获得魔攻增益。

    每次连击独立施加（Repeat），不是一次施加总层数——"获得增益时追加"类特性
    按次数分别响应。连击数走统一查询（hit_count_of）。
    """

    skill_id = 7030490
    name = "花炮"
    category = 3
    hit_count_flat = 1  # 固有 2 连击
    implemented = True

    status_effects = (
        Repeat(
            count=hit_count_of,
            effects=(
                ApplyBuff(buff_type=buffs.BuffType.SPATK, value=6),
            ),
        ),
    )


register(FlowerCannon())


# ---------------- 7080190 流沙 ----------------
class Quicksand(OpsSkill):
    """敌方获得禁足，应对防御：敌方双防大幅下降。

    禁足（LOCK）层数 = 剩余回合数，回合末逐层衰减（buffs._apply_lock），期间
    无法主动换人（buffs.can_switch）。应对从句由引擎在 STAGE_COUNTER 回合末
    结算（B46）；双防各 -6 层（属性增减益 10%/层）。
    """

    skill_id = 7080190
    name = "流沙"
    category = 3
    counter_target = COUNTER_DEFENSE
    implemented = True

    status_effects = (
        ApplyBuff(buff_type=buffs.BuffType.LOCK, value=3, target=ENEMY),
    )
    counter_effects = (
        ApplyBuff(buff_type=buffs.BuffType.DEF, value=-6, target=ENEMY),
        ApplyBuff(buff_type=buffs.BuffType.SPDEF, value=-6, target=ENEMY),
    )


register(Quicksand())


# ---------------- 7150270 风隐 ----------------
class WindHide(OpsSkill):
    """敌方和自己均脱离，先手-1。

    双方均脱离 = LeaveField(LEAVE_BOTH)：行动照常结算（B16），结算后立即进入
    收集阶段——双方各自盲选替补，集齐后按先手顺序"先退先入、后退后入"（B17）。
    若双方都先手-1，则按有效速度拼速、同速随机，先手方的换人先完成，对方尚未
    结算的行动直接作废。先手-1 = priority_delta。
    """

    skill_id = 7150270
    name = "风隐"
    category = 3
    priority_delta = -1
    implemented = True

    status_effects = (LeaveField(mode=LEAVE_BOTH),)


register(WindHide())


# ---------------- 7090470 打喷嚏 ----------------
class Sneeze(OpsSkill):
    """连击，每次连击敌方获得冻结。

    每次连击独立施加（Repeat），"获得冻结时追加"类特性按次数分别响应。
    连击数走统一查询（hit_count_of）。
    """

    skill_id = 7090470
    name = "打喷嚏"
    category = 3
    hit_count_flat = 2  # 固有 3 连击
    implemented = True

    status_effects = (
        Repeat(
            count=hit_count_of,
            effects=(
                ApplyBuff(buff_type=buffs.BuffType.FREEZE, value=1, target=ENEMY),
            ),
        ),
    )


register(Sneeze())


# ---------------- 7150300 羽化加速 ----------------
class FeatherAccel(OpsSkill):
    """自己获得全技能威力固定值增益，迅捷。

    描述无 %，走固定值通道（SKILL_POWER_FLAT），与百分比通道
    （SKILL_POWER_PERCENT）是两条独立 buff。迅捷同飞羽。
    """

    skill_id = 7150300
    name = "羽化加速"
    category = 3
    swift = True
    implemented = True

    status_effects = (ApplyBuff(buff_type=buffs.BuffType.SKILL_POWER_FLAT, value=20),)


register(FeatherAccel())


# ---------------- 7140140 化劲 ----------------
class NeutralizeForce(OpsSkill):
    """自己获得全技能威力固定值增益。

    与羽化加速同一固定值通道（SKILL_POWER_FLAT），不做百分比换算
    （SKILL_POWER_PERCENT 是另一条 buff）。
    """

    skill_id = 7140140
    name = "化劲"
    category = 3
    implemented = True

    status_effects = (ApplyBuff(buff_type=buffs.BuffType.SKILL_POWER_FLAT, value=40),)


register(NeutralizeForce())


# ---------------- 7040500 焚烧烙印 ----------------
class BurnBrand(OpsSkill):
    """驱散双方所有印记，每驱散 1 层，敌方获得灼烧。

    覆盖双方阵营的正/负印记（DispelMarks target=FIELD）；总层数经 record_as
    写入 ctx.extra，灼烧 = 5 × 总层数一次施加。驱散 0 层时不生效。
    """

    skill_id = 7040500
    name = "焚烧烙印"
    category = 3
    implemented = True

    status_effects = (
        DispelMarks(count=None, target=FIELD, record_as="dispelled_marks"),
        ApplyBuff(
            buff_type=buffs.BuffType.BURN,
            value=per_recorded("dispelled_marks", 5),
            target=ENEMY,
        ),
    )


register(BurnBrand())


# ---------------- 7090400 冰点 ----------------
class FreezingPoint(OpsSkill):
    """敌方获得冻结，应对防御：额外获得一份。

    本体（STAGE_STATUS）与应对成功（STAGE_COUNTER）各施加一次，两次独立事件。
    冻结的永久性与实时斩杀判定见 buffs.check_freeze。
    """

    skill_id = 7090400
    name = "冰点"
    category = 3
    counter_target = COUNTER_DEFENSE
    implemented = True

    status_effects = (
        ApplyBuff(buff_type=buffs.BuffType.FREEZE, value=5, target=ENEMY),
    )
    counter_effects = (
        ApplyBuff(buff_type=buffs.BuffType.FREEZE, value=5, target=ENEMY),
    )


register(FreezingPoint())


# ---------------- 7160180 赤子之心 ----------------
class InnocentHeart(OpsSkill):
    """自己获得萌化：全技能能耗永久下降。

    "获得萌化：X"族的前置条件是能获得萌化（存在前一进化阶段）：已是初始阶段时
    整组效果都不执行（When + CanCute，S22）。能萌化时先给能耗负层（PERMANENT，
    离场保留 B36），萌化在技能结算之后施加（evolution.apply_cute）。
    """

    skill_id = 7160180
    name = "赤子之心"
    category = 3
    implemented = True

    status_effects = (
        When(
            condition=CanCute(),
            effects=(
                ApplyBuff(
                    buff_type=buffs.BuffType.ENERGY_COST,
                    value=-2,
                    duration=buffs.DurationKind.PERMANENT,
                ),
                ApplyBuff(buff_type=buffs.BuffType.CUTE, value=1),
            ),
        ),
    )


register(InnocentHeart())


# ---------------- 7170240 嘲弄 ----------------
class Taunt(OpsSkill):
    """自己获得魔攻增益，若敌方本回合更换精灵，自己获得速度增益。

    换人先于行动结算（step 中 switch 先行），故敌方本回合主动换人在状态结算时
    entry_turn == 当前回合，EnemySwitchedThisTurn 即刻成立；回合中脱离换上
    （暂停续接，switch_in 刷新 entry_turn）同理。
    """

    skill_id = 7170240
    name = "嘲弄"
    category = 3
    implemented = True

    status_effects = (
        ApplyBuff(buff_type=buffs.BuffType.SPATK, value=9),
        When(
            condition=EnemySwitchedThisTurn(),
            effects=(ApplyBuff(buff_type=buffs.BuffType.SPEED, value=7),),
        ),
    )


register(Taunt())


# ---------------- 7130210 贮藏 ----------------
class Storage(OpsSkill):
    """自己获得双攻增益，每携带 1 个 0 能耗技能再加一份。

    "0 能耗"按实际能耗判定（true_energy_cost：天气/印记/buff/特性/技能自身
    修正全计入），不与 skills.json 的原始能耗比较。
    """

    skill_id = 7130210
    name = "贮藏"
    category = 3
    implemented = True

    def on_status(self, ctx):
        zero_cost = sum(
            1
            for s in ctx.actor.skills
            if skill_utils.true_energy_cost(ctx.state, ctx.actor, s) == 0
        )
        layers = 5 + 5 * zero_cost
        apply_effects(
            ctx,
            (
                ApplyBuff(buff_type=buffs.BuffType.ATK, value=layers),
                ApplyBuff(buff_type=buffs.BuffType.SPATK, value=layers),
            ),
        )


register(Storage())


# ---------------- 7050400 洗礼 ----------------
class Baptism(OpsSkill):
    """驱散自己的减益，并获得全技能降耗。

    驱散全部减益（含永久时长；特性自身的展示效果不是减益，S18）；随后获得
    ENERGY_COST 负层（降耗增益），常规时长、离场清除。
    """

    skill_id = 7050400
    name = "洗礼"
    category = 3
    implemented = True

    status_effects = (
        DispelBuffs(kind="debuff", count=None, target=SELF),
        ApplyBuff(buff_type=buffs.BuffType.ENERGY_COST, value=-1),
    )


register(Baptism())


# ---------------- 7150320 疾风连袭 ----------------
class GaleFlurry(OpsSkill):
    """重放自己释放过的迅捷技能；能耗 = 其能耗之和的一半，每次使用后 +1。

    重放集合按当前迅捷状态实时过滤（飓风特性可让任意技能获得迅捷）。
    - 动态能耗 = 集合内技能原始能耗之和整除 2 + 使用成长（SkillGrowth）。
    - 付费后按技能槽位顺序免费重放（battle.replay_used_swift_skills），与正常
      使用同等的基础结算，无应对效果；自身永远在集合外（防递归）。
    """

    skill_id = 7150320
    name = "疾风连袭"
    category = 3
    implemented = True

    def modify_energy_cost(self, ctx):
        from ...battle import used_swift_skills

        total = sum(s.energy_cost for s in used_swift_skills(ctx.actor, self.skill_id))
        return super().modify_energy_cost(ctx) + total // 2

    def on_status(self, ctx):
        from ...battle import replay_used_swift_skills

        replay_used_swift_skills(
            ctx.state, ctx.actor, self.skill_id, is_first=ctx.is_first, logs=ctx.logs
        )
        SkillGrowth(field=GROWTH_ENERGY_COST, delta=1).apply(ctx)


register(GaleFlurry())


# ---------------- 7190430 叠加态 ----------------
class Superposition(OpsSkill):
    """自己获得随机层数属性增益，巧变：幻系攻击技能。

    - RandomStatBuffs 逐次掷点，按属性汇总后逐种施加（每种一个事件）。
    - 巧变池 = skills.json 中全部幻系（element 17）攻击技能（category 0/1，
      按结构化字段过滤）；用后随机变成其一（能耗-1），用后还原。
    """

    skill_id = 7190430
    name = "叠加态"
    category = 3
    implemented = True

    status_effects = (RandomStatBuffs(layers=16),)

    def on_bind(self, pet, skill):
        pool = [
            skill_by_id(sid)
            for sid, raw in build_skill_map().items()
            if raw["element"] == 17 and raw["category"] in (0, 1)
        ]
        return {"morph_pool": pool}


register(Superposition())


# ---------------- 7110360 过载回路 ----------------
class OverloadCircuit(OpsSkill):
    """回合结束自己返场，下回合所选技能使用次数+1。

    - 返场成功后获得真实迸发：使用次数 +1 持续到下回合末，由下一次行动消费，
      未消费则到期移除；不走 overload_next 的隐藏过载计数。蓄力技能不参与重复。
    - Reenter 登记，引擎在回合末技能结算之后、力竭结算后与特性返场同一时点
      执行（离场+立即入场，完整重跑入场结算）；本回合已入场的精灵免疫
      （entry_turn == 当前回合）。
    - 回合末结算会对在场精灵的全部技能发 STAGE_ROUND_END，故用使用回合戳门控
      （skill_state）：只有本回合用过本技能才返场，返场后仍在场，戳记不会重复
      触发。
    """

    skill_id = 7110360
    name = "过载回路"
    category = 3
    implemented = True

    def on_status(self, ctx):
        ctx.set_state("used_turn", ctx.state.turn)

    def on_round_end(self, ctx):
        if ctx.state_of("used_turn") != ctx.state.turn:
            return
        ctx.set_state("used_turn", None)
        apply_effects(ctx, (Reenter(),))
        ctx.extra["reenter_bursts"] = [("skill_use_count", 1)]


register(OverloadCircuit())


# ---------------- 7040440 怒火 ----------------
class Wrath(OpsSkill):
    """自己获得双攻增益和双防减益。

    四条属性一次结算（属性 10%/层）。常规时长：离场清除、可被驱散（S18）；
    减益侧层数为负，可被"驱散减益"剥掉。
    """

    skill_id = 7040440
    name = "怒火"
    category = 3
    implemented = True

    status_effects = (
        ApplyBuff(buff_type=buffs.BuffType.ATK, value=12, target=SELF),
        ApplyBuff(buff_type=buffs.BuffType.SPATK, value=12, target=SELF),
        ApplyBuff(buff_type=buffs.BuffType.DEF, value=-4, target=SELF),
        ApplyBuff(buff_type=buffs.BuffType.SPDEF, value=-4, target=SELF),
    )


register(Wrath())


# ---------------- 本体给"下一次攻击技能威力"增益 ----------------
def _next_attack_buffs(percent: int = 0, flat: int = 0) -> tuple:
    """把"下一次攻击技能威力"数值包成效果元组（0 值不产生效果）。"""
    effects = []
    if percent:
        effects.append(
            ApplyBuff(buff_type=buffs.BuffType.NEXT_ATTACK_POWER_PERCENT, value=percent)
        )
    if flat:
        effects.append(
            ApplyBuff(buff_type=buffs.BuffType.NEXT_ATTACK_POWER_FLAT, value=flat)
        )
    return tuple(effects)


class NextAttackPowerStatus(OpsSkill):
    """ "下一次攻击时，技能威力+N%（/ +N）"类状态技（本体增益）。

    增益一次性（NEXT_ATTACK_POWER_PERCENT / _FLAT）：由引擎在下一次攻击技能
    用时取走并计入该次威力——本次行动（含本技能自身）不受影响，故"下一次"
    不需要额外门控。带 % 走百分比通道，不带 % 按固定值。常规时长：离场清除、
    可被驱散。
    """

    def __init__(self, skill_id: int, name: str, percent: int = 0, flat: int = 0):
        self.skill_id = skill_id
        self.name = name
        self.category = 3
        self.status_effects = _next_attack_buffs(percent, flat)
        self.implemented = True


# ---------------- 7040270 热身 ----------------
class WarmUp(NextAttackPowerStatus):
    """下一次攻击技能威力翻倍，应对防御：改为 4 倍。

    应对成功时在回合末（B46 应对从句）先 remove 再 apply，把 +100% 替换为
    +300%——是"改为"而非叠加。
    """

    counter_target = COUNTER_DEFENSE

    def __init__(self):
        super().__init__(7040270, "热身", percent=100)

    def on_counter(self, ctx):
        if ctx.counter_category != COUNTER_DEFENSE:
            return
        buffs.remove_buff(ctx.actor, buffs.BuffType.NEXT_ATTACK_POWER_PERCENT)
        apply_effects(
            ctx,
            _next_attack_buffs(percent=300),
        )


for _handler in (
    NextAttackPowerStatus(7020700, "伺机而动", flat=70),
):
    register(_handler)

register(WarmUp())


# ---------------- 7080200 泥浆铠甲 ----------------
class MudArmor(OpsSkill):
    """自己获得物攻和物防增益，应对防御：额外使自己的增益翻倍。

    应对成功时在回合末（B46 应对从句）把自己的全部增益层数翻倍
    （DoubleBuffs(kind="buff")）：含本技能刚给的，也含此前已有的；翻倍受层数
    钳制（B37）。
    """

    skill_id = 7080200
    name = "泥浆铠甲"
    category = 3
    counter_target = COUNTER_DEFENSE
    implemented = True

    status_effects = (
        ApplyBuff(buff_type=buffs.BuffType.ATK, value=6, target=SELF),
        ApplyBuff(buff_type=buffs.BuffType.DEF, value=6, target=SELF),
    )
    counter_effects = (DoubleBuffs(kind="buff", target=SELF),)


register(MudArmor())


# ---------------- 7160160 生日蛋糕 ----------------
class BirthdayCake(OpsSkill):
    """驱散自己的减益，自己的增益翻倍。

    两条效果按序即时结算：先驱散全部减益（含永久时长；特性自身的展示效果不是
    减益，S18），再把全部增益层数翻倍（DoubleBuffs，受层数钳制 B37）。驱散减益
    不算增益，不会被第二条翻倍。
    """

    skill_id = 7160160
    name = "生日蛋糕"
    category = 3
    implemented = True

    status_effects = (
        DispelBuffs(kind="debuff", count=None, target=SELF),
        DoubleBuffs(kind="buff", target=SELF),
    )


register(BirthdayCake())


# ---------------- 7160270 击鼓传花 ----------------
class PassTheParcel(OpsSkill):
    """自己脱离，下个入场精灵继承自己增益。

    脱离走 LeaveField 登记（B16）。离场执行时在清除 NORMAL buff 前快照全部增益
    （含 PERMANENT，按原时长；减益不继承），由续接流程逐条加给本侧下一只入场
    精灵（add_buff 统一钳制，叠加到其已有增益）。无存活替补 / 蓄力免疫时脱离
    不生效，快照作废。
    """

    skill_id = 7160270
    name = "击鼓传花"
    category = 3
    implemented = True

    status_effects = (LeaveField(mode=LEAVE_SELF, entry_inherit_buffs=True),)


register(PassTheParcel())


# ---------------- 7070060 轴承支撑 ----------------
class BearingSupport(OpsSkill):
    """左右相邻技能的永久能耗支撑。

    携带时当前技能列表中左右相邻的技能能耗 -1；每次主动使用后，该被动永久
    额外 -1。相邻关系按当前技能列表循环判定，传动重排后随位置变化：
    1 号位两侧是 4、2 号位，4 号位两侧是 3、1 号位。
    """

    skill_id = 7070060
    name = "轴承支撑"
    category = 3
    drive = 1
    implemented = True

    _PASSIVE_REDUCTION = "bearing_passive_reduction"

    def on_after_use(self, ctx):
        ctx.add_state(self._PASSIVE_REDUCTION, 1)

    def modify_other_skill_energy_cost(self, ctx, target_skill, target_index):
        skills = ctx.actor.skills
        source_index = next(
            (i for i, candidate in enumerate(skills) if candidate is ctx.skill), -1
        )
        if source_index < 0 or target_index == source_index or len(skills) <= 1:
            return 0
        neighbors = {(source_index - 1) % len(skills), (source_index + 1) % len(skills)}
        if target_index not in neighbors:
            return 0
        reduction = 1 + int(ctx.state_of(self._PASSIVE_REDUCTION, 0) or 0)
        return -reduction


register(BearingSupport())


# ---------------- 7140330 马步 ----------------
class HorseStance(OpsSkill):
    """先手-1；选择：低生命时回复，或高生命时获得物攻增益。

    两个支路的生命条件都在使用技能时实时判定（状态本体结算时），条件不成立时
    该支路无效果。先手-1 = priority_delta。
    """

    skill_id = 7140330
    name = "马步"
    category = 3
    priority_delta = -1
    choice = True
    implemented = True

    def on_status(self, ctx):
        if ctx.choice_branch == 0:
            if ctx.actor.hp * 5 < ctx.actor.max_hp:
                Heal(percent=0.60).apply(ctx)
        elif ctx.actor.hp * 5 > ctx.actor.max_hp * 4:
            ApplyBuff(buff_type=buffs.BuffType.ATK, value=15).apply(ctx)


register(HorseStance())


# ---------------- 7120130 毒雾 ----------------
class PoisonFog(OpsSkill):
    """将敌方所有增益转化为相同层数的中毒。

    "增益"按效果类型与数值判定（与时长无关；特性自身的展示效果不是增益、不参与
    转化，S18）：全部增益层数合计后一次性转为等量中毒（add_buff 统一钳制），
    增益本身被移除。
    """

    skill_id = 7120130
    name = "毒雾"
    category = 3
    implemented = True

    status_effects = (
        ConvertBuffs(
            from_kind="buff",
            to_buff_type=buffs.BuffType.POISON,
            target=ENEMY,
        ),
    )


register(PoisonFog())


# ---------------- 7100170 龙吟 ----------------
class DragonChant(OpsSkill):
    """蓄力：释放时自己获得双攻和速度增益。

    首段"开始蓄力"支付能耗并进入蓄力状态（蓄力不是状态行动，不参与应对，
    B33），下一回合再次选择该技能即释放（不再付费）。增益常规时长、离场清除。
    """

    skill_id = 7100170
    name = "龙吟"
    category = 3
    windup = True
    implemented = True

    status_effects = (
        ApplyBuff(buff_type=buffs.BuffType.ATK, value=15),
        ApplyBuff(buff_type=buffs.BuffType.SPATK, value=15),
        ApplyBuff(buff_type=buffs.BuffType.SPEED, value=8),
    )


register(DragonChant())


# ---------------- 7020720 操控 ----------------
class Manipulate(OpsSkill):
    """敌方技能能耗 +7，持续 3 回合。"""

    skill_id = 7020720
    name = "操控"
    category = 3
    implemented = True

    status_effects = (
        ApplyBuff(
            buff_type=buffs.BuffType.ENERGY_COST,
            value=7,
            target=ENEMY,
            turns=3,
        ),
    )


register(Manipulate())


# ---------------- 7030250 氧输送 ----------------
class OxygenSupply(OpsSkill):
    """自己回复 4 能量，并获得魔攻 +70%。"""

    skill_id = 7030250
    name = "氧输送"
    category = 3
    implemented = True

    status_effects = (
        GainEnergy(value=4),
        ApplyBuff(buff_type=buffs.BuffType.SPATK, value=7),
    )


register(OxygenSupply())


# ---------------- 7060240 放晴 ----------------
class ClearSky(OpsSkill):
    """光系技能威力永久 +50%；应对防御时改为永久 +100%。

    "改为"按增量实现：基础 +50，应对防御成功再 +50，合计 100%。光系 element 4。
    """

    skill_id = 7060240
    name = "放晴"
    category = 3
    counter_target = COUNTER_DEFENSE
    implemented = True

    status_effects = (
        ApplyBuff(
            buff_type=buffs.element_power_buff_type(4),
            value=50,
            duration=buffs.DurationKind.PERMANENT,
        ),
    )

    counter_effects = (
        ApplyBuff(
            buff_type=buffs.element_power_buff_type(4),
            value=50,
            duration=buffs.DurationKind.PERMANENT,
        ),
    )


register(ClearSky())


# ---------------- 7070170 啮合传递 ----------------
class MeshingTransmission(OpsSkill):
    """自己获得速度 +30；位于 1/3 号位时额外获得物攻 +80%；传动 1。"""

    skill_id = 7070170
    name = "啮合传递"
    category = 3
    drive = 1
    implemented = True

    status_effects = (
        ApplyBuff(buff_type=buffs.BuffType.SPEED, value=3),
        When(
            condition=SlotIn(slots=(1, 3)),
            effects=(ApplyBuff(buff_type=buffs.BuffType.ATK, value=8),),
        ),
    )


register(MeshingTransmission())


# ---------------- 7080370 钧势 ----------------
class GallowsStance(OpsSkill):
    """自己获得物防 +140% 和速度 -30。"""

    skill_id = 7080370
    name = "钧势"
    category = 3
    implemented = True

    status_effects = (
        ApplyBuff(buff_type=buffs.BuffType.DEF, value=14),
        ApplyBuff(buff_type=buffs.BuffType.SPEED, value=-3),
    )


register(GallowsStance())


# ---------------- 7120240 疫病吐息 ----------------
register(MarkStatus(7120240, "疫病吐息", 4, 1, ENEMY))


# ---------------- 7160200 甜心续航 ----------------
class SweetRecovery(OpsSkill):
    """自己和敌方获得萌化，并各回复 40% 生命。

    萌化按 CanCute 口径：该侧能获得萌化才给萌化与回复，不能获得时该侧不结算。
    """

    skill_id = 7160200
    name = "甜心续航"
    category = 3
    implemented = True

    def on_status(self, ctx):
        for target in (SELF, ENEMY):
            pet = ctx.actor if target == SELF else ctx.target
            if pet is None or not buffs.can_gain_buff(pet, buffs.BuffType.CUTE, 1):
                continue
            ApplyBuff(buff_type=buffs.BuffType.CUTE, value=1, target=target).apply(ctx)
            Heal(percent=0.40, target=target).apply(ctx)


register(SweetRecovery())


# ---------------- 7190280 超维投射 ----------------
register(MarkStatus(7190280, "超维投射", 7, 4, ENEMY))


# ---------------- 7190300 星链 ----------------
class StarChain(OpsSkill):
    """2 连击，每次连击使敌方获得 1 层星陨印记。"""

    skill_id = 7190300
    name = "星链"
    category = 3
    hit_count_flat = 1
    implemented = True

    status_effects = (
        Repeat(
            count=hit_count_of,
            effects=(AddMark(mark_id=7, stacks=1, target=ENEMY),),
        ),
    )


register(StarChain())


# ---------------- 7190320 心灵洞悉 ----------------
class MindInsight(OpsSkill):
    """敌方获得星陨印记，层数等于敌方已有印记总层数（正负合计）。"""

    skill_id = 7190320
    name = "心灵洞悉"
    category = 3
    implemented = True

    status_effects = (
        AddMark(mark_id=7, stacks=per_any_mark(1, ENEMY), target=ENEMY),
    )


register(MindInsight())
