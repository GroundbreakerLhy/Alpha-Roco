"""防御类技能（category 2）。

按应对成功后的效果归并为若干语义子类（纯防御 / 给自己增益 / 给敌方减益 /
印记 / 无法合并的 SpecialDefense），效果全部显式写在代码中，
``skills.json`` 只用于展示和人工校对。
"""

from __future__ import annotations

from ... import buffs
from ...data_loader import build_skill_map
from ..ops import (
    COUNTER_ATTACK,
    deal_extra_damage as _deal_extra_damage,
    ENEMY,
    LEAVE_ENEMY,
    LEAVE_SELF,
    SELF,
    AddMark,
    ApplyBuff,
    BoostCounteredSkillEnergyCost,
    DispelBuffs,
    Heal,
    LeaveField,
    LoseEnergy,
    OpsSkill,
    apply_effects,
)
from ..registry import register

GUARD = COUNTER_ATTACK


class DefenseSkill(OpsSkill):
    """防御技能的统一声明器。

    简单防御在下方表格声明一行；应对效果挂入 ``counter_effects``。
    """

    def __init__(
        self,
        skill_id: int,
        name: str,
        reduction: float,
        *,
        counter_effects: tuple = (),
        implemented: bool = False,
        drive: int = 0,
        swift: bool = False,
    ):
        self.skill_id = skill_id
        self.name = name
        self.category = 2
        self.counter_target = GUARD
        self.damage_reduction = reduction
        self.counter_effects = counter_effects
        self.implemented = implemented
        self.drive = drive
        self.swift = swift


class PureDefense(DefenseSkill):
    """只有减伤和静态属性的防御。"""


class DefenseAfterSelfBuff(DefenseSkill):
    """应对成功后给自己施加增益。"""


class DefenseAfterEnemyDebuff(DefenseSkill):
    """应对成功后给敌方施加减益。"""


class DefenseAfterSelfMark(DefenseSkill):
    """应对成功后给己方添加印记。"""


class DefenseAfterEnemyMark(DefenseSkill):
    """应对成功后给敌方添加印记。"""


class SpecialDefense(DefenseSkill):
    """无法仅靠“减伤 + 应对后效果”合并的防御。"""


# ---------------- 7060150 镜像反射 ----------------
class MirrorReflection(DefenseSkill):
    """应对攻击时复制本次被应对技能的全部当前属性（含 skill_state 成长）。"""

    def __init__(self):
        super().__init__(7060150, "镜像反射", 0.70, implemented=True)

    def on_counter(self, ctx):
        incoming = ctx.extra.get("incoming_skill")
        if incoming is None:
            return
        ctx.skill.skill_id = incoming.skill_id
        ctx.skill.name = incoming.name
        ctx.skill.element = incoming.element
        ctx.skill.category = incoming.category
        ctx.skill.power = incoming.power
        ctx.skill.energy_cost = incoming.energy_cost
        ctx.skill.drive = incoming.drive
        ctx.skill.swift = incoming.swift
        ctx.skill.windup = incoming.windup
        ctx.skill.choice = incoming.choice
        ctx.skill.usable = incoming.usable
        ctx.skill.skill_state = dict(incoming.skill_state)


register(MirrorReflection())


# ---------------- 按效果类型登记 ----------------
def _register_group(cls, rows):
    for skill_id, name, reduction, *rest in rows:
        kwargs = rest[0] if rest else {}
        register(cls(skill_id, name, reduction, **kwargs))


register(PureDefense(7020780, "防御", 0.70, implemented=True))
register(PureDefense(7150150, "风墙", 0.50, implemented=True, swift=True))

_register_group(
    DefenseAfterSelfBuff,
    (
        (
            7020790,
            "防反",
            0.70,
            {
                "counter_effects": (
                    ApplyBuff(buff_type=buffs.BuffType.ATK, value=7),
                    ApplyBuff(buff_type=buffs.BuffType.SPATK, value=7),
                ),
                "implemented": True,
            },
        ),
        (
            7021110,
            "有效预防",
            0.50,
            {
                "counter_effects": (
                    ApplyBuff(buff_type=buffs.BuffType.PRIORITY, value=1),
                ),
                "implemented": True,
            },
        ),
        (
            7030320,
            "酶浓度调整",
            0.80,
            {
                "counter_effects": (Heal(percent=0.20),),
                "implemented": True,
            },
        ),
        (
            7030520,
            "纤维化",
            0.80,
            {
                "counter_effects": (ApplyBuff(buff_type=buffs.BuffType.DEF, value=7),),
                "implemented": True,
            },
        ),
        (
            7040450,
            "淬火",
            0.80,
            {
                "counter_effects": (
                    ApplyBuff(
                        buff_type=buffs.BuffType.NEXT_ATTACK_POWER_PERCENT, value=100
                    ),
                ),
                "implemented": True,
            },
        ),
        (
            7040640,
            "暖气",
            0.70,
            {
                "counter_effects": (
                    ApplyBuff(
                        buff_type=buffs.BuffType.NEXT_ATTACK_POWER_FLAT, value=50
                    ),
                ),
                "implemented": True,
            },
        ),
        (
            7050270,
            "水泡盾",
            0.80,
            {
                "counter_effects": (
                    ApplyBuff(buff_type=buffs.BuffType.SPATK, value=7),
                ),
                "implemented": True,
            },
        ),
        (
            7050280,
            "水环",
            0.60,
            {
                "counter_effects": (
                    ApplyBuff(buff_type=buffs.BuffType.ENERGY_COST, value=-2),
                ),
                "implemented": True,
            },
        ),
        (
            7060250,
            "点亮",
            0.90,
            {
                "counter_effects": (
                    ApplyBuff(
                        buff_type=buffs.element_power_buff_type(4),
                        value=50,
                        duration=buffs.DurationKind.PERMANENT,
                    ),
                ),
                "implemented": True,
            },
        ),
        (
            7080410,
            "不动如山",
            0.90,
            {
                "counter_effects": (ApplyBuff(buff_type=buffs.BuffType.ATK, value=5),),
                "implemented": True,
            },
        ),
        (
            7140300,
            "防御反击",
            0.80,
            {
                "counter_effects": (
                    ApplyBuff(buff_type=buffs.BuffType.SKILL_POWER_FLAT, value=40),
                ),
                "implemented": True,
            },
        ),
        (
            7150310,
            "羽翼庇护",
            0.70,
            {
                "counter_effects": (
                    ApplyBuff(buff_type=buffs.BuffType.HIT_COUNT_FLAT, value=2),
                ),
                "implemented": True,
            },
        ),
        (
            7170260,
            "虚化",
            0.80,
            {
                "counter_effects": (
                    ApplyBuff(buff_type=buffs.BuffType.SPDEF, value=7),
                ),
                "implemented": True,
            },
        ),
        (
            7180210,
            "等价交换",
            0.90,
            {
                "counter_effects": (
                    ApplyBuff(buff_type=buffs.BuffType.LIFESTEAL, value=5),
                ),
                "implemented": True,
            },
        ),
    ),
)

_register_group(
    DefenseAfterEnemyDebuff,
    (
        (
            7040300,
            "火焰护盾",
            0.70,
            {
                "counter_effects": (
                    ApplyBuff(buff_type=buffs.BuffType.BURN, value=6, target=ENEMY),
                ),
                "implemented": True,
            },
        ),
        (
            7080220,
            "刺盾",
            0.70,
            {
                "counter_effects": (
                    ApplyBuff(buff_type=buffs.BuffType.ATK, value=-7, target=ENEMY),
                ),
                "implemented": True,
            },
        ),
        (
            7080400,
            "淤泥表皮",
            0.80,
            {
                "counter_effects": (
                    ApplyBuff(
                        buff_type=buffs.BuffType.HIT_COUNT_FLAT, value=-3, target=ENEMY
                    ),
                ),
                "implemented": True,
            },
        ),
        (
            7090200,
            "冰天雪地",
            0.80,
            {
                "counter_effects": (BoostCounteredSkillEnergyCost(value=3),),
                "implemented": True,
            },
        ),
        (
            7090210,
            "冰墙",
            0.80,
            {
                "counter_effects": (
                    ApplyBuff(buff_type=buffs.BuffType.FREEZE, value=2, target=ENEMY),
                ),
                "implemented": True,
            },
        ),
        (
            7160290,
            "捧杀",
            0.90,
            {
                "counter_effects": (
                    ApplyBuff(buff_type=buffs.BuffType.CUTE, value=1, target=ENEMY),
                ),
                "implemented": True,
            },
        ),
    ),
)

_register_group(
    DefenseAfterSelfMark,
    (
        (
            7050420,
            "潮汐",
            0.60,
            {
                "counter_effects": (AddMark(mark_id=11, stacks=1, target=SELF),),
                "implemented": True,
            },
        ),
        (
            7160340,
            "委屈",
            0.70,
            {
                "counter_effects": (AddMark(mark_id=8, stacks=1, target=SELF),),
                "implemented": True,
            },
        ),
    ),
)

_register_group(
    DefenseAfterEnemyMark,
    (
        (
            7090340,
            "冰蛋壳",
            0.70,
            {
                "counter_effects": (AddMark(mark_id=3, stacks=2),),
                "implemented": True,
            },
        ),
        (
            7190400,
            "冥想",
            0.80,
            {
                "counter_effects": (AddMark(mark_id=7, stacks=2),),
                "implemented": True,
            },
        ),
    ),
)


# 可驱散的增益多于 1 种时随机选取（S18）；驱散覆盖全部增益（含永久时长），
# 但特性自身的展示效果不是增益、不在范围内（S18）。
_register_group(
    SpecialDefense,
    (
        (
            7120270,
            "溶解",
            0.90,
            {
                "counter_effects": (DispelBuffs(kind="buff", count=1, target=ENEMY),),
                "implemented": True,
            },
        ),
    ),
)


# ---------------- 不好合并的防御 ----------------
_register_group(
    SpecialDefense,
    (
        (7020810, "无畏之心", 1.00),
        (7030330, "蜡质膜", 0.80),
        (7080240, "壁垒", 0.90),
        (7090350, "雪替身", 0.70),
        (7100250, "龙血", 0.70),
        (7100320, "守护咒", 0.90),
        (7110280, "集中", 0.80),
        (7110420, "电磁偏转", 0.70),
        (7120300, "毒肽", 0.70),
        (7130220, "掩护", 0.70),
        (7130320, "虫结阵", 0.80),
    ),
)


# ---------------- 7070080 能量守恒 ----------------
class EnergyConservation(SpecialDefense):
    """减伤 80%；应对攻击：两侧技能能耗永久 -1。

    "两侧"按技能环取左右相邻。永久减耗按技能 id 记入
    ``energy_cost_boost_<id>``（PERMANENT，离场保留），每人每技能最多累计到
    0 能耗（引擎最后统一钳制）。
    """

    def __init__(self):
        super().__init__(7070080, "能量守恒", 0.80, implemented=True)

    def on_counter(self, ctx):
        if ctx.counter_category != COUNTER_ATTACK:
            return
        for neighbor in ctx.actor.skills.neighbors(ctx.skill):
            ApplyBuff(
                buff_type=buffs.skill_energy_boost_type(neighbor.skill_id),
                value=-1,
                duration=buffs.DurationKind.PERMANENT,
            ).apply(ctx)


register(EnergyConservation())


# ---------------- 7140180 硬门 ----------------
class HardGate(SpecialDefense):
    """应对攻击：打断被应对技能，并造成物理伤害。

    打断是静态属性（counter_interrupt，B19）：对方本次攻击行动立即作废，故自身
    不会受到这次攻击。附加伤害在自己的行动结算时产生（防御应对攻击强制先手），
    若打死对方则直接力竭；缺省沿用本技能属性，类别显式给物理。
    """

    def __init__(self):
        super().__init__(7140180, "硬门", 0.00, implemented=True)
        self.counter_interrupt = True

    def on_defense(self, ctx):
        if not ctx.is_counter:
            return
        _deal_extra_damage(ctx, 90, category=0)


register(HardGate())


# ---------------- 7140190 听桥 ----------------
class ListenBridge(SpecialDefense):
    """应对攻击：先对敌方造成武系物理伤害，威力与被应对技能相等。

    附加伤害在自己的行动结算时产生：先给对面计算伤害，对面若还存活，自己才吃
    这次攻击——反击直接打死的话，对方的行动不再结算、自己不扣血。威力取被应对
    技能的当前威力（数据威力 + skill_state["power"] 成长，结算时读取）；被应对
    行动无技能实体（聚能等）时不结算。伤害走统一威力管线。
    """

    def __init__(self):
        super().__init__(7140190, "听桥", 0.60, implemented=True)

    def on_defense(self, ctx):
        if not ctx.is_counter:
            return
        incoming = ctx.extra.get("incoming_skill")
        if incoming is None or incoming.power is None:
            return
        growth = int(incoming.skill_state.get("power", 0) or 0)
        power = int(incoming.power) + growth
        _deal_extra_damage(ctx, power, element=11, category=0)


register(ListenBridge())


class PhaseShift(SpecialDefense):
    """应对攻击；位于 1 号位或 3 号位时额外减伤。

    槽位按使用时位置判定（传动重排每回合改变槽位，加成随之变化）；减伤总上限
    由引擎钳制到 [0, 1]。
    """

    def __init__(self):
        super().__init__(7070270, "相位移动", 0.50, drive=1, implemented=True)

    def modify_damage_reduction(self, ctx):
        extra = 0.40 if (ctx.skill_index + 1) in (1, 3) else 0.0
        return self.damage_reduction + extra


class Untouchable(SpecialDefense):
    def __init__(self):
        super().__init__(7120250, "不可接触", 0.50)

    def modify_damage_reduction(self, ctx):
        if ctx.target is None:
            return self.damage_reduction
        return self.damage_reduction + 0.10 * buffs.get_buff_value(
            ctx.target, buffs.BuffType.POISON
        )


register(
    SpecialDefense(
        7050260,
        "泡沫幻影",
        0.80,
        counter_effects=(LeaveField(mode=LEAVE_SELF),),
        implemented=True,
    )
)
register(
    SpecialDefense(
        7170150,
        "报复",
        0.70,
        counter_effects=(LoseEnergy(value=3),),
        implemented=True,
    )
)
register(
    SpecialDefense(
        7021130,
        "吓退",
        0.60,
        # 应对从句在回合末结算（B46）：敌方离场并按 B16 流程选人。若自己同时被
        # 打死，次序为"先脱离一侧（敌方）离场选入，再力竭一侧（自己）选人"（B18）。
        counter_effects=(LeaveField(mode=LEAVE_ENEMY),),
        implemented=True,
    )
)
register(PhaseShift())
register(Untouchable())


# ---------------- 7180330 虚假破产 ----------------
class FakeBankruptcy(OpsSkill):
    """应对攻击；能量不足时消耗生命代替能量。

    每代替 1 能量消耗 5% 最大生命（至少 1 点）。
    - 选择校验（dry_run）：付完生命 ≤0 时本技能不可选；
    - 结算：不做死亡保护，全额支付——支付致死即直接力竭，技能不再结算。
    走技能侧能量不足兜底管线（skills.query_energy_shortfall，特性兜底之后结算）。
    """

    skill_id = 7180330
    name = "虚假破产"
    category = 2
    counter_target = GUARD
    damage_reduction = 0.80
    implemented = True

    def modify_energy_shortfall(self, ctx, need, dry_run=False):
        cost = max(1, int(ctx.actor.max_hp * 0.05))
        if dry_run:
            return need if ctx.actor.hp > need * cost else 0
        ctx.actor.hp -= need * cost
        ctx.log(f"{ctx.actor.side} {ctx.actor.name} 失去 {need * cost} 生命")
        return need


register(FakeBankruptcy())


# ---------------- 7020800 血气 ----------------
class BloodVigor(OpsSkill):
    """应对攻击：本回合受到致命伤害时保留 1 点生命。

    必须在对方本次攻击落位前生效，是 B46 延迟结算的例外：防御应对攻击强制
    先手，本效果在 STAGE_DEFENSE 按 ctx.is_counter 即时挂上
    ``survive_lethal_turn``（记录生效回合，跨回合自动失效）。
    攻击伤害（走引擎致命判定，特性免死优先）与回合末持续伤害（buffs._apply_damage）
    都保留 1 血；冰冻是阈值力竭、非伤害，不受保护。
    """

    skill_id = 7020800
    name = "血气"
    category = 2
    counter_target = GUARD
    damage_reduction = 0.60
    implemented = True

    def on_defense(self, ctx):
        if ctx.is_counter:
            ctx.actor.survive_lethal_turn = ctx.state.turn


register(BloodVigor())


# ---------------- 7080230 硬化 ----------------
class Harden(OpsSkill):
    """应对攻击；上次使用攻击技时本技能能耗 -2。

    "上次使用"= BattlePet.last_used_skill_id（引擎在技能使用记录处更新；聚能/
    换人不改变，疾风连袭的重放不算新的使用）。类别 0/1 时减 2 费，走统一能耗
    管线；首次使用（无记录）不减。
    """

    skill_id = 7080230
    name = "硬化"
    category = 2
    counter_target = GUARD
    damage_reduction = 0.90
    implemented = True

    def modify_energy_cost(self, ctx):
        delta = super().modify_energy_cost(ctx)
        last_id = ctx.actor.last_used_skill_id
        if last_id is not None:
            raw = build_skill_map().get(last_id)
            if raw is not None and raw["category"] in (0, 1):
                delta -= 2
        return delta


register(Harden())


# ---------------- 7021120 嗜痛 ----------------
class PainTolerance(OpsSkill):
    """应对攻击：本回合每受到 1 次攻击伤害，获得双攻增益。

    "期间"= 本回合（减伤每回合开始清零，跨回合自动失效）：应对成功时在
    STAGE_DEFENSE 打上回合戳（skill_state），此后每次受到攻击伤害由引擎经
    STAGE_TAKE_DAMAGE 广播（skills.emit_take_damage），伤害即时叠加、同回合可累积。
    "1 次"按连击段数计（一次 3 连击算 3 次），层数 = 4 × ctx.hit_count。
    """

    skill_id = 7021120
    name = "嗜痛"
    category = 2
    counter_target = GUARD
    damage_reduction = 0.80
    implemented = True

    def on_defense(self, ctx):
        if ctx.is_counter and ctx.counter_category == COUNTER_ATTACK:
            ctx.set_state("active_turn", ctx.state.turn)

    def on_take_damage(self, ctx):
        if ctx.state_of("active_turn") != ctx.state.turn:
            return
        if ctx.extra.get("damage_taken", 0) <= 0 or ctx.hit_count <= 0:
            return
        layers = 4 * ctx.hit_count
        apply_effects(
            ctx,
            (
                ApplyBuff(buff_type=buffs.BuffType.ATK, value=layers),
                ApplyBuff(buff_type=buffs.BuffType.SPATK, value=layers),
            ),
        )


register(PainTolerance())
