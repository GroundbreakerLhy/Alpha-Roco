"""效果原语（Effect）与条件原语（Condition）+ 执行器。

技能效果全部由人工阅读描述后显式写成 Effect 组合，没有任何函数解析 ``desc``；
数值可以是常量，也可以是 ``(ctx) -> int`` 的运行时计算（"每有1层X"类）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, ClassVar, Optional

from .. import buffs, burst, marks, offerings, skill_utils, traits, weather as weather_mod
from ..damage import calc_damage, power_modifiers
from ..data_loader import build_skill_map, load_typechart
from ..evolution import can_cute
from ..models import BattleSkill
from .base import SkillContext, SkillHandler
from .registry import get_handler

# ==================== 作用对象 ====================

SELF = "self"
ENEMY = "enemy"
SELF_BACKLINE = "self_backline"  # 己方场下（不含使用者）
SELF_TEAM = "self_team"
ENEMY_TEAM = "enemy_team"
FIELD = "field"                  # 双方在场精灵

# ==================== 应对类别 ====================

COUNTER_ATTACK = "attack"
COUNTER_DEFENSE = "defense"
COUNTER_STATUS = "status"

# ==================== 效果激活的时机阶段 ====================

STAGE_ENTRY = "entry"
STAGE_SLOT_CHANGE = "slot_change"
STAGE_USE_START = "use_start"
STAGE_HIT = "hit"
STAGE_COUNTER = "counter"
STAGE_DEFENSE = "defense"
STAGE_STATUS = "status"
STAGE_AFTER_USE = "after_use"
STAGE_ROUND_END = "round_end"
# 受击方技能侧反应：actor 是受击方，ctx.skill 逐个取受击方自己的技能，由引擎
# 在攻击伤害落位后广播（skills.emit_take_damage），不是使用者视角的结算阶段。
STAGE_TAKE_DAMAGE = "take_damage"

# ==================== 冷却范围 ====================

COOLDOWN_USED = "used"        # 被应对的技能
COOLDOWN_DEFENSE = "defense"
COOLDOWN_ALL = "all"

# ==================== 离场方式 ====================

LEAVE_SELF = "self"
LEAVE_EMERGENCY = "emergency"    # 紧急脱离（回合结束时脱离）
LEAVE_ENEMY = "enemy"
LEAVE_BOTH = "both"

# 永久成长字段
GROWTH_POWER = "power"
GROWTH_ENERGY_COST = "energy_cost"
GROWTH_HIT_COUNT = "hit_count"


def _opponent_side(side: str) -> str:
    return "B" if side == "A" else "A"


def resolve_pets(ctx: SkillContext, target: str) -> list:
    """把作用对象解析为精灵列表。"""
    state = ctx.state
    if target == SELF:
        return [ctx.actor]
    if target == ENEMY:
        return [ctx.target] if ctx.target is not None else []
    if target == SELF_BACKLINE:
        return [p for p in state.teams[ctx.actor.side] if p is not ctx.actor]
    if target == SELF_TEAM:
        return list(state.teams[ctx.actor.side])
    if target == ENEMY_TEAM:
        return list(state.teams[_opponent_side(ctx.actor.side)])
    if target == FIELD:
        return [p for p in (ctx.actor, ctx.target) if p is not None]
    return []


def resolve_side(ctx: SkillContext, target: str) -> str:
    """把作用对象解析为阵营（印记等以阵营为单位的效果使用）。"""
    if target == ENEMY or target == ENEMY_TEAM:
        return _opponent_side(ctx.actor.side)
    return ctx.actor.side


def _value(value, ctx: SkillContext) -> int:
    """数值可以是常量，也可以是 (ctx) -> int 的运行时计算。"""
    if callable(value):
        return int(value(ctx))
    return int(value)


# ==================== 效果原语 ====================


@dataclass(kw_only=True)
class Effect:
    """效果原语基类。全部关键字构造，避免位置参数错配。

    deferrable：命中附加效果默认延迟到回合末结算（B32）；``False`` 的登记类
    原语（脱离/返场）命中时照常登记并立即生效（B16）。
    """

    deferrable: ClassVar[bool] = True

    def apply(self, ctx: SkillContext) -> None:
        raise NotImplementedError


@dataclass(kw_only=True)
class When(Effect):
    """条件门：condition 成立时依次执行 effects。"""

    condition: "Condition"
    effects: tuple = ()

    def apply(self, ctx: SkillContext) -> None:
        if self.condition.test(ctx):
            apply_effects(ctx, self.effects)


@dataclass(kw_only=True)
class ApplyBuff(Effect):
    """施加增益/减益。

    value 为负即减益（ENERGY_COST 正值 = 加费）。source_kind="trait" 是特性
    自身的展示效果——既不是增益也不是减益，不参与"获得增益/减益"判定，也不在
    驱散范围内。
    """

    buff_type: str
    value: int | Callable
    target: str = SELF
    duration: str = buffs.DurationKind.NORMAL
    source_kind: str = ""
    turns: int | None = None

    def apply(self, ctx: SkillContext) -> None:
        amount = _value(self.value, ctx)
        for pet in resolve_pets(ctx, self.target):
            extra_kw = {}
            if self.buff_type == buffs.BuffType.CUTE:
                # 萌化的 buff_gain 广播必须携带 pre_had（本次获得前是否已萌化），
                # 供"萌化状态下再获得萌化"类特性判定。
                extra_kw["pre_had"] = buffs.has_buff(pet, buffs.BuffType.CUTE)
            gained = buffs.grant_buff(pet, self.buff_type, amount, self.duration, ctx.state.turn,
                           ctx.actor.side, ctx.actor.name, self.source_kind, state=ctx.state,
                           turns=self.turns)
            if gained == 0:
                continue
            traits.on_buff_gain(ctx.state, pet, self.buff_type, gained,
                                source_side=ctx.actor.side, source_pet=ctx.actor.name,
                                source_kind=self.source_kind, **extra_kw)


@dataclass(kw_only=True)
class BoostCounteredSkillEnergyCost(Effect):
    """应对成功时使被应对技能能耗+N：按技能 id 的能耗修正 buff，作用于敌方。

    buff 类型为 ``energy_cost_boost_<被应对技能id>``（buffs.skill_energy_boost_type），
    引擎按技能 id 并入能耗（skill_utils.energy_modifier_before_traits），正值即减益。
    """

    value: int
    duration: str = buffs.DurationKind.NORMAL

    def apply(self, ctx: SkillContext) -> None:
        incoming = ctx.extra.get("incoming_skill")
        if incoming is None:
            return
        btype = buffs.skill_energy_boost_type(incoming.skill_id)
        for pet in resolve_pets(ctx, ENEMY):
            gained = buffs.grant_buff(pet, btype, self.value, self.duration,
                                      ctx.state.turn, ctx.actor.side, ctx.actor.name,
                                      state=ctx.state)
            if gained:
                traits.on_buff_gain(ctx.state, pet, btype, gained,
                                    source_side=ctx.actor.side, source_pet=ctx.actor.name)


@dataclass(kw_only=True)
class RandomStatBuffs(Effect):
    """随机 N 层属性增益。

    掷 layers 次，每次等概率落到 stats 中的一种，按属性汇总后逐种施加——每种
    属性一次施加、一个事件（不是 layers 次单独施加）。
    """

    layers: int
    stats: tuple = (buffs.BuffType.ATK, buffs.BuffType.SPATK, buffs.BuffType.DEF,
                    buffs.BuffType.SPDEF, buffs.BuffType.SPEED)
    target: str = SELF

    def apply(self, ctx: SkillContext) -> None:
        tally = {}
        for _ in range(self.layers):
            stat = random.choice(self.stats)
            tally[stat] = tally.get(stat, 0) + 1
        for pet in resolve_pets(ctx, self.target):
            for stat, n in tally.items():
                gained = buffs.grant_buff(pet, stat, n, buffs.DurationKind.NORMAL, ctx.state.turn,
                               ctx.actor.side, ctx.actor.name, state=ctx.state)
                if gained == 0:
                    continue
                traits.on_buff_gain(ctx.state, pet, stat, gained,
                                    source_side=ctx.actor.side, source_pet=ctx.actor.name)


@dataclass(kw_only=True)
class GainEnergy(Effect):
    """回复能量（含能量上限修正与 energy_gain 广播）。"""

    value: int | Callable
    target: str = SELF

    def apply(self, ctx: SkillContext) -> None:
        amount = _value(self.value, ctx)
        for pet in resolve_pets(ctx, self.target):
            traits.grant_energy(ctx.state, pet, amount)


@dataclass(kw_only=True)
class LoseEnergy(Effect):
    """失去能量（不设上限、不为负）。"""

    value: int | Callable
    target: str = ENEMY

    def apply(self, ctx: SkillContext) -> None:
        amount = _value(self.value, ctx)
        for pet in resolve_pets(ctx, self.target):
            pet.energy = max(0, pet.energy - amount)


@dataclass(kw_only=True)
class StealEnergy(Effect):
    """偷取能量：目标失去，使用者获得。"""

    value: int | Callable = 1
    target: str = ENEMY

    def apply(self, ctx: SkillContext) -> None:
        amount = _value(self.value, ctx)
        for pet in resolve_pets(ctx, self.target):
            taken = min(pet.energy, amount)
            pet.energy -= taken
            traits.grant_energy(ctx.state, ctx.actor, taken)


@dataclass(kw_only=True)
class Heal(Effect):
    """回复生命。percent 为最大生命比例（0.2 = 20%），flat 为固定值，二者相加。"""

    percent: float = 0.0
    flat: int = 0
    target: str = SELF

    def apply(self, ctx: SkillContext) -> None:
        for pet in resolve_pets(ctx, self.target):
            amount = int(pet.max_hp * self.percent) + self.flat
            amount = traits.query_heal(ctx.state, pet, amount, source="skill")
            if amount <= 0:
                continue
            pet.hp = min(pet.max_hp, pet.hp + amount)
            traits.emit(ctx.state, "heal", scope="all", side=pet.side, subject=pet,
                        heal=amount, base_heal=amount, source="skill")


@dataclass(kw_only=True)
class LoseHp(Effect):
    """失去生命（代价类），下限 0。"""

    percent: float = 0.0
    flat: int = 0
    target: str = SELF

    def apply(self, ctx: SkillContext) -> None:
        for pet in resolve_pets(ctx, self.target):
            amount = int(pet.max_hp * self.percent) + self.flat
            lost = min(amount, pet.hp)
            pet.hp = max(0, pet.hp - amount)
            if lost > 0:
                ctx.log(f"{pet.side} {pet.name} 失去 {lost} 生命")


@dataclass(kw_only=True)
class AddMark(Effect):
    """添加印记（以阵营为单位）。"""

    mark_id: int
    stacks: int | Callable = 1
    target: str = ENEMY

    def apply(self, ctx: SkillContext) -> None:
        amount = _value(self.stacks, ctx)
        side = resolve_side(ctx, self.target)
        marks.add_mark(ctx.state, side, self.mark_id, amount)


@dataclass(kw_only=True)
class DispelMarks(Effect):
    """驱散印记；count=None 表示全部驱散；steal=True 改为偷取（转给使用者一方）。

    target=FIELD 驱散双方阵营的印记（steal 不与 FIELD 同用）。record_as 非空时
    把驱散的总层数写入 ctx.extra[record_as]，供同组后续效果经 per_recorded 读取。
    """

    count: Optional[int] = None
    steal: bool = False
    target: str = ENEMY
    record_as: Optional[str] = None

    def apply(self, ctx: SkillContext) -> None:
        if self.target == FIELD:
            sides = (ctx.actor.side, _opponent_side(ctx.actor.side))
        else:
            sides = (resolve_side(ctx, self.target),)
        total = 0
        for side in sides:
            for slot in (marks.POSITIVE, marks.NEGATIVE):
                current = ctx.state.marks[side][slot]
                if current is None:
                    continue
                taken = current["stacks"] if self.count is None else min(current["stacks"], self.count)
                total += taken
                remaining = current["stacks"] - taken
                mark_id = current["id"]
                ctx.state.marks[side][slot] = None
                if remaining > 0:
                    marks.add_mark(ctx.state, side, mark_id, remaining)
                if self.steal:
                    marks.add_mark(ctx.state, ctx.actor.side, mark_id, taken)
        if self.record_as is not None:
            ctx.extra[self.record_as] = total


@dataclass(kw_only=True)
class DispelBuffs(Effect):
    """驱散增益/减益；kind ∈ "buff"/"debuff"/"all"，count=None 表示全部驱散。

    两条独立判据：①是不是增益/减益只看效果类型与数值，与时长无关——永久时长的
    同样会被驱散；②``source_kind="trait"`` 是特性自身的展示效果，既不是增益也
    不是减益，不参与驱散。
    """

    kind: str = "buff"
    count: Optional[int] = None
    target: str = ENEMY

    def apply(self, ctx: SkillContext) -> None:
        for pet in resolve_pets(ctx, self.target):
            match = []
            for buff in pet.buffs:
                if buff.source_kind == "trait":
                    continue
                if self.kind == "buff" and not buffs.is_buff(buff.buff_type, buff.value):
                    continue
                if self.kind == "debuff" and not buffs.is_debuff(
                    buff.buff_type, buff.value
                ):
                    continue
                match.append(buff)
            if self.count is not None and len(match) > self.count:
                # 指定种数且可驱散项更多时随机选取（"驱散敌方1种增益"）。
                match = random.sample(match, self.count)
            if not match:
                continue
            for buff in match:
                pet.buffs = [b for b in pet.buffs if b is not buff]


@dataclass(kw_only=True)
class ConvertBuffs(Effect):
    """把目标的一种效果按层数转化为另一种（"将敌方所有增益转化为相同层数的中毒"）。

    判据与驱散一致（S18）：增益/减益只看效果类型与数值，与时长无关（永久时长同样
    转化）；``source_kind="trait"`` 是特性自身的展示效果，不参与转化。
    """

    from_kind: str = "buff"
    to_buff_type: str = buffs.BuffType.POISON
    target: str = ENEMY

    def apply(self, ctx: SkillContext) -> None:
        for pet in resolve_pets(ctx, self.target):
            total = 0
            for buff in list(pet.buffs):
                if buff.source_kind == "trait":
                    continue
                if not buffs.is_buff(buff.buff_type, buff.value):
                    continue
                total += buff.value
                pet.buffs = [b for b in pet.buffs if b is not buff]
            if total > 0:
                buffs.grant_buff(pet, self.to_buff_type, total, state=ctx.state)


@dataclass(kw_only=True)
class ConvertBuffToMark(Effect):
    """把目标精灵某种 buff 的层数转化为该阵营的印记（"将中毒转化为中毒印记"）。

    读取层数并移除该 buff，等量加到目标阵营的印记上（0 层不生效）。
    目标精灵优先取命中快照 ``ctx.extra["hit_target"]``（"若击败敌方"类——
    回合末结算时被击败者已不在场，活跃位是空的），缺省回退 target 解析。
    """

    buff_type: str
    mark_id: int
    target: str = ENEMY

    def apply(self, ctx: SkillContext) -> None:
        anchor = ctx.extra.get("hit_target")
        pets = [anchor] if anchor is not None else resolve_pets(ctx, self.target)
        side = resolve_side(ctx, self.target)
        for pet in pets:
            layers = buffs.get_buff_value(pet, self.buff_type)
            if layers <= 0:
                continue
            buffs.remove_buff(pet, self.buff_type)
            marks.add_mark(ctx.state, side, self.mark_id, layers)


@dataclass(kw_only=True)
class DoubleBuffs(Effect):
    """目标某类效果的层数翻倍（"使敌方精灵减益的层数翻倍"）。

    翻倍 = 按现有条目再叠一份同值同来源的层数，层数钳制由写入层负责。遍历使用
    快照，新叠出的条目不会被重复处理；只有实际获得层数时才广播 buff_gain。
    """

    kind: str = "debuff"
    target: str = ENEMY

    def apply(self, ctx: SkillContext) -> None:
        for pet in resolve_pets(ctx, self.target):
            for buff in list(pet.buffs):
                is_match = (buffs.is_debuff(buff.buff_type, buff.value) if self.kind == "debuff"
                            else buffs.is_buff(buff.buff_type, buff.value))
                if not is_match:
                    continue
                extra_kw = {}
                if buff.buff_type == buffs.BuffType.CUTE:
                    # 与 ApplyBuff 同契约：萌化广播需携带 pre_had。
                    extra_kw["pre_had"] = buffs.has_buff(pet, buffs.BuffType.CUTE)
                gained = buffs.grant_buff(pet, buff.buff_type, buff.value, buff.duration,
                               ctx.state.turn, buff.source_side, buff.source_pet,
                               buff.source_kind, state=ctx.state)
                if gained == 0:
                    continue
                traits.on_buff_gain(ctx.state, pet, buff.buff_type, gained,
                                    source_side=buff.source_side,
                                    source_pet=buff.source_pet,
                                    source_kind=buff.source_kind, **extra_kw)


@dataclass(kw_only=True)
class SwapBuffs(Effect):
    """与目标交换全部增益和减益（"与敌方交换增益和减益"）。"""

    target: str = ENEMY

    def apply(self, ctx: SkillContext) -> None:
        for pet in resolve_pets(ctx, self.target):
            ctx.actor.buffs, pet.buffs = pet.buffs, ctx.actor.buffs


def deal_extra_damage(ctx: SkillContext, power: int, *,
                      element: Optional[int] = None,
                      category: Optional[int] = None,
                      hits: int = 1, target: str = ENEMY) -> None:
    """附加伤害（不占用主伤害段）：直接在调用处经 calc_damage 与统一威力管线结算。

    威力管线与主攻击段同一条：属性增减益与威力 buff 由 calc_damage 处理；印记、
    迸发（本次行动已取走的 active_bursts）与特性威力修正（query_power）在此补齐。
    element/category 缺省沿用本次技能的属性与类别。
    """
    elem = element if element is not None else ctx.skill.element
    cat = category if category is not None else ctx.skill.category
    extra_skill = BattleSkill(skill_id=-1, name=ctx.skill.name,
                              element=elem, category=cat, power=power,
                              energy_cost=0, desc="")
    typechart = load_typechart()
    for pet in resolve_pets(ctx, target):
        percent, flat = _extra_power_modifiers(ctx, pet, extra_skill)
        for _ in range(max(1, hits)):
            if pet.hp <= 0:
                break
            result = calc_damage(ctx.actor, pet, extra_skill, typechart,
                                 pet.defense_reduction, state=ctx.state,
                                 is_first=ctx.is_first,
                                 extra_power_percent=percent,
                                 extra_power_flat=flat)
            damage = result["damage"]
            if damage <= 0:
                continue
            pet.hp = max(0, pet.hp - damage)
            ctx.log(f"  额外造成 {damage} 伤害")
            traits.emit(ctx.state, "take_damage", scope="all", side=pet.side, subject=pet,
                        target=ctx.actor, skill=extra_skill, damage=damage,
                        hit_count=1, is_first=ctx.is_first)


def _extra_power_modifiers(ctx: SkillContext, target_pet, extra_skill) -> tuple:
    """附加伤害的威力修正（百分比, 固定值），与 battle 主攻击段同口径。

    这里只补 calc_damage 之外的部分：印记、迸发（ctx.extra 里的 active_bursts）、
    特性威力修正（query_power，按附加伤害自身的属性/类别匹配）。
    """
    return power_modifiers(
        ctx.state, ctx.actor, target_pet, extra_skill,
        is_first=ctx.is_first, active_bursts=ctx.extra.get("active_bursts", ()),
    )


@dataclass(kw_only=True)
class SetWeather(Effect):
    """改变天气并设置持续回合。"""

    weather_id: int
    turns: int = 8

    def apply(self, ctx: SkillContext) -> None:
        weather_mod.set_weather(ctx.state, self.weather_id, self.turns)


@dataclass(kw_only=True)
class SetCooldown(Effect):
    """安排技能冷却。scope 见 COOLDOWN_* 常量（作用于 target 一方）。"""

    scope: str = COOLDOWN_USED
    turns: int = 1
    target: str = ENEMY

    def apply(self, ctx: SkillContext) -> None:
        side = resolve_side(ctx, self.target)
        for pet in ctx.state.teams[side]:
            if self.scope == COOLDOWN_ALL:
                for skill in pet.skills:
                    for _ in range(self.turns):
                        _schedule(pet, skill.skill_id)
            elif self.scope == COOLDOWN_DEFENSE:
                for skill in pet.skills:
                    if skill.category == 2:
                        for _ in range(self.turns):
                            _schedule(pet, skill.skill_id)
            else:
                used = ctx.extra.get("countered_skill")
                if used is not None:
                    for _ in range(self.turns):
                        _schedule(pet, used.skill_id)
        ctx.extra["cooldown_pending"] = True


def _schedule(pet, skill_id: int) -> None:
    skill_utils.schedule_skill_cooldown(pet, skill_id)


@dataclass(kw_only=True)
class Overload(Effect):
    """本技能（或指定技能）本次使用次数 +1（"下回合所选技能使用次数+1"）。

    scope="next" 写入 ``overload_next["__next__"]``：下回合开始轮转到
    ``overload_current``，引擎按该回合所选技能追加执行次数（聚能不生效）。
    """

    count: int = 1
    scope: str = "self"   # "self" = 本次技能；"next" = 下回合所选技能

    def apply(self, ctx: SkillContext) -> None:
        if ctx.skill is None:
            return
        if self.scope == "self":
            buffs.add_overload(ctx.actor, ctx.skill.skill_id, self.count)
        else:
            ctx.actor.overload_next["__next__"] = ctx.actor.overload_next.get("__next__", 0) + self.count


@dataclass(kw_only=True)
class GrantBurst(Effect):
    """给予使用者迸发（入场后首次行动消耗）。"""

    burst_type: str
    value: int

    def apply(self, ctx: SkillContext) -> None:
        burst.add_burst(ctx.actor, self.burst_type, self.value)


@dataclass(kw_only=True)
class SkillGrowth(Effect):
    """本技能的永久成长（"每次使用后，本技能威力永久+45"）。

    写入 BattleSkill.skill_state，由 handler 的 modify_power /
    modify_energy_cost / modify_hit_count 读出（见 growth_value）。delta 可以是
    可调用对象（运行时取值），如"威力永久翻倍" = 增量取当前总威力。
    """

    field: str = GROWTH_POWER
    delta: int | Callable = 0
    count: int | Callable = 1

    def apply(self, ctx: SkillContext) -> None:
        times = _value(self.count, ctx)
        amount = _value(self.delta, ctx) * times
        ctx.add_state(self.field, amount)


@dataclass(kw_only=True)
class EnergyFromHp(Effect):
    """能量不足时以生命代替能量。

    记录到 skill_state，供引擎的能量兜底查询读取。
    """

    percent: float = 0.05

    def apply(self, ctx: SkillContext) -> None:
        ctx.set_state("energy_from_hp", self.percent)


@dataclass(kw_only=True)
class LeaveField(Effect):
    """脱离/返场（mode 见 LEAVE_* 常量）。

    引擎在技能结算末尾统一处理：本原语只登记请求，避免在效果中途换人。
    登记类效果（deferrable=False）：命中时照常登记并立即生效（B16），不随命中
    附加效果延迟到回合末（B32 例外）。

    entry_energy>0 时由续接流程把能量发给替换入场的精灵；
    entry_inherit_buffs=True 时在 NORMAL 清除之前快照全部增益（减益不继承），
    由续接流程按原类型/层数/时长发给下一只入场的己方精灵。
    """

    deferrable = False
    mode: str = LEAVE_SELF
    entry_energy: int = 0
    entry_inherit_buffs: bool = False

    def apply(self, ctx: SkillContext) -> None:
        ctx.extra["leave"] = self.mode
        if self.entry_energy:
            ctx.extra["leave_entry_energy"] = self.entry_energy
        if self.entry_inherit_buffs:
            ctx.extra["leave_entry_inherit_buffs"] = True


@dataclass(kw_only=True)
class Reenter(Effect):
    """返场：离场并立即入场（同一只精灵）。"""

    deferrable = False
    target: str = SELF

    def apply(self, ctx: SkillContext) -> None:
        ctx.extra["reenter"] = _opponent_side(ctx.actor.side) if self.target == ENEMY else ctx.actor.side


@dataclass(kw_only=True)
class Repeat(Effect):
    """把子效果重复执行 count 次（"3连击，每次连击…"类）。

    每次都独立执行、独立触发事件——"敌方获得冻结时追加"类特性会按次数分别响应，
    而不是只响应一次。
    """

    count: int | Callable = 1
    effects: tuple = ()

    def apply(self, ctx: SkillContext) -> None:
        for _ in range(max(0, _value(self.count, ctx))):
            apply_effects(ctx, self.effects)


# ==================== 选择 ====================


@dataclass(kw_only=True)
class Choose(Effect):
    """选择（技能二选一，明/暗）。

    输入由引擎提供（``Action.choice_branch`` → ``SkillContext.choice_branch``）；
    本原语只登记，明/暗的具体效果由 handler 读 ``ctx.choice_branch`` 后自行填写。
    """

    options: tuple = ()
    label: str = ""

    def apply(self, ctx: SkillContext) -> None:
        ctx.extra.setdefault("unimplemented", []).append(f"choose:{self.label}")


# ==================== 奉献 ====================


@dataclass(kw_only=True)
class Offering(Effect):
    """己方队伍获得奉献（队伍级资源，B40）。

    grant 为空时每次从五种效果中随机抽取；给出种类时逐种获得 times 次
    （"连击+1或威力+20"由 handler 按选择结果传入一种）。
    """

    times: int = 1
    grant: tuple = ()

    def apply(self, ctx: SkillContext) -> None:
        times = _value(self.times, ctx)
        if not self.grant:
            offerings.grant_random(ctx.state, ctx.actor.side, times)
            return
        for kind in self.grant:
            offerings.grant(ctx.state, ctx.actor.side, kind, times)


# ==================== 条件原语 ====================


@dataclass(kw_only=True)
class Condition:
    def test(self, ctx: SkillContext) -> bool:
        raise NotImplementedError


@dataclass(kw_only=True)
class Always(Condition):
    def test(self, ctx: SkillContext) -> bool:
        return True


@dataclass(kw_only=True)
class IsFirst(Condition):
    """若先于敌方行动。"""

    def test(self, ctx: SkillContext) -> bool:
        return ctx.is_first


@dataclass(kw_only=True)
class IsSecond(Condition):
    """若后于敌方行动。"""

    def test(self, ctx: SkillContext) -> bool:
        return not ctx.is_first


@dataclass(kw_only=True)
class IsCounter(Condition):
    """应对成功；category 为空表示任意被应对类别，否则需匹配 counter_category。"""

    category: str = ""

    def test(self, ctx: SkillContext) -> bool:
        if not ctx.is_counter:
            return False
        return not self.category or self.category == ctx.counter_category


@dataclass(kw_only=True)
class CanCute(Condition):
    """自己能够获得萌化（可退化，或已有萌化且特性允许再次获得）。

    "自己获得萌化：X"族的前置条件——能萌化才给 X，不能萌化时门内效果一律不
    执行（配合 When，S22）；萌化本身在技能结算之后施加。
    """

    def test(self, ctx: SkillContext) -> bool:
        return buffs.can_gain_buff(ctx.actor, buffs.BuffType.CUTE, 1)


@dataclass(kw_only=True)
class HpAtMost(Condition):
    """自己生命比例 ≤ percent。"""

    percent: float

    def test(self, ctx: SkillContext) -> bool:
        return ctx.actor.hp <= ctx.actor.max_hp * self.percent


@dataclass(kw_only=True)
class HpAtLeast(Condition):
    """自己生命比例 ≥ percent。"""

    percent: float

    def test(self, ctx: SkillContext) -> bool:
        return ctx.actor.hp >= ctx.actor.max_hp * self.percent


@dataclass(kw_only=True)
class EnemySwitchedThisTurn(Condition):
    """敌方本回合更换过精灵。"""

    def test(self, ctx: SkillContext) -> bool:
        target = ctx.target if ctx.target is not None else ctx.opponent()
        return target is not None and target.entry_turn == ctx.state.turn


@dataclass(kw_only=True)
class UsedAttackSkillLastTurn(Condition):
    """自己（ctx.actor）上回合使用过攻击技能。

    "上回合"由引擎在回合开始时平移记录（BattlePet.attack_used_last_turn）；
    蓄力回合不算使用攻击技能，释放回合照常算。
    """

    def test(self, ctx: SkillContext) -> bool:
        return ctx.actor.attack_used_last_turn


@dataclass(kw_only=True)
class HasBuff(Condition):
    """目标持有指定效果且层数 ≥ at_least。"""

    buff_type: str
    at_least: int = 1
    target: str = SELF

    def test(self, ctx: SkillContext) -> bool:
        return any(buffs.get_buff_value(pet, self.buff_type) >= self.at_least
                   for pet in resolve_pets(ctx, self.target))


@dataclass(kw_only=True)
class HasDebuff(Condition):
    """目标身上存在任意减益。"""

    target: str = SELF

    def test(self, ctx: SkillContext) -> bool:
        for pet in resolve_pets(ctx, self.target):
            for buff in pet.buffs:
                if buffs.is_debuff(buff.buff_type, buff.value):
                    return True
        return False


@dataclass(kw_only=True)
class WeatherIs(Condition):
    """当前天气属于给定集合。"""

    ids: tuple = ()

    def test(self, ctx: SkillContext) -> bool:
        return ctx.weather in self.ids


@dataclass(kw_only=True)
class SlotIn(Condition):
    """本次技能槽位属于给定集合（1-based：SlotIn(slots=(1, 3))）。"""

    slots: tuple = ()

    def test(self, ctx: SkillContext) -> bool:
        return (ctx.skill_index + 1) in self.slots


@dataclass(kw_only=True)
class EnergyAtMost(Condition):
    target: str = ENEMY
    value: int = 0

    def test(self, ctx: SkillContext) -> bool:
        return any(pet.energy <= self.value for pet in resolve_pets(ctx, self.target))


@dataclass(kw_only=True)
class MarkAtLeast(Condition):
    """指定印记层数 ≥ value（target 指定印记归属方）。"""

    mark_id: int
    value: int = 1
    target: str = ENEMY

    def test(self, ctx: SkillContext) -> bool:
        return marks.get_stacks(ctx.state, resolve_side(ctx, self.target), self.mark_id) >= self.value


@dataclass(kw_only=True)
class DamageDealt(Condition):
    """本次技能实际造成了伤害。"""

    def test(self, ctx: SkillContext) -> bool:
        return ctx.damage_dealt > 0


@dataclass(kw_only=True)
class KilledTarget(Condition):
    """本次使用击败了敌方在场精灵（"若使用本技能击败敌方，…"）。

    读命中登记时的快照（伤害结算后对方 hp≤0），B32 延迟到回合末结算也不失真。
    """

    def test(self, ctx: SkillContext) -> bool:
        return bool(ctx.extra.get("killed"))


@dataclass(kw_only=True)
class All(Condition):
    conditions: tuple = ()

    def test(self, ctx: SkillContext) -> bool:
        return all(c.test(ctx) for c in self.conditions)


@dataclass(kw_only=True)
class Any(Condition):
    conditions: tuple = ()

    def test(self, ctx: SkillContext) -> bool:
        return any(c.test(ctx) for c in self.conditions)


@dataclass(kw_only=True)
class Not(Condition):
    condition: Optional[Condition] = None

    def test(self, ctx: SkillContext) -> bool:
        return not self.condition.test(ctx)


# ==================== 数值辅助（"每有1层…"类） ====================


def per_stack(buff_type: str, per: int, target: str = ENEMY) -> Callable:
    """每有 1 层 buff_type，取值 per（"敌方每有1层中毒效果，本次技能威力+10"）。"""

    def value(ctx: SkillContext) -> int:
        return sum(buffs.get_buff_value(pet, buff_type) for pet in resolve_pets(ctx, target)) * per

    return value


def per_mark(mark_id: int, per: int, target: str = ENEMY) -> Callable:
    """每有 1 层指定印记，取值 per。"""

    def value(ctx: SkillContext) -> int:
        return marks.get_stacks(ctx.state, resolve_side(ctx, target), mark_id) * per

    return value


def per_any_mark(per: int, target: str = ENEMY) -> Callable:
    """每有 1 层印记（该阵营正/负印记层数合计，不限种类），取值 per。

    用于"敌方每有1层印记，能耗-1 / 威力+20"这类不限印记种类的描述。
    """

    def value(ctx: SkillContext) -> int:
        side = resolve_side(ctx, target)
        total = 0
        for kind in (marks.POSITIVE, marks.NEGATIVE):
            m = marks.get_mark(ctx.state, side, kind)
            if m is not None:
                total += m["stacks"]
        return total * per

    return value


def per_missing_hp(per: int, step: float = 0.1, target: str = SELF) -> Callable:
    """目标每失去 step 比例的生命，取值 per（"自己每失去10%生命，威力+10" /
    "敌方每失去5%生命，威力-5"）；满血为 0。

    步进用整数算术（lost × round(1/step) // max_hp）：float 下 1-0.9 得到
    0.0999…，直接 int(missing/step) 会把整档截少一档。
    """

    inv = round(1 / step)

    def value(ctx: SkillContext) -> int:
        total = 0
        for pet in resolve_pets(ctx, target):
            lost = pet.max_hp - pet.hp
            total += (lost * inv // pet.max_hp) * per
        return total

    return value


def per_enemy_fainted(per: int) -> Callable:
    """敌方每有 1 只力竭精灵，取值 per。"""

    def value(ctx: SkillContext) -> int:
        side = _opponent_side(ctx.actor.side)
        return sum(1 for pet in ctx.state.teams[side] if pet.hp <= 0) * per

    return value


def per_recorded(key: str, per: int) -> Callable:
    """读取同组效果先前记录到 ctx.extra 的数值，每单位取值 per。

    与效果的 record_as 配套（"每驱散1层印记，敌方获得5层灼烧" =
    DispelMarks(record_as="x") + ApplyBuff(value=per_recorded("x", 5))）；
    同组效果共享同一个 ctx，先后执行。
    """

    def value(ctx: SkillContext) -> int:
        return int(ctx.extra.get(key, 0) or 0) * per

    return value


def hit_count_of(ctx: SkillContext) -> int:
    """连击数 = 1（基础一次行动）+ buff + 特性 + 技能自身修正（含强制值）。

    攻击分支用它算伤害段数，状态技能用它做 ``Repeat(count=hit_count_of)``。
    技能固有连击数经 ``SkillHandler.modify_hit_count`` 给增量（"3连击" = 基础1 + 2），
    与 buff、特性走同一管线；汇总后的乘算交 ``SkillHandler.adjust_hit_count``。
    """
    base = 1
    flat, percent = buffs.get_hit_count_bonus(ctx.actor)
    t_flat, t_percent, t_forced = traits.query_hit_count(ctx.state, ctx.actor, ctx.target)
    flat += t_flat
    percent += t_percent
    forced = t_forced
    handler = get_handler(ctx.skill.skill_id)
    if handler is not None:
        s_flat, s_percent, s_forced = handler.modify_hit_count(ctx)
        flat += s_flat
        percent += s_percent
        if s_forced is not None:
            forced = s_forced if forced is None else max(forced, s_forced)
    if forced is not None:
        count = max(1, forced)
    else:
        count = max(1, base + flat + int(base * percent / 100))
    if handler is not None:
        count = handler.adjust_hit_count(ctx, count)
    return max(1, count)


def skill_by_id(skill_id: int) -> BattleSkill:
    """按 id 取技能模板（用于"随机变成某技能"类效果）。"""
    raw = build_skill_map()[skill_id]
    return BattleSkill(skill_id=raw["id"], name=raw["name"], element=raw["element"],
                       category=raw["category"], power=raw.get("power"),
                       energy_cost=raw.get("energyCost", 0),
                       desc=raw.get("desc", ""))


# ==================== 执行器 ====================


def apply_effects(ctx: SkillContext, effects) -> None:
    """按顺序执行一组效果原语。

    命中附加效果分两次走（B32）：ctx.extra["_defer_mode"] == "collect" 时只执行
    deferrable=False 的登记类效果（脱离/返场，命中时立即生效）；== "settle" 时
    只执行 deferrable=True 的效果（回合末统一结算）；缺省全部执行。
    """
    mode = ctx.extra.get("_defer_mode")
    for effect in effects:
        if mode == "collect" and effect.deferrable:
            continue
        if mode == "settle" and not effect.deferrable:
            continue
        effect.apply(ctx)


# ==================== 声明式技能基类 ====================


class OpsSkill(SkillHandler):
    """把效果原语挂到各阶段，无需手写钩子的技能基类。

    子类只需给出静态属性 + 各阶段效果元组；复杂技能直接继承 ``SkillHandler``
    覆写钩子。
    """

    counter_target: str = ""
    drive: int = 0
    swift: bool = False
    windup: bool = False
    usable: bool = True

    # 阶段效果（与 SkillHandler 的钩子一一对应）
    entry_effects: tuple = ()
    slot_change_effects: tuple = ()
    use_start_effects: tuple = ()
    hit_effects: tuple = ()
    counter_effects: tuple = ()
    defense_effects: tuple = ()
    status_effects: tuple = ()
    after_use_effects: tuple = ()
    round_end_effects: tuple = ()

    # 威力/连击/能耗的声明式增量（固定值；带条件的覆写钩子）
    power_percent: float = 0.0
    power_flat: float = 0.0
    hit_count_flat: int = 0
    hit_count_percent: int = 0
    energy_cost_delta: int = 0
    priority_delta: int = 0
    lifesteal: float = 0.0
    damage_reduction: float = 0.0

    def modify_power(self, ctx: SkillContext) -> tuple:
        return (self.power_percent, self.power_flat + growth_value(ctx, GROWTH_POWER))

    def modify_hit_count(self, ctx: SkillContext) -> tuple:
        return (self.hit_count_flat + growth_value(ctx, GROWTH_HIT_COUNT),
                self.hit_count_percent, None)

    def modify_energy_cost(self, ctx: SkillContext) -> int:
        return self.energy_cost_delta + growth_value(ctx, GROWTH_ENERGY_COST)

    def modify_priority(self, ctx: SkillContext) -> int:
        return self.priority_delta

    def modify_lifesteal(self, ctx: SkillContext) -> float:
        return self.lifesteal

    def modify_damage_reduction(self, ctx: SkillContext) -> float:
        return self.damage_reduction

    def on_entry(self, ctx: SkillContext) -> None:
        apply_effects(ctx, self.entry_effects)

    def on_slot_change(self, ctx: SkillContext) -> None:
        apply_effects(ctx, self.slot_change_effects)

    def on_use_start(self, ctx: SkillContext) -> None:
        apply_effects(ctx, self.use_start_effects)

    def on_hit(self, ctx: SkillContext) -> None:
        apply_effects(ctx, self.hit_effects)

    def on_counter(self, ctx: SkillContext) -> None:
        apply_effects(ctx, self.counter_effects)

    def on_defense(self, ctx: SkillContext) -> None:
        apply_effects(ctx, self.defense_effects)

    def on_status(self, ctx: SkillContext) -> None:
        apply_effects(ctx, self.status_effects)

    def on_after_use(self, ctx: SkillContext) -> None:
        apply_effects(ctx, self.after_use_effects)

    def on_round_end(self, ctx: SkillContext) -> None:
        apply_effects(ctx, self.round_end_effects)


def growth_value(ctx: SkillContext, field: str) -> int:
    """读取本技能累计的永久成长值（SkillGrowth 写入 skill_state）。"""
    return int(ctx.state_of(field, 0) or 0)
