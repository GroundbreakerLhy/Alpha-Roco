"""特性系统接口：上下文 + handler 基类。

设计约定
--------
- 事件钩子（on_*）：引擎在关键时机调用 ``traits.emit(state, "事件名", ...)``，
  广播给选定的精灵集合，handler 按需覆写。事件可以产生副作用（改 hp/能量/buff/印记）。
- 修正钩子（modify_*）：数值计算处调用 ``traits.query_*`` 汇总修正值。
  所有修正一律是"增量"语义：返回 0 表示无修正；伤害/属性乘数 = 1.0 + 增量之和。
- 未实现的特性是 no-op handler（implemented=False），不打印任何日志。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..models import Action, BattlePet, BattleSkill, BattleState


@dataclass
class TraitContext:
    """传给特性 handler 的上下文。actor 永远是特性所有者。

    常用字段（随事件/查询不同而部分生效）：
      state       战斗状态（可读写）
      actor       特性所有者精灵
      active      actor 是否在场
      side        事件发起方（"A"/"B"）
      subject     事件主体（入场/离场/力竭/受击的精灵等）
      target      作用目标（通常是被攻击方）
      action      当回合 Action（skill/charge/switch）
      skill       本次技能
      is_first    是否先手行动
      is_counter  是否由应对触发
      weather     当前天气 id（dict 结构中的 id 字段）
      energy_cost 本次技能实际能耗
      energy_gain 本次获得能量
      heal        本次回复量
      damage      本次伤害量
      hit_count   本次连击次数
      damage_dealt 本次实际造成伤害
      extra       事件专属扩展字段
    """

    state: BattleState
    actor: BattlePet
    event: str = ""
    active: bool = False
    side: str = ""
    subject: Optional[BattlePet] = None
    target: Optional[BattlePet] = None
    action: Optional[Action] = None
    skill: Optional[BattleSkill] = None
    skill_index: Optional[int] = None
    action_a: Optional[Action] = None
    action_b: Optional[Action] = None
    is_first: bool = False
    is_counter: bool = False
    weather: Optional[int] = None
    energy_cost: int = 0
    energy_gain: int = 0
    heal: int = 0
    damage: int = 0
    hit_count: int = 1
    damage_dealt: int = 0
    extra: dict = field(default_factory=dict)

    # ---- 便捷方法 ----

    def log(self, msg: str) -> None:
        self.state.log.append(msg)

    def is_active(self) -> bool:
        idx = self.state.active[self.actor.side]
        return idx >= 0 and self.state.teams[self.actor.side][idx] is self.actor

    def opponent(self) -> Optional[BattlePet]:
        opp_side = "B" if self.actor.side == "A" else "A"
        idx = self.state.active[opp_side]
        if idx < 0:
            return None
        return self.state.teams[opp_side][idx]

    # ---- 特性运行时状态（actor.trait_state） ----

    def state_of(self, key, default=None):
        return self.actor.trait_state.get(key, default)

    def set_state(self, key, value) -> None:
        self.actor.trait_state[key] = value

    def add_state(self, key, amount) -> None:
        self.actor.trait_state[key] = self.actor.trait_state.get(key, 0) + amount


class TraitHandler:
    """所有特性 handler 的基类。

    子类覆写需要的钩子，其余保持默认空实现。注册方式：
      register(MyTraitHandler(trait_id=200122, name="加个雪球", desc="...", implemented=True))
    """

    trait_id: int = 0
    name: str = ""
    desc: str = ""
    implemented: bool = False

    # ================= 事件钩子 =================

    def on_turn_start(self, ctx: TraitContext) -> None:
        """回合开始（双方行动已选定，ctx.action_a/action_b 可用）。"""

    def on_entry(self, ctx: TraitContext) -> None:
        """入场：主动换人 / 力竭替换 / 脱离换人 / 战斗开始首发（subject=入场精灵）。"""

    def on_leave(self, ctx: TraitContext) -> None:
        """离场：主动换人 / 脱离（不含力竭）；ctx.subject=离场精灵，ctx.extra["incoming"]=入场精灵。"""

    def on_charge(self, ctx: TraitContext) -> None:
        """聚能（ctx.energy_gain=回复量）。"""

    def on_skill_start(self, ctx: TraitContext) -> None:
        """技能开始：扣除能耗后、效果结算前。"""

    def on_attack(self, ctx: TraitContext) -> None:
        """造成伤害后（ctx.damage=实际伤害，ctx.target=受击方，不含连击拆分）。"""

    def on_take_damage(self, ctx: TraitContext) -> None:
        """受到伤害后（ctx.damage=实际伤害，ctx.subject=受击方）。"""

    def on_lethal(self, ctx: TraitContext) -> bool:
        """受到致命伤害判定（在伤害应用前）：返回 True 表示特性免除了本次伤害，
        由 handler 自行调整精灵血量（如保留1血/免疫）。"""
        return False

    def on_kill(self, ctx: TraitContext) -> None:
        """击败敌方精灵（ctx.subject=被击败精灵，ctx.target=击杀者）。"""

    def on_faint(self, ctx: TraitContext) -> None:
        """精灵力竭（魔力扣除已发生；ctx.subject=力竭精灵）。"""

    def on_counter(self, ctx: TraitContext) -> None:
        """应对成功（本精灵强制先手；ctx.is_counter=True）。"""

    def on_defense(self, ctx: TraitContext) -> None:
        """使用防御技能。"""

    def on_status_skill(self, ctx: TraitContext) -> None:
        """使用状态技能（含聚能类，效果未实现前也会触发）。"""

    def on_skill_end(self, ctx: TraitContext) -> None:
        """技能结束：全部效果（含离场）结算后；ctx.damage_dealt/hit_count 可用。"""

    def on_round_end(self, ctx: TraitContext) -> None:
        """回合结束（按主客场顺序广播）。"""

    def on_buff_gain(self, ctx: TraitContext) -> None:
        """获得增益/减益（ctx.extra["buff_type"]/["value"]）。"""

    def on_energy_gain(self, ctx: TraitContext) -> None:
        """获得能量（ctx.energy_gain）。"""

    def on_dot_damage(self, ctx: TraitContext) -> None:
        """属性伤害结算（中毒/灼烧/寄生/冻结等回合结束伤害；ctx.subject=受击精灵）。
        接入点：buffs.py 的 _apply_poison/_apply_burn/_apply_leech/_apply_freeze（待接入）。"""

    def on_heal(self, ctx: TraitContext) -> None:
        """回复生命（ctx.heal，ctx.extra["base_heal"]=修正前数值）。"""

    def on_weather_change(self, ctx: TraitContext) -> None:
        """天气变化（ctx.weather=新天气 id）。"""

    def on_revive(self, ctx: TraitContext) -> None:
        """复活（力竭后复活，可上场）。"""

    # ================= 修正钩子 =================
    # 修正查询默认只作用于"被查询精灵自身"的特性；
    # 仅能耗（query_energy_cost）与连击（query_hit_count）两类会额外查询敌方在场精灵
    # 的特性（冰封"敌方能耗+1"、无差别过滤"全场连击固定"等光环）。
    # ctx.actor = 特性所有者，ctx.target = 被查询的精灵（自身效果判断用 ctx.target is ctx.actor）。

    def modify_stat(self, ctx: TraitContext, stat: str) -> float:
        """属性百分比增量（0.2 = +20%）；stat ∈ atk/spatk/def/spdef。"""
        return 0.0

    def modify_power(self, ctx: TraitContext) -> tuple:
        """技能威力增量，返回 (百分比, 固定值)；ctx.skill 可用。"""
        return (0.0, 0.0)

    def modify_energy_cost(self, ctx: TraitContext) -> int:
        """技能能耗增量；ctx.skill 可用。"""
        return 0

    def modify_speed(self, ctx: TraitContext) -> int:
        """速度增量。"""
        return 0

    def modify_damage_dealt(self, ctx: TraitContext) -> float:
        """造成伤害乘数增量（-0.4 = -40%）。"""
        return 0.0

    def modify_damage_taken(self, ctx: TraitContext) -> float:
        """受到伤害乘数增量；ctx.skill（攻击技能）/ctx.subject（攻击方）/ctx.is_first 可用。"""
        return 0.0

    def modify_hit_count(self, ctx: TraitContext) -> tuple:
        """连击增量，返回 (固定值, 百分比)。"""
        return (0, 0)

    def force_hit_count(self, ctx: TraitContext) -> Optional[int]:
        """强制连击数（"固定为N"类，全场光环在 ctx.is_active() 时生效）：返回非 None 时覆盖连击数。"""
        return None

    def modify_lifesteal(self, ctx: TraitContext) -> float:
        """吸血增量（0.1 = +10% 吸血比例）；ctx.skill 可用。"""
        return 0.0

    def modify_energy_limit(self, ctx: TraitContext) -> int:
        """能量上限增量（突破上限）。"""
        return 0

    def modify_heal(self, ctx: TraitContext) -> int:
        """回复量增量（负值可把回复削减为0；ctx.heal=修正前数值）。"""
        return 0

    def modify_energy_gain(self, ctx: TraitContext) -> int:
        """获得能量增量。"""
        return 0

    def modify_energy_shortfall(self, ctx: TraitContext, need: int) -> int:
        """能量不足兜底（"能量不足时消耗5%生命代替1能量"类）：
        返回可补充的能量数，handler 自行支付代价；need=缺口。"""
        return 0

    def is_skill_usable(self, ctx: TraitContext, skill: BattleSkill) -> Optional[bool]:
        """技能可用性（ctx.skill_index=槽位，正位宝剑类按槽位限制）：返回 True/False 强制，None 不干预。"""
        return None

    def modify_skill_element(self, ctx: TraitContext, skill: BattleSkill) -> Optional[int]:
        """技能属性改写（"普通系技能变为翼系"类）：返回新属性 id 或 None。"""
        return None

    def modify_round_end_repeats(self, ctx: TraitContext) -> int:
        """回合结束效果额外触发次数（"双方回合结束效果额外触发1次"类）。"""
        return 0

    def modify_poison_repeats(self, ctx: TraitContext) -> int:
        """中毒效果额外触发次数（在 buffs.py 结算中毒时查询）。"""
        return 0
