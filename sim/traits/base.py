"""特性系统接口：上下文 + handler 基类。

事件钩子（on_*）经 traits.emit 广播，可产生副作用（改 hp/能量/buff/印记）。
修正钩子（modify_*）经 traits.query_* 汇总；一律增量语义：0 无修正，乘数 = 1.0 + 增量。
未实现的特性是 no-op handler（implemented=False），不打印任何日志。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..models import Action, BattlePet, BattleSkill, BattleState


@dataclass
class TraitContext:
    """传给特性 handler 的上下文。actor 永远是特性所有者，字段随事件/查询部分生效。"""

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


    def state_of(self, key, default=None):
        return self.actor.trait_state.get(key, default)

    def set_state(self, key, value) -> None:
        self.actor.trait_state[key] = value

    def add_state(self, key, amount) -> None:
        self.actor.trait_state[key] = self.actor.trait_state.get(key, 0) + amount


class TraitHandler:
    """所有特性 handler 的基类：子类按需覆写钩子，其余为空实现。"""

    trait_id: int = 0
    name: str = ""
    desc: str = ""
    implemented: bool = False

    # ================= 事件钩子 =================

    def on_battle_start(self, ctx: TraitContext) -> None:
        """战斗创建完成后的初始化事件，双方所有精灵都会收到。"""

    def on_evolution(self, ctx: TraitContext) -> None:
        """首领化完成（subject=完成进化的精灵）。"""

    def on_turn_start(self, ctx: TraitContext) -> None:
        """回合开始（双方行动已选定，ctx.action_a/action_b 可用）。"""

    def on_entry(self, ctx: TraitContext) -> None:
        """入场：主动换人 / 力竭替换 / 脱离换人 / 战斗开始首发（subject=入场精灵）。"""

    def on_leave(self, ctx: TraitContext) -> None:
        """离场：主动换人 / 脱离（不含力竭）；ctx.subject=离场精灵，ctx.extra["incoming"]=入场精灵。"""

    def on_charge(self, ctx: TraitContext) -> None:
        """聚能（ctx.energy_gain=回复量）。"""

    def on_windup(self, ctx: TraitContext) -> None:
        """进入蓄力状态（首次选择蓄力技能、支付能耗后；ctx.skill=蓄力技能）。"""

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
        """使用状态技能（含聚能类）。"""

    def on_skill_end(self, ctx: TraitContext) -> None:
        """技能结束：全部效果（含离场）结算后；ctx.damage_dealt/hit_count 可用。"""

    def on_round_end(self, ctx: TraitContext) -> None:
        """特性效果阶段（按双方有效速度顺序广播）。"""

    def on_round_ready(self, ctx: TraitContext) -> None:
        """回合末全部结算完成后，为下一回合准备状态。"""

    def on_buff_gain(self, ctx: TraitContext) -> None:
        """获得增益/减益（ctx.extra["buff_type"]/["value"]）。"""

    def on_energy_gain(self, ctx: TraitContext) -> None:
        """获得能量（ctx.energy_gain）。"""

    def on_dot_damage(self, ctx: TraitContext) -> None:
        """属性伤害结算（中毒/灼烧/寄生等回合结束伤害；冻结斩杀不属于伤害；ctx.subject=受击精灵）。"""

    def on_heal(self, ctx: TraitContext) -> None:
        """回复生命（ctx.heal，ctx.extra["base_heal"]=修正前数值）。"""

    def on_weather_change(self, ctx: TraitContext) -> None:
        """天气变化（ctx.weather=新天气 id）。"""

    def on_revive(self, ctx: TraitContext) -> None:
        """复活（力竭后复活，可上场）。"""

    def allow_cute_reapplication(self, pet: BattlePet) -> bool:
        """已有萌化时是否允许再次获得萌化而不继续退化；默认不允许。"""
        return False

    def allow_cute_beyond_limit(self, pet: BattlePet) -> bool:
        """无法继续退化时是否仍可获得萌化 buff；默认不允许。"""
        return False

    def modify_buff_gain_layers(self, pet: BattlePet, buff_type: str, value: int) -> int:
        """获得效果时额外层数，沿原数值方向增加；默认无修正。"""
        return 0

    def modify_base_energy_cost(self, ctx: TraitContext) -> Optional[int]:
        """技能基础能耗覆盖：返回非 None 时替代技能自身的基础能耗，其余修正照常叠加。"""
        return None

    def consume_skip_windup(self, pet: BattlePet) -> bool:
        """是否跳过本次蓄力；返回 True 时消费该状态。"""
        return False

    # ================= 修正钩子 =================
    # 修正查询默认只查被查询精灵自身；仅能耗与连击额外查敌方在场精灵（敌方光环）。
    # ctx.actor = 特性所有者，ctx.target = 被查询精灵（自身效果判断 ctx.target is ctx.actor）。

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

    def modify_priority(self, ctx: TraitContext) -> int:
        """行动先手值增量（"迅捷技能先手+1"类）；ctx.skill 可用。"""
        return 0

    def modify_star_consume(self, ctx: TraitContext, stacks: int) -> Optional[int]:
        """星陨印记触发时消耗的层数；返回非 None 覆盖默认全部消耗；ctx.target=触发方（攻击方）。"""
        return None

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

    def modify_energy_shortfall(
        self, ctx: TraitContext, need: int, dry_run: bool = False
    ) -> int:
        """能量不足兜底：返回可补充的能量数，handler 自行支付代价；need=缺口。

        dry_run=True 只回答能否补足（不得支付代价）；dry_run=False 全额支付，
        生命 ≤0 由引擎直接力竭。"""
        return 0

    def is_skill_usable(self, ctx: TraitContext, skill: BattleSkill) -> Optional[bool]:
        """技能可用性（ctx.skill_index=槽位）：返回 True/False 强制，None 不干预。"""
        return None

    def allow_any_skill_in_windup(self, ctx: TraitContext) -> bool:
        """蓄力状态下是否允许使用任意携带技能；默认 False。"""
        return False

    def modify_skill_element(
        self, ctx: TraitContext, skill: BattleSkill
    ) -> Optional[int]:
        """技能属性改写（"普通系技能变为翼系"类）：返回新属性 id 或 None。"""
        return None

    def modify_round_end_repeats(self, ctx: TraitContext) -> int:
        """回合结束效果额外触发次数（"双方回合结束效果额外触发1次"类）。"""
        return 0

    def modify_poison_repeats(self, ctx: TraitContext) -> int:
        """中毒效果额外触发次数（在 buffs.py 结算中毒时查询）。"""
        return 0

    def modify_burn_growth(self, ctx: TraitContext) -> bool:
        """是否将灼烧回合结束时的衰减改为增长。"""
        return False

    # ================= 显示 =================

    def display(self, state, pet) -> list | None:
        """当前特性效果的显示行（序列化给客户端 buff 区）。

        返回 [{"name", "layers", "per", "gain": bool}]；None/[] = 无显示。
        """
        return None
