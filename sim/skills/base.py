"""技能系统：上下文与 handler 基类。

`modify_*` 返回增量（0 = 无修正）；`on_*` 钩子按固定阶段调用，可改血量/能量/buff/印记。
描述不参与运行时。阶段顺序：on_use_start → 伤害计算 → on_hit → on_counter →
on_defense / on_status → on_after_use → on_round_end。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..models import BattlePet, BattleSkill, BattleState


@dataclass
class SkillContext:
    """传给技能 handler 的上下文；actor 恒为使用者。

    skill 在技能阶段非空；logs=None 时写 state.log；choice_branch 由 Action 透传。
    """

    state: BattleState
    actor: BattlePet
    skill: BattleSkill
    skill_index: int = -1
    target: Optional[BattlePet] = None
    category: str = ""
    is_first: bool = False
    is_counter: bool = False
    choice_branch: int = 0
    counter_category: str = ""
    energy_cost: int = 0
    damage_dealt: int = 0
    hit_count: int = 1
    weather: Optional[int] = None
    logs: Optional[list] = None
    extra: dict = field(default_factory=dict)

    # ---- 便捷方法 ----

    def log(self, msg: str) -> None:
        """写日志（**技能 handler 不得使用**，见 SPEC.md V3）。"""
        (self.logs if self.logs is not None else self.state.log).append(msg)

    def is_active(self) -> bool:
        idx = self.state.active[self.actor.side]
        return idx >= 0 and self.state.teams[self.actor.side][idx] is self.actor

    def opponent(self) -> Optional[BattlePet]:
        opp_side = "B" if self.actor.side == "A" else "A"
        idx = self.state.active[opp_side]
        if idx < 0:
            return None
        return self.state.teams[opp_side][idx]

    # ---- 技能自身运行时状态（BattleSkill.skill_state） ----

    def state_of(self, key, default=None):
        return self.skill.skill_state.get(key, default)

    def set_state(self, key, value) -> None:
        self.skill.skill_state[key] = value

    def add_state(self, key, amount) -> None:
        self.skill.skill_state[key] = self.skill.skill_state.get(key, 0) + amount


class SkillHandler:
    """技能 handler 基类：覆写需要的钩子，其余为空实现；声明式写法见 ``ops.OpsSkill``。"""

    skill_id: int = 0
    name: str = ""
    category: int = -1
    desc: str = ""          # 仅展示/校对，引擎不解析
    implemented: bool = False   # 全部效果已在代码定义（对应 skills.json 的 done）

    # ================= 静态属性（构造/重绑时确定） =================

    counter_target: str = ""    # 应对类别；空=不应对
    counter_interrupt: bool = False  # 打断：应对成功时被应对的技能/聚能行动作废
    drive: int = 0              # 传动值
    swift: bool = False         # 迅捷：主动换人入场时自动释放
    windup: bool = False        # 蓄力：需先蓄力 1 回合
    choice: bool = False          # 选择：明/暗二选一
    usable: bool = True         # False = 无法主动使用
    element: Optional[int] = None   # 固有属性改写；None = 用 skills.json

    def on_bind(self, pet: BattlePet, skill: BattleSkill) -> dict:
        """构造/重绑时覆盖静态属性，返回写回 BattleSkill 的字段字典。

        键：counter_target / drive / swift / windup / usable / element / morph_pool；
        需要的动态判定放 modify_* 查询。
        """
        return {}

    # ================= 修正查询（增量语义） =================

    def modify_power(self, ctx: SkillContext) -> tuple:
        """技能威力增量，返回 ``(百分比, 固定值)``。"""
        return (0.0, 0.0)

    def absolute_power(self, ctx: SkillContext) -> Optional[int]:
        """威力绝对值覆盖；None = 用 skills.json 的 power，非 None 作为基础威力。"""
        return None

    def modify_hit_count(self, ctx: SkillContext) -> tuple:
        """连击增量，返回 ``(固定值, 百分比, 强制值|None)``；强制值非 None 时覆盖连击数。"""
        return (0, 0, None)

    def adjust_hit_count(self, ctx: SkillContext, count: int) -> int:
        """连击数的最终修正（乘算/翻倍）；modify_hit_count 只给增量，故乘算在此。"""
        return count

    def modify_energy_cost(self, ctx: SkillContext) -> int:
        """技能能耗增量。"""
        return 0

    def modify_other_skill_energy_cost(
        self, ctx: SkillContext, target_skill: BattleSkill, target_index: int
    ) -> int:
        """携带本技能时，对同一精灵另一技能能耗的增量。"""
        return 0

    def modify_energy_shortfall(self, ctx: SkillContext, need: int, dry_run: bool = False) -> int:
        """能量不足时可补充的能量（代价由 handler 结算）；dry_run=True 只探测。"""
        return 0

    def modify_priority(self, ctx: SkillContext) -> int:
        """先手增量。"""
        return 0

    def modify_lifesteal(self, ctx: SkillContext) -> float:
        """吸血比例增量（0.5 = +50%）。"""
        return 0.0

    def modify_damage_reduction(self, ctx: SkillContext) -> float:
        """防御技能减伤（0.7 = 减伤70%）；仅防御技能分支查询。"""
        return 0.0

    def modify_element(self, ctx: SkillContext) -> Optional[int]:
        """本次技能属性改写；None = 不改写。"""
        return None

    def ignore_resist(self, ctx: SkillContext) -> bool:
        """本次伤害无视敌方系别抵抗（克制倍率 < 1 时按 1 计）。"""
        return False

    def modify_use_count(self, ctx: SkillContext) -> int:
        """本次行动的额外执行次数（"使用次数永久+1"类）。"""
        return 0

    # ================= 每回合替换（借用/取念/复写） =================

    # 与巧变分离：在场时每回合末重抽，离场还原。
    round_replace: bool = False
    round_replace_cost_delta: int = 0  # 取念/复写为 -2

    def round_replace_pool(self, state: BattleState, pet: BattlePet,
                           skill: BattleSkill) -> list:
        """返回每回合替换的候选技能模板池；空 = 本回合不替换。"""
        return []

    # ================= 事件钩子 =================

    def on_entry(self, ctx: SkillContext) -> None:
        """精灵入场时向携带技能广播。"""

    def on_slot_change(self, ctx: SkillContext) -> None:
        """传动重排后槽位变化；ctx.extra 带 old_index / new_index。"""

    def on_use_start(self, ctx: SkillContext) -> None:
        """支付能耗后、造成伤害前（对应引擎的 skill_start 时点）。"""

    def on_hit(self, ctx: SkillContext) -> None:
        """本次伤害结算后（含未造成伤害的情况）；ctx.damage_dealt 可用。"""

    def on_counter(self, ctx: SkillContext) -> None:
        """应对成功；三条分支共用，按 ctx.counter_category 区分奖励。"""

    def on_defense(self, ctx: SkillContext) -> None:
        """防御技能本体（减伤已通过 modify_damage_reduction 取出）。"""

    def on_status(self, ctx: SkillContext) -> None:
        """状态技能本体结算。"""

    def on_after_use(self, ctx: SkillContext) -> None:
        """使用后：永久成长、巧变、位置、冷却、脱离/返场等。"""

    def on_round_end(self, ctx: SkillContext) -> None:
        """回合结束（技能登记的持续效果）。"""

    def on_take_damage(self, ctx: SkillContext) -> None:
        """自己受到攻击伤害后广播到本技能；ctx.skill 为本技能，ctx.target 为攻击方。

        extra 带 damage_taken / incoming_skill，hit_count 为连击段数；
        触发条件由 handler 判定，引擎只在攻击伤害落位后广播。
        """
