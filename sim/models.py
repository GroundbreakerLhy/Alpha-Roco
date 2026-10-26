"""战斗核心数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SkillData:
    id: int
    name: str
    element: int
    category: int
    energy_cost: int
    power: Optional[int]
    desc: str
    effects: list = field(default_factory=list)


@dataclass
class BattleSkill:
    skill_id: int
    name: str
    element: int
    category: int
    power: Optional[int]
    energy_cost: int
    desc: str
    have_counter: bool = False
    counter_target: str = ""
    counter_interrupt: bool = False  # 打断：应对成功时被应对的技能/聚能行动作废
    drive: int = 0  # 传动：回合开始时向下移动的格数
    swift: bool = False  # 迅捷：主动换人入场时自动释放
    # 巧变：使用后替换，临时技能用后还原
    morph_pool: list = field(default_factory=list)  # 随机池：BattleSkill 模板
    morph_origin: Optional["BattleSkill"] = None  # 临时技能记录的原技能
    # 每回合替换（借用/取念/复写）：入场/回合末抽取，离场还原
    round_replace_origin: Optional["BattleSkill"] = None
    windup: bool = False  # 蓄力：需先蓄力 1 回合
    choice: bool = False  # 选择：明/暗二选一
    usable: bool = (
        True  # False = 无法主动使用
    )
    # 技能自身持久状态（成长值/使用计数等），由 sim.skills 读写
    skill_state: dict = field(default_factory=dict)


@dataclass
class BattlePet:
    side: str
    spirit_id: int
    name: str
    level: int
    stats: dict
    hp: int
    max_hp: int
    energy: int
    skills: list
    attributes: list = field(default_factory=list)
    speed_range: list = field(default_factory=lambda: [0, 0])
    defense_reduction: float = 0.0
    buffs: list = field(default_factory=list)
    overload_next: dict = field(default_factory=dict)
    overload_current: dict = field(default_factory=dict)
    ivs: dict = field(default_factory=dict)
    nature: int | None = None
    bloodline: int | None = (
        None  # 血脉编号：0-17 元素血脉；18=首领血脉
    )
    skill_cooldowns: set = field(default_factory=set)
    skill_cooldowns_pending: set = field(default_factory=set)
    has_acted_since_entry: bool = False
    entry_turn: int = (
        0  # 最近入场回合（本回合入场者免疫返场效果）
    )
    light_heal_rounds: int = 0  # 光合治愈剩余回合数
    pending_dizzy: int = 0  # 下回合施加的眩晕层数（B45）
    windup_skill: Optional[BattleSkill] = None  # 当前正在蓄力的技能
    cute_history: list[int] = field(default_factory=list)  # 萌化前的形态 ID 栈（后进先出）
    # 技能侧免死的生效回合；跨回合自动失效，无需清理。
    survive_lethal_turn: int = 0
    bursts: list = field(default_factory=list)
    wish_original_skill: Optional[BattleSkill] = None
    trait_id: Optional[int] = None
    trait_state: dict = field(default_factory=dict)
    # 应对统计：{"count": 累计次数, "types": {"attack"/"defense"/"status": n}}
    counter_stats: dict = field(default_factory=lambda: {"count": 0, "types": {}})
    # 本场释放过的技能：{id: 次数}（插入序=首次使用顺序；含迅捷与巧变临时技能）
    used_skill_counts: dict = field(default_factory=dict)
    revealed_skills: set = field(default_factory=set)
    # 上一次使用的技能 id（聚能/换人不变，重放不算新使用）
    last_used_skill_id: Optional[int] = None
    # 回合开始时 this 平移到 last 并清零；蓄力回合不算使用攻击技能，重放也算。
    attack_used_this_turn: bool = False
    attack_used_last_turn: bool = False
    # 本场离场次数（换人/脱离/返场都算，力竭不算）
    leave_count: int = 0
    # 最近一次使用技能的属性/类别；回合开始时平移到 prev_*
    last_skill_element: int | None = None
    last_skill_category: str = ""
    prev_skill_element: int | None = None
    prev_skill_category: str = ""

    def __setattr__(self, name, value) -> None:
        object.__setattr__(self, name, value)
        # 生命写入统一入口：实时检查冻结；构造阶段 buffs 尚未赋值，跳过。
        if name == "hp" and "buffs" in self.__dict__:
            from .buffs import check_freeze
            check_freeze(self)

    @property
    def alive(self) -> bool:
        return self.hp > 0

    @property
    def speed(self) -> int:
        return self.stats["speed"]


@dataclass
class BattleState:
    teams: dict = field(default_factory=dict)
    active: dict = field(default_factory=lambda: {"A": 0, "B": 0})
    magic: dict = field(default_factory=lambda: {"A": 4, "B": 4})
    offerings: dict = field(default_factory=lambda: {"A": {}, "B": {}})
    marks: dict = field(
        default_factory=lambda: {
            "A": {"positive": None, "negative": None},
            "B": {"positive": None, "negative": None},
        }
    )
    round_end_order: list = field(default_factory=lambda: ["A", "B"])
    trait_round_end_turn: int = -1  # 防止脱离续接重复执行本回合特性阶段
    weather: int | None = None
    resonance_magic: dict = field(
        default_factory=lambda: {
            "A": None,
            "B": None,
        }
    )
    resonance_usage: dict = field(
        default_factory=lambda: {
            "A": {},
            "B": {},
        }
    )
    resonance_cooldown: dict = field(
        default_factory=lambda: {
            "A": {},
            "B": {},
        }
    )
    turn: int = 0
    log: list = field(default_factory=list)
    winner: Optional[str] = None
    # 本回合行动顺序（先手方、后手方）
    last_order: list = field(default_factory=list)
    # 双方本回合/上回合技能能耗和
    turn_costs: dict = field(default_factory=lambda: {"A": 0, "B": 0})
    prev_turn_costs: dict = field(default_factory=lambda: {"A": 0, "B": 0})
    costs_settled_turn: int = -1
    entry_done: bool = False
    # 特性请求换人：服务端发 choose_replacement 由玩家选人
    pending_switch: dict = field(default_factory=lambda: {"A": False, "B": False})
    # 特性指定的自动替换：值为要换上场的精灵，回合末自动执行
    pending_switch_to: dict = field(
        default_factory=lambda: {"A": None, "B": None}
    )
    # 返场请求：回合末自动执行，无需玩家输入
    pending_reenter: dict = field(default_factory=lambda: {"A": False, "B": False})
    # 返场成功后的迸发，记录请求者，避免误发给替换后的精灵。
    pending_reenter_bursts: dict = field(default_factory=lambda: {"A": [], "B": []})
    # 技能脱离：该侧行动结算后立即离场并重新选人；回合中生效、跳过离场者的回合末结算。
    pending_action_leave: dict = field(default_factory=lambda: {"A": False, "B": False})
    # 技能脱离附带"入场精灵回复 N 能量"，与 pending_action_leave 同生命周期。
    pending_entry_energy: dict = field(default_factory=lambda: {"A": 0, "B": 0})
    # 击鼓传花：离场前把 NORMAL 增益快照到 pending_entry_buffs，由下一只入场精灵继承；
    # 脱离未执行则作废。
    pending_entry_inherit: dict = field(default_factory=lambda: {"A": False, "B": False})
    pending_entry_buffs: dict = field(default_factory=lambda: {"A": [], "B": []})
    # 回合因脱离暂停：含续接上下文与应选人的一侧（leave_side）
    paused_turn: Optional[dict] = None
    # 应对奖励登记：回合末第一步统一结算（B46）。
    pending_counters: list = field(default_factory=list)

    @property
    def pets(self) -> dict:
        return {side: self.teams[side][self.active[side]] for side in ("A", "B")}


@dataclass
class Action:
    kind: str
    skill_index: Optional[int] = None
    pet_index: Optional[int] = None
    magic_id: Optional[int] = None
    magic_branch: int = 0
    choice_branch: int = 0  # 选择分支：0=明，1=暗
