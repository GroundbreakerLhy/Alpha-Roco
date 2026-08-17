"""Core data models for the headless battle simulator."""

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
    # 技能机制(从描述解析)
    swift: bool = False        # 迅捷:使用该技能时先手+1
    drive: int = 0             # 传动N:使用后技能位置移动N格
    qiaobian: str = ""         # 巧变:X(技能元素/效果替换)
    choices: list = field(default_factory=list)  # 选择:A或B(选项文本列表)


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
    defense_cooldowns: set = field(default_factory=set)
    defense_used_this_turn: set = field(default_factory=set)
    has_acted_since_entry: bool = False
    bursts: list = field(default_factory=list)
    # 特性(精灵被动)支持
    feature_id: int | None = None
    feature_desc: str = ""
    energy_cap: int = 10                 # 能量上限(默认 10,特性可修改)
    entry_count: int = 0                 # 本场入场次数
    energy_spent_total: int = 0          # 本场累计消耗能量
    skills_used: set = field(default_factory=set)          # 本场使用过的技能 id
    feature_state: dict = field(default_factory=dict)      # 特性私有状态
    feature_skill_index_restriction: list | None = None    # 仅可使用哪些技能位(None=不限)
    revive_turn: int | None = None       # 力竭后复活的目标回合
    blocked_skills: set = field(default_factory=set)       # 冷却/禁用的技能 id(本回合)
    # 缺失机制支持
    form_type: str = ""                  # 形态(主/首领/特殊) → 血脉判定
    capture_ball: str = "普通咕噜球"      # 捕捉用球
    disguised: bool = False              # 伪装状态
    bag_pet_index: int | None = None     # 背包随机精灵在队伍中的索引

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
    revealed: dict = field(default_factory=lambda: {"A": set(), "B": set()})
    marks: dict = field(default_factory=lambda: {
        "A": {"positive": None, "negative": None},
        "B": {"positive": None, "negative": None},
    })
    home_side: str = "A"
    turn: int = 0
    log: list = field(default_factory=list)
    winner: Optional[str] = None
    # 特性系统跨回合状态
    feature_pending_inherit: dict = field(default_factory=lambda: {"A": [], "B": []})
    feature_pending_entry: dict = field(default_factory=lambda: {"A": [], "B": []})
    # 环境与时间(周末/王国入夜)
    day_of_week: int = 0                 # 0=周一 ... 6=周日
    is_night: bool = True                # 王国入夜
    # 印记叠加槽(赋予的印记不替换时使用)
    marks_extra: dict = field(default_factory=lambda: {
        "A": {"positive": [], "negative": []},
        "B": {"positive": [], "negative": []},
    })

    @property
    def pets(self) -> dict:
        return {side: self.teams[side][self.active[side]] for side in ("A", "B")}


@dataclass
class Action:
    kind: str = "charge"
    skill_index: Optional[int] = None
    pet_index: Optional[int] = None
