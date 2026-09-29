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
    drive: int = 0  # 传动值：回合开始时技能向下移动的格数
    swift: bool = False  # 迅捷标记：主动换人入场时自动释放
    morph_pool: list = field(default_factory=list)  # 巧变随机池：BattleSkill 模板列表
    morph_origin: Optional["BattleSkill"] = None  # 巧变临时技能记录的原技能
    windup: bool = False  # 蓄力技能标记：需要先蓄力1回合才能释放
    usable: bool = True  # False = 无法主动使用（技能代码定义，如"使用3次翼系技能后自动使用"）
    # 技能自身的持久运行时状态（永久成长值/使用计数等），由 sim.skills 的效果原语读写
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
    bloodline: int | None = None  # 血脉编号：0-17 元素血脉；18=首领血脉（enums.LORD_BLOODLINE）。一只精灵只有一种血脉
    skill_cooldowns: set = field(default_factory=set)
    skill_cooldowns_pending: set = field(default_factory=set)
    has_acted_since_entry: bool = False
    entry_turn: int = 0  # 最近一次入场的回合（返场免疫判定：本回合入场的精灵免疫返场效果）
    light_heal_rounds: int = 0  # 光合治愈持续回合数（回合结束回复）
    windup_skill: Optional[BattleSkill] = None  # 当前正在蓄力的技能
    bursts: list = field(default_factory=list)
    wish_original_skill: Optional[BattleSkill] = None
    trait_id: Optional[int] = None
    trait_state: dict = field(default_factory=dict)
    # 应对统计（按精灵）：{"count": 累计次数, "types": {"attack": n, "defense": n, "status": n}}
    # 由 counter 模块记录，供"每应对成功N次"类特性读取
    counter_stats: dict = field(default_factory=lambda: {"count": 0, "types": {}})

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
    weather: int | None = None
    resonance_magic: dict = field(default_factory=lambda: {
        "A": None,
        "B": None,
    })
    resonance_usage: dict = field(default_factory=lambda: {
        "A": {},
        "B": {},
    })
    resonance_cooldown: dict = field(default_factory=lambda: {
        "A": {},
        "B": {},
    })
    turn: int = 0
    log: list = field(default_factory=list)
    winner: Optional[str] = None
    entry_done: bool = False
    # 特性请求的换人（如警惕：回合结束能量为0时脱离）——服务端发 choose_replacement 由玩家选人
    pending_switch: dict = field(default_factory=lambda: {"A": False, "B": False})
    # 特性请求的返场（精灵离场并立即入场，如安可 200290）——引擎自动执行，无需玩家输入
    pending_reenter: dict = field(default_factory=lambda: {"A": False, "B": False})
    # 技能驱动的脱离请求（如移花接木"随后脱离"、吓退"敌方脱离"）：该侧行动结算后立即
    # 离场并重新选人。与 pending_switch 的区别：它在**回合中**生效并跳过离场者的回合末结算。
    pending_action_leave: dict = field(default_factory=lambda: {"A": False, "B": False})
    # 回合因脱离暂停：非 None 时 step 已中断等待选人，含续接上下文与应选人的一侧（leave_side）
    paused_turn: Optional[dict] = None

    @property
    def pets(self) -> dict:
        return {side: self.teams[side][self.active[side]] for side in ("A", "B")}


@dataclass
class Action:
    kind: str = "charge"
    skill_index: Optional[int] = None
    pet_index: Optional[int] = None
    magic_id: Optional[int] = None
    magic_branch: int = 0
