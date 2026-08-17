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

    @property
    def pets(self) -> dict:
        return {side: self.teams[side][self.active[side]] for side in ("A", "B")}


@dataclass
class Action:
    kind: str = "charge"
    skill_index: Optional[int] = None
    pet_index: Optional[int] = None
