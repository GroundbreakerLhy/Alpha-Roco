"""技能注册表：skill_id -> handler。

默认把 data/skills.json 的全部技能注册为 no-op（implemented=False），
实现时用 ``register()`` 覆盖。desc 仅供展示/校对，运行时不读取。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .base import SkillHandler

ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SKILLS_PATH = ROOT / "data" / "skills.json"

_REGISTRY: dict[int, SkillHandler] = {}
_RAW: dict[int, dict] = {}
_loaded = False


def _raw_skills() -> list:
    with open(SKILLS_PATH, encoding="utf-8") as f:
        return json.load(f)["skills"]


class _NoopSkill(SkillHandler):
    """未实现的技能占位：不产生任何效果。"""


def _load_defaults() -> None:
    global _loaded
    if _loaded:
        return
    for raw in _raw_skills():
        _RAW[raw["id"]] = raw
        h = _NoopSkill()
        h.skill_id = raw["id"]
        h.name = raw["name"]
        h.category = raw["category"]
        h.desc = raw.get("desc", "")
        h.implemented = False
        _REGISTRY[raw["id"]] = h
    _loaded = True


def register(handler: SkillHandler, skill_id: int | None = None) -> SkillHandler:
    """注册/覆盖一个技能 handler。skill_id 缺省取 handler.skill_id。"""
    _load_defaults()
    sid = skill_id if skill_id is not None else handler.skill_id
    handler.skill_id = sid
    raw = _RAW.get(sid)
    if raw is not None:
        handler.name = handler.name or raw["name"]
        handler.desc = handler.desc or raw.get("desc", "")
        if handler.category < 0:
            handler.category = raw["category"]
    _REGISTRY[sid] = handler
    return handler


def get_handler(skill_id: int | None) -> SkillHandler | None:
    if skill_id is None:
        return None
    _load_defaults()
    return _REGISTRY.get(skill_id)


def raw_of(skill_id: int) -> dict | None:
    """skills.json 中的原始记录（仅供展示/校对，不参与效果计算）。"""
    _load_defaults()
    return _RAW.get(skill_id)


def static_of(skill_id: int) -> dict:
    """构造 BattleSkill 用的静态属性字典。"""
    handler = get_handler(skill_id)
    if handler is None:
        return {"counter_target": "", "counter_interrupt": False, "drive": 0,
                "swift": False, "windup": False, "usable": True, "element": None,
                "choice": False}
    return {
        "counter_target": handler.counter_target,
        "counter_interrupt": handler.counter_interrupt,
        "drive": handler.drive,
        "swift": handler.swift,
        "windup": handler.windup,
        "usable": handler.usable,
        "element": handler.element,
        "choice": handler.choice,
    }


def bind_skill(skill, pet=None):
    """把静态定义写入 BattleSkill；只写非默认值，不覆盖其他来源的标记。"""
    handler = get_handler(getattr(skill, "skill_id", None))
    if handler is None:
        return skill
    if handler.counter_target:
        skill.counter_target = handler.counter_target
    if handler.counter_interrupt:
        skill.counter_interrupt = True
    if handler.drive:
        skill.drive = handler.drive
    if handler.swift:
        skill.swift = True
    if handler.windup:
        skill.windup = True
    if not handler.usable:
        skill.usable = False
    if handler.element is not None:
        skill.element = handler.element
    if handler.choice:
        skill.choice = True
    overrides = handler.on_bind(pet, skill)
    for key, value in overrides.items():
        setattr(skill, key, value)
    return skill


def all_handlers() -> list:
    _load_defaults()
    return [_REGISTRY[sid] for sid in sorted(_REGISTRY)]


def report() -> dict:
    """进度统计：以 data/skills.json 的 done/tested 为准。"""
    raws = _raw_skills()
    done = [s for s in raws if s.get("done")]
    tested = [s for s in raws if s.get("tested")]
    return {
        "total": len(raws),
        "done": len(done),
        "tested": len(tested),
        "done_ids": sorted(s["id"] for s in done),
        "tested_ids": sorted(s["id"] for s in tested),
    }


def check() -> list:
    """一致性自检：核对代码定义与 data/skills.json，返回问题清单（空=通过）。"""
    _load_defaults()
    problems = []
    for sid, handler in sorted(_REGISTRY.items()):
        raw = _RAW.get(sid)
        if raw is None:
            # 负数 id 是运行时虚拟技能（愿力冲击等特性/共鸣替换用），不属于 skills.json
            if sid < 0:
                continue
            problems.append(f"{sid}: 注册表存在但 skills.json 中没有该技能")
            continue
        if handler.name and handler.name != raw["name"]:
            problems.append(f"{sid}: 名称不一致 代码={handler.name} 数据={raw['name']}")
        if handler.category >= 0 and handler.category != raw["category"]:
            problems.append(f"{sid}: 类别不一致 代码={handler.category} 数据={raw['category']}")
        if handler.implemented and handler.category < 0:
            problems.append(f"{sid}: 已实现但未声明 category")
    return problems
