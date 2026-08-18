"""特性注册表：trait_id -> handler 实例。

默认把 data/traits.json 中的全部特性注册为 no-op handler（implemented=False），
实现某个特性时用 register() 覆盖即可。这样注册表始终完整，引擎分发不会缺项。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .base import TraitHandler

ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TRAITS_PATH = ROOT / "data" / "traits.json"

_REGISTRY: dict[int, TraitHandler] = {}
_loaded = False


def _raw_traits() -> list:
    with open(TRAITS_PATH, encoding="utf-8") as f:
        return json.load(f)["traits"]


class _NoopTrait(TraitHandler):
    """未实现的特性占位：不产生任何效果。"""


def _load_defaults() -> None:
    global _loaded
    if _loaded:
        return
    for raw in _raw_traits():
        h = _NoopTrait()
        h.trait_id = raw["id"]
        h.name = raw["name"]
        h.desc = raw["desc"]
        h.implemented = False
        _REGISTRY[raw["id"]] = h
    _loaded = True


def register(handler: TraitHandler, trait_id: int | None = None) -> TraitHandler:
    """注册/覆盖一个特性 handler。trait_id 缺省取 handler.trait_id。"""
    _load_defaults()
    tid = trait_id if trait_id is not None else handler.trait_id
    handler.trait_id = tid
    _REGISTRY[tid] = handler
    return handler


def get_handler(trait_id: int | None) -> TraitHandler | None:
    if trait_id is None:
        return None
    _load_defaults()
    return _REGISTRY.get(trait_id)


def info(trait_id: int | None) -> dict | None:
    """特性展示信息 {id, name, desc}，用于状态序列化。"""
    if trait_id is None:
        return None
    h = get_handler(trait_id)
    if h is None:
        return {"id": trait_id, "name": str(trait_id), "desc": ""}
    return {"id": h.trait_id, "name": h.name, "desc": h.desc}


def report() -> dict:
    """实现进度统计（开发用）。"""
    _load_defaults()
    done = [h for h in _REGISTRY.values() if h.implemented]
    return {
        "total": len(_REGISTRY),
        "implemented": len(done),
        "ids": sorted(h.trait_id for h in done),
    }
