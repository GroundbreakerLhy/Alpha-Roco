"""特性效果来源的上下文记录。"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

_CURRENT = ContextVar[tuple[str, str, int | None] | None](
    "current_trait_source", default=None
)


@contextmanager
def trait_source_context(side: str, pet: str, trait_id: int | None):
    token = _CURRENT.set((side, pet, trait_id))
    try:
        yield
    finally:
        _CURRENT.reset(token)


def current_trait_source() -> tuple[str, str, int | None] | None:
    return _CURRENT.get()
