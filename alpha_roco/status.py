from __future__ import annotations

DURATION_CODES = {"normal": 0.0, "temporary": 1.0, "permanent": 2.0}


def duration_code(value) -> float:
    return DURATION_CODES.get(value, 0.0)
