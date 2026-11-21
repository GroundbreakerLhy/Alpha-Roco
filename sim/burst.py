"""迸发：入场后首次行动消耗的额外效果。

type 取值：attack_power_flat / attack_power_percent / skill_use_count /
energy_cost_flat / enemy_energy_cost。
"""

from __future__ import annotations

from . import events


def _state(state):
    if state is not None:
        return state
    context = events.current_context()
    return None if context is None else context["state"]


def add_burst(pet, burst_type: str, value: int, *, expires_on_turn: int | None = None,
              state=None) -> None:
    effect = {"type": burst_type, "value": value}
    if expires_on_turn is not None:
        effect["expires_on_turn"] = expires_on_turn
    pet.bursts.append(effect)
    state = _state(state)
    if state is not None:
        context = events.current_context()
        events.emit(
            state,
            "burst",
            action="grant",
            source=(
                context["source"]
                if context is not None and context["state"] is state
                else events.field_entity()
            ),
            target=events.pet_entity(state, pet),
            attribute=burst_type,
            value=value,
            duration=expires_on_turn or "",
        )


def expire_bursts(pet, current_turn: int, state=None) -> None:
    """移除到本回合末到期、且尚未消费的限时迸发。"""
    expired = [
        effect
        for effect in pet.bursts
        if effect.get("expires_on_turn") is not None
        and effect["expires_on_turn"] <= current_turn
    ]
    pet.bursts[:] = [effect for effect in pet.bursts if effect not in expired]
    state = _state(state)
    if state is not None:
        for effect in expired:
            events.emit(
                state,
                "burst",
                action="expire",
                target=events.pet_entity(state, pet),
                attribute=effect["type"],
                value=effect["value"],
            )


def clear_bursts(pet, state=None) -> None:
    state = _state(state)
    if state is not None:
        for effect in pet.bursts:
            events.emit(
                state,
                "burst",
                action="remove",
                target=events.pet_entity(state, pet),
                attribute=effect["type"],
                value=effect["value"],
            )
    pet.bursts.clear()


def take_bursts(pet, state=None) -> list:
    bursts = list(pet.bursts)
    pet.bursts.clear()
    state = _state(state)
    if state is not None:
        for effect in bursts:
            events.emit(
                state,
                "burst",
                action="consume",
                target=events.pet_entity(state, pet),
                attribute=effect["type"],
                value=effect["value"],
            )
    return bursts


def get_bursts(pet) -> list:
    return list(pet.bursts)
