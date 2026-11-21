"""结构化战斗事件：所有事件在结算点按发生顺序编号写入 ``BattleState.events``。"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

PET = "pet"
TEAM = "team"
FIELD = "field"
SYSTEM = "system"
MARK = "mark"
WEATHER = "weather"
TRAIT = "trait"
ENTITY_KINDS = (PET, TEAM, FIELD, SYSTEM, MARK, WEATHER, TRAIT)

EVENT_KINDS = (
    "unknown",
    "log",
    "private",
    "skill",
    "damage",
    "heal",
    "life",
    "energy",
    "buff",
    "mark",
    "weather",
    "switch",
    "entry",
    "leave",
    "faint",
    "windup",
    "counter",
    "trait",
    "resonance",
    "charge",
    "cooldown",
    "burst",
    "slot",
    "choice",
    "status",
    "system",
    "flee",
    "interrupt",
    "turn",
    "win",
    "growth",
    "offering",
    "overload",
    "evolve",
)

EVENT_ACTIONS = (
    "unknown",
    "use",
    "replay",
    "damage",
    "heal",
    "apply",
    "remove",
    "consume",
    "expire",
    "convert",
    "swap",
    "gain",
    "loss",
    "set",
    "clear",
    "add",
    "schedule",
    "settle",
    "trigger",
    "enter",
    "leave",
    "switch",
    "charge",
    "faint",
    "counter",
    "interrupt",
    "flee",
    "limit",
    "win",
    "evolve",
    "grant",
    "decay",
    "reflect",
    "move",
    "start",
    "end",
    "ready",
    "replace",
    "block",
    "restore",
    "reduce",
    "increase",
    "fail",
)

EVENT_PHASES = (
    "setup",
    "action",
    "round_end",
    "resonance",
    "replacement",
    "lead",
)

EVENT_EFFECTS = (
    "",
    "skill",
    "counter",
    "replay",
    "dot",
    "mark",
    "reflect",
    "extra",
    "lifesteal",
    "thorns",
    "weather",
    "entry",
    "leave",
    "faint",
    "windup",
    "burst",
    "cooldown",
    "slot",
    "growth",
    "offering",
    "overload",
    "resonance",
    "evolve",
    "wish",
    "turn_limit",
    "winner",
    "interrupt",
    "blocked",
    "invalid",
    "need_selection",
    "dizzy",
    "unusable",
    "unimplemented",
    "energy_stop",
    "faint_replacement",
    "force_switch",
    "trait_switch",
    "leave_switch",
    "freeze",
    "reconcile",
    "attack",
    "battle_start",
    "buff_gain",
    "charge",
    "cost",
    "defense",
    "energy_shortfall",
    "kill",
    "round_end",
    "round_ready",
    "skill_end",
    "skill_start",
    "status",
    "status_skill",
    "take_damage",
    "trait",
    "lethal",
)

_CURRENT = ContextVar("battle_event_context", default=None)


def entity(
    kind: str,
    side: str | None = None,
    index: int = -1,
    name: str = "",
    spirit_id: int | None = None,
    trait_id: int | None = None,
) -> dict:
    return {
        "kind": kind,
        "side": side,
        "index": index,
        "name": name,
        "spirit_id": spirit_id,
        "trait_id": trait_id,
    }


def field_entity(name: str = "") -> dict:
    return entity(FIELD, None, -1, name)


def system_entity(side: str | None = None, name: str = "") -> dict:
    return entity(SYSTEM, side, -1, name)


def team_entity(side: str | None = None, name: str = "") -> dict:
    return entity(TEAM, side, -1, name)


def mark_entity(side: str | None = None, name: str = "") -> dict:
    return entity(MARK, side, -1, name)


def weather_entity(name: str = "") -> dict:
    return entity(WEATHER, None, -1, name)


def trait_entity(
    side: str | None = None,
    index: int = -1,
    name: str = "",
    spirit_id: int | None = None,
    trait_id: int | None = None,
) -> dict:
    return entity(TRAIT, side, index, name, spirit_id, trait_id)


def pet_entity(state, pet) -> dict:
    if pet is None:
        return field_entity()
    for side in ("A", "B"):
        for index, item in enumerate(state.teams.get(side, [])):
            if item is pet:
                return entity(
                    PET, side, index, pet.name, pet.spirit_id, pet.trait_id
                )
    return entity(PET, pet.side, -1, pet.name, pet.spirit_id, pet.trait_id)


def pet_entity_by_name(state, side: str, name: str) -> dict:
    for index, item in enumerate(state.teams.get(side, [])):
        if item.name == name:
            return entity(
                PET, side, index, name, item.spirit_id, item.trait_id
            )
    spirit_id = _spirit_id_by_name(name)
    if spirit_id is not None:
        for index, item in enumerate(state.teams.get(side, [])):
            if item.spirit_id == spirit_id or spirit_id in item.cute_history:
                return entity(
                    PET, side, index, item.name, item.spirit_id, item.trait_id
                )
    return entity(PET, side, -1, name)


_SPIRIT_IDS: dict[str, int] | None = None

EVENT_WINDOW = 2048
_SKILL_REF = "skill"
_TRAIT_REF = "trait"


def _spirit_id_by_name(name: str) -> int | None:
    global _SPIRIT_IDS
    if _SPIRIT_IDS is None:
        from .data_loader import load_spirits

        _SPIRIT_IDS = {spirit["name"]: spirit["id"] for spirit in load_spirits()}
    return _SPIRIT_IDS.get(name)


def _references(event) -> set:
    keys = set()
    for role in ("source", "target"):
        entity = event.get(role) or {}
        index = entity.get("index")
        if (
            entity.get("kind") == PET
            and entity.get("side") in ("A", "B")
            and isinstance(index, int)
            and index >= 0
        ):
            keys.add((PET, entity["side"], index))
        elif entity.get("kind") == TRAIT and entity.get("trait_id") is not None:
            keys.add((_TRAIT_REF, entity["trait_id"]))
    if event.get("skill_id") is not None:
        keys.add((_SKILL_REF, event["skill_id"]))
    if event.get("trait_id") is not None:
        keys.add((_TRAIT_REF, event["trait_id"]))
    return keys


def _touch(state, event) -> None:
    references = _references(event)
    if not references:
        return
    for old in state.events:
        if old is event:
            continue
        if references & _references(old):
            old["last_used"] = event["seq"]


def _trim(state) -> None:
    while len(state.events) > EVENT_WINDOW:
        victim = min(
            range(len(state.events)),
            key=lambda position: (
                state.events[position]["last_used"],
                state.events[position]["seq"],
            ),
        )
        del state.events[victim]


def emit(
    state,
    kind: str,
    *,
    action: str = "",
    effect: str = "",
    source: dict | None = None,
    target: dict | None = None,
    skill_id: int | None = None,
    skill_index: int = -1,
    trait_id: int | None = None,
    attribute: str | int = "",
    value: int | float = 0,
    duration: str | int = "",
    stacks: int = 0,
    replay: bool = False,
    choice_branch: int = 0,
    public: bool = True,
    owner: str | None = None,
    line: str = "",
    logs: list | None = None,
    phase: str = "",
    extra: dict | None = None,
):
    source = source if source is not None else field_entity()
    target = target if target is not None else field_entity()
    if not isinstance(source, dict):
        source = pet_entity(state, source)
    if not isinstance(target, dict):
        target = pet_entity(state, target)
    current = _CURRENT.get()
    if current is not None and current["state"] is state:
        if skill_id is None:
            skill_id = current.get("skill_id")
        if trait_id is None:
            trait_id = current.get("trait_id")
        if not effect:
            effect = current.get("effect", "")
    if owner is None and not public:
        owner = source.get("side") or target.get("side")
    event = {
        "seq": state.event_seq,
        "turn": state.turn,
        "kind": kind,
        "phase": phase or state.event_phase,
        "action": action,
        "source": source,
        "target": target,
        "skill_id": skill_id,
        "skill_index": skill_index,
        "trait_id": trait_id,
        "effect": effect,
        "attribute": attribute,
        "value": value,
        "duration": duration,
        "stacks": stacks,
        "replay": bool(replay),
        "choice_branch": choice_branch,
        "public": bool(public),
        "line": line,
        "last_used": state.event_seq,
    }
    if owner is not None:
        event["owner"] = owner
    if extra:
        event.update(extra)
    state.event_seq += 1
    state.events.append(event)
    _touch(state, event)
    _trim(state)
    if line:
        (logs if logs is not None else state.log).append(line)
    return event


@contextmanager
def context(state, source: dict | None = None, target: dict | None = None, **fields):
    token = _CURRENT.set(
        {"state": state, "source": source, "target": target, **fields}
    )
    try:
        yield
    finally:
        _CURRENT.reset(token)


def current_context():
    return _CURRENT.get()


def snapshot(state) -> dict:
    """结算上下文内的状态快照，供 :func:`reconcile` 补齐未显式发出的事件。"""
    result = {}
    for side in ("A", "B"):
        for index, pet in enumerate(state.teams.get(side, [])):
            totals = {}
            durations = {}
            for buff in pet.buffs:
                totals[buff.buff_type] = totals.get(buff.buff_type, 0) + buff.value
                durations.setdefault(buff.buff_type, buff.duration)
            result[(side, index)] = {
                "hp": pet.hp,
                "max_hp": pet.max_hp,
                "energy": pet.energy,
                "spirit_id": pet.spirit_id,
                "buffs": totals,
                "durations": durations,
            }
    return result


def _pet_target_matches(event, side: str, index: int) -> bool:
    target = event.get("target") or {}
    return (
        target.get("kind") == PET
        and target.get("side") == side
        and target.get("index") == index
    )


def flush_freeze_loss(state, pet, logs: list | None = None) -> None:
    """冻结斩杀不是伤害，但实际减少了生命；在此统一补报。"""
    lost = pet.freeze_loss
    if lost <= 0:
        return
    pet.freeze_loss = 0
    emit(
        state,
        "life",
        action="loss",
        effect="freeze",
        source=system_entity(pet.side, "freeze"),
        target=pet_entity(state, pet),
        attribute="freeze",
        value=lost,
        logs=logs,
    )


def reconcile(
    state,
    before: dict,
    start_seq: int,
    *,
    source: dict | None = None,
    effect: str = "reconcile",
    logs: list | None = None,
    include_hp: bool = True,
) -> None:
    """把结算过程中未被显式事件覆盖的生命、能量、buff 与形态变化补成事件。

    只读取实际结算后的状态与本次结算新增的事件，不解析日志。
    """
    source = source if source is not None else field_entity()
    for (side, index), old in before.items():
        team = state.teams.get(side, [])
        if index >= len(team):
            continue
        pet = team[index]
        target = pet_entity(state, pet)
        hp = old["hp"]
        energy = old["energy"]
        buff_types = set()
        for event in state.events:
            if event["seq"] < start_seq or not _pet_target_matches(event, side, index):
                continue
            value = int(event.get("value") or 0)
            if event["kind"] == "damage" or (
                event["kind"] == "life" and event.get("action") == "loss"
            ):
                hp = max(0, hp - value)
            elif event["kind"] == "heal":
                hp = min(pet.max_hp, hp + value)
            elif event["kind"] == "energy":
                if event.get("action") == "gain":
                    energy += value
                elif event.get("action") == "loss":
                    energy -= value
            elif event["kind"] == "buff":
                buff_types.add(event.get("attribute"))
        if include_hp:
            if pet.freeze_loss > 0:
                hp = max(0, hp - pet.freeze_loss)
                flush_freeze_loss(state, pet, logs)
            if pet.spirit_id != old["spirit_id"]:
                emit(
                    state,
                    "evolve",
                    action="evolve",
                    effect="evolve",
                    source=target,
                    target=target,
                    attribute=pet.spirit_id,
                    value=old["spirit_id"],
                    extra={
                        "hp_before": old["hp"],
                        "max_hp_before": old["max_hp"],
                        "hp_after": pet.hp,
                    },
                )
            elif pet.max_hp == old["max_hp"]:
                residual = pet.hp - hp
                if residual > 0:
                    emit(
                        state,
                        "heal",
                        action="heal",
                        effect=effect,
                        source=source,
                        target=target,
                        value=residual,
                        logs=logs,
                    )
                elif residual < 0:
                    emit(
                        state,
                        "life",
                        action="loss",
                        effect=effect,
                        source=source,
                        target=target,
                        value=-residual,
                        logs=logs,
                    )
        elif pet.spirit_id != old["spirit_id"]:
            emit(
                state,
                "evolve",
                action="evolve",
                effect="evolve",
                source=target,
                target=target,
                attribute=pet.spirit_id,
                value=old["spirit_id"],
            )
        residual = pet.energy - energy
        if residual > 0:
            emit(
                state,
                "energy",
                action="gain",
                effect=effect,
                source=source,
                target=target,
                value=residual,
                logs=logs,
            )
        elif residual < 0:
            emit(
                state,
                "energy",
                action="loss",
                effect=effect,
                source=source,
                target=target,
                value=-residual,
                logs=logs,
            )
        totals = {}
        durations = {}
        for buff in pet.buffs:
            totals[buff.buff_type] = totals.get(buff.buff_type, 0) + buff.value
            durations.setdefault(buff.buff_type, buff.duration)
        for buff_type in dict.fromkeys((*old["buffs"], *totals)):
            if buff_type in buff_types:
                continue
            delta = totals.get(buff_type, 0) - old["buffs"].get(buff_type, 0)
            if delta > 0:
                emit(
                    state,
                    "buff",
                    action="apply",
                    effect=effect,
                    source=source,
                    target=target,
                    attribute=buff_type,
                    value=delta,
                    stacks=delta,
                    duration=durations.get(
                        buff_type, old["durations"].get(buff_type, "")
                    ),
                    logs=logs,
                )
            elif delta < 0:
                emit(
                    state,
                    "buff",
                    action="remove",
                    effect=effect,
                    source=source,
                    target=target,
                    attribute=buff_type,
                    value=delta,
                    stacks=-delta,
                    duration=old["durations"].get(
                        buff_type, durations.get(buff_type, "")
                    ),
                    logs=logs,
                )



@contextmanager
def phase(state, value: str):
    previous = state.event_phase
    state.event_phase = value
    try:
        yield
    finally:
        state.event_phase = previous
