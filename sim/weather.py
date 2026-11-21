"""天气系统（data/weather.json）。"""

from __future__ import annotations

from . import buffs, events, traits

RAIN = 0
BLIZZARD = 1
SANDSTORM = 2
THUNDERSTORM = 3

# 天气对应系别（"本技能系别和天气相同"类技能读取）。
WEATHER_ELEMENTS = {
    RAIN: 3,
    BLIZZARD: 6,
    SANDSTORM: 5,
    THUNDERSTORM: 8,
}


def element_of(weather) -> int | None:
    """当前天气对应的系别；无天气或未知天气返回 None。"""
    if weather is None:
        return None
    weather_id = weather["id"] if isinstance(weather, dict) else weather
    return WEATHER_ELEMENTS.get(weather_id)


def set_weather(state, weather_id: int, turns: int = 1) -> None:
    state.weather = {"id": weather_id, "remaining": turns}
    events.emit(
        state,
        "weather",
        action="set",
        source=events.weather_entity(str(weather_id)),
        target=events.field_entity(),
        attribute=weather_id,
        value=weather_id,
        duration=turns,
        line=f"天气变为 {weather_id}，持续 {turns} 回合",
    )


def clear_weather(state) -> None:
    state.weather = None
    events.emit(
        state,
        "weather",
        action="clear",
        source=events.weather_entity(),
        target=events.field_entity(),
        line="天气消失",
    )


def get_weather(state):
    return state.weather


def get_weather_id(state):
    if state.weather is None:
        return None
    return state.weather["id"]


def water_power_multiplier(weather) -> float:
    if weather is None:
        return 1.0
    weather_id = weather["id"] if isinstance(weather, dict) else weather
    return 1.5 if weather_id == RAIN else 1.0


def sandstorm_energy_modifier(weather, skill_element: int) -> int:
    if weather is None:
        return 0
    weather_id = weather["id"] if isinstance(weather, dict) else weather
    if weather_id == SANDSTORM and skill_element == 5:
        return -2
    return 0


def on_round_end(state) -> None:
    if state.weather is None:
        return
    order = list(getattr(state, "round_end_order", ["A", "B"]))
    weather_id = state.weather["id"]

    if weather_id == BLIZZARD:
        # 暴风雪：双方在场精灵各获得1层冻结（按 round_end_order）。
        for side in order:
            if state.active[side] < 0:
                continue
            pet = state.teams[side][state.active[side]]
            if pet.hp > 0:
                gained = buffs.grant_buff(
                    pet,
                    buffs.BuffType.FREEZE,
                    1,
                    buffs.DurationKind.PERMANENT,
                    state=state,
                )
                if gained == 0:
                    continue
                traits.on_buff_gain(state, pet, buffs.BuffType.FREEZE, gained)
                events.emit(
                    state,
                    "buff",
                    action="apply",
                    effect="weather",
                    source=events.weather_entity(str(weather_id)),
                    target=events.pet_entity(state, pet),
                    attribute=buffs.BuffType.FREEZE,
                    value=gained,
                    stacks=gained,
                    line=f"暴风雪：{pet.name} 获得1层冻结",
                )

    if weather_id == THUNDERSTORM:
        # 雷鸣：双方在场精灵各获得1层引电，电系免疫（按 round_end_order）。
        for side in order:
            if state.active[side] < 0:
                continue
            pet = state.teams[side][state.active[side]]
            if pet.hp <= 0 or 8 in pet.attributes:
                continue
            if not buffs.grant_buff(
                pet, buffs.BuffType.LIGHTNING, 1, state=state
            ):
                continue
            events.emit(
                state,
                "buff",
                action="apply",
                effect="weather",
                source=events.weather_entity(str(weather_id)),
                target=events.pet_entity(state, pet),
                attribute=buffs.BuffType.LIGHTNING,
                value=1,
                stacks=1,
                line=f"雷鸣：{pet.name} 获得1层引电",
            )

    state.weather["remaining"] -= 1
    if state.weather["remaining"] <= 0:
        events.emit(
            state,
            "weather",
            action="clear",
            source=events.weather_entity(str(weather_id)),
            target=events.field_entity(),
            line="天气消失",
        )
        state.weather = None
