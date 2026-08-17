"""Weather system for data/weather.json."""

from __future__ import annotations

from . import buffs

RAIN = 0
BLIZZARD = 1
SANDSTORM = 2
THUNDERSTORM = 3


def set_weather(state, weather_id: int, turns: int = 1) -> None:
    state.weather = {"id": weather_id, "remaining": turns}
    state.log.append(f"天气变为 {weather_id}，持续 {turns} 回合")


def clear_weather(state) -> None:
    state.weather = None
    state.log.append("天气消失")


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
    order = ["A", "B"] if state.home_side == "A" else ["B", "A"]
    weather_id = state.weather["id"]

    if weather_id == BLIZZARD:
        # 暴风雪：双方当前出战精灵都获得1层冻结；按主客场顺序结算
        for side in order:
            pet = state.teams[side][state.active[side]]
            if pet.hp > 0:
                buffs.add_buff(pet, buffs.BuffType.FREEZE, 1, buffs.DurationKind.PERMANENT)
                state.log.append(f"暴风雪：{pet.name} 获得1层冻结")

    if weather_id == THUNDERSTORM:
        # 雷鸣：双方每回合结束获得1层引电；电系精灵免疫；按主客场顺序结算
        for side in order:
            pet = state.teams[side][state.active[side]]
            if pet.hp <= 0 or 8 in pet.attributes:
                continue
            buffs.add_buff(pet, buffs.BuffType.LIGHTNING, 1)
            state.log.append(f"雷鸣：{pet.name} 获得1层引电")

    state.weather["remaining"] -= 1
    if state.weather["remaining"] <= 0:
        state.log.append("天气消失")
        state.weather = None
