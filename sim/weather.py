"""Weather system for data/weather.json."""

from __future__ import annotations

from . import buffs
from .data_loader import load_typechart
from .typechart import type_multiplier

RAIN = 0
BLIZZARD = 1
SANDSTORM = 2
THUNDERSTORM = 3


def set_weather(state, weather_id) -> None:
    state.weather = weather_id
    state.log.append(f"天气变为 {weather_id}")


def clear_weather(state) -> None:
    state.weather = None
    state.log.append("天气消失")


def get_weather(state):
    return state.weather


def water_power_multiplier(weather) -> float:
    return 1.5 if weather == RAIN else 1.0


def sandstorm_energy_modifier(weather, skill_element: int) -> int:
    if weather == SANDSTORM and skill_element == 5:
        return -2
    return 0


def on_round_end(state) -> None:
    order = ["A", "B"] if state.home_side == "A" else ["B", "A"]

    if state.weather == BLIZZARD:
        # 每回合为敌方叠加1层冰冻；按主客场顺序结算
        for side in order:
            opponent_side = "B" if side == "A" else "A"
            pet = state.teams[side][state.active[side]]
            target = state.teams[opponent_side][state.active[opponent_side]]
            if pet.hp > 0 and target.hp > 0:
                buffs.add_buff(target, buffs.BuffType.FREEZE, 1, buffs.DurationKind.PERMANENT)
                state.log.append(f"暴风雪：{target.name} 获得1层冻结")

    if state.weather == THUNDERSTORM:
        # 双方每回合结束获得1层引电；电系精灵免疫；按主客场顺序结算
        for side in order:
            pet = state.teams[side][state.active[side]]
            if pet.hp <= 0 or 8 in pet.attributes:
                continue
            buffs.add_buff(pet, buffs.BuffType.LIGHTNING, 1)
            state.log.append(f"雷鸣：{pet.name} 获得1层引电")
