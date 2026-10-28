from __future__ import annotations

from alpha_roco.config import RewardConfig
from alpha_roco.losses import terminal_rewards
from sim.buffs import classify_buff

DEFAULT_REWARD = RewardConfig()


def _hp_ratio(pet: dict, own: bool) -> float:
    if not own and "hp_percent" in pet:
        return pet["hp_percent"] / 100.0
    max_hp = pet.get("max_hp")
    return pet.get("hp", max_hp) / max_hp if max_hp else 1.0


def _team_hp(pets: list[dict], own: bool, config: RewardConfig) -> float:
    if not pets:
        return 0.0
    return sum(_hp_ratio(pet, own) for pet in pets) / min(
        len(pets), config.team_size
    )


def _team_alive(pets: list[dict]) -> int:
    return sum(
        pet.get("hp", 0) > 0 or pet.get("hp_percent", 0) > 0 for pet in pets
    )


def _effect_score(pets: list[dict]) -> float:
    score = 0.0
    for pet in pets:
        for effect in pet.get("buffs", []):
            value = effect.get("value")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            kind = classify_buff(effect.get("type", ""), value)
            if kind == "buff":
                score += abs(value)
            elif kind == "debuff":
                score -= abs(value)
    return score / 10.0


def _effect_map(pets: list[dict]) -> dict[tuple[str, str, str], float]:
    result = {}
    for pet in pets:
        name = pet.get("name", "")
        for effect in pet.get("buffs", []):
            value = effect.get("value")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            result[
                (name, effect.get("type", ""), effect.get("source_kind", ""))
            ] = value
    return result


def _mark_map(marks: dict) -> dict[tuple[str, int], float]:
    result = {}
    for polarity in ("positive", "negative"):
        mark = marks.get(polarity)
        if mark is not None:
            result[(polarity, int(mark["id"]))] = float(mark.get("stacks", 0))
    return result


def _active_energy(state: dict, side: str) -> float:
    teams = state.get("teams", {})
    active = state.get("active", {}).get(side, -1)
    if active < 0 or active >= len(teams.get(side, [])):
        return 0.0
    return float(teams[side][active].get("energy") or 0.0)


def effect_gains(
    previous: dict, current: dict
) -> list[tuple[str, tuple[str, str, str], float]]:
    gains = []
    previous_own = previous.get("own_effects", {})
    for key, value in current.get("own_effects", {}).items():
        gained = value - previous_own.get(key, 0)
        if gained > 0 and classify_buff(key[1], value) == "buff":
            gains.append(("own", key, gained))
    previous_enemy = previous.get("enemy_effects", {})
    for key, value in current.get("enemy_effects", {}).items():
        gained = value - previous_enemy.get(key, 0)
        if gained > 0 and classify_buff(key[1], value) == "debuff":
            gains.append(("enemy", key, gained))
    return gains


def state_summary(
    state: dict, side: str, config: RewardConfig = DEFAULT_REWARD
) -> dict:
    other = "B" if side == "A" else "A"
    teams = state.get("teams", {})
    own = teams.get(side, [])
    enemy = teams.get(other, [])
    effect = _effect_score(own) - _effect_score(enemy)
    return {
        "own_hp": _team_hp(own, True, config),
        "enemy_hp": _team_hp(enemy, False, config),
        "own_alive": _team_alive(own),
        "enemy_alive": _team_alive(enemy),
        "effect": max(-config.effect_clip, min(config.effect_clip, effect)),
        "own_effects": _effect_map(own),
        "enemy_effects": _effect_map(enemy),
        "own_marks": _mark_map(state.get("marks", {}).get(side, {})),
        "enemy_marks": _mark_map(state.get("marks", {}).get(other, {})),
        "own_energy": _active_energy(state, side),
        "enemy_energy": _active_energy(state, other),
    }


def potential(summary: dict, config: RewardConfig = DEFAULT_REWARD) -> float:
    return (
        config.hp_potential * (summary["enemy_hp"] - summary["own_hp"])
        + config.alive_potential
        * (summary["enemy_alive"] - summary["own_alive"])
        / config.team_size
        + config.effect_potential * summary.get("effect", 0.0)
    )


def assign_rewards(episode: list[dict], winner: str, gamma: float = 1.0) -> None:
    reward = terminal_rewards(winner)[0]
    episode[0]["reward"] = reward + episode[0].get("lead_bonus", 0.0)
    episode[0]["terminal"] = True
    battle_entries = [entry for entry in episode if entry["phase"] != "lead"]
    for index, entry in enumerate(battle_entries):
        following = (
            gamma * battle_entries[index + 1]["potential"]
            if index + 1 < len(battle_entries)
            else 0.0
        )
        entry["reward"] = following - entry["potential"]
        entry["terminal"] = index + 1 == len(battle_entries)
    if battle_entries:
        battle_entries[-1]["reward"] += reward


def switch_action_penalty(
    phase: str,
    kind: str,
    first_action: bool,
    previous_kind: str | None,
    config: RewardConfig = DEFAULT_REWARD,
) -> float:
    if phase != "action" or kind != "switch":
        return 0.0
    if first_action:
        return -config.switch_first_turn_penalty
    if previous_kind == "switch":
        return -config.switch_consecutive_penalty
    return 0.0


def apply_exploration_bonus(
    episode: list[dict],
    config: RewardConfig = DEFAULT_REWARD,
    cap: float | None = None,
) -> None:
    remaining = cap if cap is not None else config.rare_skill_cap
    for entry in episode:
        raw = max(entry.get("exploration_bonus", 0.0), 0.0)
        exploration = raw if remaining is None else min(raw, remaining)
        penalty = min(entry.get("switch_penalty", 0.0), 0.0)
        entry["reward"] = entry.get("reward", 0.0) + exploration + penalty
        if remaining is not None:
            remaining -= exploration
