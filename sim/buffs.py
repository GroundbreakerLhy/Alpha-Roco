"""Buff/debuff system: data structures and round-end resolution.

This module intentionally does not depend on battle flow details.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .data_loader import load_typechart
from .evolution import apply_cute
from .typechart import type_multiplier


class BuffType:
    ATK = "atk"                                 # 物攻
    SPATK = "spatk"                             # 魔攻
    DEF = "def"                                 # 物抗
    SPDEF = "spdef"                             # 魔抗
    SKILL_POWER_PERCENT = "skill_power_percent" # 技能威力_百分比
    LIFESTEAL = "lifesteal"                     # 吸血
    HIT_COUNT_PERCENT = "hit_count_percent"     # 连击数_百分比
    SPEED = "speed"                             # 速度
    SKILL_POWER_FLAT = "skill_power_flat"       # 技能威力_固定值
    HIT_COUNT_FLAT = "hit_count_flat"           # 连击数_固定值
    ENERGY_COST = "energy_cost"                 # 能耗
    OVERLOAD = "overload"                       # 超载
    PRIORITY = "priority"                       # 先手
    PRIORITY_DEBUFF = "priority_debuff"         # 先手减益
    POISON = "poison"                           # 中毒
    BURN = "burn"                               # 灼伤
    DIZZY = "dizzy"                             # 眩晕
    LEECH = "leech"                             # 寄生
    FREEZE = "freeze"                           # 冰冻
    CUTE = "cute"                               # 萌化
    LOCK = "lock"                               # 禁足
    LIGHTNING = "lightning"                     # 引电


ALWAYS_BUFF_TYPES = {
    BuffType.LIFESTEAL,
}

SIGNED_TYPES = {
    BuffType.ATK,
    BuffType.SPATK,
    BuffType.DEF,
    BuffType.SPDEF,
    BuffType.SKILL_POWER_PERCENT,
    BuffType.HIT_COUNT_PERCENT,
    BuffType.SPEED,
    BuffType.SKILL_POWER_FLAT,
    BuffType.HIT_COUNT_FLAT,
    BuffType.OVERLOAD,
    BuffType.PRIORITY,
    BuffType.ENERGY_COST,
}

DEBUFF_TYPES = {
    BuffType.PRIORITY_DEBUFF,
    BuffType.POISON,
    BuffType.BURN,
    BuffType.DIZZY,
    BuffType.LEECH,
    BuffType.FREEZE,
    BuffType.CUTE,
    BuffType.LOCK,
    BuffType.LIGHTNING,
}


def classify_buff(buff_type: str, value: int = 0) -> str:
    if buff_type in ALWAYS_BUFF_TYPES:
        return "buff"
    if buff_type in DEBUFF_TYPES:
        return "debuff"
    if buff_type in SIGNED_TYPES:
        if buff_type == BuffType.ENERGY_COST:
            if value > 0:
                return "debuff"
            if value < 0:
                return "buff"
            return "unknown"
        if value > 0:
            return "buff"
        if value < 0:
            return "debuff"
        return "unknown"
    return "unknown"


def is_buff(buff_type: str, value: int = 0) -> bool:
    return classify_buff(buff_type, value) in ("buff", "both")


def is_debuff(buff_type: str, value: int = 0) -> bool:
    return classify_buff(buff_type, value) in ("debuff", "both")


class DebuffType:
    POISON = BuffType.POISON
    BURN = BuffType.BURN
    DIZZY = BuffType.DIZZY
    LEECH = BuffType.LEECH
    FREEZE = BuffType.FREEZE
    CUTE = BuffType.CUTE
    LOCK = BuffType.LOCK
    LIGHTNING = BuffType.LIGHTNING


class DurationKind:
    PERMANENT = "permanent"                     # 永久
    NORMAL = "normal"                           # 常规
    TEMPORARY = "temporary"                     # 临时


@dataclass
class Buff:
    buff_type: str
    value: int
    duration: str = DurationKind.NORMAL
    source_side: str = ""
    source_pet: str = ""
    expire_turn: int | None = None


def add_buff(pet, buff_type: str, value: int, duration: str = DurationKind.NORMAL,
             current_turn: int | None = None, source_side: str = "", source_pet: str = "",
             _no_hook: bool = False) -> None:
    if value == 0:
        return

    if buff_type == BuffType.LIFESTEAL and value < 0:
        return

    if buff_type != BuffType.LOCK and buff_type != BuffType.CUTE:
        current = get_buff_value(pet, buff_type)
        new_total = max(-99, min(99, current + value))
        if new_total == 0:
            remove_buff(pet, buff_type)
            return
        if new_total == current:
            return
        value = new_total - current

    if buff_type == BuffType.LOCK:
        if has_buff(pet, BuffType.LOCK):
            return
        pet.buffs.append(Buff(
            buff_type=buff_type,
            value=value,
            duration=duration,
            source_side=source_side,
            source_pet=source_pet,
        ))
        return

    if buff_type == BuffType.CUTE:
        if not _no_hook:
            from . import features
            extra_cute = features.on_buff_gain(pet, buff_type, value)
            if extra_cute < 0:
                remove_buff(pet, BuffType.CUTE)
                return
        applied = 0
        for _ in range(value):
            if not apply_cute(pet):
                break
            applied += 1
        if applied > 0:
            pet.buffs.append(Buff(buff_type=BuffType.CUTE, value=applied, duration=DurationKind.PERMANENT))
        return

    if buff_type == BuffType.FREEZE:
        duration = DurationKind.PERMANENT
    expire_turn = current_turn + 1 if duration == DurationKind.TEMPORARY and current_turn is not None else None
    pet.buffs.append(Buff(
        buff_type=buff_type,
        value=value,
        duration=duration,
        source_side=source_side,
        source_pet=source_pet,
        expire_turn=expire_turn,
    ))

    if buff_type == BuffType.LIGHTNING and get_buff_value(pet, BuffType.LIGHTNING) >= 2:
        _trigger_lightning(pet)

    if not _no_hook:
        from . import features
        extra = features.on_buff_gain(pet, buff_type, value)
        if extra != 0:
            add_buff(pet, buff_type, extra, duration, current_turn, source_side, source_pet, _no_hook=True)


def remove_buff(pet, buff_type: str) -> None:
    pet.buffs = [b for b in pet.buffs if b.buff_type != buff_type]


def get_buff_value(pet, buff_type: str) -> int:
    return sum(b.value for b in pet.buffs if b.buff_type == buff_type)


def has_buff(pet, buff_type: str) -> bool:
    return any(b.buff_type == buff_type for b in pet.buffs)


def clear_normal_buffs(pet) -> None:
    pet.buffs = [b for b in pet.buffs if b.duration != DurationKind.NORMAL]


def expire_temporary_buffs(pet, current_turn: int) -> None:
    pet.buffs = [
        b for b in pet.buffs
        if b.duration != DurationKind.TEMPORARY or b.expire_turn is None or b.expire_turn > current_turn
    ]


def can_act(pet) -> bool:
    return get_buff_value(pet, BuffType.DIZZY) <= 0


def can_switch(pet) -> bool:
    return not has_buff(pet, BuffType.LOCK)


def add_overload(pet, skill_id: int, amount: int = 1) -> None:
    pet.overload_next[skill_id] = pet.overload_next.get(skill_id, 0) + amount


def get_overload(pet, skill_id: int) -> int:
    return pet.overload_current.get(skill_id, 0) + pet.overload_next.get(skill_id, 0)


def consume_overload(pet, skill_id: int) -> int:
    if pet.overload_current.get(skill_id, 0) > 0:
        pet.overload_current[skill_id] -= 1
        return 1
    if pet.overload_next.get(skill_id, 0) > 0:
        pet.overload_next[skill_id] -= 1
        return 1
    return 0


def start_turn_overload(pet) -> None:
    pet.overload_current = pet.overload_next
    pet.overload_next = {}


def _immune(pet, element: int) -> bool:
    return element in pet.attributes


def _apply_damage(pet, damage: int) -> None:
    pet.hp = max(0, pet.hp - damage)


def _find_source_pet(state, source_side: str, source_pet: str):
    if not source_side or not source_pet:
        return None
    for pet in state.teams.get(source_side, []):
        if pet.name == source_pet:
            return pet
    return None


def _trigger_lightning(pet) -> None:
    if _immune(pet, 8):
        remove_buff(pet, BuffType.LIGHTNING)
        return
    typechart = load_typechart()
    mult = type_multiplier(8, pet.attributes, typechart)
    damage = int(pet.max_hp * 0.25 * mult)
    _apply_damage(pet, damage)
    remove_buff(pet, BuffType.LIGHTNING)


def on_round_end(state) -> None:
    typechart = load_typechart()
    order = ["A", "B"] if state.home_side == "A" else ["B", "A"]

    for side in order:
        for pet in state.teams[side]:
            if pet.hp <= 0:
                continue

            from . import features
            poison_times = 2 if features.poison_extra_trigger(state) else 1
            for _ in range(poison_times):
                _apply_poison(pet, typechart, state)
            _apply_burn(pet, typechart, features.burn_mode(state, side), state)
            _apply_leech(pet, state, typechart)
            _apply_freeze(pet, typechart)
            _apply_dizzy(pet)
            _apply_lock(pet)
            expire_temporary_buffs(pet, state.turn)


def _apply_poison(pet, typechart, state=None) -> None:
    layers = get_buff_value(pet, BuffType.POISON)
    if layers <= 0 or _immune(pet, 9):
        return
    mult = type_multiplier(9, pet.attributes, typechart)
    damage = int(pet.max_hp * 0.03 * layers * mult)
    _apply_damage(pet, damage)
    if state is not None:
        _feature_heal_on_doom(state, pet, damage, "敌方受到中毒效果伤害时，自己回复等量生命")


def _apply_burn(pet, typechart, mode: str = "normal", state=None) -> None:
    layers = get_buff_value(pet, BuffType.BURN)
    if layers <= 0 or _immune(pet, 2):
        return
    mult = type_multiplier(2, pet.attributes, typechart)
    damage = int(pet.max_hp * 0.02 * layers * mult)
    _apply_damage(pet, damage)
    if state is not None:
        _feature_heal_on_doom(state, pet, damage, "敌方受到灼烧伤害时，自己回复等量生命")
    remove_buff(pet, BuffType.BURN)
    if mode == "grow":
        # 衰减变为增长:层数翻倍
        new_layers = layers * 2
        if new_layers > 0:
            pet.buffs.append(Buff(buff_type=BuffType.BURN, value=new_layers, duration=DurationKind.NORMAL))
    elif mode == "to_poison":
        # 衰减的灼烧变为相同层数的中毒
        if layers > 0:
            pet.buffs.append(Buff(buff_type=BuffType.POISON, value=layers, duration=DurationKind.NORMAL))
    else:
        new_layers = layers // 2
        if new_layers > 0:
            pet.buffs.append(Buff(buff_type=BuffType.BURN, value=new_layers, duration=DurationKind.NORMAL))


def _feature_heal_on_doom(state, pet, damage: int, key: str) -> None:
    """敌方因持续伤害(毒/灼烧)扣血时,若对方在场精灵特性匹配则回复等量。"""
    if damage <= 0:
        return
    from . import features
    opp_side = "B" if pet.side == "A" else "A"
    idx = state.active[opp_side]
    if idx < 0:
        return
    opp = state.teams[opp_side][idx]
    if opp.hp <= 0:
        return
    if key in features.desc_of(opp):
        opp.hp = min(opp.max_hp, opp.hp + damage)
        state.log.append(f"  [特性] {opp.name} 回复 {damage} 生命(敌方受持续伤害)")


def _apply_leech(pet, state, typechart) -> None:
    layers = get_buff_value(pet, BuffType.LEECH)
    if layers <= 0 or _immune(pet, 1):
        return
    mult = type_multiplier(1, pet.attributes, typechart)
    damage = int(pet.max_hp * 0.02 * layers * mult)
    _apply_damage(pet, damage)
    source = _find_source_pet(state, pet.buffs[0].source_side if pet.buffs else "", pet.buffs[0].source_pet if pet.buffs else "")
    if source is not None and source.hp > 0:
        source.hp = min(source.max_hp, source.hp + damage)


def _apply_freeze(pet, typechart) -> None:
    layers = get_buff_value(pet, BuffType.FREEZE)
    if layers <= 0 or _immune(pet, 6):
        return
    mult = type_multiplier(6, pet.attributes, typechart)
    damage = int(pet.max_hp * 0.05 * layers * mult)
    _apply_damage(pet, damage)


def _apply_dizzy(pet) -> None:
    layers = get_buff_value(pet, BuffType.DIZZY)
    if layers <= 0:
        return
    remove_buff(pet, BuffType.DIZZY)
    if layers > 1:
        pet.buffs.append(Buff(buff_type=BuffType.DIZZY, value=layers - 1, duration=DurationKind.NORMAL))


def _apply_lock(pet) -> None:
    if not has_buff(pet, BuffType.LOCK):
        return
    for buff in pet.buffs:
        if buff.buff_type == BuffType.LOCK:
            buff.value -= 1
            if buff.value <= 0:
                pet.buffs = [b for b in pet.buffs if b is not buff]
            return


def get_speed_value(pet, mark_speed_bonus: int = 0) -> int:
    return pet.stats["speed"] + get_buff_value(pet, BuffType.SPEED) * 10 + mark_speed_bonus


def get_stat_multiplier(pet, stat: str) -> float:
    value = get_buff_value(pet, stat)
    return 1.0 + value * 0.1


def get_skill_power_modifier(pet) -> tuple:
    percent = get_buff_value(pet, BuffType.SKILL_POWER_PERCENT)
    flat = get_buff_value(pet, BuffType.SKILL_POWER_FLAT)
    return percent, flat


def get_hit_count_bonus(pet) -> tuple:
    flat = get_buff_value(pet, BuffType.HIT_COUNT_FLAT)
    percent = get_buff_value(pet, BuffType.HIT_COUNT_PERCENT)
    return flat, percent


def get_energy_cost_modifier(pet) -> int:
    return get_buff_value(pet, BuffType.ENERGY_COST)


def get_priority_bonus(pet) -> int:
    return get_buff_value(pet, BuffType.PRIORITY)


def get_lifesteal(pet) -> float:
    return get_buff_value(pet, BuffType.LIFESTEAL) * 0.1
