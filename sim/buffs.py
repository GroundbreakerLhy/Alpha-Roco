"""增减益系统：数据结构与回合末结算。"""

from __future__ import annotations

from dataclasses import dataclass, field

from .data_loader import load_typechart
from .evolution import apply_cute, can_cute
from .typechart import type_multiplier


class BuffType:
    ATK = "atk"
    SPATK = "spatk"
    DEF = "def"
    SPDEF = "spdef"
    SKILL_POWER_PERCENT = "skill_power_percent"
    NEXT_ATTACK_POWER_PERCENT = "next_attack_power_percent"  # 一次性，攻击技能用时消耗
    NEXT_ATTACK_POWER_FLAT = "next_attack_power_flat"  # 一次性，攻击技能用时消耗
    LIFESTEAL = "lifesteal"
    HIT_COUNT_PERCENT = "hit_count_percent"
    SPEED = "speed"
    SPEED_PERCENT = "speed_percent"
    SKILL_POWER_FLAT = "skill_power_flat"
    HIT_COUNT_FLAT = "hit_count_flat"
    ENERGY_COST = "energy_cost"
    ATTACK_ENERGY_COST = "attack_energy_cost"  # 仅攻击技能生效
    OVERLOAD = "overload"
    PRIORITY = "priority"
    PRIORITY_DEBUFF = "priority_debuff"
    POISON = "poison"
    BURN = "burn"
    DIZZY = "dizzy"
    LEECH = "leech"
    FREEZE = "freeze"
    FREEZE_IMMUNE = "freeze_immune"
    BURN_IMMUNE = "burn_immune"
    LEECH_IMMUNE = "leech_immune"
    CUTE = "cute"
    LOCK = "lock"
    LIGHTNING = "lightning"


ALWAYS_BUFF_TYPES = {
    BuffType.LIFESTEAL,
    BuffType.FREEZE_IMMUNE,
    BuffType.BURN_IMMUNE,
    BuffType.LEECH_IMMUNE,
}

# 1%/层的 buff：钳制边界 ±990，与 10%/层 buff 的有效上限一致（B37）；其余 ±99。
PERCENT_LAYER_BUFF_TYPES = frozenset(
    {
        BuffType.SKILL_POWER_PERCENT,
        BuffType.NEXT_ATTACK_POWER_PERCENT,
        BuffType.HIT_COUNT_PERCENT,
    }
)


def layer_clamp_bound(buff_type: str) -> int:
    """单条增减益的层数钳制边界（B37）。"""
    if buff_type in PERCENT_LAYER_BUFF_TYPES or buff_type.startswith("power_boost_"):
        return 990
    return 99


# 减益 -> 对应免疫 buff（特性给予）
IMMUNE_BUFF_FOR = {
    BuffType.FREEZE: BuffType.FREEZE_IMMUNE,
    BuffType.BURN: BuffType.BURN_IMMUNE,
    BuffType.LEECH: BuffType.LEECH_IMMUNE,
}


def is_immune_to(pet, buff_type: str) -> bool:
    """该精灵是否免疫指定减益（特性给予的免疫 buff）。"""
    immune_type = IMMUNE_BUFF_FOR.get(buff_type)
    return immune_type is not None and get_buff_value(pet, immune_type) > 0


SIGNED_TYPES = {
    BuffType.ATK,
    BuffType.SPATK,
    BuffType.DEF,
    BuffType.SPDEF,
    BuffType.SKILL_POWER_PERCENT,
    BuffType.NEXT_ATTACK_POWER_PERCENT,
    BuffType.HIT_COUNT_PERCENT,
    BuffType.SPEED,
    BuffType.SPEED_PERCENT,
    BuffType.SKILL_POWER_FLAT,
    BuffType.NEXT_ATTACK_POWER_FLAT,
    BuffType.HIT_COUNT_FLAT,
    BuffType.OVERLOAD,
    BuffType.PRIORITY,
    BuffType.ENERGY_COST,
    BuffType.ATTACK_ENERGY_COST,
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
        if buff_type in (BuffType.ENERGY_COST, BuffType.ATTACK_ENERGY_COST):
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
    # 按元素威力增益：power_boost_<元素id>
    if buff_type.startswith("power_boost_"):
        return "buff" if value >= 0 else "debuff"
    # 按技能能耗修正：energy_cost_boost_<技能id>
    if buff_type.startswith("energy_cost_boost_"):
        return "debuff" if value > 0 else "buff"
    return "unknown"


def element_power_buff_type(element: int) -> str:
    """按元素的威力增益 buff 类型名。"""
    return f"power_boost_{element}"


def get_element_power_percent(pet, element: int) -> int:
    """该精灵对指定元素技能的威力百分比增益。"""
    return get_buff_value(pet, element_power_buff_type(element))


def skill_energy_boost_type(skill_id: int) -> str:
    """按技能能耗修正的 buff 类型名。"""
    return f"energy_cost_boost_{skill_id}"


def get_skill_energy_boost(pet, skill_id: int) -> int:
    """该精灵对指定技能的能耗修正。"""
    return get_buff_value(pet, skill_energy_boost_type(skill_id))


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
    PERMANENT = "permanent"
    NORMAL = "normal"
    TEMPORARY = "temporary"


@dataclass
class Buff:
    buff_type: str
    value: int
    duration: str = DurationKind.NORMAL
    source_side: str = ""
    source_pet: str = ""
    expire_turn: int | None = None
    applied_turn: int = 0  # 获得时的回合（禁足等按回合递减的效果据此跳过当回合）
    source_kind: str = ""  # ""=技能/常规；"debuff"=特性施加的减益；"trait"=特性展示

    def is_gain(self) -> bool:
        """是否属于增益；source_kind="trait" 不算。"""
        if self.source_kind == "trait":
            return False
        return classify_buff(self.buff_type, self.value) in ("buff", "both")

    def is_loss(self) -> bool:
        """是否属于减益；source_kind="trait" 不算。"""
        if self.source_kind == "trait":
            return False
        return classify_buff(self.buff_type, self.value) in ("debuff", "both")


def _allows_cute_reapplication(pet) -> bool:
    """已有萌化且当前特性允许再获得萌化时，不要求还能退化。"""
    if not has_buff(pet, BuffType.CUTE):
        return False
    from .traits.registry import get_handler

    handler = get_handler(pet.trait_id)
    return handler is not None and handler.allow_cute_reapplication(pet)


def _allows_cute_beyond_limit(pet) -> bool:
    """特性允许萌化层数超过可退化次数（无忧无虑 200182）。"""
    from .traits.registry import get_handler

    handler = get_handler(pet.trait_id)
    return handler is not None and handler.allow_cute_beyond_limit(pet)


def can_gain_buff(pet, buff_type: str, value: int) -> bool:
    """获得规则探测，不修改状态；必须在调用 add_buff 前执行。"""
    if value == 0 or (buff_type == BuffType.LIFESTEAL and value < 0):
        return False
    if is_immune_to(pet, buff_type):
        return False
    elemental_immunity = {
        BuffType.FREEZE: 6,
        BuffType.BURN: 2,
        BuffType.POISON: 9,
        BuffType.LEECH: 1,
        BuffType.LIGHTNING: 8,
    }
    element = elemental_immunity.get(buff_type)
    if element is not None and element in pet.attributes:
        return False
    if buff_type == BuffType.LOCK and has_buff(pet, BuffType.LOCK):
        return False
    if buff_type == BuffType.CUTE:
        return value > 0 and (
            _allows_cute_reapplication(pet)
            or _allows_cute_beyond_limit(pet)
            or can_cute(pet)
        )
    return True


def grant_buff(
    pet,
    buff_type: str,
    value: int,
    duration: str = DurationKind.NORMAL,
    current_turn: int | None = None,
    source_side: str = "",
    source_pet: str = "",
    source_kind: str = "",
    state=None,
    turns: int | None = None,
) -> int:
    """规则层入口：先拦截/执行即时效果，再调用纯状态写入；返回实际获得层数。

    不广播 buff_gain；调用方只在返回值非零时广播，特性内部仍可不广播。
    """
    if not can_gain_buff(pet, buff_type, value):
        return 0
    if source_kind != "trait" and is_buff(buff_type, value):
        from .traits.registry import get_handler

        handler = get_handler(pet.trait_id)
        if handler is not None:
            extra_layers = handler.modify_buff_gain_layers(pet, buff_type, value)
            value += extra_layers if value > 0 else -extra_layers
    if buff_type == BuffType.CUTE:
        applied = 0
        for _ in range(value):
            # 可再获得萌化：只加状态不退化，避免重绑定特性丢失本次响应者。
            if _allows_cute_reapplication(pet):
                applied += 1
                continue
            if can_cute(pet):
                apply_cute(pet)
                applied += 1
                continue
            # 层数不受可退化次数限制：形态不变，层数照加
            if _allows_cute_beyond_limit(pet):
                applied += 1
                continue
            break
        value = applied
        if value == 0:
            return 0
    before = get_buff_value(pet, buff_type)
    add_buff(
        pet,
        buff_type,
        value,
        duration,
        current_turn,
        source_side,
        source_pet,
        source_kind,
        turns=turns,
    )
    gained = get_buff_value(pet, buff_type) - before
    if gained and buff_type == BuffType.FREEZE:
        check_freeze(pet)
    if gained and buff_type == BuffType.LIGHTNING:
        if get_buff_value(pet, BuffType.LIGHTNING) >= 2:
            _trigger_lightning(pet, state)
    return gained


def add_buff(
    pet,
    buff_type: str,
    value: int,
    duration: str = DurationKind.NORMAL,
    current_turn: int | None = None,
    source_side: str = "",
    source_pet: str = "",
    source_kind: str = "",
    turns: int | None = None,
) -> None:
    if value == 0:
        return

    if buff_type != BuffType.LOCK and buff_type != BuffType.CUTE:
        current = get_buff_value(pet, buff_type)
        bound = layer_clamp_bound(buff_type)
        new_total = max(-bound, min(bound, current + value))
        if new_total == 0:
            remove_buff(pet, buff_type)
            return
        if new_total == current:
            return
        value = new_total - current

    if buff_type == BuffType.LOCK:
        pet.buffs.append(
            Buff(
                buff_type=buff_type,
                value=value,
                duration=duration,
                source_side=source_side,
                source_pet=source_pet,
                applied_turn=current_turn or 0,
                source_kind=source_kind,
            )
        )
        return

    if buff_type in (BuffType.FREEZE, BuffType.CUTE):
        duration = DurationKind.PERMANENT
    if turns is not None:
        duration = DurationKind.TEMPORARY
        expire_turn = (current_turn or 0) + turns
    else:
        expire_turn = (
            current_turn + 1
            if duration == DurationKind.TEMPORARY and current_turn is not None
            else None
        )
    pet.buffs.append(
        Buff(
            buff_type=buff_type,
            value=value,
            duration=duration,
            source_side=source_side,
            source_pet=source_pet,
            expire_turn=expire_turn,
            source_kind=source_kind,
        )
    )


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
        b
        for b in pet.buffs
        if b.duration != DurationKind.TEMPORARY
        or b.expire_turn is None
        or b.expire_turn > current_turn
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


def _apply_damage(pet, damage: int, current_turn: int | None = None) -> None:
    """持续伤害结算；技能侧免死对中毒/灼烧/寄生生效，冰冻阈值判定不受保护。"""
    if (
        current_turn is not None
        and damage >= pet.hp
        and getattr(pet, "survive_lethal_turn", 0) == current_turn
    ):
        pet.hp = 1
        return
    pet.hp = max(0, pet.hp - damage)


def _find_source_pet(state, source_side: str, source_pet: str):
    if not source_side or not source_pet:
        return None
    for pet in state.teams.get(source_side, []):
        if pet.name == source_pet:
            return pet
    return None


def _trigger_lightning(pet, state=None) -> None:
    if _immune(pet, 8):
        remove_buff(pet, BuffType.LIGHTNING)
        return
    typechart = load_typechart()
    mult = type_multiplier(8, pet.attributes, typechart)
    damage = int(pet.max_hp * 0.25 * mult)
    hp_before = pet.hp
    _apply_damage(pet, damage)
    lost = hp_before - pet.hp
    if state is not None and lost > 0:
        state.log.append(f"{pet.side} {pet.name} 受到引电 {lost} 伤害")
    remove_buff(pet, BuffType.LIGHTNING)


def on_round_end(state) -> None:
    typechart = load_typechart()
    order = list(getattr(state, "round_end_order", ["A", "B"]))

    for side in order:
        for pet in state.teams[side]:
            if pet.hp <= 0:
                continue

            _apply_poison(pet, typechart, state)
            _apply_burn(pet, typechart, state)
            _apply_leech(pet, state, typechart)
            _apply_dizzy(pet)
            _apply_lock(pet, state.turn)
            expire_temporary_buffs(pet, state.turn)


def _apply_poison(pet, typechart, state=None) -> None:
    layers = get_buff_value(pet, BuffType.POISON)
    if layers <= 0 or _immune(pet, 9):
        return
    mult = type_multiplier(9, pet.attributes, typechart)
    damage = int(pet.max_hp * 0.03 * layers * mult)
    hp_before = pet.hp
    _apply_damage(pet, damage, state.turn if state is not None else None)
    lost = hp_before - pet.hp
    if lost > 0 and state is not None:
        state.log.append(f"{pet.side} {pet.name} 受到中毒 {lost} 伤害")
        from . import traits  # 延迟导入，避免与 traits.impl 循环依赖

        traits.emit(
            state,
            "dot_damage",
            scope="all",
            side=pet.side,
            subject=pet,
            damage=damage,
            dot_type="poison",
        )


def _apply_burn(pet, typechart, state=None) -> None:
    layers = get_buff_value(pet, BuffType.BURN)
    if layers <= 0 or _immune(pet, 2):
        return
    if is_immune_to(pet, BuffType.BURN):
        return
    mult = type_multiplier(2, pet.attributes, typechart)
    damage = int(pet.max_hp * 0.02 * layers * mult)
    hp_before = pet.hp
    _apply_damage(pet, damage, state.turn if state is not None else None)
    lost = hp_before - pet.hp
    if lost > 0 and state is not None:
        state.log.append(f"{pet.side} {pet.name} 受到灼烧 {lost} 伤害")
        from . import traits  # 延迟导入，避免与 traits.impl 循环依赖

        traits.emit(
            state,
            "dot_damage",
            scope="all",
            side=pet.side,
            subject=pet,
            damage=damage,
            dot_type="burn",
        )
    if state is not None:
        from . import traits  # 延迟导入，避免与 traits.impl 循环依赖

        if traits.query_burn_growth(state, pet):
            new_layers = layers * 2
        else:
            new_layers = layers // 2
    else:
        new_layers = layers // 2
    remove_buff(pet, BuffType.BURN)
    if new_layers > 0:
        pet.buffs.append(
            Buff(
                buff_type=BuffType.BURN, value=new_layers, duration=DurationKind.NORMAL
            )
        )


def _apply_leech(pet, state, typechart) -> None:
    layers = get_buff_value(pet, BuffType.LEECH)
    if layers <= 0 or _immune(pet, 1):
        return
    if is_immune_to(pet, BuffType.LEECH):
        return
    mult = type_multiplier(1, pet.attributes, typechart)
    damage = int(pet.max_hp * 0.02 * layers * mult)
    hp_before = pet.hp
    _apply_damage(pet, damage, state.turn if state is not None else None)
    lost = hp_before - pet.hp
    if lost > 0:
        state.log.append(f"{pet.side} {pet.name} 受到寄生 {lost} 伤害")
        from . import traits  # 延迟导入，避免与 traits.impl 循环依赖

        traits.emit(
            state,
            "dot_damage",
            scope="all",
            side=pet.side,
            subject=pet,
            damage=damage,
            dot_type="leech",
        )
    source = _find_source_pet(
        state,
        pet.buffs[0].source_side if pet.buffs else "",
        pet.buffs[0].source_pet if pet.buffs else "",
    )
    if source is not None and source.hp > 0:
        source.hp = min(source.max_hp, source.hp + damage)


def check_freeze(pet) -> None:
    """实时冻结斩杀判定：生命变动或获得冻结后立即检查，不属于伤害结算。"""
    if pet.hp <= 0:
        return
    layers = get_buff_value(pet, BuffType.FREEZE)
    if layers <= 0 or _immune(pet, 6):
        return
    if is_immune_to(pet, BuffType.FREEZE):
        return
    threshold = int(pet.max_hp * 0.05 * layers)
    if threshold >= pet.hp:
        # 绕过 hp 写入入口避免递归；冻结不是伤害，不走免死/减伤。
        object.__setattr__(pet, "hp", 0)


def _apply_dizzy(pet) -> None:
    layers = get_buff_value(pet, BuffType.DIZZY)
    if layers <= 0:
        return
    remove_buff(pet, BuffType.DIZZY)
    if layers > 1:
        pet.buffs.append(
            Buff(
                buff_type=BuffType.DIZZY, value=layers - 1, duration=DurationKind.NORMAL
            )
        )


def _apply_lock(pet, current_turn: int = 0) -> None:
    """禁足按回合递减：施加当回合不递减，"3 回合"=接下来 3 个回合结束各减 1。"""
    if not has_buff(pet, BuffType.LOCK):
        return
    for buff in pet.buffs:
        if buff.buff_type == BuffType.LOCK:
            if buff.applied_turn >= current_turn:
                return
            buff.value -= 1
            if buff.value <= 0:
                pet.buffs = [b for b in pet.buffs if b is not buff]
            return


def get_speed_value(pet, mark_speed_bonus: int = 0) -> int:
    base = (
        pet.stats["speed"] + get_buff_value(pet, BuffType.SPEED) * 10 + mark_speed_bonus
    )
    percent = get_buff_value(pet, BuffType.SPEED_PERCENT)
    return int(base * (1.0 + percent * 0.1))


def get_stat_multiplier(pet, stat: str) -> float:
    value = get_buff_value(pet, stat)
    return 1.0 + value * 0.1


def get_skill_power_modifier(pet) -> tuple:
    percent = get_buff_value(pet, BuffType.SKILL_POWER_PERCENT)
    flat = get_buff_value(pet, BuffType.SKILL_POWER_FLAT)
    return percent, flat


def consume_next_attack_power(pet) -> tuple:
    """取用一次性"下一次攻击技能威力"增益，返回 (百分比, 固定值) 并清空。

    无增益返回 (0, 0)；NORMAL 时长，离场清除（P11 增量语义）。
    """
    percent = get_buff_value(pet, BuffType.NEXT_ATTACK_POWER_PERCENT)
    flat = get_buff_value(pet, BuffType.NEXT_ATTACK_POWER_FLAT)
    if percent:
        remove_buff(pet, BuffType.NEXT_ATTACK_POWER_PERCENT)
    if flat:
        remove_buff(pet, BuffType.NEXT_ATTACK_POWER_FLAT)
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
