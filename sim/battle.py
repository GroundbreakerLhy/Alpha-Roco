"""Headless 6v6 battle loop and JSON state serialization."""

from __future__ import annotations

import random
import re

from . import buffs, burst, counter, features, marks, skill_utils
from .damage import calc_damage, level_coefficient
from .models import Action, BattlePet, BattleState
from .data_loader import load_typechart
from .typechart import type_multiplier

CHARGE_ENERGY = 5
MAGIC_START = 4
MAX_TURN = 50

# random.seed(42)


def create_battle(pet_a: BattlePet, pet_b: BattlePet) -> BattleState:
    return create_team_battle([pet_a], [pet_b])


def create_team_battle(team_a: list, team_b: list) -> BattleState:
    return BattleState(
        teams={"A": team_a, "B": team_b},
        active={"A": 0, "B": 0},
        magic={"A": MAGIC_START, "B": MAGIC_START},
        revealed={"A": set(), "B": set()},
        home_side=random.choice(["A", "B"]),
        turn=1,
        log=[],
    )


def _active_pet(state: BattleState, side: str) -> BattlePet | None:
    idx = state.active[side]
    if idx < 0:
        return None
    return state.teams[side][idx]


def _first_alive_index(state: BattleState, side: str):
    for i, pet in enumerate(state.teams[side]):
        if pet.hp > 0:
            return i
    return None


def _add_energy(state: BattleState, side: str, pet: BattlePet, amount: int, logs: list) -> None:
    features._energy(state, side, pet, amount, logs)


def _parse_reduction(desc: str) -> float:
    match = re.search(r"减伤(\d+)%", desc or "")
    if not match:
        return 0.0
    return int(match.group(1)) / 100.0


def _settle_defense_cooldowns(state) -> None:
    for side in ("A", "B"):
        for pet in state.teams[side]:
            defense_ids = [skill.skill_id for skill in pet.skills if skill.category == 2]
            if pet.defense_used_this_turn:
                for skill_id in defense_ids:
                    pet.defense_cooldowns.add(skill_id)
            else:
                for skill_id in defense_ids:
                    pet.defense_cooldowns.discard(skill_id)
            pet.defense_used_this_turn.clear()


def _apply_faint(state: BattleState, side: str) -> None:
    pet = _active_pet(state, side)
    if pet is None or pet.hp > 0:
        return
    state.active[side] = -1
    delta = features.faint_magic_delta(pet)
    state.magic[side] = max(0, state.magic[side] - 1 + delta)
    features.on_faint(state, side, state.log)
    state.log.append(f"{side} {pet.name} 倒下了，剩余魔力 {state.magic[side]}")
    if state.magic[side] <= 0:
        state.winner = "B" if side == "A" else "A"
        state.log.append(f"{state.winner} 获胜")


def _apply_turn_limit(state: BattleState) -> None:
    if state.winner is not None:
        return
    magic_a = state.magic["A"]
    magic_b = state.magic["B"]
    if magic_a != magic_b:
        state.winner = "A" if magic_a > magic_b else "B"
        state.log.append(f"达到{MAX_TURN + 1}回合，魔力值 {magic_a}:{magic_b}，{state.winner} 获胜")
        return
    hp_a = sum(pet.hp for pet in state.teams["A"])
    hp_b = sum(pet.hp for pet in state.teams["B"])
    if hp_a != hp_b:
        state.winner = "A" if hp_a > hp_b else "B"
        state.log.append(f"达到{MAX_TURN + 1}回合，剩余血量 {hp_a}:{hp_b}，{state.winner} 获胜")
        return
    state.winner = "draw"
    state.log.append(f"达到{MAX_TURN + 1}回合，魔力与剩余血量均相同，平局")


def _skill_causes_self_leave(skill) -> bool:
    desc = skill.desc or ""
    return any(key in desc for key in ("自己脱离", "自己离场", "紧急脱离"))


def _skill_causes_enemy_leave(skill) -> bool:
    desc = skill.desc or ""
    return any(key in desc for key in ("敌方脱离", "敌方紧急脱离", "使敌方精灵返场", "敌方精灵返场"))


# 离场换人通用结算：蓄电印记(6)迸发、棘刺印记(1)、降灵印记(5)
def _force_switch_after_leave(state: BattleState, side: str, logs: list) -> None:
    current_idx = state.active[side]
    if current_idx < 0:
        return
    outgoing = state.teams[side][current_idx]
    features.on_leave(state, side, logs)
    buffs.clear_normal_buffs(outgoing)
    burst.clear_bursts(outgoing)
    idx = None
    for i, pet in enumerate(state.teams[side]):
        if i != current_idx and pet.hp > 0:
            idx = i
            break
    if idx is None:
        return
    state.active[side] = idx
    incoming = state.teams[side][idx]
    incoming.has_acted_since_entry = False
    positive_mark = marks.find_mark(state, side, marks.POSITIVE, 6)
    if positive_mark is not None:
        burst.add_burst(incoming, "attack_power_flat", 10 * positive_mark["stacks"])
    logs.append(f"{side} 因脱离换上 {incoming.name}")
    features.apply_pending_entry(state, side, logs)
    features.on_entry(state, side, logs)
    negative = marks.find_mark(state, side, marks.NEGATIVE, 1)
    if negative is not None:
        loss = int(incoming.max_hp * 0.06 * negative["stacks"])
        incoming.hp = max(0, incoming.hp - loss)
        logs.append(f"  棘刺印记：{incoming.name} 入场失去 {loss} 生命")
    negative = marks.find_mark(state, side, marks.NEGATIVE, 5)
    if negative is not None:
        loss = negative["stacks"]
        incoming.energy = max(0, incoming.energy - loss)
        logs.append(f"  降灵印记：{incoming.name} 入场失去 {loss} 能量")


def _resolve_action(state: BattleState, side: str, action: Action, is_first: bool = False) -> list:
    logs = []
    team = state.teams[side]
    current = state.active[side]

    if current < 0:
        if action.kind == "switch":
            target = action.pet_index
            if target is not None and 0 <= target < len(team) and team[target].hp > 0:
                state.active[side] = target
                logs.append(f"{side} 换上 {team[target].name}")
                return logs
        logs.append(f"{side} 需要选择上场的精灵")
        return logs

    pet = team[current]

    if action.kind == "switch":
        if not buffs.can_switch(pet):
            logs.append(f"{side} {pet.name} 被禁足，无法主动换人")
            return logs
        target = action.pet_index
        if target is not None and 0 <= target < len(team) and target != current and team[target].hp > 0:
            # 背包随机精灵特性:只能与该随机精灵互换
            if pet.bag_pet_index is not None and target != pet.bag_pet_index:
                logs.append(f"{side} {pet.name} 只能与背包随机精灵互换")
                return logs
            features.on_leave(state, side, logs)
            buffs.clear_normal_buffs(pet)
            burst.clear_bursts(pet)
            state.active[side] = target
            incoming = team[target]
            incoming.has_acted_since_entry = False
            positive_mark = marks.get_mark(state, side, marks.POSITIVE)
            if positive_mark is not None and positive_mark["id"] == 6:
                burst.add_burst(incoming, "attack_power_flat", 10 * positive_mark["stacks"])
            logs.append(f"{side} 换上 {incoming.name}")
            features.apply_pending_entry(state, side, logs)
            features.on_entry(state, side, logs)
            # 印记 6：蓄电印记 —— 入场获得迸发
            # 印记 1：棘刺印记 —— 入场失去6%最大生命/层
            # 印记 5：降灵印记 —— 入场失去1能量/层
            negative = marks.get_mark(state, side, marks.NEGATIVE)
            if negative is not None and negative["id"] == 1:
                loss = int(incoming.max_hp * 0.06 * negative["stacks"])
                incoming.hp = max(0, incoming.hp - loss)
                logs.append(f"  棘刺印记：{incoming.name} 入场失去 {loss} 生命")
            if negative is not None and negative["id"] == 5:
                loss = negative["stacks"]
                incoming.energy = max(0, incoming.energy - loss)
                logs.append(f"  降灵印记：{incoming.name} 入场失去 {loss} 能量")
            return logs
        idx = _first_alive_index(state, side)
        if idx is not None and idx != current:
            state.active[side] = idx
            logs.append(f"{side} 换人无效，自动换上 {team[idx].name}")
        return logs

    opponent_side = "B" if side == "A" else "A"
    opponent = _active_pet(state, opponent_side)

    if not buffs.can_act(pet):
        logs.append(f"{side} {pet.name} 眩晕，无法行动")
        return logs

    if action.kind == "charge":
        pet.has_acted_since_entry = True
        active_bursts = features.take_bursts(pet)
        _add_energy(state, side, pet, CHARGE_ENERGY, logs)
        logs.append(f"{side} {pet.name} 聚能，回复 {CHARGE_ENERGY} 能量")
        features.on_charge(state, side, logs)
        return logs

    if action.kind != "skill" or action.skill_index is None:
        _add_energy(state, side, pet, CHARGE_ENERGY, logs)
        logs.append(f"{side} {pet.name} 行动无效，改为聚能")
        return logs

    if not (0 <= action.skill_index < len(pet.skills)):
        _add_energy(state, side, pet, CHARGE_ENERGY, logs)
        logs.append(f"{side} {pet.name} 技能索引无效，改为聚能")
        return logs

    skill = pet.skills[action.skill_index]
    # 特性技能位限制(仅可使用1号位 / 1、3号位技能)
    usable = features.usable_indices(pet)
    if usable is not None and action.skill_index not in usable:
        _add_energy(state, side, pet, CHARGE_ENERGY, logs)
        logs.append(f"{side} {pet.name} 该技能位被特性限制，改为聚能")
        return logs
    if skill.skill_id in pet.blocked_skills:
        _add_energy(state, side, pet, CHARGE_ENERGY, logs)
        logs.append(f"{side} {pet.name} {skill.name} 冷却中，改为聚能")
        return logs

    # 印记 2：蓄势印记 —— 全技能能耗+1/层；印记 11：湿润印记 —— 全技能能耗-1/层
    mark_energy_bonus = 0
    pos2 = marks.find_mark(state, side, marks.POSITIVE, 2)
    if pos2 is not None:
        mark_energy_bonus += pos2["stacks"]
    pos11 = marks.find_mark(state, side, marks.POSITIVE, 11)
    if pos11 is not None:
        mark_energy_bonus -= pos11["stacks"]
    feature_cost = features.modify_energy_cost(pet, skill, 0) + features.modify_energy_cost_state(state, side, skill, 0)
    total_cost = max(0, skill.energy_cost + buffs.get_energy_cost_modifier(pet) + mark_energy_bonus + feature_cost)
    if pet.energy < total_cost:
        d = features.desc_of(pet)
        if "能量不足时，消耗5%生命，代替1能量" in d or "能量不足时，消耗5%最大生命，代替1能量" in d:
            use_max = "最大生命" in d
            need = total_cost - pet.energy
            loss = 0
            ok = True
            for _ in range(need):
                base = pet.max_hp if use_max else pet.hp
                cost = max(1, int(base * 0.05))
                if pet.hp <= cost:
                    ok = False
                    break
                pet.hp -= cost
                loss += cost
                pet.energy += 1
            if not ok:
                logs.append(f"{side} {pet.name} 生命不足，无法以生命代替能量使用 {skill.name}")
                return logs
            pet.energy -= total_cost
            pet.energy_spent_total += total_cost
            logs.append(f"{side} {pet.name} 能量不足，消耗 {loss} 生命代替能量")
            if use_max and pet.hp < pet.max_hp * 0.5 and "生命低于50%时" in d:
                buffs.add_buff(pet, buffs.BuffType.LIFESTEAL, 10)
                logs.append(f"  [特性] {pet.name} 生命低于50%，获得吸血100%")
        else:
            logs.append(f"{side} {pet.name} 能量不足，无法使用 {skill.name}")
            return logs
    else:
        pet.energy -= total_cost
        pet.energy_spent_total += total_cost

    active_bursts = features.take_bursts(pet)
    # 印记 9：龙噬印记 —— 释放3能耗技能后双攻+30%/层
    positive_mark = marks.find_mark(state, side, marks.POSITIVE, 9)
    if positive_mark is not None and total_cost == 3:
        gain = 3 * positive_mark["stacks"]
        buffs.add_buff(pet, buffs.BuffType.ATK, gain)
        buffs.add_buff(pet, buffs.BuffType.SPATK, gain)
        logs.append(f"  龙噬印记：双攻+{gain * 10}%")
    state.revealed[side].add(skill.skill_id)
    logs.append(f"{side} {pet.name} 使用 {skill.name}")

    if opponent is None:
        logs.append("  对方没有在场精灵")
        return logs

    if skill.category in (0, 1) and skill.power is not None:
        # 印记 0：攻击印记 —— 全技能威力+10%/层
        # 印记 2：蓄势印记 —— 全技能威力+30%/层
        # 印记 12：风气印记 —— 先手攻击时技能威力+20%/层
        extra_power_percent = 0.0
        extra_power_flat = 0.0
        pos0 = marks.find_mark(state, side, marks.POSITIVE, 0)
        if pos0 is not None:
            extra_power_percent += 10.0 * pos0["stacks"]
        pos2 = marks.find_mark(state, side, marks.POSITIVE, 2)
        if pos2 is not None:
            extra_power_percent += 30.0 * pos2["stacks"]
        pos12 = marks.find_mark(state, side, marks.POSITIVE, 12)
        if pos12 is not None and is_first:
            extra_power_percent += 20.0 * pos12["stacks"]
        for active_burst in active_bursts:
            if active_burst["type"] == "attack_power_flat":
                extra_power_flat += active_burst["value"]
        # 特性威力修正
        extra_power_percent += features.modify_power_percent(pet, skill, 0.0, is_first)
        extra_power_percent += features.modify_power_percent_state(state, side, skill, is_first)
        if features._fs(pet, "next_attack_x2", False):
            extra_power_percent += 100.0
            features._fss(pet, "next_attack_x2", False)
        if features._fs(pet, "first_attack_pending", False) and "携带的攻击技能获得迸发：威力+40" in features.desc_of(pet):
            extra_power_flat += 40
        pet.has_acted_since_entry = True
        eff_elem = features.effective_element(pet, skill)
        result = calc_damage(
            pet, opponent, skill, load_typechart(),
            opponent.defense_reduction,
            extra_power_percent=extra_power_percent,
            extra_power_flat=extra_power_flat,
            state=state,
            element_override=eff_elem,
        )
        hit_flat, hit_percent = buffs.get_hit_count_bonus(pet)
        hit_count = max(1, 1 + hit_flat + int(1 * hit_percent / 100))
        hit_count = features.modify_hit_count(pet, skill, hit_count)
        hit_count = features.modify_hit_count_state(state, side, hit_count, skill)
        total_damage = result["damage"] * hit_count
        if total_damage > 0:
            adjusted = features.on_take_damage(state, opponent_side, side, skill, total_damage, True, logs)
            if adjusted > 0:
                opponent.hp = max(0, opponent.hp - adjusted)
                logs.append(f"  造成 {adjusted} 伤害")
                features.on_deal_damage(state, side, skill, adjusted, result["type_multiplier"], logs)
                lifesteal = buffs.get_lifesteal(pet)
                if lifesteal > 0:
                    heal = int(adjusted * lifesteal)
                    if heal > 0:
                        actual = features.on_heal(state, side, pet, heal, logs)
                        if actual > 0:
                            pet.hp = min(pet.max_hp, pet.hp + actual)
                            logs.append(f"  吸血回复 {actual}")
            else:
                logs.append("  伤害被特性免疫")
        else:
            logs.append("  没有造成伤害")
        if opponent.hp <= 0:
            features.on_kill(state, side, logs)

        negative_mark = marks.find_mark(state, opponent_side, marks.NEGATIVE, 7)
        # 印记 7：星陨印记 —— 非幻系攻击触发额外幻系伤害
        if negative_mark is not None and eff_elem != 17 and opponent.hp > 0:
            n = negative_mark["stacks"]
            consume = n
            if "触发星陨印记时仅消耗一半层数" in features.desc_of(pet):
                consume = n - n // 2
            # 星陨伤害同样吃技能威力类 buff/印记（含攻击印记、蓄势印记、迸发等）
            power_percent_buff, power_flat_buff = buffs.get_skill_power_modifier(pet)
            star_power = (n * n + 24 * n - 24 + extra_power_flat + power_flat_buff) * (
                1.0 + (extra_power_percent + power_percent_buff) / 100.0
            )
            atk_key = "atk" if skill.category == 0 else "spatk"
            def_key = "def" if skill.category == 0 else "spdef"
            atk_stat = (pet.stats[atk_key] * buffs.get_stat_multiplier(pet, atk_key)
                        * features.modify_stat(pet, atk_key, 1.0)
                        * features.modify_stat_state(state, side, atk_key, 1.0))
            def_stat = (opponent.stats[def_key] * buffs.get_stat_multiplier(opponent, def_key)
                        * features.modify_stat(opponent, def_key, 1.0)
                        * features.modify_stat_state(state, opponent_side, def_key, 1.0))
            star_eff = type_multiplier(17, opponent.attributes, load_typechart())
            star_dmg = int(
                star_power * (atk_stat / def_stat) * level_coefficient(pet.level)
                * star_eff * (1 - opponent.defense_reduction)
            )
            if star_dmg > 0:
                opponent.hp = max(0, opponent.hp - star_dmg)
                logs.append(f"  星陨印记触发，造成 {star_dmg} 幻系伤害")
                if opponent.hp <= 0:
                    features.on_kill(state, side, logs)
            marks.remove_mark(state, opponent_side, 7)
            if consume > 0:
                features._add_mark(state, side, opponent_side, 7, consume, logs)
            logs.append(f"  星陨印记触发，消耗 {consume} 层")
    elif skill.category == 2:
        pet.has_acted_since_entry = True
        active_bursts = features.take_bursts(pet)
        if skill.skill_id in pet.defense_cooldowns:
            logs.append(f"  {skill.name} 冷却中")
            return logs
        reduction = _parse_reduction(skill.desc)
        pet.defense_reduction = reduction
        pet.defense_used_this_turn.add(skill.skill_id)
        logs.append(f"  防御，减伤 {int(reduction * 100)}%")
    else:
        pet.has_acted_since_entry = True
        active_bursts = features.take_bursts(pet)
        logs.append("  非伤害技能：效果暂未实现")

    want_switch = features.after_skill_use(state, side, skill, is_first, logs)
    if want_switch:
        _force_switch_after_leave(state, side, logs)
    if _skill_causes_self_leave(skill):
        _force_switch_after_leave(state, side, logs)
    if _skill_causes_enemy_leave(skill) and opponent is not None:
        _force_switch_after_leave(state, opponent_side, logs)

    return logs


# 印记 3：减速印记 —— 每层速度-10
def _effective_speed(state: BattleState, side: str) -> int:
    pet = _active_pet(state, side)
    if pet is None:
        return -1
    mark_speed_bonus = 0
    negative = marks.find_mark(state, side, marks.NEGATIVE, 3)
    if negative is not None:
        mark_speed_bonus = -10 * negative["stacks"]
    speed = buffs.get_speed_value(pet, mark_speed_bonus)
    speed = features.modify_speed(pet, speed)
    speed = features.modify_speed_state(state, side, speed)
    return speed


def round_end_order(state: BattleState) -> list:
    return ["A", "B"] if state.home_side == "A" else ["B", "A"]


def _action_skill_of(state: BattleState, side: str, action: Action):
    if action.kind != "skill" or action.skill_index is None:
        return None
    pet = state.teams[side][state.active[side]]
    if not (0 <= action.skill_index < len(pet.skills)):
        return None
    return pet.skills[action.skill_index]


def _round_end_settlement(state: BattleState) -> None:
    """回合结束统一结算:防御冷却、印记/buff/特性回合结束效果、力竭判定。"""
    _settle_defense_cooldowns(state)
    if features.round_end_disabled(state):
        state.log.append("  在场特性使双方回合结束效果不触发")
    else:
        times = 1 + features.round_end_extra(state)
        for _ in range(times):
            marks.on_round_end(state)
            buffs.on_round_end(state)
        for side in ("A", "B"):
            pet = _active_pet(state, side)
            if pet is not None and pet.hp > 0:
                want = features.on_turn_end(state, side, state.log)
                if want:
                    _force_switch_after_leave(state, side, state.log)
    for side in ("A", "B"):
        _apply_faint(state, side)


def step(state: BattleState, action_a: Action, action_b: Action) -> BattleState:
    if state.winner is not None:
        return state

    state.turn += 1
    state.log = []

    for team in state.teams.values():
        for pet in team:
            pet.defense_reduction = 0.0
            pet.defense_used_this_turn.clear()
            buffs.start_turn_overload(pet)

    # 回合开始:技能冷却衰减、复活、首次入场、回合开始特性
    for side in ("A", "B"):
        for pet in state.teams[side]:
            if pet.hp <= 0:
                continue
            turns = pet.feature_state.setdefault("blocked_turns", {})
            new_blocked = set()
            for sid in pet.blocked_skills:
                turns[sid] = turns.get(sid, 1) - 1
                if turns[sid] > 0:
                    new_blocked.add(sid)
            pet.blocked_skills = new_blocked
    features.check_revive(state, "A", state.log)
    features.check_revive(state, "B", state.log)
    for side in ("A", "B"):
        pet = _active_pet(state, side)
        if pet is not None and pet.hp > 0:
            if pet.entry_count == 0:
                features.on_entry(state, side, state.log)
            features.on_turn_start(state, side, state.log)

    for side, action in (("A", action_a), ("B", action_b)):
        if action.kind == "flee":
            state.winner = "B" if side == "A" else "A"
            state.log.append(f"{side} 逃跑了，{state.winner} 获胜")
            return state

    # 切换精灵优先级最高：先处理所有换人，再处理攻击/聚能
    actions = {"A": action_a, "B": action_b}
    for side in ("A", "B"):
        if actions[side].kind == "switch":
            state.log.extend(_resolve_action(state, side, actions[side]))
            if state.winner is not None:
                return state

    remaining = {side: action for side, action in actions.items() if action.kind != "switch"}
    if not remaining:
        _round_end_settlement(state)
        if state.winner is not None:
            return state
        if state.turn > MAX_TURN:
            _apply_turn_limit(state)
        return state

    a = _active_pet(state, "A")
    b = _active_pet(state, "B")
    a_speed = _effective_speed(state, "A")
    b_speed = _effective_speed(state, "B")
    # 迅捷:使用迅捷技能时该侧先手(+100)
    if "A" in remaining and a is not None:
        sk = _action_skill_of(state, "A", remaining["A"])
        if sk is not None and (features.skill_is_swift(a, sk) or features.skill_is_swift_state(state, "A", a, sk)):
            a_speed += 100
    if "B" in remaining and b is not None:
        sk = _action_skill_of(state, "B", remaining["B"])
        if sk is not None and (features.skill_is_swift(b, sk) or features.skill_is_swift_state(state, "B", b, sk)):
            b_speed += 100

    # 应对：按独立应对模块判定强制先手；其余按速度/先手决定
    forced = None
    if "A" in remaining and "B" in remaining:
        forced = counter.forced_first(state, remaining["A"], remaining["B"])
    if forced == "A":
        first, second = "A", "B"
        features.on_counter_success(
            state, "A",
            _action_skill_of(state, "A", remaining["A"]),
            _action_skill_of(state, "B", remaining["B"]),
            state.log,
        )
    elif forced == "B":
        first, second = "B", "A"
        features.on_counter_success(
            state, "B",
            _action_skill_of(state, "B", remaining["B"]),
            _action_skill_of(state, "A", remaining["A"]),
            state.log,
        )
    else:
        pa = buffs.get_priority_bonus(a) if a is not None else 0
        pb = buffs.get_priority_bonus(b) if b is not None else 0
        if pa != pb:
            first, second = ("A", "B") if pa > pb else ("B", "A")
        elif a_speed > b_speed:
            first, second = "A", "B"
        elif b_speed > a_speed:
            first, second = "B", "A"
        else:
            first, second = ("A", "B") if random.random() < 0.5 else ("B", "A")

    if first in remaining:
        state.log.extend(_resolve_action(state, first, remaining[first], is_first=True))
        _apply_faint(state, "B" if first == "A" else "A")
        if state.winner is not None:
            return state

    if second in remaining and state.active[second] >= 0:
        state.log.extend(_resolve_action(state, second, remaining[second], is_first=False))
        _apply_faint(state, "B" if second == "A" else "A")
        if state.winner is not None:
            return state

    _round_end_settlement(state)
    if state.winner is not None:
        return state

    if state.turn > MAX_TURN:
        _apply_turn_limit(state)

    return state


def pet_to_dict(pet: BattlePet, opponent: BattlePet | None = None, typechart: dict | None = None) -> dict:
    skills = []
    for i, skill in enumerate(pet.skills):
        item = {
            "index": i,
            "skill_id": skill.skill_id,
            "name": skill.name,
            "element": skill.element,
            "category": skill.category,
            "power": skill.power,
            "energy_cost": skill.energy_cost,
            "desc": skill.desc,
            "display_power": None,
        }
        if opponent is not None and typechart is not None and skill.category in (0, 1) and skill.power is not None:
            item["display_power"] = calc_damage(pet, opponent, skill, typechart)["display_power"]
        skills.append(item)
    return {
        "name": pet.name,
        "side": pet.side,
        "hp": pet.hp,
        "max_hp": pet.max_hp,
        "energy": pet.energy,
        "level": pet.level,
        "stats": pet.stats,
        "attributes": pet.attributes,
        "speed_range": pet.speed_range,
        "defense_cooldowns": sorted(pet.defense_cooldowns),
        "feature": {
            "id": pet.feature_id,
            "desc": pet.feature_desc,
        } if pet.feature_id is not None else None,
        "buffs": [
            {"type": buff.buff_type, "value": buff.value, "duration": buff.duration}
            for buff in pet.buffs
        ],
        "skills": skills,
    }


def state_to_dict(state: BattleState, view_side: str | None = None) -> dict:
    typechart = load_typechart()
    teams = {}
    for side in ("A", "B"):
        opponent_side = "B" if side == "A" else "A"
        opponent_idx = state.active[opponent_side]
        opponent = state.teams[opponent_side][opponent_idx] if opponent_idx >= 0 else None
        side_list = []
        for pet in state.teams[side]:
            d = pet_to_dict(pet, opponent, typechart)
            if view_side is not None and side == view_side:
                mark_energy_bonus = 0
                pos2 = marks.find_mark(state, side, marks.POSITIVE, 2)
                if pos2 is not None:
                    mark_energy_bonus += pos2["stacks"]
                pos11 = marks.find_mark(state, side, marks.POSITIVE, 11)
                if pos11 is not None:
                    mark_energy_bonus -= pos11["stacks"]
                for skill_item in d["skills"]:
                    skill_obj = next((s for s in pet.skills if s.skill_id == skill_item["skill_id"]), None)
                    if skill_obj is not None:
                        skill_item["energy_cost"] = skill_utils.actual_energy_cost(pet, skill_obj, mark_energy_bonus)
            if view_side is not None and side != view_side:
                revealed_ids = state.revealed[side]
                d["skills"] = [
                    {"name": skill["name"], "energy_cost": skill["energy_cost"]}
                    for skill in d["skills"] if skill["skill_id"] in revealed_ids
                ]
            side_list.append(d)
        teams[side] = side_list
    return {
        "turn": state.turn,
        "winner": state.winner,
        "log": state.log,
        "magic": state.magic,
        "active": state.active,
        "home_side": state.home_side,
        "marks": state.marks,
        "marks_extra": state.marks_extra,
        "teams": teams,
    }
