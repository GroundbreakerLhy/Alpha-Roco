"""Headless 6v6 battle loop and JSON state serialization."""

from __future__ import annotations

import copy
import random

from . import (
    buffs,
    burst,
    counter,
    marks,
    resonance,
    skill_utils,
    skills,
    traits,
    weather,
)
from .damage import calc_damage, level_coefficient
from .models import Action, BattlePet, BattleState
from .data_loader import load_typechart
from .traits import impl as _traits_impl  # noqa: F401  特性实现接入点（注册 handler）
from .skills import impl as _skills_impl  # noqa: F401  技能实现接入点（注册 handler）
from .typechart import type_multiplier

CHARGE_ENERGY = 5
MAGIC_START = 4
MAX_TURN = 50

# random.seed(42)


def create_battle(pet_a: BattlePet, pet_b: BattlePet) -> BattleState:
    return create_team_battle([pet_a], [pet_b])


def create_team_battle(team_a: list, team_b: list) -> BattleState:
    state = BattleState(
        teams={"A": team_a, "B": team_b},
        active={"A": 0, "B": 0},
        magic={"A": MAGIC_START, "B": MAGIC_START},
        revealed={"A": set(), "B": set()},
        home_side=random.choice(["A", "B"]),
        turn=1,
        log=[],
    )
    # 首发精灵的入场回合（返场免疫判定用）
    for side in ("A", "B"):
        idx = state.active[side]
        if idx >= 0 and state.teams[side]:
            state.teams[side][idx].entry_turn = state.turn
    # 技能静态属性（应对类别/传动/迅捷/蓄力等）来自 sim/skills 注册表；
    # 须在特性 battle_start 之前绑定，以便特性仍可覆盖（bind 只写非默认值）。
    for side in ("A", "B"):
        for pet in state.teams[side]:
            for skill in pet.skills:
                skills.bind_skill(skill, pet)
    # 战斗创建时先初始化需要预计算的技能运行时标记（如飓风共享技能的迅捷）。
    traits.on_battle_start(state)
    apply_drive(state)
    return state


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


def _drive_reorder(skills: list, drives: list) -> list:
    """按传动值重排技能列表。

    规则：
      从 1 号位开始依次处理；
      目标位置 = 原位置 + 传动值，循环到列表长度内；
      目标被占时，从目标位置往前找最近的空位。
    """
    n = len(skills)
    if n <= 1:
        return list(skills)
    slots: list = [None] * n
    for old_pos, (skill, drive) in enumerate(zip(skills, drives)):
        target = (old_pos + drive) % n
        if slots[target] is None:
            slots[target] = skill
        else:
            pos = target
            while slots[pos] is not None:
                pos = (pos - 1) % n
            slots[pos] = skill
    return slots


def apply_drive(state: BattleState, side: str | None = None) -> None:
    """对指定方（或双方）当前在场精灵执行回合开始的传动重排。"""
    sides = ("A", "B") if side is None else (side,)
    for s in sides:
        pet = _active_pet(state, s)
        if pet is None:
            continue
        drives = [getattr(skill, "drive", 0) for skill in pet.skills]
        pet.skills = _drive_reorder(pet.skills, drives)


def _add_energy(state: BattleState, pet: BattlePet, amount: int) -> int:
    """回复能量（含特性能量上限修正），返回实际回复量。"""
    return traits.grant_energy(state, pet, amount)


def _settle_skill_cooldowns(state) -> None:
    """回合结束统一结算所有技能冷却。"""
    for side in ("A", "B"):
        for pet in state.teams[side]:
            skill_utils.settle_skill_cooldowns(pet)


def _apply_faint(state: BattleState, side: str, logs: list | None = None) -> None:
    pet = _active_pet(state, side)
    if pet is None or pet.hp > 0:
        return
    state.active[side] = -1
    state.magic[side] -= 1
    # logs 缺省写 state.log；行动本地日志流（疾风连袭重放中途的力竭）传入以保持顺序
    stream = logs if logs is not None else state.log
    stream.append(f"{side} {pet.name} 倒下了，剩余魔力 {state.magic[side]}")
    # 特性：力竭事件（广播；诈死/赎价类特性可在此回调魔力）
    traits.emit(state, "faint", scope="all", side=side, subject=pet)
    # 必须在 faint 特性结算后再判定胜负；诈死可能返还本次损失的魔力。
    if state.magic[side] <= 0:
        state.winner = "B" if side == "A" else "A"
        stream.append(f"{state.winner} 获胜")


def _apply_turn_limit(state: BattleState) -> None:
    if state.winner is not None:
        return
    magic_a = state.magic["A"]
    magic_b = state.magic["B"]
    if magic_a != magic_b:
        state.winner = "A" if magic_a > magic_b else "B"
        state.log.append(
            f"达到{MAX_TURN + 1}回合，魔力值 {magic_a}:{magic_b}，{state.winner} 获胜"
        )
        return
    hp_a = sum(pet.hp for pet in state.teams["A"])
    hp_b = sum(pet.hp for pet in state.teams["B"])
    if hp_a != hp_b:
        state.winner = "A" if hp_a > hp_b else "B"
        state.log.append(
            f"达到{MAX_TURN + 1}回合，剩余血量 {hp_a}:{hp_b}，{state.winner} 获胜"
        )
        return
    state.winner = "draw"
    state.log.append(f"达到{MAX_TURN + 1}回合，魔力与剩余血量均相同，平局")


def _skill_cost_after_bursts(state: BattleState, side: str, skill) -> int:
    """计算技能在当前迸发影响下的实际能耗，不处理能量不足兜底。"""
    total_cost = current_skill_cost(state, side, skill)
    pet = _active_pet(state, side)
    if pet is None:
        return total_cost
    for active_burst in pet.bursts:
        if active_burst["type"] == "energy_cost_flat":
            total_cost = max(0, total_cost - active_burst["value"])
    return total_cost


def _find_swift_skill(state: BattleState, side: str, incoming: BattlePet) -> int | None:
    """主动换人入场后，逐槽查找第一个可用的迅捷技能，返回技能索引；没有则 None。"""
    if incoming.hp <= 0 or not buffs.can_act(incoming):
        return None

    # 队伍技能槽最多为 4 个；当前候选不可用时必须继续检查后续槽位。
    for skill_index, skill in enumerate(incoming.skills[:4]):
        if not skill_utils.is_swift(skill):
            continue
        # 迅捷只按实际当前能量判断，不触发“能量不足时用生命补能量”的兜底。
        if incoming.energy < _skill_cost_after_bursts(state, side, skill):
            continue
        if skill_utils.is_skill_on_cooldown(incoming, skill.skill_id):
            continue
        if not traits.query_skill_usable(
            state, incoming, skill, skill_index=skill_index
        ):
            continue
        return skill_index
    return None


def _apply_morph(pet: BattlePet, skill) -> None:
    """巧变结算：临时技能用后还原；原技能用后替换为随机池中的临时技能，能耗-1。"""
    if skill is None:
        return

    origin = getattr(skill, "morph_origin", None)
    if origin is not None:
        if skill in pet.skills:
            pet.skills[pet.skills.index(skill)] = origin
        return

    pool = getattr(skill, "morph_pool", None) or []
    if not pool:
        return

    template = random.choice(pool)
    temp = copy.deepcopy(template)
    temp.energy_cost = max(0, temp.energy_cost - 1)
    temp.morph_pool = []
    temp.morph_origin = skill
    # 巧变临时技能同样绑定静态属性（应对类别/传动等），模板未绑定时需要补齐。
    skills.bind_skill(temp, pet)
    if skill in pet.skills:
        pet.skills[pet.skills.index(skill)] = temp


def _is_windup(pet: BattlePet) -> bool:
    return getattr(pet, "windup_skill", None) is not None


# 入场统一结算：蓄电印记(6)迸发、棘刺印记(1)、降灵印记(5)、特性入场事件
# thorn=False 用于力竭后的替换（棘刺/降灵只对"离场"生效，力竭不算离场）
# active_switch=True 仅表示本次入场来自玩家主动换人；返回可用迅捷技能索引或 None。
def switch_in(
    state: BattleState,
    side: str,
    incoming: BattlePet,
    logs: list,
    thorn: bool = True,
    active_switch: bool = False,
) -> int | None:
    incoming.has_acted_since_entry = False
    incoming.entry_turn = state.turn  # 记录入场回合（返场免疫判定用）
    positive_mark = marks.get_mark(state, side, marks.POSITIVE)
    if positive_mark is not None and positive_mark["id"] == 6:
        burst.add_burst(incoming, "attack_power_flat", 10 * positive_mark["stacks"])
    if thorn:
        negative = marks.get_mark(state, side, marks.NEGATIVE)
        if negative is not None and negative["id"] == 1:
            loss = int(incoming.max_hp * 0.06 * negative["stacks"])
            incoming.hp = max(0, incoming.hp - loss)
            logs.append(f"  棘刺印记：{incoming.name} 入场失去 {loss} 生命")
        if negative is not None and negative["id"] == 5:
            loss = negative["stacks"]
            incoming.energy = max(0, incoming.energy - loss)
            logs.append(f"  降灵印记：{incoming.name} 入场失去 {loss} 能量")
        _apply_dark_surge(state, side, incoming, logs)
    traits.emit(state, "entry", scope="all", side=side, subject=incoming)
    if active_switch:
        return _find_swift_skill(state, side, incoming)
    return None


# 离场换人通用结算：离场事件广播 + 换人 + 入场结算
def _apply_dark_surge(state, side: str, incoming, logs: list) -> None:
    negative = marks.get_mark(state, side, marks.NEGATIVE)
    if negative is None or negative["id"] != 13:
        return
    for _ in range(5):
        debuff_type = random.choice(["atk", "spatk", "def", "spdef", "speed"])
        buffs.add_buff(incoming, debuff_type, -1)
    logs.append(f"  暗涌印记：{incoming.name} 获得5层随机属性减益")


def _force_switch_after_leave(state: BattleState, side: str, logs: list) -> None:
    current_idx = state.active[side]
    if current_idx < 0:
        return
    outgoing = state.teams[side][current_idx]
    if _is_windup(outgoing):
        logs.append(f"{side} {outgoing.name} 蓄力中，免疫离场效果")
        return
    idx = None
    for i, pet in enumerate(state.teams[side]):
        if i != current_idx and pet.hp > 0:
            idx = i
            break
    if idx is None:
        return
    incoming = state.teams[side][idx]
    # 特性：离场事件（广播，subject=离场精灵，incoming=入场精灵）
    traits.emit(
        state, "leave", scope="all", side=side, subject=outgoing, incoming=incoming
    )
    buffs.clear_normal_buffs(outgoing)
    burst.clear_bursts(outgoing)
    outgoing.light_heal_rounds = 0
    state.active[side] = idx
    logs.append(f"{side} 因脱离换上 {incoming.name}")
    switch_in(state, side, incoming, logs, active_switch=False)


def trait_leave_switch(
    state: BattleState, side: str, incoming: BattlePet, logs: list
) -> None:
    """特性请求的脱离换人（如警惕：回合结束能量为0，玩家选择后调用）。

    与 _force_switch_after_leave 等价，但由服务端传入玩家选定的 incoming。
    """
    outgoing = state.teams[side][state.active[side]]
    if _is_windup(outgoing):
        logs.append(f"{side} {outgoing.name} 蓄力中，免疫离场效果")
        return
    traits.emit(
        state, "leave", scope="all", side=side, subject=outgoing, incoming=incoming
    )
    buffs.clear_normal_buffs(outgoing)
    burst.clear_bursts(outgoing)
    outgoing.light_heal_rounds = 0
    state.active[side] = state.teams[side].index(incoming)
    logs.append(f"{side} 因特性脱离换上 {incoming.name}")
    switch_in(state, side, incoming, logs, active_switch=False)


def reenter(state: BattleState, side: str, logs: list) -> bool:
    """返场：精灵离场并立即入场（同一只精灵）。

    完整走离场结算（广播 leave、清 NORMAL buff/迸发）与入场结算（switch_in）。
    「本回合入场的精灵免疫此效果」：该精灵本回合已入场（entry_turn == state.turn）则不生效。
    返回是否执行了返场。
    """
    pet = _active_pet(state, side)
    if pet is None or pet.hp <= 0:
        return False
    if pet.entry_turn == state.turn:
        logs.append(f"{side} {pet.name} 本回合已入场，免疫返场效果")
        return False
    # 离场结算
    traits.emit(state, "leave", scope="all", side=side, subject=pet, incoming=pet)
    buffs.clear_normal_buffs(pet)
    burst.clear_bursts(pet)
    pet.light_heal_rounds = 0
    logs.append(f"{side} {pet.name} 返场")
    # 立即入场结算（同一只精灵；switch_in 会刷新 entry_turn）
    switch_in(state, side, pet, logs, active_switch=False)
    return True


def _resolve_trait_reenter(state: BattleState) -> None:
    """回合结束结算后处理特性请求的返场（如安可 200290）。"""
    for side in ("A", "B"):
        if state.pending_reenter[side]:
            state.pending_reenter[side] = False
            reenter(state, side, state.log)


def _resolve_trait_leave(state: BattleState) -> None:
    """回合结束结算后处理特性请求的脱离（如警惕：能量为0时脱离）。

    特性 handler 已在 on_round_end 里设置 state.pending_switch[side]=True，
    引擎在此不做换人——由服务端发 choose_replacement 让玩家手动选择上场精灵。
    本函数保留为空语义占位（如未来有纯引擎自动脱离特性可在此接入）。
    """


def resume_after_leave(
    state: BattleState, side: str, incoming: BattlePet
) -> BattleState:
    """脱离暂停后，由服务端传入玩家选定的替补：入场并续完本回合。

    step() 在某侧行动后因脱离而暂停（state.paused_turn 非空、该侧 active=-1）时调用。
    新精灵本回合即上场——若暂停发生在先手方行动后，续接时后手方的行动会打到新精灵上。
    续接过程中若再次触发脱离（如另一方也用了移花接木），会再次暂停（paused_turn 重新置位），
    由服务端循环处理。
    """
    ctx = state.paused_turn
    state.paused_turn = None
    logs = state.log
    # 入场（替换 active=-1 的离场位）；脱离换人对入场的棘刺/降灵印记生效（thorn=True）
    state.active[side] = state.teams[side].index(incoming)
    logs.append(f"{side} 脱离后换上 {incoming.name}")
    switch_in(state, side, incoming, logs, thorn=True, active_switch=False)
    if ctx.get("do_second"):
        second = ctx["second"]
        state.log.extend(
            _resolve_repeated(
                state,
                second,
                ctx["second_action"],
                is_first=False,
                is_counter=ctx["second_is_counter"],
                counter_category=ctx["second_counter_category"],
            )
        )
        _apply_faint(state, "B" if second == "A" else "A")
        if state.winner is not None:
            return state
        # 续接的后手方也可能触发脱离（再次暂停）
        if _try_pause_for_leave(state, second, {"do_second": False}):
            return state
    _round_end(state)
    return state


def _consume_skill_leave(state: BattleState, ctx, side: str, logs: list) -> None:
    """登记技能脱离请求（LeaveField）：在该侧行动结算后**立即离场**并重新选人。

    脱离不取消本次行动（行动照常结算，见 SPEC.md B36）；离场精灵跳过自己的回合末
    结算。无存活替补时脱离不生效。紧急脱离/返场的消费待接入。
    """
    mode = ctx.extra.pop("leave", None)
    if not mode:
        return
    opponent_side = "B" if side == "A" else "A"
    sides = []
    if mode in (skills.LEAVE_SELF, skills.LEAVE_BOTH):
        sides.append(side)
    if mode in (skills.LEAVE_ENEMY, skills.LEAVE_BOTH):
        sides.append(opponent_side)
    for s in sides:
        has_substitute = any(
            i != state.active[s] and p.hp > 0 for i, p in enumerate(state.teams[s])
        )
        if has_substitute:
            state.pending_action_leave[s] = True


def _execute_action_leave(state: BattleState, side: str, logs: list) -> int:
    """执行技能脱离：精灵立即离场（出战位 -1），跳过其回合末结算。

    返回离场精灵原本的出战位下标；未离场（蓄力免疫/无存活替补）返回 -1。
    """
    idx = state.active[side]
    if idx < 0:
        return -1
    pet = state.teams[side][idx]
    if _is_windup(pet):
        logs.append(f"{side} {pet.name} 蓄力中，免疫离场效果")
        return -1
    if not any(i != idx and p.hp > 0 for i, p in enumerate(state.teams[side])):
        return -1
    traits.emit(state, "leave", scope="all", side=side, subject=pet, incoming=None)
    buffs.clear_normal_buffs(pet)
    burst.clear_bursts(pet)
    pet.light_heal_rounds = 0
    pet.windup_skill = None
    state.active[side] = -1
    logs.append(f"{side} {pet.name} 脱离了战斗")
    return idx


def _try_pause_for_leave(state: BattleState, side: str, resume_ctx: dict) -> bool:
    """该侧行动后若有脱离请求：立即离场并暂停本回合等待选人。返回是否暂停。"""
    if not state.pending_action_leave[side]:
        return False
    state.pending_action_leave[side] = False
    leaver_idx = _execute_action_leave(state, side, state.log)
    if leaver_idx < 0:
        return False
    ctx = dict(resume_ctx)
    ctx["leave_side"] = side
    ctx["leaver_index"] = leaver_idx
    state.paused_turn = ctx
    return True


def _settle_counters(state: BattleState) -> None:
    """应对奖励从句的统一结算：回合末结算的**第一步**（B38）。

    效果按结算时的在场状态解析（对方本回合的行动已全部完成，扣能类从句不会
    把行动"吸"到无法支付）。从句登记的脱离请求随即照常执行（B36 流程）并
    暂停等待选人；续接后重入回合末结算时本函数自然跳过（登记已被消费）。
    """
    pending = state.pending_counters
    state.pending_counters = []
    for item in pending:
        pet = item["pet"]
        counter_ctx = skills.emit(
            state,
            skills.STAGE_COUNTER,
            pet,
            item["skill"],
            skill_index=item["skill_index"],
            is_first=item["is_first"],
            is_counter=True,
            counter_category=item["counter_category"],
            energy_cost=item["energy_cost"],
            damage_dealt=item["damage_dealt"],
            hit_count=item["hit_count"],
            logs=state.log,
        )
        _consume_skill_leave(state, counter_ctx, pet.side, state.log)
    # 从句伤害可能致命（如"应对攻击：造成90威力物伤"类）
    for side in ("A", "B"):
        _apply_faint(state, side)
    # 从句登记的脱离（吓退"敌方脱离"等）：立即离场并暂停等待选人
    for side in ("A", "B"):
        if _try_pause_for_leave(state, side, {"do_second": False}):
            return


def _settle_hit_effects(state: BattleState) -> None:
    """命中附加效果的统一结算：回合末结算紧随应对奖励之后（B39）。

    命中时只登记（含未造成伤害的情形，S9），此处重放 STAGE_HIT；效果按结算时
    的在场状态解析（含 damage_dealt 等命中当时的快照字段）。脱离请求同样
    按 B36 流程执行并暂停等待选人。
    """
    pending = state.pending_hit_effects
    state.pending_hit_effects = []
    for item in pending:
        pet = item["pet"]
        hit_ctx = skills.emit(
            state,
            skills.STAGE_HIT,
            pet,
            item["skill"],
            skill_index=item["skill_index"],
            is_first=item["is_first"],
            is_counter=False,
            counter_category=item["counter_category"],
            energy_cost=item["energy_cost"],
            damage_dealt=item["damage_dealt"],
            hit_count=item["hit_count"],
            logs=state.log,
            _defer_mode="settle",
            killed=item["killed"],
            hit_target=item["hit_target"],
        )
        _consume_skill_leave(state, hit_ctx, pet.side, state.log)
    for side in ("A", "B"):
        _apply_faint(state, side)
    for side in ("A", "B"):
        if _try_pause_for_leave(state, side, {"do_second": False}):
            return


def _round_end(state: BattleState) -> None:
    """回合结束结算（step 主路径与脱离续接共用；离场精灵因出战位 -1 被跳过）。"""
    # 第一步：应对奖励从句（B38），紧随其后是命中附加效果（B39）；
    # 触发脱离暂停时中止，续接后重入完成其余步骤
    _settle_counters(state)
    if state.paused_turn is not None or state.winner is not None:
        return
    _settle_hit_effects(state)
    if state.paused_turn is not None or state.winner is not None:
        return
    _settle_skill_cooldowns(state)
    marks.on_round_end(state)
    buffs.on_round_end(state)
    resonance.on_round_end(state)
    weather.on_round_end(state)
    traits.on_round_end(state)
    for team in state.teams.values():
        for pet in team:
            pet.overload_current.clear()
    for side in ("A", "B"):
        _apply_faint(state, side)
    _resolve_trait_leave(state)
    _resolve_trait_reenter(state)
    if state.winner is not None:
        return
    if state.turn > MAX_TURN:
        _apply_turn_limit(state)
    if state.winner is None:
        apply_drive(state)


def _resolve_action(
    state: BattleState,
    side: str,
    action: Action,
    is_first: bool = False,
    is_counter: bool = False,
    counter_category: str = "",
    followups: list | None = None,
    used: list | None = None,
    pay_energy: bool = True,
    replay: bool = False,
) -> list:
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
        if (
            target is not None
            and 0 <= target < len(team)
            and target != current
            and team[target].hp > 0
        ):
            incoming = team[target]
            # 特性：离场事件（广播，subject=离场精灵，incoming=入场精灵）
            traits.emit(
                state, "leave", scope="all", side=side, subject=pet, incoming=incoming
            )
            buffs.clear_normal_buffs(pet)
            burst.clear_bursts(pet)
            pet.light_heal_rounds = 0
            pet.windup_skill = None
            state.active[side] = target
            logs.append(f"{side} 换上 {incoming.name}")
            swift_index = switch_in(state, side, incoming, logs, active_switch=True)
            if swift_index is not None:
                logs.append(f"{side} {incoming.name} 触发迅捷")
                if followups is not None:
                    followups.append(Action(kind="skill", skill_index=swift_index))
            return logs
        logs.append(f"{side} 换人目标无效")
        return logs

    opponent_side = "B" if side == "A" else "A"
    opponent = _active_pet(state, opponent_side)

    if not buffs.can_act(pet):
        logs.append(f"{side} {pet.name} 眩晕，无法行动")
        return logs

    if action.kind == "charge":
        pet.has_acted_since_entry = True
        active_bursts = burst.take_bursts(pet)
        gained = _add_energy(state, pet, CHARGE_ENERGY)
        logs.append(f"{side} {pet.name} 聚能，回复 {gained} 能量")
        traits.emit(
            state,
            "charge",
            scope="all",
            side=side,
            subject=pet,
            energy_gain=gained,
            is_first=is_first,
            is_counter=is_counter,
        )
        # 特性：行动完成后的条件性脱离（如哨兵）。
        if pet.trait_state.pop("sentinel_leave_after_action", False):
            _force_switch_after_leave(state, side, logs)
        return logs

    if action.kind != "skill" or action.skill_index is None:
        _add_energy(state, pet, CHARGE_ENERGY)
        logs.append(f"{side} {pet.name} 行动无效，改为聚能")
        return logs

    if not (0 <= action.skill_index < len(pet.skills)):
        _add_energy(state, pet, CHARGE_ENERGY)
        logs.append(f"{side} {pet.name} 技能索引无效，改为聚能")
        return logs

    skill = pet.skills[action.skill_index]
    # 蓄力后的回合只能使用已经蓄力的技能、聚能或换人。
    # 特性可放宽（嫉妒 200174：蓄力状态下可使用任一携带技能）。
    if (
        pet.windup_skill is not None
        and skill is not pet.windup_skill
        and not traits.query_windup_any(state, pet)
    ):
        logs.append(f"{side} {pet.name} 蓄力中，只能使用已蓄力技能、聚能或换人")
        return logs
    # 离场是本次行动的结算结果，不是技能的静态属性。
    # 后续技能效果代码化后，应按具体条件（如应对成功）在这里设置为 True。
    leave = False
    if not replay and skill_utils.is_skill_on_cooldown(pet, skill.skill_id):
        logs.append(f"  {skill.name} 冷却中")
        return logs
    if not traits.query_skill_usable(state, pet, skill, skill_index=action.skill_index):
        logs.append(f"{side} {pet.name} 无法使用 {skill.name}（特性限制）")
        return logs

    # 蓄力：首次选择时支付能量并进入蓄力状态；下一回合再次选择该技能时释放。
    # 重放（疾风连袭）跳过蓄力，直接按释放结算。
    windup_release = (
        bool(getattr(skill, "windup", False))
        and getattr(pet, "windup_skill", None) is skill
    ) or (replay and bool(getattr(skill, "windup", False)))

    # 迸发：本次技能能耗修正（如生物电，不消耗迸发本身）
    total_cost = _skill_cost_after_bursts(state, side, skill)

    if windup_release:
        # 能量已在蓄力回合支付，释放回合不再扣费。
        pet.windup_skill = None
    elif pay_energy:
        if pet.energy < total_cost:
            # 特性：能量不足兜底（如消耗生命代替能量）
            shortfall = traits.query_energy_shortfall(
                state, pet, total_cost - pet.energy, skill
            )
            if shortfall > 0:
                pet.energy += shortfall
            else:
                logs.append(f"{side} {pet.name} 能量不足，无法使用 {skill.name}")
                return logs
        pet.energy -= total_cost

    if getattr(skill, "windup", False) and not windup_release:
        pet.has_acted_since_entry = True
        pet.windup_skill = skill
        # 对外只暴露“使用了蓄力”，不暴露具体技能。
        logs.append(f"{side} {pet.name} 开始蓄力")
        # 特性：进入蓄力状态（洄游 200114 等）
        traits.emit(
            state,
            "windup",
            scope="all",
            side=side,
            subject=pet,
            skill=skill,
            is_first=is_first,
        )
        return logs

    active_bursts = burst.take_bursts(pet)
    # 迸发：攻击后给敌方施加能耗+（如超负荷）
    for active_burst in active_bursts:
        if active_burst["type"] == "enemy_energy_cost" and opponent is not None:
            buffs.add_buff(
                opponent,
                buffs.BuffType.ENERGY_COST,
                active_burst["value"],
                buffs.DurationKind.NORMAL,
                state.turn,
                side,
                pet.name,
            )
            traits.on_buff_gain(
                state,
                opponent,
                buffs.BuffType.ENERGY_COST,
                active_burst["value"],
                side,
                pet.name,
            )
            logs.append(f"  {opponent.name} 全技能能耗+{active_burst['value']}")
    # 印记 9：龙噬印记 —— 释放3能耗技能后双攻+40%/层
    positive_mark = marks.get_mark(state, side, marks.POSITIVE)
    if positive_mark is not None and positive_mark["id"] == 9 and total_cost == 3:
        gain = 4 * positive_mark["stacks"]
        buffs.add_buff(pet, buffs.BuffType.ATK, gain)
        traits.on_buff_gain(state, pet, buffs.BuffType.ATK, gain, side, pet.name)
        buffs.add_buff(pet, buffs.BuffType.SPATK, gain)
        traits.on_buff_gain(state, pet, buffs.BuffType.SPATK, gain, side, pet.name)
        logs.append(f"  龙噬印记：双攻+{gain * 10}%")
    if skill.skill_id != -1:
        state.revealed[side].add(skill.skill_id)
        if not replay:
            # 使用记录（疾风连袭重放集合 / "每使用1次X技能"类的来源）：实时更新；
            # 重放是疾风连袭自身的效果，不算一次新的"使用"
            pet.used_skill_counts[skill.skill_id] = (
                pet.used_skill_counts.get(skill.skill_id, 0) + 1
            )
    logs.append(f"{side} {pet.name} 使用 {skill.name}")
    if used is not None:
        used.append(skill)
    # 特性：技能开始
    traits.emit(
        state,
        "skill_start",
        scope="all",
        side=side,
        subject=pet,
        target=opponent,
        action=action,
        skill=skill,
        skill_index=action.skill_index,
        energy_cost=total_cost,
        is_first=is_first,
        is_counter=is_counter,
    )

    if opponent is None:
        logs.append("  对方没有在场精灵")
        return logs

    total_damage = 0
    applied_damage = 0
    hit_count = 1
    if skill.category in (0, 1) and skill.power is not None:
        skill_element = traits.query_skill_element(state, pet, skill)
        # 天气：雨天使水系技能威力提升至150%
        extra_power_percent = (
            50.0
            if weather.get_weather_id(state) == weather.RAIN and skill_element == 3
            else 0.0
        )
        # 印记 0：攻击印记 —— 全技能威力+10%/层
        # 印记 2：蓄势印记 —— 全技能威力+30%/层
        # 印记 12：风气印记 —— 先手攻击时技能威力+20%/层
        extra_power_flat = 0.0
        if marks.get_mark(state, side, marks.POSITIVE) is not None:
            positive = marks.get_mark(state, side, marks.POSITIVE)
            if positive["id"] == 0:
                extra_power_percent += 10.0 * positive["stacks"]
            if positive["id"] == 2:
                extra_power_percent += 30.0 * positive["stacks"]
            if positive["id"] == 12 and is_first:
                extra_power_percent += 20.0 * positive["stacks"]
        for active_burst in active_bursts:
            if active_burst["type"] == "attack_power_flat":
                extra_power_flat += active_burst["value"]
            if active_burst["type"] == "attack_power_percent":
                extra_power_percent += active_burst["value"]
        pet.has_acted_since_entry = True
        # 特性：威力/属性/连击修正汇总
        extra_power_percent, extra_power_flat = traits.query_power(
            state,
            pet,
            opponent,
            extra_power_percent,
            extra_power_flat,
            is_first=is_first,
            skill=skill,
        )
        # 技能自身威力修正（"敌方每有1层冻结威力+20"、应对状态威力N倍等）；
        # 与特性同一增量语义。is_counter 传入供"应对状态：本次威力N倍"类即时判定
        # （作用于本次行动自身，是 B38 延迟结算的例外，不走回合末登记）。
        # skill_index 供槽位条件威力（"位于3号位时威力+40"，机械传动族）按
        # 使用时槽位判定——传动重排会改变槽位，加成随之变化。
        # 只喂给本次伤害，不并入 extra_power_*——星陨印记的附加伤害吃通用威力
        # buff/印记，但不吃技能描述自带的专属威力修正。
        skill_power_percent, skill_power_flat = skills.query_power(
            state,
            pet,
            skill,
            skill_index=action.skill_index,
            is_first=is_first,
            is_counter=is_counter,
        )
        # 威力由规则算出的技能（闪击/鸣沙按差值查表等）：绝对覆盖基础威力（S17）。
        absolute_power = skills.query_absolute_power(
            state, pet, skill, is_first=is_first
        )
        result = calc_damage(
            pet,
            opponent,
            skill,
            load_typechart(),
            opponent.defense_reduction,
            extra_power_percent=extra_power_percent + skill_power_percent,
            extra_power_flat=extra_power_flat + skill_power_flat,
            element=skill_element,
            state=state,
            is_first=is_first,
            power_override=absolute_power,
        )
        # 连击数 = 基础1 + buff + 特性 + 技能自身修正（共用 skills.compute_hit_count）。
        hit_count = skills.compute_hit_count(state, pet, skill, opponent)
        total_damage = result["damage"] * hit_count
        if total_damage > 0:
            prevented = False
            if total_damage >= opponent.hp:
                # 特性：致命伤害判定（免死类）
                prevented = traits.emit_lethal(
                    state,
                    side,
                    opponent,
                    total_damage,
                    attacker=pet,
                    skill=skill,
                    is_first=is_first,
                )
            if not prevented:
                opponent.hp = max(0, opponent.hp - total_damage)
                applied_damage = total_damage
                logs.append(f"  造成 {applied_damage} 伤害")
                traits.emit(
                    state,
                    "attack",
                    scope="all",
                    side=side,
                    subject=pet,
                    target=opponent,
                    skill=skill,
                    damage=applied_damage,
                    hit_count=hit_count,
                    is_first=is_first,
                    type_mult=result["type_multiplier"],
                )
                traits.emit(
                    state,
                    "take_damage",
                    scope="all",
                    side=opponent_side,
                    subject=opponent,
                    target=pet,
                    skill=skill,
                    damage=applied_damage,
                    hit_count=hit_count,
                    is_first=is_first,
                )
                lifesteal = buffs.get_lifesteal(pet) + traits.query_lifesteal(
                    state, pet, opponent, skill=skill
                )
                if lifesteal > 0:
                    heal = int(applied_damage * lifesteal)
                    if heal > 0:
                        base_heal = heal
                        heal = traits.query_heal(state, pet, heal, source="lifesteal")
                        pet.hp = min(pet.max_hp, pet.hp + heal)
                        logs.append(f"  吸血回复 {heal}")
                        traits.emit(
                            state,
                            "heal",
                            scope="all",
                            side=side,
                            subject=pet,
                            heal=heal,
                            base_heal=base_heal,
                            source="lifesteal",
                        )
                if opponent.hp <= 0:
                    traits.emit(
                        state,
                        "kill",
                        scope="all",
                        side=side,
                        subject=opponent,
                        target=pet,
                        skill=skill,
                        damage=applied_damage,
                        hit_count=hit_count,
                    )
            else:
                logs.append(f"  {opponent.name} 的特性使其免疫了致命伤害")
        else:
            logs.append("  没有造成伤害")

        # 命中后先跑"登记类"效果（脱离/返场，deferrable=False）：立即生效（B36）；
        # 其余命中附加效果登记到回合末统一结算（B39）；含未造成伤害的情形（S9）。
        hit_ctx = skills.emit(
            state,
            skills.STAGE_HIT,
            pet,
            skill,
            skill_index=action.skill_index,
            is_first=is_first,
            is_counter=is_counter,
            counter_category=counter_category,
            energy_cost=total_cost,
            damage_dealt=applied_damage,
            hit_count=hit_count,
            logs=logs,
            _defer_mode="collect",
        )
        _consume_skill_leave(state, hit_ctx, side, logs)
        state.pending_hit_effects.append(
            {
                "pet": pet,
                "skill": skill,
                "skill_index": action.skill_index,
                "is_first": is_first,
                "counter_category": counter_category,
                "energy_cost": total_cost,
                "damage_dealt": applied_damage,
                "hit_count": hit_count,
                # "若使用本技能击败敌方"类的快照：伤害已结算，对方 hp≤0 即本次击败
                "killed": opponent.hp <= 0,
                # 命中当时的对方在场精灵（"若击败敌方"类的锚点——回合末结算时
                # 被击败者已不在场，活跃位是空的）
                "hit_target": opponent,
            }
        )

        negative_mark = marks.get_mark(state, opponent_side, marks.NEGATIVE)
        # 印记 7：星陨印记 —— 非幻系攻击触发额外幻系伤害
        if (
            negative_mark is not None
            and negative_mark["id"] == 7
            and skill.element != 17
        ):
            n = negative_mark["stacks"]
            # 星陨伤害同样吃技能威力类 buff/印记（含攻击印记、蓄势印记、迸发等）
            power_percent_buff, power_flat_buff = buffs.get_skill_power_modifier(pet)
            star_power = (n * n + 24 * n - 24 + extra_power_flat + power_flat_buff) * (
                1.0 + (extra_power_percent + power_percent_buff) / 100.0
            )
            atk_key = "atk" if skill.category == 0 else "spatk"
            def_key = "def" if skill.category == 0 else "spdef"
            atk_stat = pet.stats[atk_key] * buffs.get_stat_multiplier(pet, atk_key)
            def_stat = opponent.stats[def_key] * buffs.get_stat_multiplier(
                opponent, def_key
            )
            star_eff = type_multiplier(17, opponent.attributes, load_typechart())
            star_dmg = int(
                star_power
                * (atk_stat / def_stat)
                * level_coefficient(pet.level)
                * star_eff
                * (1 - opponent.defense_reduction)
            )
            opponent.hp = max(0, opponent.hp - star_dmg)
            marks.remove_mark(state, opponent_side, 7)
            logs.append(f"  星陨印记触发，造成 {star_dmg} 幻系伤害")
    elif skill.category == 2:
        pet.has_acted_since_entry = True
        active_bursts = burst.take_bursts(pet)
        # 减伤来自技能注册表（sim/skills/impl/defense.py），不解析 desc。
        # 钳制到 [0, 1]：技能减伤可叠乘（不可接触"敌方每有1层中毒减伤+10%"），
        # 超过 100% 时星陨伤害的 (1 - defense_reduction) 会变成负数而反过来给对手回血。
        reduction = skills.query_damage_reduction(
            state, pet, skill, skill_index=action.skill_index
        )
        pet.defense_reduction = max(0.0, min(1.0, reduction))
        # 防御技能复用通用冷却：任意防御使用后，本精灵所有防御技能进入冷却。
        for defense_skill in pet.skills:
            if defense_skill.category == 2:
                skill_utils.schedule_skill_cooldown(pet, defense_skill.skill_id)
        logs.append(f"  防御，减伤 {int(reduction * 100)}%")
        traits.emit(
            state,
            "defense",
            scope="all",
            side=side,
            subject=pet,
            skill=skill,
            is_first=is_first,
            is_counter=is_counter,
        )
    else:
        pet.has_acted_since_entry = True
        active_bursts = burst.take_bursts(pet)
        # 状态技能本体：已实现的技能走 STAGE_STATUS 结算；未实现的仍为占位（不生效）。
        handler = skills.get_handler(skill.skill_id)
        if handler is None or not handler.implemented:
            logs.append("  非伤害技能：效果暂未实现")
        status_ctx = skills.emit(
            state,
            skills.STAGE_STATUS,
            pet,
            skill,
            skill_index=action.skill_index,
            is_first=is_first,
            is_counter=is_counter,
            counter_category=counter_category,
            energy_cost=total_cost,
            logs=logs,
        )
        # 状态技能可登记脱离（移花接木"随后脱离"等），统一消费。
        _consume_skill_leave(state, status_ctx, side, logs)
        traits.emit(
            state,
            "status_skill",
            scope="all",
            side=side,
            subject=pet,
            skill=skill,
            is_first=is_first,
            is_counter=is_counter,
        )

    if is_counter:
        # 应对判定成功：只登记奖励从句，延迟到回合末结算的第一步统一生效（B38）。
        # 判定与强制先手在行动排序时已生效；从句不在此立即结算——否则扣能类从句
        # 会把对方本回合的行动"吸"到无法支付。
        state.pending_counters.append(
            {
                "pet": pet,
                "skill": skill,
                "skill_index": action.skill_index,
                "is_first": is_first,
                "counter_category": counter_category,
                "energy_cost": total_cost,
                "damage_dealt": applied_damage,
                "hit_count": hit_count,
            }
        )

    if pet.trait_state.pop("sentinel_leave_after_action", False):
        leave = True
    if leave:
        _force_switch_after_leave(state, side, logs)

    # 特性：技能结束（含离场结算后）
    traits.emit(
        state,
        "skill_end",
        scope="all",
        side=side,
        subject=pet,
        target=opponent,
        action=action,
        skill=skill,
        skill_index=action.skill_index,
        energy_cost=total_cost,
        is_first=is_first,
        is_counter=is_counter,
        damage_dealt=applied_damage,
        hit_count=hit_count,
    )

    return logs


# 技能实际能耗（印记 2/11、天气沙暴、buff、特性共同修正）
def current_skill_cost(state: BattleState, side: str, skill) -> int:
    pet = _active_pet(state, side)
    if pet is None:
        return 0
    return skill_utils.true_energy_cost(state, pet, skill)


def can_use_skill(
    state: BattleState, side: str, skill, skill_index: int | None = None
) -> bool:
    """技能是否可用（能耗/技能冷却/特性限制），供服务端校验与客户端提示。"""
    pet = _active_pet(state, side)
    if pet is None:
        return False
    # 蓄力后的回合只能使用已蓄力技能、聚能或换人。
    # 特性可放宽（嫉妒 200174：蓄力状态下可使用任一携带技能）。
    if (
        pet.windup_skill is not None
        and skill is not pet.windup_skill
        and not traits.query_windup_any(state, pet)
    ):
        return False
    if pet.windup_skill is skill:
        # 蓄力技能的能耗已在蓄力回合支付，释放时不再校验能量。
        if skill_utils.is_skill_on_cooldown(pet, skill.skill_id):
            return False
        return traits.query_skill_usable(state, pet, skill, skill_index=skill_index)
    if pet.energy < current_skill_cost(state, side, skill):
        # 特性兜底：能量不足但特性可补充（如石头大餐消耗生命代替能量）
        gap = current_skill_cost(state, side, skill) - pet.energy
        if traits.query_energy_shortfall(state, pet, gap, skill) <= 0:
            return False
    if skill_utils.is_skill_on_cooldown(pet, skill.skill_id):
        return False
    return traits.query_skill_usable(state, pet, skill, skill_index=skill_index)


def _resolve_repeated(
    state: BattleState,
    side: str,
    action: Action,
    is_first: bool,
    is_counter: bool = False,
    counter_category: str = "",
) -> list:
    logs = []
    pet = _active_pet(state, side)
    if pet is None:
        return logs
    if action.kind == "skill":
        skill = pet.skills[action.skill_index]
        key = skill.skill_id
    elif action.kind == "charge":
        key = "charge"
    else:
        key = None

    count = max(1, pet.overload_current.get(key, 0)) if key is not None else 1
    # 迸发：本次行动技能使用次数+1（入场后首次行动，如噼啪！）
    if action.kind == "skill":
        for b in pet.bursts:
            if b["type"] == "skill_use_count":
                count += b["value"]
        # 蓄力技能不参与连击/过载重复：否则会出现蓄力与释放同一回合结算。
        if getattr(skill, "windup", False):
            count = 1
    used = []
    for _ in range(count):
        if state.winner is not None or state.active[side] < 0:
            break
        if action.kind == "skill":
            skill = pet.skills[action.skill_index]
            # 蓄力释放的能量已在蓄力回合支付，释放回合不再校验。
            if pet.windup_skill is not skill and pet.energy < current_skill_cost(
                state, side, skill
            ):
                shortfall = traits.query_energy_shortfall(
                    state,
                    pet,
                    current_skill_cost(state, side, skill) - pet.energy,
                    skill,
                )
                if shortfall > 0:
                    pet.energy += shortfall
                else:
                    logs.append(f"{side} {pet.name} 能量不足，停止重复使用")
                    break
        logs.extend(
            _resolve_action(
                state, side, action, is_first, is_counter, counter_category, used=used
            )
        )
        if state.winner is not None or state.active[side] < 0:
            break
    if used:
        _apply_morph(pet, used[0])
    return logs


def _effective_speed(state: BattleState, side: str) -> int:
    pet = _active_pet(state, side)
    if pet is None:
        return -1
    mark_speed_bonus = 0
    negative = marks.get_mark(state, side, marks.NEGATIVE)
    if negative is not None and negative["id"] == 3:
        mark_speed_bonus = -10 * negative["stacks"]
    base = buffs.get_speed_value(pet, mark_speed_bonus)
    return base + traits.query_speed(state, pet)


def used_swift_skills(pet: BattlePet, exclude_id: int) -> list:
    """疾风连袭的重放集合：本场释放过（used_skill_counts 的键）且当前为迅捷的技能，
    按技能槽位顺序（技能排布顺序）；exclude_id 排除自身防递归。

    迅捷状态读取时判定（飓风 200211 可让任意技能获得迅捷），记录实时更新。
    """
    return [
        s
        for s in pet.skills
        if s.skill_id in pet.used_skill_counts
        and getattr(s, "swift", False)
        and s.skill_id != exclude_id
    ]


def replay_used_swift_skills(
    state: BattleState, pet: BattlePet, exclude_id: int, is_first: bool = False
) -> list:
    """疾风连袭：按槽位顺序依次免费重放 used_swift_skills 集合。

    与正常使用同等的基础结算（伤害/命中附加/状态效果/防御减伤），但**无应对
    效果**（is_counter=False，不结算应对从句）、不付费、不校验冷却、不进入
    蓄力。每次重放后结算力竭（B33）；胜负已分、使用者力竭或被换下场即中断
    后续重放——对方力竭且无替补在场时，后续攻击重放与正常行动一样扑空。
    """
    side = pet.side
    opp_side = "B" if side == "A" else "A"
    logs = []
    for skill in used_swift_skills(pet, exclude_id):
        if state.winner is not None or state.active[side] < 0 or pet.hp <= 0:
            break
        idx = pet.skills.index(skill)
        logs.extend(
            _resolve_action(
                state,
                side,
                Action(kind="skill", skill_index=idx),
                is_first=is_first,
                pay_energy=False,
                replay=True,
            )
        )
        _apply_faint(state, opp_side, logs)
        _apply_faint(state, side, logs)
        if state.winner is not None:
            break
    return logs


def _action_priority(state: BattleState, side: str, action: Action) -> int:
    """行动的先手值：默认 0，与无修正的技能同一层级正常比速度。

    技能行动经 skills.query_priority 查 handler 的 modify_priority（"先手+1"类）；
    聚能没有技能实体可查，自然落在默认 0。换人行动在排序前已被优先处理（B9），
    不会进入这里。特性侧暂无先手查询（数据里仅 2 条未实现特性引用先手），
    有消费者时在返回值处并列追加。
    """
    if action.kind != "skill" or action.skill_index is None:
        return 0
    pet = _active_pet(state, side)
    if pet is None or action.skill_index >= len(pet.skills):
        return 0
    skill = pet.skills[action.skill_index]
    return skills.query_priority(state, pet, skill, skill_index=action.skill_index)


def round_end_order(state: BattleState) -> list:
    return ["A", "B"] if state.home_side == "A" else ["B", "A"]


def step(state: BattleState, action_a: Action, action_b: Action) -> BattleState:
    if state.winner is not None:
        return state

    state.turn += 1
    state.log = []
    # 上一回合的脱离暂停/请求不应残留（正常应已被续接消费，这里防御性清空）
    state.paused_turn = None
    state.pending_counters = []
    state.pending_hit_effects = []
    for side in ("A", "B"):
        state.pending_action_leave[side] = False

    # 战斗开始为首发精灵补发入场事件；随后广播回合开始事件（双方行动已选定）
    traits.ensure_entry(state)
    traits.on_turn_start(state, action_a, action_b)

    for team in state.teams.values():
        for pet in team:
            pet.defense_reduction = 0.0
            buffs.start_turn_overload(pet)

    for side, action in (("A", action_a), ("B", action_b)):
        if action.kind == "flee":
            state.winner = "B" if side == "A" else "A"
            state.log.append(f"{side} 逃跑了，{state.winner} 获胜")
            return state

    # 共鸣魔法：不占回合操作，双方都选完技能后才结算，且先于本回合行动
    actions_for_magic = {"A": action_a, "B": action_b}
    for side, action in actions_for_magic.items():
        if action.magic_id is None:
            continue
        opponent_side = "B" if side == "A" else "A"
        opponent_action = actions_for_magic[opponent_side]
        opponent_is_status = (
            counter.action_category(state, opponent_side, opponent_action)
            == counter.STATUS
        )
        resonance.use_magic(
            state, side, action.magic_id, opponent_is_status, action.magic_branch
        )

    # 切换精灵优先级最高：先处理所有换人，再处理攻击/聚能
    actions = {"A": action_a, "B": action_b}
    remaining = {}
    for side in ("A", "B"):
        action = actions[side]
        if action.kind == "switch":
            followups = []
            state.log.extend(_resolve_action(state, side, action, followups=followups))
            if state.winner is not None:
                return state
            # 主动换人可能触发迅捷，作为该方本回合的行动参与正常速度排序。
            if followups:
                remaining[side] = followups[0]
        else:
            remaining[side] = action
    if not remaining:
        # 双方都只换人：无行动，直接进入回合结束结算
        _round_end(state)
        return state

    if len(remaining) == 1:
        # 只剩一方有行动时，行动方直接先手，不进行速度判定。
        first = next(iter(remaining))
        second = None
        forced = None
        counter_category = ""
    else:
        a_speed = _effective_speed(state, "A")
        b_speed = _effective_speed(state, "B")
        a_prio = _action_priority(state, "A", remaining["A"])
        b_prio = _action_priority(state, "B", remaining["B"])

        # 应对：按独立应对模块判定强制先手；其余按先手值/速度决定
        forced = counter.forced_first(state, remaining["A"], remaining["B"])
        counter_category = ""
        if forced == "A":
            first, second = "A", "B"
            counter_category = counter.action_category(state, "B", remaining["B"])
            counter.record_counter(_active_pet(state, "A"), counter_category)
            traits.emit(
                state,
                "counter",
                scope="all",
                side="A",
                subject=_active_pet(state, "A"),
                action=remaining["A"],
                is_counter=True,
            )
        elif forced == "B":
            first, second = "B", "A"
            counter_category = counter.action_category(state, "A", remaining["A"])
            counter.record_counter(_active_pet(state, "B"), counter_category)
            traits.emit(
                state,
                "counter",
                scope="all",
                side="B",
                subject=_active_pet(state, "B"),
                action=remaining["B"],
                is_counter=True,
            )
        elif a_prio > b_prio:
            first, second = "A", "B"
        elif b_prio > a_prio:
            first, second = "B", "A"
        elif a_speed > b_speed:
            first, second = "A", "B"
        elif b_speed > a_speed:
            first, second = "B", "A"
        else:
            first, second = ("A", "B") if random.random() < 0.5 else ("B", "A")

    if first in remaining:
        state.log.extend(
            _resolve_repeated(
                state,
                first,
                remaining[first],
                is_first=True,
                is_counter=(forced == first),
                counter_category=counter_category if forced == first else "",
            )
        )
        _apply_faint(state, "B" if first == "A" else "A")
        if state.winner is not None:
            return state
        # 先手方行动后请求脱离（如移花接木"随后脱离"）：立即离场并暂停等待选人
        if _try_pause_for_leave(
            state,
            first,
            {
                "do_second": second is not None
                and second in remaining
                and state.active[second] >= 0,
                "second": second,
                "second_action": remaining.get(second),
                "second_is_counter": forced == second,
                "second_counter_category": counter_category if forced == second else "",
            },
        ):
            return state

    if second is not None and second in remaining and state.active[second] >= 0:
        state.log.extend(
            _resolve_repeated(
                state,
                second,
                remaining[second],
                is_first=False,
                is_counter=(forced == second),
                counter_category=counter_category if forced == second else "",
            )
        )
        _apply_faint(state, "B" if second == "A" else "A")
        if state.winner is not None:
            return state
        # 后手方行动后请求脱离（吓退使敌方脱离、移花接木后手等）：立即离场并暂停等待选人
        if _try_pause_for_leave(state, second, {"do_second": False}):
            return state

    _round_end(state)
    return state


def pet_to_dict(
    pet: BattlePet,
    opponent: BattlePet | None = None,
    typechart: dict | None = None,
    hide_wish: bool = False,
) -> dict:
    skills_source = pet.skills
    if hide_wish and pet.wish_original_skill is not None:
        skills_source = list(pet.skills)
        skills_source[0] = pet.wish_original_skill
    skills = []
    for i, skill in enumerate(skills_source):
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
        if (
            opponent is not None
            and typechart is not None
            and skill.category in (0, 1)
            and skill.power is not None
        ):
            item["display_power"] = calc_damage(pet, opponent, skill, typechart)[
                "display_power"
            ]
        skills.append(item)
    return {
        "name": pet.name,
        "spirit_id": pet.spirit_id,
        "side": pet.side,
        "hp": pet.hp,
        "max_hp": pet.max_hp,
        "energy": pet.energy,
        "level": pet.level,
        "stats": pet.stats,
        "attributes": pet.attributes,
        "bloodline": pet.bloodline,
        "speed_range": pet.speed_range,
        "skill_cooldowns": sorted(pet.skill_cooldowns),
        "buffs": [
            {
                "type": buff.buff_type,
                "value": buff.value,
                "duration": buff.duration,
                "source_kind": buff.source_kind,
            }
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
        opponent = (
            state.teams[opponent_side][opponent_idx] if opponent_idx >= 0 else None
        )
        active_idx = state.active[side]
        active_pet = state.teams[side][active_idx] if active_idx >= 0 else None
        side_list = []
        for pet in state.teams[side]:
            d = pet_to_dict(
                pet,
                opponent,
                typechart,
                hide_wish=(view_side is not None and side != view_side),
            )
            # 特性效果显示行（客户端 buff 区，如 +双攻 * n (20%/trait)）：只给在场精灵，场下不下发
            d["trait_effects"] = (
                traits.describe(state, pet) if pet is active_pet else []
            )
            if view_side is not None and side == view_side:
                # 己方真实速度：buff/印记/特性修正全部计入（流沙统治者+50、变形活画等）
                mark_speed_bonus = 0
                negative = marks.get_mark(state, side, marks.NEGATIVE)
                if negative is not None and negative["id"] == 3:
                    mark_speed_bonus = -10 * negative["stacks"]
                d["eff_speed"] = buffs.get_speed_value(
                    pet, mark_speed_bonus
                ) + traits.query_speed(state, pet)
                for skill_item in d["skills"]:
                    skill_obj = next(
                        (s for s in pet.skills if s.skill_id == skill_item["skill_id"]),
                        None,
                    )
                    if skill_obj is not None:
                        # 己方显示真实能耗：天气/印记/buff/特性修正全部计入，最后统一钳制
                        # （缩壳-2、冰封敌方光环+1、对流翻转等）
                        skill_item["energy_cost"] = skill_utils.true_energy_cost(
                            state, pet, skill_obj
                        )
                        # 己方显示真实威力：含特性威力/属性改写修正（目空/涂鸦/展翅等；
                        # 顺风/破空等先手条件按非先手计算，实际出招时另算）
                        # 对方力竭/缺席（active<0）时无法计算显示威力，保持 None
                        if opponent is not None:
                            extra_percent, extra_flat = traits.query_power(
                                state,
                                pet,
                                opponent,
                                0,
                                0,
                                is_first=False,
                                skill=skill_obj,
                            )
                            # 技能自身威力修正同样计入显示威力（按敌方冻结层数、
                            # 所在槽位加威力等；槽位取当前显示顺序）
                            s_percent, s_flat = skills.query_power(
                                state,
                                pet,
                                skill_obj,
                                skill_index=skill_item["index"],
                                is_first=False,
                            )
                            skill_item["display_power"] = calc_damage(
                                pet,
                                opponent,
                                skill_obj,
                                typechart,
                                extra_power_percent=extra_percent + s_percent,
                                extra_power_flat=extra_flat + s_flat,
                                element=traits.query_skill_element(
                                    state, pet, skill_obj
                                ),
                                state=state,
                                power_override=skills.query_absolute_power(
                                    state, pet, skill_obj, is_first=False
                                ),
                            )["display_power"]
            if view_side is not None and side != view_side:
                # 对方速度范围也显示真实范围（含 buff/印记/特性速度修正，如流沙统治者+50）
                mark_speed_bonus = 0
                negative = marks.get_mark(state, side, marks.NEGATIVE)
                if negative is not None and negative["id"] == 3:
                    mark_speed_bonus = -10 * negative["stacks"]
                flat = (
                    buffs.get_buff_value(pet, buffs.BuffType.SPEED) * 10
                    + mark_speed_bonus
                    + traits.query_speed(state, pet)
                )
                percent = buffs.get_buff_value(pet, buffs.BuffType.SPEED_PERCENT)
                d["eff_speed_range"] = [
                    int((pet.speed_range[0] + flat) * (1.0 + percent * 0.1)),
                    int((pet.speed_range[1] + flat) * (1.0 + percent * 0.1)),
                ]
                revealed_ids = state.revealed[side]
                d["skills"] = [
                    {
                        "name": skill["name"],
                        "energy_cost": skill["energy_cost"],
                        "display_power": skill.get("display_power"),
                        "element": skill.get("element"),  # 原始属性（不做特性改写）
                    }
                    for skill in d["skills"]
                    if skill["skill_id"] in revealed_ids
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
        "weather": state.weather,
        "marks": state.marks,
        "resonance_usage": state.resonance_usage,
        "resonance_magic": state.resonance_magic,
        "resonance_cooldown": state.resonance_cooldown,
        "teams": teams,
    }
