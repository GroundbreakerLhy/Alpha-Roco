"""Pet feature (精灵特性) system.

Every spirit in data/spirits.json may carry a "feature" described by a Chinese
sentence. This module implements those descriptions against the battle engine:

  * load_features()  - feature_id -> {id, desc, name}
  * event hooks      - on_entry / on_leave / on_turn_start / on_turn_end /
                       on_charge / after_skill_use / on_deal_damage /
                       on_take_damage / on_faint / on_kill / on_counter_success /
                       on_energy_gain / on_heal / on_buff_gain
  * modifiers        - modify_power_percent / modify_damage_taken /
                       modify_energy_cost / modify_hit_count / modify_speed /
                       modify_stat / energy_cap_of / usable_indices /
                       can_act / can_switch / round_end_behavior

Effects that depend on mechanisms absent from the headless engine (weather,
bloodline, 巧变/传动/选择/奉献, Pokéball capture, weekend, etc.) are either
approximated with a documented stand-in or left inert with a log note.
"""

from __future__ import annotations

import random
import re

from . import buffs, burst, marks
from .data_loader import load_spirits, calc_all_stats, get_attributes, round_half_up
from .typechart import type_multiplier
from .damage import calc_damage, level_coefficient

_FEATURES = None


def load_features() -> dict:
    global _FEATURES
    if _FEATURES is None:
        _FEATURES = {}
        for sp in load_spirits():
            f = sp.get("feature")
            if f:
                _FEATURES[f["id"]] = {
                    "id": f["id"],
                    "desc": f.get("desc", ""),
                    "name": sp["name"],
                }
    return _FEATURES


def feature_of(pet) -> dict | None:
    if pet is None or getattr(pet, "feature_id", None) is None:
        return None
    return load_features().get(pet.feature_id)


def desc_of(pet) -> str:
    f = feature_of(pet)
    return f["desc"] if f else ""


def _fs(pet, key: str, default=None):
    """Read feature private state."""
    return pet.feature_state.get(key, default)


def _fss(pet, key: str, value) -> None:
    """Write feature private state."""
    pet.feature_state[key] = value


def _active(state, side: str):
    idx = state.active[side]
    if idx < 0:
        return None
    return state.teams[side][idx]


def _opp_side(side: str) -> str:
    return "B" if side == "A" else "A"


def _opp(state, side: str):
    return _active(state, _opp_side(side))


def _team_alive(state, side: str):
    return [p for p in state.teams[side] if p.hp > 0]


def _log(state, side: str, text: str, logs=None) -> None:
    entry = f"{side} [特性] {text}"
    if logs is None:
        state.log.append(entry)
    else:
        logs.append(entry)


def _add_buff(pet, buff_type: str, value: int, logs=None, side: str = "") -> None:
    buffs.add_buff(pet, buff_type, value)
    if logs is not None and value != 0:
        logs.append(f"  [特性] {pet.name} 获得 {buff_type} +{value}")


def _heal(state, side: str, pet, amount: int, logs=None) -> int:
    """回复生命(经 on_heal 特性钩子:无法回复/随机分配/过量转化),返回实际回复量。"""
    if amount <= 0:
        return 0
    actual = on_heal(state, side, pet, amount, logs)
    if actual > 0:
        pet.hp = min(pet.max_hp, pet.hp + actual)
    return actual


def _energy(state, side: str, pet, amount: int, logs=None) -> None:
    """Add energy respecting cap; called instead of _add_energy so features
    that change the energy cap or react to energy gain can hook in."""
    if amount == 0:
        return
    cap = energy_cap_of(pet)
    before = pet.energy
    pet.energy = min(cap, pet.energy + amount)
    gained = pet.energy - before
    if gained <= 0:
        return
    if logs is not None:
        logs.append(f"  [特性] {pet.name} 回复 {gained} 能量")
    on_energy_gain(state, side, pet, gained, logs)


def on_heal(state, side: str, pet, amount: int, logs) -> int:
    """Return the amount that actually heals (features may redirect/negate)."""
    if amount <= 0:
        return 0
    d = desc_of(pet)
    if "自己无法回复生命，而是将回复生命变为敌方扣除等量生命" in d:
        op = _opp(state, side)
        if op is not None:
            op.hp = max(0, op.hp - amount)
            if logs is not None:
                logs.append(f"  [特性] {pet.name} 将回复转化为敌方损失 {amount} 生命")
        return 0
    if "获得能量或生命时，会将等量的能量或生命随机分配给场下的精灵" in d:
        bench = [p for p in state.teams[side] if p.hp > 0 and p is not pet]
        if bench:
            target = random.choice(bench)
            target.hp = min(target.max_hp, target.hp + amount)
            if logs is not None:
                logs.append(f"  [特性] {pet.name} 将 {amount} 生命分配给场下 {target.name}")
        return amount
    if "入场时获得50%吸血，每过量回复5%生命转化为10%物攻" in d:
        overflow = max(0, pet.hp + amount - pet.max_hp)
        if overflow > 0:
            layers = int(overflow / (pet.max_hp * 0.05))
            if layers > 0:
                buffs.add_buff(pet, buffs.BuffType.ATK, layers)
                if logs is not None:
                    logs.append(f"  [特性] {pet.name} 过量回复转化为物攻+{layers * 10}%")
    return amount


# ---------------------------------------------------------------------------
# Modifiers
# ---------------------------------------------------------------------------

def energy_cap_of(pet) -> int:
    d = desc_of(pet)
    if d == "自己的能量可以超过能量上限。":
        return 99
    if "突破能量上限并立即回复10能量" in d:
        return 99
    return pet.energy_cap


def usable_indices(pet):
    if pet.feature_skill_index_restriction is not None:
        return pet.feature_skill_index_restriction
    return None


def can_act(pet) -> bool:
    d = desc_of(pet)
    # 能量不足时消耗生命代替能量(战斗内由 on_energy 缺口处理),此处不拦截
    return True


def can_switch(pet) -> bool:
    d = desc_of(pet)
    if "每次行动后脱离" in d or "每次行动后脱离。" in d:
        return True
    return True


def modify_power_percent(pet, skill, base_percent: float, is_first: bool) -> float:
    """Extra skill-power percent contributed by features. base_percent is the
    value already accumulated by buffs/marks (not needed here)."""
    extra = 0.0
    d = desc_of(pet)
    if not d:
        return 0.0

    # 若先于敌方攻击，本次技能威力+N%
    m = re.search(r"若先于敌方攻击，本次技能威力\+(\d+)%", d)
    if m and is_first:
        extra += float(m.group(1))

    # 携带的能耗为1/大于3 的技能威力+N%(含鸭吉吉国王变体)
    cost = getattr(skill, "energy_cost", None)
    if "能耗为1的技能威力+50%" in d and cost == 1:
        extra += 50.0
    if "携带的能耗为1的技能，威力+50%" in d and cost == 1:
        extra += 50.0
    if "携带的能耗大于3的技能，威力+40%" in d and cost is not None and cost > 3:
        extra += 40.0
    # 携带的无额外效果的攻击技能，威力+30%(近似:所有攻击技能)
    if "携带的无额外效果的攻击技能，威力+30%" in d and skill.category in (0, 1):
        extra += 30.0

    # 携带的非光系技能，威力+25%(含"额外获得三个随机技能"变体)
    if "非光系技能威力+25%" in d and skill.element != 4:
        extra += 25.0

    # 使用非本系技能时威力+50%
    if "使用非本系技能时威力+50%" in d and skill.element not in pet.attributes:
        extra += 50.0

    # 携带的「虫鸣」技能威力+20
    if "「虫鸣」技能威力+20" in d and skill.name == "虫鸣":
        extra += 20.0
    if "音波弹/音爆/金属噪音/午夜噪音威力提升" in d and skill.name in (
        "音波弹", "音爆", "金属噪音", "午夜噪音",
    ):
        extra += 40.0

    # 能耗为0的技能威力+30%(己方水系技能计数特性)
    if "能耗为0的技能威力+30%" in d and getattr(skill, "energy_cost", None) == 0:
        extra += 30.0

    # 天气为暴风雪时，冰系技能威力+100%
    if "天气为暴风雪时" in d and skill.element == 6:
        extra += 100.0

    # 每携带1个冰系技能进入战斗，地系技能威力+10%
    m = re.search(r"每携带1个冰系技能进入战斗，地系技能威力\+(\d+)%", d)
    if m and skill.element == 5:
        ice_count = sum(1 for s in pet.skills if s.element == 6)
        extra += float(m.group(1)) * ice_count

    # 己方精灵每使用1次防御技能，自己入场时机械系和地系技能威力+10%
    if "机械系和地系技能威力+10%" in d and skill.element in (16, 5):
        extra += 10.0 * _fs(pet, "ally_defense_skill_count", 0)

    # 己方精灵每应对1次，自己入场时水系和武系技能威力+20%
    if "水系和武系技能威力+20%" in d and skill.element in (3, 11):
        extra += 20.0 * _fs(pet, "ally_counter_count", 0)

    # 己方精灵每使用1次状态技能，自己入场时毒系和萌系技能威力+10
    if "毒系和萌系技能威力+10" in d and skill.element in (9, 13):
        extra += 10.0 * _fs(pet, "ally_status_count", 0)

    # 己方精灵每使用1次火系技能，自己入场时获得全技能威力+10
    if "全技能威力+10" in d:
        extra += 10.0 * _fs(pet, "ally_fire_count", 0)

    # 敌方携带技能总能耗每有1点，自己攻击时威力+10%
    if "敌方携带技能总能耗每有1点" in d:
        pass  # 需要 state,由 modify_power_percent_state 补充

    # 行动时，敌方每有1层增益，本次行动技能威力+10%，速度+5
    if "敌方每有1层增益" in d:
        pass  # 需要 state,由 modify_power_percent_state 补充

    # 光系技能威力永久+30(火系选择特性的另一选项)
    if _fs(pet, "light_power_bonus", False) and skill.element == 4:
        extra += 30.0

    return extra


def modify_power_percent_state(state, side: str, skill, is_first: bool) -> float:
    """State-dependent power bonuses (star-mark, opponent team skills, etc.)."""
    pet = _active(state, side)
    if pet is None:
        return 0.0
    d = desc_of(pet)
    extra = 0.0

    # 敌方每有1层星陨印记，自己的地系技能威力+20% / 自己的技能威力+20%
    m = re.search(r"敌方每有1层星陨印记，自己的(?:地系)?技能威力\+(\d+)%", d)
    if m:
        neg = marks.get_mark(state, _opp_side(side), marks.NEGATIVE)
        if neg is not None and neg["id"] == 7:
            extra += float(m.group(1)) * neg["stacks"]

    # 敌方每携带1种系别的技能，自己攻击时威力+10
    if "敌方每携带1种系别的技能，自己攻击时威力+10" in d:
        opp = _opp(state, side)
        if opp is not None:
            kinds = {s.element for s in opp.skills}
            extra += 10.0 * len(kinds)

    # 敌方携带技能总能耗每有1点，自己攻击时威力+10%
    if "敌方携带技能总能耗每有1点" in d:
        opp = _opp(state, side)
        if opp is not None:
            total = sum(s.energy_cost for s in opp.skills)
            extra += 10.0 * total

    # 行动时，敌方每有1层增益，本次行动技能威力+10%，速度+5
    if "敌方每有1层增益" in d:
        opp = _opp(state, side)
        if opp is not None:
            kinds = {b.buff_type for b in opp.buffs if b.value > 0}
            extra += 10.0 * len(kinds)

    # 若敌方血脉是非本系/首领/污染血脉，技能威力+100%(血脉=formType)
    opp = _opp(state, side)
    opp_bl = bloodline_of(opp) if opp is not None else "normal"
    if "血脉是首领血脉" in d and opp_bl == "leader":
        extra += 100.0
    if "血脉是污染血脉" in d and opp_bl == "polluted":
        extra += 100.0
    if "血脉是非本系的系别血脉" in d and opp_bl == "special":
        extra += 100.0

    # 攻击时，将敌方所有印记变为相同层数的星陨印记(威力类不适用)
    return extra


def modify_damage_taken(pet, attacker, skill, damage: int) -> int:
    """Reduce damage per feature text. Returns adjusted damage."""
    d = desc_of(pet)
    if not d or damage <= 0:
        return damage

    # 木桶状态:减伤50%
    if _fs(pet, "barrel", False):
        return max(0, int(damage * 0.5))

    # 受到自己携带技能系别的攻击伤害-40% / 抵抗自己携带技能系别的攻击伤害
    if "受到自己携带技能系别的攻击伤害-40%" in d:
        if skill.element in [s.element for s in pet.skills]:
            return max(0, int(damage * 0.6))
    if d == "抵抗自己携带技能系别的攻击伤害。":
        if skill.element in [s.element for s in pet.skills]:
            return max(0, int(damage * 0.5))

    # 受到非敌方系别的技能攻击时伤害-50%
    if "受到非敌方系别的技能攻击时伤害-50%" in d:
        if skill.element not in attacker.attributes:
            return max(0, int(damage * 0.5))

    # 能量等于0的精灵，无法对自己造成伤害(攻击方能量为0时免疫)
    if d == "能量等于0的精灵，无法对自己造成伤害。":
        if attacker is not None and attacker.energy == 0:
            return 0
    if "能耗小于等于1的攻击技能，无法对自己造成伤害" in d:
        if skill.category in (0, 1) and skill.energy_cost <= 1:
            return 0

    # 本精灵受到的克制伤害+25%(背包随机精灵特性)
    if "本精灵受到的克制伤害+25%" in d:
        typechart = __import__("sim.typechart", fromlist=["type_multiplier"])
        mult = type_multiplier(skill.element, pet.attributes, _typechart())
        if mult > 1.0:
            return int(damage * 1.25)

    # 若后于对手行动，自己受到的伤害+25%(普通系变翼系特性)
    if "若后于对手行动，自己受到的伤害+25%" in d:
        # 后手信息由 battle 侧传入,此处保守处理
        pass

    return damage


def _typechart():
    from .data_loader import load_typechart
    return load_typechart()


def modify_energy_cost(pet, skill, base_cost: int) -> int:
    """Extra energy-cost modifier from features (applied on top of buffs)."""
    d = desc_of(pet)
    if not d:
        return 0

    # 携带的防御技能能耗-2
    if "防御技能能耗-2" in d and skill.category == 2:
        return -2

    # 携带的水系技能获得选择…(跳过)
    # 非幻系技能能耗-2(复写/借用/取念)
    if "非幻系技能能耗-2" in d and skill.element != 17:
        return -2

    # 使用水系技能后，全技能能耗-1/-2 → 永久型由特性状态记录
    if "全技能能耗-1" in d or "全技能能耗-2" in d:
        pass  # 由 after_skill_use 累积进 feature_state,此处不处理

    # 敌方精灵离场时，自己获得全技能能耗-3
    if "全技能能耗-3" in d:
        pass  # 由 on_kill/on_leave 处理,此处由 state 记录

    return 0


def modify_hit_count(pet, skill, count: int) -> int:
    d = desc_of(pet)
    if not d:
        return count
    # 使用翼系技能后，获得连击数+1 → 触发型(非永久),在 after_skill_use 处理
    # 自己每有1层萌化，获得连击数+3
    if "自己每有1层萌化，获得连击数+3" in d:
        cute = buffs.get_buff_value(pet, buffs.BuffType.CUTE)
        return count + 3 * cute
    # 敌方每有1层中毒效果，自己获得连击数+1
    if "敌方每有1层中毒效果，自己获得连击数+1" in d:
        return count  # 需要 opponent 信息,由 battle 侧补充
    # 自己每失去25%生命，连击数+2 → 触发型
    # 在场时，所有精灵连击数固定为2/1 → 全局,battle 侧处理
    return count


def modify_speed(pet, speed: int) -> int:
    d = desc_of(pet)
    if not d:
        return speed
    # 天气为沙暴时，自己获得速度+50
    if "天气为沙暴时" in d:
        return speed + 50  # 天气缺失,默认视为触发(近似)
    # 若敌方技能足够击败自己，回合开始时自己获得速度+50
    if "自己获得速度+50" in d:
        return speed + 50
    # 使敌方获得中毒时，也会使其获得物攻-40%和速度-40 → 对敌方,不在此
    return speed


def modify_stat(pet, stat: str, mult: float) -> float:
    """Extra stat multiplier from features (multiplied on top of buffs)."""
    d = desc_of(pet)
    if not d:
        return mult
    # 每有1能量，获得双防+10%
    if "每有1能量，获得双防+10%" in d and stat in ("def", "spdef"):
        return mult * (1.0 + 0.1 * pet.energy)
    # 总技能能耗小于4时，自己获得双防+80%
    if "总技能能耗小于4时，自己获得双防+80%" in d and stat in ("def", "spdef"):
        total = sum(s.energy_cost for s in pet.skills)
        if total < 4:
            return mult * 1.8
    # 队伍存在虫系精灵，自己获得双攻+50%
    if "队伍存在虫系精灵，自己获得双攻+50%" in d and stat in ("atk", "spatk"):
        return mult * 1.5  # 需要队伍信息,近似为触发
    # 双方场上每有1种不同的增益，自己获得物防+20%
    if "双方场上每有1种不同的增益，自己获得物防+20%" in d and stat == "def":
        return mult  # 需要 state,由 battle 侧补充
    # 双方场上每有1种不同的印记，自己获得魔攻+50%
    if "双方场上每有1种不同的印记，自己获得魔攻+50%" in d and stat == "spatk":
        return mult  # 需要 state,由 battle 侧补充
    # 己方队伍中每有1只力竭的精灵，自己获得双攻+30%
    if "己方队伍中每有1只力竭的精灵，自己获得双攻+30%" in d and stat in ("atk", "spatk"):
        return mult  # 需要 state
    # 双方队伍中每有1只力竭的精灵，自己获得双攻+30%
    if "双方队伍中每有1只力竭的精灵，自己获得双攻+30%" in d and stat in ("atk", "spatk"):
        return mult  # 需要 state
    # 入场时，若自己魔力值为1，自己获得双攻+100%
    if "自己获得双攻+100%" in d and stat in ("atk", "spatk"):
        return mult * 2.0
    # 入场时，若敌方魔力值为1，自己获得双防+100%
    if "自己获得双防+100%" in d and stat in ("def", "spdef"):
        return mult * 2.0
    return mult


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

_OFFERING_BUFFS = [
    ("atk", 1), ("spatk", 1), ("def", 1), ("spdef", 1),
    ("speed", 1), ("skill_power_percent", 10), ("lifesteal", 1),
    ("hit_count_flat", 1),
]


def _random_offering(state, side: str, logs) -> None:
    """奉献:随机增益。给己方一只随机存活精灵一个随机增益。"""
    alive = _team_alive(state, side)
    if not alive:
        return
    target = random.choice(alive)
    buff_type, value = random.choice(_OFFERING_BUFFS)
    buffs.add_buff(target, buff_type, value)
    if logs is not None:
        logs.append(f"  [特性] 奉献：{target.name} 获得 {buff_type} +{value}")


def _transform_spirit(state, side: str, target_name: str, logs) -> None:
    """将当前精灵替换为指定精灵形态(类似萌化退化),保留生命比例与技能。"""
    pet = _active(state, side)
    if pet is None:
        return
    for sp in load_spirits():
        if sp["name"] == target_name:
            old_max = pet.max_hp
            pet.spirit_id = sp["id"]
            pet.name = sp["name"]
            pet.attributes = get_attributes(sp)
            new_stats = calc_all_stats(sp["stats"], ivs=pet.ivs, nature=pet.nature)
            pet.stats = new_stats
            pet.max_hp = new_stats["hp"]
            pet.hp = min(pet.max_hp, round_half_up(pet.hp * pet.max_hp / old_max)) if old_max else pet.max_hp
            pet.energy = min(energy_cap_of(pet), pet.energy)
            if logs is not None:
                logs.append(f"  [特性] {pet.name} 变为 {target_name}")
            return
    if logs is not None:
        logs.append(f"  [特性] 找不到目标形态 {target_name}")


def _qiyi_queen_name(pet) -> str:
    """棋绮后形态名:根据当前精灵名判断白子/黑子。"""
    if "黑" in pet.name:
        return "棋绮后（黑子）"
    return "棋绮后（白子）"


def _replace_first_skill(pet, element: int, logs) -> None:
    """将首个技能替换为指定元素的基础攻击技能(愿力冲击数据缺失,用同元素替代)。"""
    if not pet.skills:
        return
    from .data_loader import load_skills
    candidates = [
        s for s in load_skills()
        if s.get("element") == element and s.get("category") in (0, 1)
        and s.get("power") and s.get("name") != "蓄能"
    ]
    if not candidates:
        if logs is not None:
            logs.append(f"  [特性] 无元素 {element} 攻击技能可替换")
        return
    raw = sorted(candidates, key=lambda s: -s["power"])[0]
    from .models import BattleSkill
    pet.skills[0] = BattleSkill(
        skill_id=raw["id"], name=raw["name"], element=raw["element"],
        category=raw["category"], power=raw.get("power"),
        energy_cost=raw.get("energyCost", 0), desc=raw.get("desc", ""),
    )
    if logs is not None:
        logs.append(f"  [特性] 首个技能替换为 {raw['name']}")


def _entry_buffs_for(state, side: str, pet, logs) -> None:
    """入场时根据己方/敌方累计技能使用情况获得的增益。"""
    d = desc_of(pet)
    if not d:
        return
    # 队伍中每有1只其他的虫系精灵，入场时获得攻防速+10%/15%
    m = re.search(r"队伍中每有1只其他的虫系精灵，自己入场时获得攻防速\+(\d+)%", d)
    if m:
        team = state.teams[side]
        bug_count = sum(1 for p in team if p is not pet and p.hp > 0 and 10 in p.attributes)
        if "攻防速+10%" in d:
            buffs.add_buff(pet, buffs.BuffType.ATK, bug_count)
            buffs.add_buff(pet, buffs.BuffType.DEF, bug_count)
            buffs.add_buff(pet, buffs.BuffType.SPEED, bug_count)
        else:
            _fss(pet, "entry_bug_pct", 15 * bug_count)
        if bug_count and logs is not None:
            logs.append(f"  [特性] 虫系队友 {bug_count} 只，入场获得攻防速加成")
    # 己方精灵每使用1次水系技能，自己入场时获得全技能能耗-1
    if "己方精灵每使用1次水系技能，自己入场时获得全技能能耗-1" in d:
        n = _fs(pet, "ally_water_count", 0)
        if n > 0:
            buffs.add_buff(pet, buffs.BuffType.ENERGY_COST, -n)
    # 己方其他精灵每有1层萌化，自己入场时全技能能耗-1
    if "己方其他精灵每有1层萌化，自己入场时全技能能耗-1" in d:
        cute = sum(
            buffs.get_buff_value(p, buffs.BuffType.CUTE)
            for p in state.teams[side] if p is not pet and p.hp > 0
        )
        if cute > 0:
            buffs.add_buff(pet, buffs.BuffType.ENERGY_COST, -cute)
    # 己方精灵每使用1次火系技能，自己入场时获得全技能威力+10
    if "己方精灵每使用1次火系技能，自己入场时获得全技能威力+10" in d:
        n = _fs(pet, "ally_fire_count", 0)
        if n > 0:
            buffs.add_buff(pet, buffs.BuffType.SKILL_POWER_FLAT, 10 * n)
    # 敌方每使用1次「聚能」技能或更换精灵，自己入场时获得魔攻+20%[和魔防+10%]
    if "敌方每使用1次「聚能」技能或更换精灵" in d:
        n = _fs(pet, "opp_charge_switch_count", 0)
        if n > 0:
            buffs.add_buff(pet, buffs.BuffType.SPATK, 2 * n)
            if "魔防+10%" in d:
                buffs.add_buff(pet, buffs.BuffType.SPDEF, n)
    # 入场前己方精灵每使用1次火系技能，获得攻防+10%，速度+10(最多10次)
    if "入场前己方精灵每使用1次火系技能，获得攻防+10%，速度+10" in d:
        n = min(_fs(pet, "ally_fire_count", 0), 10)
        if n > 0:
            buffs.add_buff(pet, buffs.BuffType.ATK, n)
            buffs.add_buff(pet, buffs.BuffType.DEF, n)
            buffs.add_buff(pet, buffs.BuffType.SPEED, n)
    # 初始生命为10%，入场前己方精灵每使用1次草系技能，回复30%生命
    if "初始生命为10%" in d:
        if pet.entry_count == 0:
            pet.hp = max(1, int(pet.max_hp * 0.1))
        else:
            n = _fs(pet, "ally_grass_count", 0)
            if n > 0:
                _heal(state, side, pet, int(pet.max_hp * 0.3 * n), logs)
    # 初始能量为0，入场前己方精灵每放1次X系技能，回复3能量
    m = re.search(r"初始能量为0，入场前己方精灵每放1次([\u4e00-\u9fa5]+)系技能，回复3能量", d)
    if m:
        if pet.entry_count == 0:
            pet.energy = 0
        kind = {"地": "ally_earth_count", "冰": "ally_ice_count", "火": "ally_fire_count"}.get(m.group(1))
        n = _fs(pet, kind, 0) if kind else 0
        if n > 0:
            _energy(state, side, pet, 3 * n, logs)
    # 突破能量上限并立即回复10能量，入场前己方精灵每放1次地系技能，回复3能量
    if "突破能量上限并立即回复10能量" in d:
        _energy(state, side, pet, 10, logs)
        n = _fs(pet, "ally_earth_count", 0)
        if n > 0:
            _energy(state, side, pet, 3 * n, logs)
    # 初始能量为0，入场前己方精灵每成功应对1次，回复5能量
    if "初始能量为0，入场前己方精灵每成功应对1次，回复5能量" in d:
        if pet.entry_count == 0:
            pet.energy = 0
        n = _fs(pet, "ally_counter_count", 0)
        if n > 0:
            _energy(state, side, pet, 5 * n, logs)
    # 每入场1次，永久获得双攻+30%/40%
    m = re.search(r"每入场1次，获得双攻永久\+(\d+)%", d)
    if m:
        gain = int(m.group(1)) // 10
        buffs.add_buff(pet, buffs.BuffType.ATK, gain * pet.entry_count, buffs.DurationKind.PERMANENT)
        buffs.add_buff(pet, buffs.BuffType.SPATK, gain * pet.entry_count, buffs.DurationKind.PERMANENT)


def _freeze_opponent(state, side: str, n: int, logs) -> None:
    """使敌方获得冻结;施加者带冻结增强特性时追加效果。"""
    op = _opp(state, side)
    if op is None:
        return
    d = desc_of(_active(state, side))
    if "使敌方获得冻结时，也会使其获得2层冻结" in d:
        n += 2
    if "使敌方获得冻结时，也会使其获得全技能能耗+1" in d:
        buffs.add_buff(op, buffs.BuffType.ENERGY_COST, 1)
        if logs is not None:
            logs.append(f"  [特性] 敌方获得全技能能耗+1")
    buffs.add_buff(op, buffs.BuffType.FREEZE, n)
    if logs is not None:
        logs.append(f"  [特性] 敌方获得 {n} 层冻结")


# ---------------------------------------------------------------------------
# 入场
# ---------------------------------------------------------------------------

def on_entry(state, side: str, logs=None) -> None:
    pet = _active(state, side)
    if pet is None:
        return
    d = desc_of(pet)
    if not d:
        return
    pet.entry_count += 1
    logs = logs if logs is not None else []

    if d == "入场时获得50%吸血。":
        buffs.add_buff(pet, buffs.BuffType.LIFESTEAL, 5)
        logs.append(f"  [特性] {pet.name} 入场获得 50% 吸血")
    if d == "入场时获得50%吸血，每过量回复5%生命转化为10%物攻。":
        buffs.add_buff(pet, buffs.BuffType.LIFESTEAL, 5)
        _fss(pet, "overheal_to_atk", True)
        logs.append(f"  [特性] {pet.name} 入场获得 50% 吸血(过量回复转化物攻)")
    m = re.search(r"入场时偷取敌方场上所有精灵(\d+)能量", d)
    if m:
        n = int(m.group(1))
        stolen = 0
        for op in _team_alive(state, _opp_side(side)):
            take = min(n, op.energy)
            op.energy -= take
            stolen += take
        if stolen > 0:
            _energy(state, side, pet, stolen, logs)
            logs.append(f"  [特性] {pet.name} 偷取敌方 {stolen} 能量")
    if d == "入场首回合，获得物攻+100%。":
        buffs.add_buff(pet, buffs.BuffType.ATK, 10, buffs.DurationKind.TEMPORARY, current_turn=state.turn)
        logs.append(f"  [特性] {pet.name} 入场首回合物攻+100%")
    if d == "入场时，获得物攻+100%，每次行动后-20%。":
        buffs.add_buff(pet, buffs.BuffType.ATK, 10)
        _fss(pet, "entry_atk_bonus", True)
        logs.append(f"  [特性] {pet.name} 入场物攻+100%(行动后衰减)")
    if d == "入场时，若自己魔力值为1，自己获得双攻+100%。":
        if state.magic[side] == 1:
            buffs.add_buff(pet, buffs.BuffType.ATK, 10)
            buffs.add_buff(pet, buffs.BuffType.SPATK, 10)
            logs.append(f"  [特性] {pet.name} 魔力为1,双攻+100%")
    if d == "入场时，若敌方魔力值为1，自己获得双防+100%。":
        if state.magic[_opp_side(side)] == 1:
            buffs.add_buff(pet, buffs.BuffType.DEF, 10)
            buffs.add_buff(pet, buffs.BuffType.SPDEF, 10)
            logs.append(f"  [特性] {pet.name} 敌方魔力为1,双防+100%")
    if "自己入场时敌方获得2层冻结" in d:
        _freeze_opponent(state, side, 2, logs)
    if "全技能能耗+2" in d and "王国入夜后" in d:
        if state.is_night:
            buffs.add_buff(pet, buffs.BuffType.ENERGY_COST, 2, buffs.DurationKind.PERMANENT)
            logs.append(f"  [特性] {pet.name} 王国入夜,全技能能耗+2")
    elif "可以学习全部攻击技能石" in d:
        buffs.add_buff(pet, buffs.BuffType.ENERGY_COST, 2, buffs.DurationKind.PERMANENT)
        logs.append(f"  [特性] {pet.name} 全技能能耗+2")
    # 根据自己的血脉，入场时获得不同效果
    if d == "根据自己的血脉，入场时获得不同效果。":
        bl = bloodline_of(pet)
        if bl == "leader":
            buffs.add_buff(pet, buffs.BuffType.ATK, 2)
            buffs.add_buff(pet, buffs.BuffType.DEF, 2)
            buffs.add_buff(pet, buffs.BuffType.SPEED, 2)
            _energy(state, side, pet, 2, logs)
            logs.append(f"  [特性] {pet.name} 首领血脉,入场攻防速+20%,回2能量")
        elif bl == "polluted":
            buffs.add_buff(pet, buffs.BuffType.ATK, 3)
            buffs.add_buff(pet, buffs.BuffType.SPATK, 3)
            logs.append(f"  [特性] {pet.name} 污染血脉,入场双攻+30%")
        elif bl == "special":
            for st in ("atk", "spatk", "def", "spdef"):
                buffs.add_buff(pet, st, 1)
            logs.append(f"  [特性] {pet.name} 特殊血脉,入场四维+10%")
        else:
            _energy(state, side, pet, 3, logs)
            logs.append(f"  [特性] {pet.name} 普通血脉,入场回3能量")
    # 根据捕捉所用的咕噜球，入场时获得不同效果
    if "根据捕捉所用的咕噜球" in d:
        ball = pet.capture_ball
        if "高级" in ball:
            buffs.add_buff(pet, buffs.BuffType.ATK, 3)
            buffs.add_buff(pet, buffs.BuffType.SPEED, 1)
            logs.append(f"  [特性] {pet.name} 高级球,入场物攻+30%,速度+10")
        elif "中级" in ball:
            buffs.add_buff(pet, buffs.BuffType.ATK, 2)
            logs.append(f"  [特性] {pet.name} 中级球,入场物攻+20%")
        elif "国王" in ball:
            for st in ("atk", "spatk", "def", "spdef", "speed"):
                buffs.add_buff(pet, st, 2)
            logs.append(f"  [特性] {pet.name} 国王球,入场全属性+20%")
        else:
            buffs.add_buff(pet, buffs.BuffType.ATK, 1)
            logs.append(f"  [特性] {pet.name} 普通球,入场物攻+10%")
    # 额外获得三个未携带的随机技能(首次入场)
    if "额外获得三个未携带的随机技能" in d and pet.entry_count == 1:
        _append_random_skills(pet, 3, logs=logs)
    # 可以学习全部攻击技能石(近似:补充攻击技能至8个)
    if "可以学习全部攻击技能石" in d and pet.entry_count == 1 and len(pet.skills) < 8:
        _append_random_skills(pet, 8 - len(pet.skills), attack_only=True, logs=logs)
    # 己方精灵每完整使用1次「选择」技能(「明」和「暗」各1次)
    if "「明」和「暗」各1次" in d:
        ming = sum(p.feature_state.get("ming_count", 0) for p in state.teams[side] if p.hp > 0)
        an = sum(p.feature_state.get("an_count", 0) for p in state.teams[side] if p.hp > 0)
        complete = min(ming, an)
        if complete > 0:
            buffs.add_buff(pet, buffs.BuffType.ATK, 4 * complete, buffs.DurationKind.PERMANENT)
            for p in state.teams[side]:
                p.feature_state["ming_count"] = max(0, p.feature_state.get("ming_count", 0) - complete)
                p.feature_state["an_count"] = max(0, p.feature_state.get("an_count", 0) - complete)
            logs.append(f"  [特性] {pet.name} 明暗各{complete}次,入场物攻永久+{complete * 40}%")
    # 背包里会变化出随机精灵
    if "背包里会变化出随机精灵" in d and pet.bag_pet_index is None:
        from . import data_loader
        playable = [
            s for s in data_loader.load_spirits()
            if s.get("stats") and s["stats"].get("hp") and s["name"] != pet.name
        ]
        if playable:
            sp = random.choice(playable)
            new_pet = data_loader.make_battle_pet(sp, side)
            state.teams[side].append(new_pet)
            pet.bag_pet_index = len(state.teams[side]) - 1
            logs.append(f"  [特性] 背包变化出随机精灵 {sp['name']}")
    # 1号位/2号位技能 传动/迅捷
    if ("1号位技能获得传动1" in d or "1号位技能获得迅捷和传动1" in d
            or "1号和2号位技能获得传动1" in d) and pet.skills:
        pet.skills[0].drive = 1
        if "1号位技能获得迅捷和传动1" in d:
            pet.skills[0].swift = True
        if "1号和2号位技能获得传动1" in d and len(pet.skills) > 1:
            pet.skills[1].drive = 1
    if d == "入场时，复制敌方的增益。在场时，若敌方获得增益自己也会获得。":
        op = _opp(state, side)
        if op is not None:
            copied = 0
            for b in op.buffs:
                if b.value > 0:
                    buffs.add_buff(pet, b.buff_type, b.value)
                    copied += 1
            if copied:
                logs.append(f"  [特性] {pet.name} 复制敌方 {copied} 种增益")
    if d == "首次入场时，失去自己一半的当前生命。":
        if pet.entry_count == 1:
            pet.hp = max(1, pet.hp - pet.hp // 2)
            logs.append(f"  [特性] {pet.name} 首次入场失去一半当前生命")
    if d == "根据捕捉所用的咕噜球，入场时获得不同效果。":
        logs.append(f"  [特性] {pet.name} 咕噜球机制缺失,入场效果未实现")
    if d == "根据自己的血脉，入场时获得不同效果。":
        logs.append(f"  [特性] {pet.name} 血脉机制缺失,入场效果未实现")
    _entry_buffs_for(state, side, pet, logs)

    # 携带的攻击技能获得迸发：威力+40(入场后首次攻击生效)
    if "携带的攻击技能获得迸发：威力+40" in d:
        _fss(pet, "first_attack_pending", True)

    # 入场继承(离场精灵遗留)
    pending = state.feature_pending_inherit.get(side) if hasattr(state, "feature_pending_inherit") else None
    if pending:
        for b in pending:
            if b.value != 0:
                buffs.add_buff(pet, b.buff_type, b.value)
        if logs is not None:
            logs.append(f"  [特性] {pet.name} 继承离场精灵的增益减益")
        state.feature_pending_inherit[side] = []


# ---------------------------------------------------------------------------
# 离场
# ---------------------------------------------------------------------------

def on_leave(state, side: str, logs=None) -> None:
    """主动离场(换人)时结算。outgoing 仍为当前 active 精灵。"""
    pet = _active(state, side)
    if pet is None:
        return
    d = desc_of(pet)
    logs = logs if logs is not None else []

    if d == "离场时回复10能量。":
        pet.energy = min(energy_cap_of(pet), pet.energy + 10)
        logs.append(f"  [特性] {pet.name} 离场回复10能量")
    if d == "离场后，自己的增益和减益会被更换入场的精灵继承。":
        if not hasattr(state, "feature_pending_inherit"):
            state.feature_pending_inherit = {"A": [], "B": []}
        state.feature_pending_inherit[side] = [b for b in pet.buffs if b.value != 0]
        logs.append(f"  [特性] {pet.name} 离场,增益减益将由下一位继承")
    if d == "离场后，更换入场的精灵以木桶状态登场。":
        if not hasattr(state, "feature_pending_entry"):
            state.feature_pending_entry = {"A": [], "B": []}
        state.feature_pending_entry[side].append(("barrel",))
        logs.append(f"  [特性] {pet.name} 离场,下一位以木桶状态登场")
    if d == "离场后，更换入场的精灵回复20%生命且免疫寄生。":
        if not hasattr(state, "feature_pending_entry"):
            state.feature_pending_entry = {"A": [], "B": []}
        state.feature_pending_entry[side].append(("heal_pct", 0.2))
        logs.append(f"  [特性] {pet.name} 离场,下一位入场回复20%生命")
    if d == "离场后，更换入场的精灵获得双防+20%且免疫冻结。":
        if not hasattr(state, "feature_pending_entry"):
            state.feature_pending_entry = {"A": [], "B": []}
        state.feature_pending_entry[side].append(("stat", ("def", "spdef"), 2))
        logs.append(f"  [特性] {pet.name} 离场,下一位入场双防+20%")
    if d == "离场后，更换入场的精灵获得双攻+20%且免疫灼烧。":
        if not hasattr(state, "feature_pending_entry"):
            state.feature_pending_entry = {"A": [], "B": []}
        state.feature_pending_entry[side].append(("stat", ("atk", "spatk"), 2))
        logs.append(f"  [特性] {pet.name} 离场,下一位入场双攻+20%")
    if d == "自己或其他精灵离场时，更换入场的精灵获得萌化。":
        if not hasattr(state, "feature_pending_entry"):
            state.feature_pending_entry = {"A": [], "B": []}
        state.feature_pending_entry[side].append(("cute", 1))
        logs.append(f"  [特性] {pet.name} 离场,下一位入场获得萌化")
    if d == "自己或其他精灵离场时，自己与更换入场的精灵交换血量百分比。":
        idx = state.active[side]
        nxt = None
        for i, p in enumerate(state.teams[side]):
            if i != idx and p.hp > 0:
                nxt = p
                break
        if nxt is not None:
            pct_a = pet.hp / pet.max_hp
            pct_b = nxt.hp / nxt.max_hp
            pet.hp = max(1, round_half_up(pet.max_hp * pct_b))
            nxt.hp = max(1, round_half_up(nxt.max_hp * pct_a))
            logs.append(f"  [特性] {pet.name} 与换入精灵交换血量百分比")

    # 敌方离场触发类(检查对方在场精灵特性)
    opp_side = _opp_side(side)
    opp = _active(state, opp_side)
    if opp is not None:
        od = desc_of(opp)
        if od == "敌方精灵离场后，更换入场的精灵失去3能量。":
            # 由 battle 在换入敌方精灵后处理(入场所需),此处记录请求
            if not hasattr(state, "feature_pending_entry"):
                state.feature_pending_entry = {"A": [], "B": []}
            state.feature_pending_entry[side].append(("energy_loss", 3))
            logs.append(f"  [特性] {opp.name} 使离场方换入精灵失去3能量")
        if od == "敌方精灵离场后，更换入场的精灵获得5层中毒。":
            if not hasattr(state, "feature_pending_entry"):
                state.feature_pending_entry = {"A": [], "B": []}
            state.feature_pending_entry[side].append(("poison", 5))
            logs.append(f"  [特性] {opp.name} 使离场方换入精灵获得5层中毒")
        if od == "敌方精灵离场后，其增益和减益会被更换入场的精灵继承。":
            if not hasattr(state, "feature_pending_inherit"):
                state.feature_pending_inherit = {"A": [], "B": []}
            state.feature_pending_inherit[side] = [b for b in pet.buffs if b.value != 0]
            logs.append(f"  [特性] {opp.name} 使离场方增益减益被继承")
        if od == "敌方精灵离场时，自己获得全技能能耗-3。":
            buffs.add_buff(opp, buffs.BuffType.ENERGY_COST, -3, buffs.DurationKind.PERMANENT)
            logs.append(f"  [特性] {opp.name} 因敌方离场获得全技能能耗-3")


def apply_pending_entry(state, side: str, logs) -> None:
    """换入精灵入场后应用离场方遗留效果。"""
    pet = _active(state, side)
    if pet is None:
        return
    if not hasattr(state, "feature_pending_entry"):
        state.feature_pending_entry = {"A": [], "B": []}
    for item in state.feature_pending_entry.get(side, []):
        if item[0] == "heal_pct":
            _heal(state, side, pet, int(pet.max_hp * item[1]), logs)
        elif item[0] == "stat":
            for st in item[1]:
                buffs.add_buff(pet, st, item[2])
            logs.append(f"  [特性] {pet.name} 入场获得属性提升")
        elif item[0] == "cute":
            buffs.add_buff(pet, buffs.BuffType.CUTE, item[1])
        elif item[0] == "energy_loss":
            pet.energy = max(0, pet.energy - item[1])
            logs.append(f"  [特性] {pet.name} 入场失去 {item[1]} 能量")
        elif item[0] == "poison":
            buffs.add_buff(pet, buffs.BuffType.POISON, item[1])
            logs.append(f"  [特性] {pet.name} 入场获得 {item[1]} 层中毒")
        elif item[0] == "barrel":
            _fss(pet, "barrel", True)
            logs.append(f"  [特性] {pet.name} 以木桶状态登场(减伤50%)")
    state.feature_pending_entry[side] = []


# ---------------------------------------------------------------------------
# 回合开始
# ---------------------------------------------------------------------------

def check_revive(state, side: str, logs) -> None:
    for pet in state.teams[side]:
        if pet.hp <= 0 and pet.revive_turn is not None and pet.revive_turn <= state.turn:
            pet.hp = pet.max_hp
            pet.revive_turn = None
            logs.append(f"  [特性] {pet.name} 复活")


def on_turn_start(state, side: str, logs=None) -> None:
    pet = _active(state, side)
    if pet is None:
        return
    d = desc_of(pet)
    if not d:
        return
    logs = logs if logs is not None else []

    if d == "回合开始时，技能顺序打乱，4号位的技能能耗-4。":
        if len(pet.skills) >= 4:
            _fss(pet, "slot4_cost_bonus", -4)
            logs.append(f"  [特性] {pet.name} 4号位技能本回合能耗-4")
    if d == "若敌方技能足够击败自己，回合开始时自己获得速度+50。":
        if _enemy_can_ko(state, side):
            buffs.add_buff(pet, buffs.BuffType.SPEED, 5, buffs.DurationKind.TEMPORARY, current_turn=state.turn)
            logs.append(f"  [特性] {pet.name} 敌方足以击败自己,速度+50")
    if d == "若敌方技能足够击败自己，回合开始时自己获得速度+50，双攻+50%。":
        if _enemy_can_ko(state, side):
            buffs.add_buff(pet, buffs.BuffType.SPEED, 5, buffs.DurationKind.TEMPORARY, current_turn=state.turn)
            buffs.add_buff(pet, buffs.BuffType.ATK, 5, buffs.DurationKind.TEMPORARY, current_turn=state.turn)
            buffs.add_buff(pet, buffs.BuffType.SPATK, 5, buffs.DurationKind.TEMPORARY, current_turn=state.turn)
            logs.append(f"  [特性] {pet.name} 敌方足以击败自己,速度+50,双攻+50%")
    if d == "回合开始时若敌方技能足够击败自己，自己获得速度+50，行动后脱离。":
        if _enemy_can_ko(state, side):
            buffs.add_buff(pet, buffs.BuffType.SPEED, 5, buffs.DurationKind.TEMPORARY, current_turn=state.turn)
            _fss(pet, "leave_after_action", True)
            logs.append(f"  [特性] {pet.name} 敌方足以击败自己,速度+50,行动后脱离")
    # 在场时,识破精灵的变化效果,解除其伪装
    if "识破精灵的变化效果，解除其伪装" in d:
        op = _opp(state, side)
        if op is not None and op.disguised:
            op.disguised = False
            logs.append(f"  [特性] {pet.name} 识破 {op.name} 的伪装,解除其变化效果")


def _enemy_can_ko(state, side: str) -> bool:
    pet = _active(state, side)
    opp = _opp(state, side)
    if pet is None or opp is None:
        return False
    typechart = _typechart()
    from .models import BattleSkill
    for sk in opp.skills:
        if sk.category in (0, 1) and sk.power:
            fake = BattleSkill(skill_id=sk.skill_id, name=sk.name, element=sk.element,
                               category=sk.category, power=sk.power, energy_cost=sk.energy_cost, desc=sk.desc)
            r = calc_damage(opp, pet, fake, typechart)
            if r["damage"] >= pet.hp:
                return True
    return False


# ---------------------------------------------------------------------------
# 回合结束
# ---------------------------------------------------------------------------

def on_turn_end(state, side: str, logs=None) -> bool:
    """回合结束时特性结算。返回 True 表示请求本侧精灵脱离/换人。"""
    pet = _active(state, side)
    if pet is None:
        return False
    d = desc_of(pet)
    if not d:
        return False
    logs = logs if logs is not None else []
    want_switch = False

    if d == "回合结束时，回复3能量。":
        _energy(state, side, pet, 3, logs)
    if d == "回合结束时，回复6能量。":
        _energy(state, side, pet, 6, logs)
    if d == "回合结束时，回复12%生命。":
        _heal(state, side, pet, int(pet.max_hp * 0.12), logs)
    if d == "回合结束时，若自己能量为0则脱离。":
        if pet.energy == 0:
            want_switch = True
            logs.append(f"  [特性] {pet.name} 能量为0,回合结束脱离")
    if d == "回合结束时，若自己的能量为0，则回复10能量。":
        if pet.energy == 0:
            _energy(state, side, pet, 10, logs)
    if d == "回合结束时，若场上的己方精灵能量等于0，自己立即替换此精灵。":
        if pet.energy == 0:
            want_switch = True
            logs.append(f"  [特性] {pet.name} 能量为0,立即替换")
    if d == "回合结束时，偷取敌方场上所有精灵1能量。":
        stolen = 0
        for op in _team_alive(state, _opp_side(side)):
            take = min(1, op.energy)
            op.energy -= take
            stolen += take
        if stolen > 0:
            _energy(state, side, pet, stolen, logs)
            logs.append(f"  [特性] {pet.name} 回合结束偷取敌方 {stolen} 能量")
    if d == "回合结束时，双方队伍中的所有精灵回复1能量。":
        for s in ("A", "B"):
            for p in _team_alive(state, s):
                p.energy = min(energy_cap_of(p), p.energy + 1)
        logs.append(f"  [特性] {pet.name} 双方全体回复1能量")
    if d == "回合结束时偷取敌方1层印记。":
        for slot in (marks.POSITIVE, marks.NEGATIVE):
            cur = marks.get_mark(state, _opp_side(side), slot)
            if cur is not None:
                marks.add_mark(state, side, cur["id"], 1)
                logs.append(f"  [特性] {pet.name} 偷取敌方1层印记")
                break
    if d == "回合结束时，敌方获得2层星陨印记。":
        _add_mark(state, side, _opp_side(side), 7, 2, logs)
        logs.append(f"  [特性] {pet.name} 回合结束敌方获得2层星陨印记")
    if d == "回合结束时，敌方每2层中毒转化为1层中毒印记。":
        op = _opp(state, side)
        if op is not None:
            poison = buffs.get_buff_value(op, buffs.BuffType.POISON)
            converted = poison // 2
            if converted > 0:
                _add_mark(state, side, _opp_side(side), 4, converted, logs)
                buffs.add_buff(op, buffs.BuffType.POISON, -(converted * 2))
                logs.append(f"  [特性] {pet.name} 敌方 {converted} 层中毒转化为中毒印记")
    if d == "回合结束时驱散敌方1层印记，且驱散后己方队伍获得1次随机奉献。":
        for slot in (marks.POSITIVE, marks.NEGATIVE):
            cur = marks.get_mark(state, _opp_side(side), slot)
            if cur is not None:
                marks.clear_side(state, _opp_side(side), slot)
                logs.append(f"  [特性] {pet.name} 驱散敌方1层印记")
                break
        _random_offering(state, side, logs)
    if d == "回合结束时，己方队伍获得1次随机奉献。":
        _random_offering(state, side, logs)
    if d == "每场战斗1次，回合结束时，若自己累计消耗的能量恰好为12，则回满能量和生命。":
        if pet.energy_spent_total == 12 and not _fs(pet, "twelve_used", False):
            pet.energy = energy_cap_of(pet)
            pet.hp = pet.max_hp
            _fss(pet, "twelve_used", True)
            logs.append(f"  [特性] {pet.name} 累计消耗12能量,回满能量和生命")
    if d == "王国入夜后，进入战斗时获得全技能能耗+2，回合结束时自己回复5%生命和1能量。" and state.is_night:
        _heal(state, side, pet, int(pet.max_hp * 0.05), logs)
        _energy(state, side, pet, 1, logs)
    if d == "使用防御技能后，回合结束时脱离。":
        if _fs(pet, "used_defense_this_turn", False):
            want_switch = True
            logs.append(f"  [特性] {pet.name} 使用防御技能,回合结束脱离")
    if d == "使用光系技能后，回合结束时自己返场。":
        logs.append(f"  [特性] {pet.name} 返场机制在无头引擎中无实际效果")
    if d == "若使用技能能耗高于敌方，回合结束敌方失去能耗之差的能量。":
        my_skill = _fs(pet, "last_skill_cost", None)
        opp = _opp(state, side)
        opp_cost = _fs(opp, "last_skill_cost", None) if opp else None
        if my_skill is not None and opp_cost is not None and my_skill > opp_cost:
            diff = my_skill - opp_cost
            opp.energy = max(0, opp.energy - diff)
            logs.append(f"  [特性] {pet.name} 敌方失去 {diff} 能量(能耗差)")
    if d == "本回合与敌方使用的技能在系别/类型/能耗上每有1项相同，回合结束时获得物攻和物防永久+10%。":
        my_sk = _fs(pet, "last_skill", None)
        opp = _opp(state, side)
        opp_sk = _fs(opp, "last_skill", None) if opp else None
        if my_sk is not None and opp_sk is not None:
            same = 0
            if my_sk.element == opp_sk.element:
                same += 1
            if my_sk.category == opp_sk.category:
                same += 1
            if my_sk.energy_cost == opp_sk.energy_cost:
                same += 1
            if same > 0:
                buffs.add_buff(pet, buffs.BuffType.ATK, same, buffs.DurationKind.PERMANENT)
                buffs.add_buff(pet, buffs.BuffType.DEF, same, buffs.DurationKind.PERMANENT)
                logs.append(f"  [特性] {pet.name} 与敌方技能 {same} 项相同,物攻物防永久+{same * 10}%")
    if d == "自己使用2次不同的冰系技能后，对手获得4层冻结，随后特性重置。":
        if len(_fs(pet, "ice_skills_used", set())) >= 2:
            _freeze_opponent(state, side, 4, logs)
            _fss(pet, "ice_skills_used", set())
            logs.append(f"  [特性] {pet.name} 使用2次不同冰系技能,对手获得4层冻结")
    if d == "自己使用2次不同的火系技能后，下次技能无需蓄力，随后特性重置。":
        if len(_fs(pet, "fire_skills_used", set())) >= 2:
            _fss(pet, "next_skill_free", True)
            _fss(pet, "fire_skills_used", set())
            logs.append(f"  [特性] {pet.name} 使用2次不同火系技能,下次技能无需蓄力")

    return want_switch


# ---------------------------------------------------------------------------
# 聚能
# ---------------------------------------------------------------------------

def on_charge(state, side: str, logs=None) -> None:
    pet = _active(state, side)
    if pet is None:
        return
    d = desc_of(pet)
    if not d:
        return
    logs = logs if logs is not None else []
    if "蓄力时可以使用任一携带技能，且获得双防+100%" in d:
        buffs.add_buff(pet, buffs.BuffType.DEF, 10, buffs.DurationKind.TEMPORARY, current_turn=state.turn)
        buffs.add_buff(pet, buffs.BuffType.SPDEF, 10, buffs.DurationKind.TEMPORARY, current_turn=state.turn)
        logs.append(f"  [特性] {pet.name} 蓄力获得双防+100%")
    if d == "每次进入蓄力状态，获得全技能能耗永久-2。":
        buffs.add_buff(pet, buffs.BuffType.ENERGY_COST, -2, buffs.DurationKind.PERMANENT)
        logs.append(f"  [特性] {pet.name} 蓄力,全技能能耗永久-2")
    if d == "自己的聚能获得选择：偷取敌方3能量。":
        op = _opp(state, side)
        if op is not None:
            take = min(3, op.energy)
            op.energy -= take
            _energy(state, side, pet, take, logs)
            logs.append(f"  [特性] {pet.name} 聚能偷取敌方 {take} 能量")


# ---------------------------------------------------------------------------
# 技能使用后
# ---------------------------------------------------------------------------

def after_skill_use(state, side: str, skill, is_first: bool, logs=None) -> bool:
    """技能使用并结算后触发。返回 True 表示请求本侧精灵脱离。"""
    pet = _active(state, side)
    if pet is None:
        return False
    d = desc_of(pet)
    if not d:
        return False
    logs = logs if logs is not None else []
    opp = _opp(state, side)

    # 通用状态记录
    pet.skills_used.add(skill.skill_id)
    _fss(pet, "last_skill", skill)
    _fss(pet, "last_skill_cost", skill.energy_cost)

    # 己方/敌方技能使用计数(供"入场前己方每使用1次X系技能"类特性)
    _count_ally_skill(state, side, skill)

    # 本场战斗首次使用的技能获得迅捷
    if "本场战斗首次使用的技能获得迅捷" in d and _fs(pet, "swift_first_skill_id", None) is None:
        _fss(pet, "swift_first_skill_id", skill.skill_id)

    # 选择技能:随机执行一个选项,记录明/暗计数,按特性额外触发
    if getattr(skill, "choices", None):
        use_choice_skill(state, side, skill, logs)

    # 传动:技能位置移动;位置变化特性能耗-1
    moved = apply_drive(state, side, pet, skill, logs)
    if moved and "若回合内自己携带的技能位置发生变化，该技能能耗永久-1" in d:
        skill.energy_cost = max(0, skill.energy_cost - 1)
        logs.append(f"  [特性] {pet.name} {skill.name} 位置变化,能耗永久-1")

    # 使用草系技能时，敌方获得2层中毒
    if d == "使用草系技能时，敌方获得2层中毒。" and skill.element == 1 and opp is not None:
        buffs.add_buff(opp, buffs.BuffType.POISON, 2)
        logs.append(f"  [特性] {pet.name} 草系技能使敌方获得2层中毒")
    if d == "使用技能时，敌方获得2层中毒。" and opp is not None:
        buffs.add_buff(opp, buffs.BuffType.POISON, 2)
        logs.append(f"  [特性] {pet.name} 技能使敌方获得2层中毒")
    if d == "使用能耗小于等于1的技能时，敌方获得4层中毒。" and skill.energy_cost <= 1 and opp is not None:
        buffs.add_buff(opp, buffs.BuffType.POISON, 4)
        logs.append(f"  [特性] {pet.name} 低能耗技能使敌方获得4层中毒")
    if "使用草系技能后，回复10%生命" in d and skill.element == 1:
        _heal(state, side, pet, int(pet.max_hp * 0.10), logs)
    if "使用草系技能后，回复15%生命" in d and skill.element == 1:
        _heal(state, side, pet, int(pet.max_hp * 0.15), logs)
    if d == "使用火系技能后，获得双攻+20%。" and skill.element == 2:
        buffs.add_buff(pet, buffs.BuffType.ATK, 2)
        buffs.add_buff(pet, buffs.BuffType.SPATK, 2)
        logs.append(f"  [特性] {pet.name} 火系技能后双攻+20%")
    if d == "使用火系技能后，获得双攻永久+30%。" and skill.element == 2:
        buffs.add_buff(pet, buffs.BuffType.ATK, 3, buffs.DurationKind.PERMANENT)
        buffs.add_buff(pet, buffs.BuffType.SPATK, 3, buffs.DurationKind.PERMANENT)
        logs.append(f"  [特性] {pet.name} 火系技能后双攻永久+30%")
    if d == "使用水系技能后，全技能能耗-1。" and skill.element == 3:
        buffs.add_buff(pet, buffs.BuffType.ENERGY_COST, -1, buffs.DurationKind.PERMANENT)
        logs.append(f"  [特性] {pet.name} 水系技能后全技能能耗永久-1")
    if d == "使用水系技能后，全技能能耗-2。" and skill.element == 3:
        buffs.add_buff(pet, buffs.BuffType.ENERGY_COST, -2, buffs.DurationKind.PERMANENT)
        logs.append(f"  [特性] {pet.name} 水系技能后全技能能耗永久-2")
    if d == "自己使用恶系技能后，敌方失去2能量。" and skill.element == 15 and opp is not None:
        opp.energy = max(0, opp.energy - 2)
        logs.append(f"  [特性] {pet.name} 恶系技能使敌方失去2能量")
    if d == "使用翼系技能后，获得连击数+1。" and skill.element == 12:
        buffs.add_buff(pet, buffs.BuffType.HIT_COUNT_FLAT, 1, buffs.DurationKind.TEMPORARY, current_turn=state.turn)
        logs.append(f"  [特性] {pet.name} 翼系技能后连击数+1")
    if d == "使用能耗为3的技能时，获得攻防+20%。" and skill.energy_cost == 3:
        buffs.add_buff(pet, buffs.BuffType.ATK, 2)
        buffs.add_buff(pet, buffs.BuffType.DEF, 2)
        logs.append(f"  [特性] {pet.name} 3能耗技能获得攻防+20%")
    if d == "使用能耗为3的技能后，获得攻防永久+20%。" and skill.energy_cost == 3:
        buffs.add_buff(pet, buffs.BuffType.ATK, 2, buffs.DurationKind.PERMANENT)
        buffs.add_buff(pet, buffs.BuffType.DEF, 2, buffs.DurationKind.PERMANENT)
        logs.append(f"  [特性] {pet.name} 3能耗技能后攻防永久+20%")
    if d == "使用水系技能后，敌方获得中毒，获得层数等于中毒印记层数的2倍。" and skill.element == 3 and opp is not None:
        neg = marks.get_mark(state, _opp_side(side), marks.NEGATIVE)
        n = neg["stacks"] * 2 if neg is not None and neg["id"] == 4 else 0
        if n > 0:
            buffs.add_buff(opp, buffs.BuffType.POISON, n)
            logs.append(f"  [特性] {pet.name} 水系技能使敌方中毒 {n} 层")
    if d == "冰系技能使敌方获得4层灼烧，火系技能使敌方获得2层冻结。" and opp is not None:
        if skill.element == 6:
            buffs.add_buff(opp, buffs.BuffType.BURN, 4)
            logs.append(f"  [特性] {pet.name} 冰系技能使敌方4层灼烧")
        if skill.element == 2:
            _freeze_opponent(state, side, 2, logs)
    if d == "草系技能使敌方获得4层灼烧，火系技能使敌方获得1层寄生。" and opp is not None:
        if skill.element == 1:
            buffs.add_buff(opp, buffs.BuffType.BURN, 4)
            logs.append(f"  [特性] {pet.name} 草系技能使敌方4层灼烧")
        if skill.element == 2:
            buffs.add_buff(opp, buffs.BuffType.LEECH, 1)
            logs.append(f"  [特性] {pet.name} 火系技能使敌方1层寄生")
    m = re.search(r"每携带1个毒系技能进入战斗，水系技能使敌方获得(\d+)层中毒", d)
    if m and skill.element == 3 and opp is not None:
        poison_count = sum(1 for s in pet.skills if s.element == 9)
        n = int(m.group(1)) * poison_count
        if n > 0:
            buffs.add_buff(opp, buffs.BuffType.POISON, n)
            logs.append(f"  [特性] {pet.name} 水系技能使敌方中毒 {n} 层")
    if "1号位技能获得传动1，且使用后使敌方获得6层灼烧" in d and opp is not None:
        if pet.skills and pet.skills[0].skill_id == skill.skill_id:
            buffs.add_buff(opp, buffs.BuffType.BURN, 6)
            logs.append(f"  [特性] {pet.name} 1号位技能使敌方6层灼烧")
    if d == "使用攻击技能时，敌方每有1层冻结，在攻击前使其获得1层星陨印记。" and opp is not None and skill.category in (0, 1):
        frozen = buffs.get_buff_value(opp, buffs.BuffType.FREEZE)
        if frozen > 0:
            _add_mark(state, side, _opp_side(side), 7, frozen, logs)
            logs.append(f"  [特性] {pet.name} 攻击前使敌方获得 {frozen} 层星陨印记")
    if "使用「选择」技能后" in d:
        # 冷却1回合:使用任意技能后该技能冷却
        pet.blocked_skills.add(skill.skill_id)
        logs.append(f"  [特性] {pet.name} {skill.name} 冷却1回合")
    if d == "对手本回合使用的技能，冷却1回合。":
        opp = _opp(state, side)
        opp_last = _fs(opp, "last_skill", None) if opp else None
        if opp_last is not None:
            opp.blocked_skills.add(opp_last.skill_id)
            opp.feature_state.setdefault("blocked_turns", {})[opp_last.skill_id] = 2
            logs.append(f"  [特性] {pet.name} 对手 {opp_last.name} 冷却1回合")
    if d == "使用状态技能后，敌方获得「聒噪」技能的效果，持续3回合。" and skill.category == 3 and opp is not None:
        buffs.add_buff(opp, buffs.BuffType.ENERGY_COST, 2)
        logs.append(f"  [特性] {pet.name} 状态技能后敌方全技能能耗+2(聒噪)")
    if d == "自己使用2次不同的冰系技能后，对手获得4层冻结，随后特性重置。":
        if skill.element == 6:
            used = set(_fs(pet, "ice_skills_used", set()))
            used.add(skill.skill_id)
            _fss(pet, "ice_skills_used", used)
    if d == "自己使用2次不同的火系技能后，下次技能无需蓄力，随后特性重置。":
        if skill.element == 2:
            used = set(_fs(pet, "fire_skills_used", set()))
            used.add(skill.skill_id)
            _fss(pet, "fire_skills_used", used)
    if "入场后首次行动，所选技能使用次数+1" in d:
        if _fs(pet, "first_action_done", False) is False:
            _fss(pet, "first_action_done", True)
            logs.append(f"  [特性] {pet.name} 入场后首次行动,技能使用次数+1")
            if "该回合每次行动后回复2能量" in d:
                _energy(state, side, pet, 2, logs)
    if "本场战斗首次使用的技能获得迅捷" in d:
        if not _fs(pet, "swift_used", False):
            _fss(pet, "swift_used", True)
            buffs.add_buff(pet, buffs.BuffType.PRIORITY, 1, buffs.DurationKind.TEMPORARY, current_turn=state.turn)
            logs.append(f"  [特性] {pet.name} 本场首次技能获得迅捷(先手+1)")
    if "携带的火系技能获得选择" in d and skill.element == 2:
        if random.random() < 0.5:
            pet.hp = max(1, pet.hp - int(pet.max_hp * 0.15))
            logs.append(f"  [特性] {pet.name} 选择:火系技能使用后失去15%生命")
        else:
            _fss(pet, "light_power_bonus", True)
            logs.append(f"  [特性] {pet.name} 选择:光系技能威力永久+30")
    if "携带的翼系攻击技能获得选择" in d and skill.element == 12 and skill.category in (0, 1):
        _fss(pet, "wing_choice", "drain" if random.random() < 0.5 else "cost")
        if _fs(pet, "wing_choice") == "cost":
            logs.append(f"  [特性] {pet.name} 选择:翼系技能能耗+1(本次)")
        else:
            logs.append(f"  [特性] {pet.name} 选择:翼系攻击吸血50%")
    if "天气为雷鸣时，使用电系技能后敌方获得引电" in d and skill.element == 8 and opp is not None:
        buffs.add_buff(opp, buffs.BuffType.LIGHTNING, 1)
        logs.append(f"  [特性] {pet.name} 电系技能使敌方获得引电")
    if d == "若先于敌方行动，行动后获得连击数+1。" and is_first:
        buffs.add_buff(pet, buffs.BuffType.HIT_COUNT_FLAT, 1, buffs.DurationKind.TEMPORARY, current_turn=state.turn)
        logs.append(f"  [特性] {pet.name} 先手行动后连击数+1")

    # 攻击后重置(每回复1能量双攻永久+20%)
    if d == "每回复1能量，物攻和魔攻永久+20%，攻击后重置。":
        buffs.remove_buff(pet, buffs.BuffType.ATK)
        buffs.remove_buff(pet, buffs.BuffType.SPATK)

    # 入场物攻+100%每次行动后-20%
    if _fs(pet, "entry_atk_bonus", False):
        buffs.add_buff(pet, buffs.BuffType.ATK, -2)

    # 入场后首次攻击迸发标记清除
    if _fs(pet, "first_attack_pending", False):
        _fss(pet, "first_attack_pending", False)

    want_switch = False
    if d == "每次行动后脱离。":
        want_switch = True
        logs.append(f"  [特性] {pet.name} 行动后脱离")
    if _fs(pet, "leave_after_action", False):
        want_switch = True
        _fss(pet, "leave_after_action", False)
        logs.append(f"  [特性] {pet.name} 行动后脱离")
    if d == "使用防御技能后，回合结束时脱离。":
        if skill.category == 2:
            _fss(pet, "used_defense_this_turn", True)
    return want_switch


def _count_ally_skill(state, side: str, skill) -> None:
    """累计己方技能使用计数到己方全部精灵(供入场类特性)。"""
    elem = skill.element
    cat = skill.category
    for p in state.teams[side]:
        if p.hp <= 0:
            continue
        if elem == 3:
            p.feature_state["ally_water_count"] = p.feature_state.get("ally_water_count", 0) + 1
        if elem == 2:
            p.feature_state["ally_fire_count"] = p.feature_state.get("ally_fire_count", 0) + 1
        if elem == 1:
            p.feature_state["ally_grass_count"] = p.feature_state.get("ally_grass_count", 0) + 1
        if elem == 5:
            p.feature_state["ally_earth_count"] = p.feature_state.get("ally_earth_count", 0) + 1
        if elem == 6:
            p.feature_state["ally_ice_count"] = p.feature_state.get("ally_ice_count", 0) + 1
        if cat == 2:
            p.feature_state["ally_defense_skill_count"] = p.feature_state.get("ally_defense_skill_count", 0) + 1
        if cat == 3:
            p.feature_state["ally_status_count"] = p.feature_state.get("ally_status_count", 0) + 1
        if elem in (11, 5):
            p.feature_state["ally_fight_earth_count"] = p.feature_state.get("ally_fight_earth_count", 0) + 1
    # 敌方计数(供"敌方每使用1次聚能或更换精灵"特性)
    for p in state.teams[_opp_side(side)]:
        if p.hp > 0:
            p.feature_state["opp_charge_switch_count"] = p.feature_state.get("opp_charge_switch_count", 0) + 1


# ---------------------------------------------------------------------------
# 造成伤害后
# ---------------------------------------------------------------------------

def on_deal_damage(state, side: str, skill, damage: int, type_mult: float, logs=None) -> None:
    pet = _active(state, side)
    if pet is None:
        return
    d = desc_of(pet)
    if not d or damage <= 0:
        return
    logs = logs if logs is not None else []

    if "造成克制伤害后" in d and type_mult > 1.0:
        buffs.add_buff(pet, buffs.BuffType.ATK, 2)
        buffs.add_buff(pet, buffs.BuffType.DEF, 2)
        buffs.add_buff(pet, buffs.BuffType.SPEED, 2)
        _energy(state, side, pet, 2, logs)
        logs.append(f"  [特性] {pet.name} 克制伤害后攻防速+20%,回复2能量")
        if "首个技能替换为" in d:
            elem_map = {"光": 4, "草": 1, "火": 2, "水": 3}
            m = re.search(r"首个技能替换为([\u4e00-\u9fa5])系愿力冲击", d)
            if m and m.group(1) in elem_map:
                if not _fs(pet, "wish_replaced", False):
                    _replace_first_skill(pet, elem_map[m.group(1)], logs)
                    _fss(pet, "wish_replaced", True)
    if d == "攻击会使敌方已有的减益层数+3。":
        op = _opp(state, side)
        if op is not None:
            for b in op.buffs:
                if buffs.is_debuff(b.buff_type, b.value):
                    b.value += 3
            logs.append(f"  [特性] {pet.name} 攻击使敌方减益层数+3")
    if "携带的翼系攻击技能获得选择：能耗+1，攻击时吸血50%" in d and skill.element == 12:
        heal = int(damage * 0.5)
        actual = on_heal(state, side, pet, heal, logs)
        if actual > 0:
            pet.hp = min(pet.max_hp, pet.hp + actual)
            logs.append(f"  [特性] {pet.name} 翼系攻击吸血 {actual}")
    if "天气为雨天时，使用水系攻击技能时吸血50%" in d and skill.element == 3:
        heal = int(damage * 0.5)
        actual = on_heal(state, side, pet, heal, logs)
        if actual > 0:
            pet.hp = min(pet.max_hp, pet.hp + actual)
            logs.append(f"  [特性] {pet.name} 雨天水系攻击吸血 {actual}")
    if d == "攻击时，将敌方所有印记变为相同层数的星陨印记。":
        op_side = _opp_side(side)
        total = 0
        for slot in (marks.POSITIVE, marks.NEGATIVE):
            cur = marks.get_mark(state, op_side, slot)
            if cur is not None:
                total += cur["stacks"]
        if total > 0:
            marks.clear_side(state, op_side, marks.POSITIVE)
            marks.clear_side(state, op_side, marks.NEGATIVE)
            _add_mark(state, op_side, 7, total, logs)
            logs.append(f"  [特性] {pet.name} 敌方印记全部变为 {total} 层星陨印记")


# ---------------------------------------------------------------------------
# 受到伤害后
# ---------------------------------------------------------------------------

def on_take_damage(state, side: str, attacker_side: str, skill, damage: int, include_hit_count: bool, logs=None) -> int:
    """受到伤害后触发。返回调整后的伤害(可免疫/减为0)。"""
    pet = _active(state, side)
    if pet is None:
        return damage
    d = desc_of(pet)
    if not d or damage <= 0:
        return damage
    logs = logs if logs is not None else []
    attacker = _active(state, attacker_side)
    new_damage = damage

    if d == "每受到1次攻击伤害，对攻击自己的精灵造成50威力物理伤害。":
        if attacker is not None:
            from .models import BattleSkill
            fake = BattleSkill(skill_id=-1, name="反伤", element=0, category=0, power=50,
                               energy_cost=0, desc="反伤")
            r = calc_damage(attacker, pet, fake, _typechart())
            if r["damage"] > 0:
                attacker.hp = max(0, attacker.hp - r["damage"])
                logs.append(f"  [特性] {pet.name} 反伤 {r['damage']}")
    if d == "每受到1次技能攻击（不含连击），敌方获得1层棘刺印记。" and include_hit_count:
        _add_mark(state, side, _opp_side(side), 1, 1, logs)
        logs.append(f"  [特性] {pet.name} 受到攻击,敌方获得1层棘刺印记")
    if d == "每受到1次技能攻击（不含连击），敌方获得1层暗涌印记。" and include_hit_count:
        logs.append(f"  [特性] {pet.name} 暗涌印记未定义,效果缺失")
    if d == "每受到1次攻击伤害，己方队伍获得1次随机奉献。":
        _random_offering(state, side, logs)
    if d == "受到致命伤害时，获得1层萌化，并免疫此次伤害。（最多触发2次）":
        if damage >= pet.hp:
            used = _fs(pet, "faint_protect_used", 0)
            if used < 2:
                _fss(pet, "faint_protect_used", used + 1)
                new_damage = 0
                buffs.add_buff(pet, buffs.BuffType.CUTE, 1)
                logs.append(f"  [特性] {pet.name} 免疫致命伤害,获得1层萌化({used + 1}/2)")
    if d == "每场战斗1次，受到致命伤害时保留1血，且敌方获得15层灼烧。":
        if damage >= pet.hp and not _fs(pet, "faint_keep1_used", False):
            _fss(pet, "faint_keep1_used", True)
            new_damage = max(0, pet.hp - 1)
            op = _opp(state, side)
            if op is not None:
                buffs.add_buff(op, buffs.BuffType.BURN, 15)
                logs.append(f"  [特性] {pet.name} 保留1血,敌方获得15层灼烧")
    if d == "自己每失去25%生命，连击数+2。":
        marker = _fs(pet, "hp_quarter_marker", 0)
        quarters_lost = int((pet.max_hp - pet.hp) / (pet.max_hp * 0.25))
        if quarters_lost > marker:
            gain = quarters_lost - marker
            _fss(pet, "hp_quarter_marker", quarters_lost)
            buffs.add_buff(pet, buffs.BuffType.HIT_COUNT_FLAT, 2 * gain, buffs.DurationKind.TEMPORARY, current_turn=state.turn)
            logs.append(f"  [特性] {pet.name} 每失去25%生命连击数+2")

    return new_damage


# ---------------------------------------------------------------------------
# 力竭 / 击败
# ---------------------------------------------------------------------------

def faint_magic_delta(pet) -> int:
    """魔力损失修正:正值=少损失,负值=额外损失。"""
    d = desc_of(pet)
    if not d:
        return 0
    if d == "自己力竭时，少损失1点魔力。":
        return 1
    if "棋契陛下大幅提升种族资质，力竭时扣除4魔力" in d:
        return -4
    if "被敌方精灵击败时，自己额外损失1点魔力" in d:
        return -1
    return 0


def on_faint(state, side: str, logs=None) -> None:
    pet = _active(state, side)
    if pet is None:
        return
    d = desc_of(pet)
    if not d:
        return
    logs = logs if logs is not None else []
    if d == "自己力竭时，敌方获得攻防+20%。":
        op = _opp(state, side)
        if op is not None:
            buffs.add_buff(op, buffs.BuffType.ATK, 2)
            buffs.add_buff(op, buffs.BuffType.DEF, 2)
            logs.append(f"  [特性] {pet.name} 力竭,敌方获得攻防+20%")
    if d == "力竭4回合后复活。":
        pet.revive_turn = state.turn + 4
        logs.append(f"  [特性] {pet.name} 将于 {pet.revive_turn} 回合复活")


def on_kill(state, side: str, logs=None) -> None:
    pet = _active(state, side)
    if pet is None:
        return
    d = desc_of(pet)
    if not d:
        return
    logs = logs if logs is not None else []
    if d == "主动击败敌方精灵时，自己永久获得双攻+50%。":
        buffs.add_buff(pet, buffs.BuffType.ATK, 5, buffs.DurationKind.PERMANENT)
        buffs.add_buff(pet, buffs.BuffType.SPATK, 5, buffs.DurationKind.PERMANENT)
        logs.append(f"  [特性] {pet.name} 击败敌方,双攻永久+50%")
    if d == "主动击败敌方后，己方队伍获得5次随机奉献。":
        for _ in range(5):
            _random_offering(state, side, logs)
    if "击败敌方精灵时，敌方额外损失1点魔力" in d:
        opp_side = _opp_side(side)
        state.magic[opp_side] = max(0, state.magic[opp_side] - 1)
        logs.append(f"  [特性] {pet.name} 击败敌方,敌方额外损失1点魔力")


# ---------------------------------------------------------------------------
# 应对成功
# ---------------------------------------------------------------------------

def on_counter_success(state, side: str, counter_skill, interrupted_skill, logs=None) -> None:
    """应对成功:counter_skill 是本侧使用的技能,interrupted_skill 是对方被打断技能。"""
    pet = _active(state, side)
    if pet is None:
        return
    d = desc_of(pet)
    logs = logs if logs is not None else []

    # 己方应对计数
    for p in state.teams[side]:
        if p.hp > 0:
            p.feature_state["ally_counter_count"] = p.feature_state.get("ally_counter_count", 0) + 1
    if not d:
        return

    if d == "应对成功后，下次行动先手+1。":
        buffs.add_buff(pet, buffs.BuffType.PRIORITY, 1, buffs.DurationKind.TEMPORARY, current_turn=state.turn)
        logs.append(f"  [特性] {pet.name} 应对成功,下次行动先手+1")
    if d == "应对成功后，下次攻击威力翻倍。":
        _fss(pet, "next_attack_x2", True)
        logs.append(f"  [特性] {pet.name} 应对成功,下次攻击威力翻倍")
    if d == "应对成功后，下次行动技能能耗-5。":
        buffs.add_buff(pet, buffs.BuffType.ENERGY_COST, -5, buffs.DurationKind.TEMPORARY, current_turn=state.turn)
        logs.append(f"  [特性] {pet.name} 应对成功,下次行动能耗-5")
    if d == "应对成功后，获得全技能威力永久+30。":
        buffs.add_buff(pet, buffs.BuffType.SKILL_POWER_FLAT, 30, buffs.DurationKind.PERMANENT)
        logs.append(f"  [特性] {pet.name} 应对成功,全技能威力永久+30")
    if d == "应对成功后，永久获得双攻+30%。":
        buffs.add_buff(pet, buffs.BuffType.ATK, 3, buffs.DurationKind.PERMANENT)
        buffs.add_buff(pet, buffs.BuffType.SPATK, 3, buffs.DurationKind.PERMANENT)
        logs.append(f"  [特性] {pet.name} 应对成功,双攻永久+30%")
    if "自己防御应对成功时，敌方获得萌化。" in d:
        op = _opp(state, side)
        if op is not None:
            buffs.add_buff(op, buffs.BuffType.CUTE, 1)
            logs.append(f"  [特性] {pet.name} 防御应对成功,敌方获得萌化")
    if "打断敌方时，被打断的技能进入2回合冷却。" in d and interrupted_skill is not None:
        op = _opp(state, side)
        if op is not None:
            op.blocked_skills.add(interrupted_skill.skill_id)
            op.feature_state.setdefault("blocked_extra", {})[interrupted_skill.skill_id] = 2
            logs.append(f"  [特性] {pet.name} 打断敌方,{interrupted_skill.name} 冷却2回合")

    # 棋绮后变身(应对次数累积)
    m = re.search(r"(攻击|防御|状态)技能应对(\d+)次后，回满能量和生命，变为棋绮后", d)
    if m:
        kind = {"攻击": 0, "防御": 2, "状态": 3}[m.group(1)]
        need = int(m.group(2))
        key = f"counter_{kind}"
        if counter_skill is not None and counter_skill.category == kind:
            n = _fs(pet, key, 0) + 1
            _fss(pet, key, n)
            if n >= need:
                pet.energy = energy_cap_of(pet)
                pet.hp = pet.max_hp
                _transform_spirit(state, side, _qiyi_queen_name(pet), logs)
                _fss(pet, key, 0)


# ---------------------------------------------------------------------------
# 能量/增益钩子
# ---------------------------------------------------------------------------

def on_energy_gain(state, side: str, pet, gained: int, logs=None) -> None:
    d = desc_of(pet)
    if not d:
        return
    logs = logs if logs is not None else []
    if d == "每回复1能量，同时回复5%生命。":
        actual = on_heal(state, side, pet, int(pet.max_hp * 0.05 * gained), logs)
        if actual > 0:
            pet.hp = min(pet.max_hp, pet.hp + actual)
    if d == "每回复1能量，物攻和魔攻永久+20%，攻击后重置。":
        buffs.add_buff(pet, buffs.BuffType.ATK, 2, buffs.DurationKind.PERMANENT)
        buffs.add_buff(pet, buffs.BuffType.SPATK, 2, buffs.DurationKind.PERMANENT)
    if "获得能量或生命时，会将等量的能量或生命随机分配给场下的精灵" in d:
        bench = [p for p in state.teams[side] if p.hp > 0 and p is not pet]
        if bench:
            target = random.choice(bench)
            target.energy = min(energy_cap_of(target), target.energy + gained)
            if logs is not None:
                logs.append(f"  [特性] {pet.name} 将 {gained} 能量分配给场下 {target.name}")


def on_buff_gain(pet, buff_type: str, value: int) -> int:
    """额外层数返回。buffs.add_buff 在添加增益后调用。"""
    d = desc_of(pet)
    if not d or value == 0:
        return 0
    extra = 0
    if d == "获得增益时，额外获得层数+2。":
        if buffs.is_buff(buff_type, value):
            extra += 2
    if d == "每回合各1次，获得增益时，同时获得全技能能耗-1，获得减益时，同时获得全技能能耗+1。":
        if buffs.is_buff(buff_type, value):
            buffs.add_buff(pet, buffs.BuffType.ENERGY_COST, -1)
        elif buffs.is_debuff(buff_type, value):
            buffs.add_buff(pet, buffs.BuffType.ENERGY_COST, 1)
    if d == "若自己在萌化状态下再获得萌化会解除萌化。":
        if buff_type == buffs.BuffType.CUTE and buffs.has_buff(pet, buffs.BuffType.CUTE):
            buffs.remove_buff(pet, buffs.BuffType.CUTE)
            extra = -value
    return extra


# ---------------------------------------------------------------------------
# 回合结束全局行为查询
# ---------------------------------------------------------------------------

def round_end_disabled(state) -> bool:
    for side in ("A", "B"):
        pet = _active(state, side)
        if pet is not None and desc_of(pet) == "在场时，双方回合结束时的效果不会触发。":
            return True
    return False


def round_end_extra(state) -> int:
    """需要额外整体结算的次数。"""
    for side in ("A", "B"):
        pet = _active(state, side)
        if pet is not None and desc_of(pet) == "在场时，双方回合结束时的效果会额外触发1次。":
            return 1
    return 0


def poison_extra_trigger(state) -> bool:
    for side in ("A", "B"):
        pet = _active(state, side)
        if pet is not None and desc_of(pet) == "在场时，双方回合结束时的中毒效果会额外触发1次。":
            return True
    return False


def burn_mode(state, side: str) -> str:
    """灼烧结算模式: normal / grow / to_poison。"""
    for s in ("A", "B"):
        pet = _active(state, s)
        if pet is None:
            continue
        d = desc_of(pet)
        if d == "在场时，所有灼烧的衰减变为增长。":
            return "grow"
        if d == "在场时，衰减的灼烧变为相同层数的中毒。":
            return "to_poison"
    return "normal"


def modify_stat_state(state, side: str, stat: str, mult: float) -> float:
    """需要 state 的属性倍率修正。"""
    pet = _active(state, side)
    if pet is None:
        return mult
    d = desc_of(pet)
    if not d:
        return mult
    if "己方队伍中每有1只力竭的精灵，自己获得双攻+30%" in d and stat in ("atk", "spatk"):
        n = sum(1 for p in state.teams[side] if p.hp <= 0)
        return mult * (1.0 + 0.3 * n)
    if "双方队伍中每有1只力竭的精灵，自己获得双攻+30%" in d and stat in ("atk", "spatk"):
        n = sum(1 for s in ("A", "B") for p in state.teams[s] if p.hp <= 0)
        return mult * (1.0 + 0.3 * n)
    if "双方场上每有1种不同的增益，自己获得物防+20%" in d and stat == "def":
        kinds = set()
        for s in ("A", "B"):
            p = _active(state, s)
            if p is not None:
                for b in p.buffs:
                    if b.value > 0:
                        kinds.add(b.buff_type)
        return mult * (1.0 + 0.2 * len(kinds))
    if "双方场上每有1种不同的印记，自己获得魔攻+50%" in d and stat == "spatk":
        kinds = set()
        for s in ("A", "B"):
            for slot in (marks.POSITIVE, marks.NEGATIVE):
                cur = marks.get_mark(state, s, slot)
                if cur is not None:
                    kinds.add(cur["id"])
        return mult * (1.0 + 0.5 * len(kinds))
    if "队伍中每有1只其他的虫系精灵，自己入场时获得攻防速+15%" in d and stat in ("atk", "def"):
        bug_count = sum(1 for p in state.teams[side] if p is not pet and p.hp > 0 and 10 in p.attributes)
        return mult * (1.0 + 0.15 * bug_count)
    if "己方精灵每使用1次武系或地系技能，自己入场时获得攻防+5%" in d and stat in ("atk", "def"):
        n = _fs(pet, "ally_fight_earth_count", 0)
        return mult * (1.0 + 0.05 * n)
    is_weekend = state.day_of_week in (5, 6)
    if "周末时自己获得双攻+40%" in d and stat in ("atk", "spatk") and is_weekend:
        return mult * 1.4
    if "其他时间获得双防+40%" in d and stat in ("def", "spdef") and not is_weekend:
        return mult * 1.4
    return mult


def modify_speed_state(state, side: str, speed: int) -> int:
    pet = _active(state, side)
    if pet is None:
        return speed
    d = desc_of(pet)
    if not d:
        return speed
    if "行动时，敌方每有1层增益，本次行动技能威力+10%，速度+5" in d:
        op = _opp(state, side)
        if op is not None:
            kinds = {b.buff_type for b in op.buffs if b.value > 0}
            return speed + 5 * len(kinds)
    if "队伍中每有1只其他的虫系精灵，自己入场时获得攻防速+15%" in d:
        bug_count = sum(1 for p in state.teams[side] if p is not pet and p.hp > 0 and 10 in p.attributes)
        return speed + int(speed * 0.15 * bug_count)
    if "入场前己方精灵每使用1次火系技能，获得攻防+10%，速度+10" in d:
        n = min(_fs(pet, "ally_fire_count", 0), 10)
        return speed + 10 * n
    return speed


def modify_hit_count_state(state, side: str, count: int, skill=None) -> int:
    for s in ("A", "B"):
        p = _active(state, s)
        if p is None:
            continue
        d = desc_of(p)
        if d == "在场时，所有精灵连击数固定为2。":
            return 2
        if d == "在场时，所有精灵连击数固定为1。":
            return 1
    pet = _active(state, side)
    if pet is not None:
        d = desc_of(pet)
        if "敌方每有1层中毒效果，自己获得连击数+1" in d:
            op = _opp(state, side)
            if op is not None:
                count += buffs.get_buff_value(op, buffs.BuffType.POISON)
        # 巧变:虫鸣 —— 队伍中每携带1个虫鸣,虫系技能连击+1
        if skill is not None and skill.element == 10 and "己方精灵携带的虫系技能获得巧变：虫鸣" in d:
            bug_count = sum(1 for p in state.teams[side] for sk in p.skills if sk.name == "虫鸣")
            count += bug_count
    return count


def modify_energy_cost_state(state, side: str, skill, base_cost: int) -> int:
    """敌方在场特性影响的能耗。"""
    bonus = 0
    op = _opp(state, side)
    if op is not None and desc_of(op) == "在场时，敌方全技能能耗+1。":
        bonus += 1
    pet = _active(state, side)
    if pet is not None:
        d = desc_of(pet)
        if d == "携带的技能受能耗变化效果的影响翻倍。":
            from .buffs import get_energy_cost_modifier
            bonus += get_energy_cost_modifier(pet)
        if d == "自己的能耗增加变为能耗降低；能耗降低变为能耗增加。":
            from .buffs import get_energy_cost_modifier
            bonus -= 2 * get_energy_cost_modifier(pet)
        if d == "携带的电系技能获得迸发：能耗-2。" and skill.element == 8:
            bonus -= 2
        if d == "每携带1个水系技能进入战斗，地系技能能耗-1。" and skill.element == 5:
            water_count = sum(1 for s in pet.skills if s.element == 3)
            bonus -= water_count
        if _fs(pet, "next_skill_free", False):
            bonus -= 999
            _fss(pet, "next_skill_free", False)
    return bonus


# ===========================================================================
# 缺失机制实现(迅捷/传动/巧变/选择/血脉/咕噜球/木桶/背包/时间/伪装/
# 技能石/随机技能/迸发延长/印记叠加)
# ===========================================================================

def bloodline_of(pet) -> str:
    """血脉:首领形态→leader;被污染→polluted;特殊形态→special;其余→normal。"""
    name = pet.name or ""
    ft = getattr(pet, "form_type", "") or ""
    if ft == "首领形态" or "首领" in name:
        return "leader"
    if "被污染" in name:
        return "polluted"
    if ft == "特殊形态":
        return "special"
    return "normal"


def skill_is_swift(pet, skill) -> bool:
    """技能是否迅捷(使用该技能时先手+1)。"""
    if skill is None:
        return False
    if getattr(skill, "swift", False):
        return True
    d = desc_of(pet)
    if not d:
        return False
    if "携带的龙系技能获得迅捷" in d and skill.element == 7:
        return True
    if "携带的能耗小于3的技能，获得迅捷" in d and skill.energy_cost < 3:
        return True
    if "1号位技能获得迅捷和传动1" in d and pet.skills and pet.skills[0].skill_id == skill.skill_id:
        return True
    if "本场战斗首次使用的技能获得迅捷" in d and _fs(pet, "swift_first_skill_id", None) == skill.skill_id:
        return True
    return False


def skill_is_swift_state(state, side: str, pet, skill) -> bool:
    """需要队伍信息的迅捷判定(其他翼系精灵携带相同技能)。"""
    if skill is None:
        return False
    d = desc_of(pet)
    if d and "其他翼系精灵携带相同技能" in d:
        team = state.teams[side]
        others = [p for p in team if p is not pet and p.hp > 0 and 12 in p.attributes]
        for p in others:
            if any(s.skill_id == skill.skill_id for s in p.skills):
                return True
    return False


def apply_drive(state, side: str, pet, skill, logs) -> bool:
    """传动N:使用后技能位置移动N格(与右侧相邻技能交换,越界向左)。返回是否移动。"""
    drive = getattr(skill, "drive", 0) if skill is not None else 0
    if drive <= 0 or pet is None or not pet.skills:
        return False
    idx = None
    for i, s in enumerate(pet.skills):
        if s.skill_id == skill.skill_id:
            idx = i
            break
    if idx is None:
        return False
    moved = False
    for _ in range(drive):
        nxt = idx + 1
        if nxt >= len(pet.skills):
            nxt = idx - 1
        if nxt < 0 or nxt == idx:
            break
        pet.skills[idx], pet.skills[nxt] = pet.skills[nxt], pet.skills[idx]
        idx = nxt
        moved = True
    if moved and logs is not None:
        logs.append(f"  [特性] 传动{drive}:{skill.name} 移动 {drive} 格")
    return moved


def effective_element(pet, skill) -> int:
    """技能实际元素(巧变:同系别 / 普通系技能变为翼系技能)。"""
    elem = getattr(skill, "element", None)
    if elem is None:
        return elem
    d = desc_of(pet)
    if d and "普通系技能变为翼系技能" in d and elem == 0:
        return 12
    q = getattr(skill, "qiaobian", "") or ""
    if "同系别技能" in q and pet.attributes:
        return pet.attributes[0]
    if d and "巧变：同系别技能" in d and pet.attributes:
        return pet.attributes[0]
    return elem


# ---------- 选择 ----------

def _exec_choice(state, side: str, pet, option_text: str, logs) -> None:
    """执行一个选择选项(近似识别常见效果模式)。"""
    if not option_text:
        return
    opp = _opp(state, side)
    if "回复" in option_text:
        m = re.search(r"回复(\d+)%生命", option_text)
        if m and not option_text.startswith("敌方"):
            _heal(state, side, pet, int(pet.max_hp * int(m.group(1)) / 100), logs)
            return
        m = re.search(r"回复(\d+)能量", option_text)
        if m:
            _energy(state, side, pet, int(m.group(1)), logs)
            return
    m = re.search(r"敌方获得(\d+)层([\u4e00-\u9fa5]+)", option_text)
    if m and opp is not None:
        btype = {"中毒": "poison", "灼烧": "burn", "冻结": "freeze", "寄生": "leech"}.get(m.group(2))
        if btype:
            buffs.add_buff(opp, btype, int(m.group(1)))
            if logs is not None:
                logs.append(f"  [特性] 选择:敌方获得 {m.group(1)} 层{m.group(2)}")
            return
    m = re.search(r"偷取敌方(\d+)能量", option_text)
    if m and opp is not None:
        take = min(int(m.group(1)), opp.energy)
        opp.energy -= take
        _energy(state, side, pet, take, logs)
        return
    if logs is not None:
        logs.append(f"  [特性] 选择选项「{option_text}」效果未识别")


def use_choice_skill(state, side: str, skill, logs) -> None:
    """使用带「选择:」的技能:随机选一个选项执行,记录明/暗计数,按特性额外触发。"""
    choices = getattr(skill, "choices", None) or []
    if not choices:
        return
    pet = _active(state, side)
    if pet is None:
        return
    pick = random.randrange(len(choices))
    _exec_choice(state, side, pet, choices[pick], logs)
    for p in state.teams[side]:
        if p.hp > 0:
            key = "ming_count" if pick == 0 else "an_count"
            p.feature_state[key] = p.feature_state.get(key, 0) + 1
    d = desc_of(pet)
    if d and "会额外使用1次另一种「选择」效果" in d and len(choices) > 1:
        _exec_choice(state, side, pet, choices[1 - pick], logs)
    elif d and "会额外使用1次相同的「选择」效果" in d:
        _exec_choice(state, side, pet, choices[pick], logs)


# ---------- 迸发延长 ----------

def take_bursts(pet) -> list:
    """迸发延长1回合:首次行动保留迸发,第二次行动再消耗。"""
    if pet is None:
        return []
    if "自己技能的迸发效果延长1回合" in desc_of(pet) and pet.bursts:
        if not _fs(pet, "burst_extended", False):
            _fss(pet, "burst_extended", True)
            return list(pet.bursts)
        bursts = list(pet.bursts)
        pet.bursts.clear()
        _fss(pet, "burst_extended", False)
        return bursts
    return burst.take_bursts(pet)


# ---------- 印记叠加 ----------

def _add_mark(state, source_side: str, target_side: str, mark_id: int, amount: int, logs) -> None:
    """施加印记;施加方带"赋予的印记不替换"特性时改为叠加,同时生效。"""
    pet = _active(state, source_side)
    stack = False
    if pet is not None and "赋予的印记不会替换其他印记" in desc_of(pet):
        stack = True
    marks.add_mark(state, target_side, mark_id, amount, stack=stack)


# ---------- 随机技能 / 技能石 ----------

def _append_random_skills(pet, n: int, attack_only: bool = False, logs=None) -> None:
    """为精灵补充随机技能(未携带的)。attack_only=True 时只补攻击技能。"""
    from . import data_loader
    owned = {s.skill_id for s in pet.skills}
    pool = [
        s for s in data_loader.load_skills()
        if s["id"] not in owned and s.get("name") != "蓄能"
        and (not attack_only or s.get("category") in (0, 1))
    ]
    if not pool:
        return
    picks = random.sample(pool, min(n, len(pool)))
    from .models import BattleSkill
    for raw in picks:
        mech = data_loader._parse_skill_mechanics(raw)
        pet.skills.append(BattleSkill(
            skill_id=raw["id"], name=raw["name"], element=raw["element"],
            category=raw["category"], power=raw.get("power"),
            energy_cost=raw.get("energyCost", 0), desc=raw.get("desc", ""),
            counter_target=raw.get("counterTarget") or "",
            swift=mech["swift"], drive=mech["drive"],
            qiaobian=mech["qiaobian"], choices=mech["choices"],
        ))
    if logs is not None:
        logs.append(f"  [特性] {pet.name} 额外获得 {len(picks)} 个技能")
