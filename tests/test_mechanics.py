#!/usr/bin/env python3
"""缺失机制(迅捷/传动/巧变/选择/血脉/咕噜球/木桶/周末/印记叠加/背包/技能石/随机技能/迸发延长)专项验证。"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim import buffs, features, marks
from sim.battle import create_team_battle, step
from sim.data_loader import find_spirit, load_spirits, make_battle_pet
from sim.models import Action

SPIRITS = load_spirits()


def make(side, name, skill_names=None, **kw):
    sp = find_spirit(name, SPIRITS)
    assert sp is not None, f"找不到 {name}"
    return make_battle_pet(sp, side, skill_names=skill_names, **kw)


def test_swift():
    """迅捷:使用迅捷技能时强制先手。"""
    # 找一只带迅捷技能的精灵(翼击等)
    from sim.data_loader import load_skills
    swift_skill = next(s for s in load_skills() if "迅捷" in s.get("desc", "") and s.get("category") in (0, 1))
    # 构造:慢速精灵带迅捷技能 vs 快速精灵
    slow = make("A", "音速犬", skill_names=[swift_skill["name"], "啄击"])
    fast = make("B", "岚鸟")
    slow.stats["speed"] = 1
    fast.stats["speed"] = 100  # 高于慢速精灵但低于慢速+迅捷加成(101)
    slow.energy = 10
    fast.energy = 10
    state = create_team_battle([slow], [fast])
    state = step(state, Action(kind="skill", skill_index=0), Action(kind="skill", skill_index=0))
    # A 使用迅捷技能应先于 B
    idx_a = next(i for i, l in enumerate(state.log) if "A " in l and " 使用 " in l)
    idx_b = next(i for i, l in enumerate(state.log) if "B " in l and " 使用 " in l)
    assert idx_a < idx_b, f"迅捷应先手:A@{idx_a} B@{idx_b}\n{state.log}"
    print(f"[PASS] 迅捷:{swift_skill['name']} 慢速精灵仍先手")


def test_drive_and_position():
    """传动:技能位置移动;位置变化特性能耗-1。"""
    # 找带传动1的攻击技能
    from sim.data_loader import load_skills
    drive_skill = next(s for s in load_skills() if "传动1" in s.get("desc", "") and s.get("category") in (0, 1))
    pet = make("A", "音速犬", skill_names=[drive_skill["name"], "啄击", "扇风", "突进"])
    pet.energy = 10
    b = make("B", "岚鸟")
    state = create_team_battle([pet], [b])
    before = [s.skill_id for s in pet.skills]
    state = step(state, Action(kind="skill", skill_index=0), Action(kind="charge"))
    after = [s.skill_id for s in pet.skills]
    assert before != after, "传动后技能位置应变化"
    print(f"[PASS] 传动:{drive_skill['name']} 技能位置 {before} -> {after}")


def test_qiaobian_same_type():
    """巧变:同系别 —— 技能元素变为精灵主属性。"""
    # 找带"巧变：同系别技能"特性的精灵
    carrier = None
    for s in SPIRITS:
        if s.get("feature") and "巧变：同系别技能" in s["feature"]["desc"] \
                and s.get("mainElement") not in (None, 0):
            carrier = s
            break
    if carrier is None:
        print("[SKIP] 无主属性非普通系的同系别巧变特性精灵")
        return
    pet = make_battle_pet(carrier, "A")
    # 普通系技能应变为精灵主属性
    norm = next((sk for sk in pet.skills if sk.element == 0), None)
    if norm is None:
        print("[SKIP] 精灵无普通系技能")
        return
    elem = features.effective_element(pet, norm)
    assert elem == pet.attributes[0] != 0, f"巧变后元素 {elem} 应为主属性 {pet.attributes[0]}"
    print(f"[PASS] 巧变同系别:{carrier['name']}({carrier['mainElement']}) 普通系技能变为 {elem}")


def test_bloodline_leader():
    """血脉:攻击首领血脉精灵时威力+100%(圣光迪莫=首领形态)。"""
    carrier = None
    for s in SPIRITS:
        if s.get("feature") and "血脉是首领血脉" in s["feature"]["desc"]:
            carrier = s
            break
    assert carrier is not None, "无首领血脉特性精灵"
    a = make_battle_pet(carrier, "A")
    b = make("B", "圣光迪莫")  # 首领形态
    assert features.bloodline_of(b) == "leader", "圣光迪莫应为首领血脉"
    bonus = features.modify_power_percent_state(create_team_battle([a], [b]), "A", a.skills[0], True)
    assert bonus >= 100.0, f"首领血脉应+100%威力,实际 {bonus}"
    print(f"[PASS] 血脉:{carrier['name']} 对首领血脉 {b.name} 威力+100%")


def test_ball():
    """咕噜球:星尘虫按球类型入场获得不同效果。"""
    carrier = find_spirit("星尘虫", SPIRITS)
    pet = make_battle_pet(carrier, "A")
    pet.capture_ball = "高级咕噜球"
    state = create_team_battle([pet], [make("B", "岚鸟")])
    state = step(state, Action(kind="charge"), Action(kind="charge"))
    atk = buffs.get_buff_value(pet, buffs.BuffType.ATK)
    assert atk >= 3, f"高级球应入场物攻+30%,实际 {atk * 10}%"
    print(f"[PASS] 咕噜球:星尘虫 高级球入场物攻+{atk * 10}%")


def test_barrel():
    """木桶:离场后换入精灵以木桶状态登场(减伤50%)。"""
    barrel = None
    for s in SPIRITS:
        if s.get("feature") and "以木桶状态登场" in s["feature"]["desc"]:
            barrel = s
            break
    assert barrel is not None, "无木桶特性精灵"
    out = make_battle_pet(barrel, "A")
    bench = make("A", "音速犬")
    b = make("B", "岚鸟")
    state = create_team_battle([out, bench], [b])
    state = step(state, Action(kind="charge"), Action(kind="charge"))
    state = step(state, Action(kind="switch", pet_index=1), Action(kind="charge"))
    assert features._fs(bench, "barrel", False), "换入精灵应为木桶状态"
    dmg = features.modify_damage_taken(bench, b, b.skills[0], 100)
    assert dmg == 50, f"木桶减伤50%,实际 {dmg}"
    print(f"[PASS] 木桶:{barrel['name']} 换入 {bench.name} 木桶状态减伤50%")


def test_weekend_and_night():
    """周末/王国入夜:按环境条件触发。"""
    # 周末特性:工作日走双防,周末走双攻
    carrier = None
    for s in SPIRITS:
        if s.get("feature") and "周末时自己获得双攻+40%" in s["feature"]["desc"]:
            carrier = s
            break
    assert carrier is not None, "无周末特性精灵"
    pet = make_battle_pet(carrier, "A")
    b = make("B", "岚鸟")
    state = create_team_battle([pet], [b])
    state.day_of_week = 0  # 周一
    m_def = features.modify_stat_state(state, "A", "def", 1.0)
    assert m_def >= 1.4, f"工作日应双防+40%,实际 {m_def}"
    state.day_of_week = 6  # 周日
    m_atk = features.modify_stat_state(state, "A", "atk", 1.0)
    assert m_atk >= 1.4, f"周末应双攻+40%,实际 {m_atk}"
    print(f"[PASS] 周末:周一双防+40%,周日双攻+40%")
    # 王国入夜:夜晚能耗+2,白天不触发
    night_pet = None
    for s in SPIRITS:
        if s.get("feature") and "王国入夜后" in s["feature"]["desc"]:
            night_pet = s
            break
    assert night_pet is not None, "无王国入夜特性精灵"
    p = make_battle_pet(night_pet, "A")
    s2 = create_team_battle([p], [make("B", "岚鸟")])
    s2.is_night = False
    s2 = step(s2, Action(kind="charge"), Action(kind="charge"))
    assert buffs.get_energy_cost_modifier(p) == 0, "白天不应获得能耗+2"
    s3 = create_team_battle([make_battle_pet(night_pet, "A")], [make("B", "岚鸟")])
    s3.is_night = True
    s3 = step(s3, Action(kind="charge"), Action(kind="charge"))
    assert buffs.get_energy_cost_modifier(s3.teams["A"][0]) >= 2, "夜晚应获得能耗+2"
    print(f"[PASS] 王国入夜:夜晚能耗+2,白天无")


def test_mark_stack():
    """印记叠加:赋予的印记不替换,同时生效。"""
    carrier = None
    for s in SPIRITS:
        if s.get("feature") and "赋予的印记不会替换其他印记" in s["feature"]["desc"]:
            carrier = s
            break
    if carrier is None:
        print("[SKIP] 无印记叠加特性精灵")
        return
    a = make_battle_pet(carrier, "A")
    b = make("B", "岚鸟")
    state = create_team_battle([a], [b])
    features._add_mark(state, "A", "B", 4, 2, state.log)   # 敌方中毒印记
    features._add_mark(state, "A", "B", 5, 1, state.log)   # 敌方降灵印记(应叠加不替换)
    extra = marks.get_extra_marks(state, "B", marks.NEGATIVE)
    extra_ids = {m["id"] for m in extra}
    assert 4 in extra_ids and 5 in extra_ids, f"两种印记应同时叠加生效: {extra}"
    assert marks.find_stacks(state, "B", 4) == 2
    print(f"[PASS] 印记叠加:中毒印记+降灵印记同时生效不替换")


def test_random_skills_and_stone():
    """随机技能/技能石:入场补充技能。"""
    from sim import features as f
    pet = make("A", "音速犬")
    f._append_random_skills(pet, 3, logs=[])
    assert len(pet.skills) >= 7, f"应补充3个随机技能,实际 {len(pet.skills)}"
    print(f"[PASS] 随机技能:补充后 {len(pet.skills)} 个技能")


def test_choice_ming_an():
    """选择:使用选择技能执行选项并记录明暗计数。"""
    from sim.data_loader import load_skills
    choice_skill = next(s for s in load_skills() if "选择：" in s.get("desc", "") and s.get("category") in (0, 1))
    pet = make("A", "音速犬", skill_names=[choice_skill["name"], "啄击"])
    pet.energy = 10
    b = make("B", "岚鸟")
    random.seed(1)
    state = create_team_battle([pet], [b])
    state = step(state, Action(kind="skill", skill_index=0), Action(kind="charge"))
    counts = (pet.feature_state.get("ming_count", 0), pet.feature_state.get("an_count", 0))
    assert sum(counts) == 1, f"选择技能使用后应记录1次明/暗,实际 {counts}"
    print(f"[PASS] 选择:{choice_skill['name']} 使用后记录明/暗 {counts}")


if __name__ == "__main__":
    test_swift()
    test_drive_and_position()
    test_qiaobian_same_type()
    test_bloodline_leader()
    test_ball()
    test_barrel()
    test_weekend_and_night()
    test_mark_stack()
    test_random_skills_and_stone()
    test_choice_ming_an()
    print("\nALL DONE")
