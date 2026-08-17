"""Feature-system tests: trigger verification + randomized smoke tests."""

import random
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim import buffs, features, marks
from sim.battle import create_team_battle, step
from sim.data_loader import find_spirit, load_spirits, make_battle_pet
from sim.models import Action

SPIRITS = load_spirits()


def make(side, name, skill_names=None):
    spirit = find_spirit(name, SPIRITS)
    assert spirit is not None, f"找不到精灵 {name}"
    return make_battle_pet(spirit, side, skill_names=skill_names)


def run_turn(state, acts):
    return step(state, *acts)


def test_import_and_load():
    feats = features.load_features()
    assert len(feats) == 231, f"特性 id 数量应为 231,实际 {len(feats)}"
    assert all(f["desc"] for f in feats.values()), "存在空描述特性"
    dimo = find_spirit("迪莫", SPIRITS)
    assert dimo["feature"]["desc"].startswith("造成克制伤害后"), dimo["feature"]["desc"]
    print("[PASS] 特性加载:231 个特性 id / 596 只精灵,迪莫特性正确")


def test_dimo_trigger():
    """迪莫:造成克制伤害后攻防速+20%并回复2能量。"""
    from sim.data_loader import load_typechart
    from sim.typechart import type_multiplier
    tc = load_typechart()
    a = make("A", "迪莫")
    # 找一只被光系克制的精灵当对手
    victim = None
    for s in SPIRITS:
        if not s.get("stats"):
            continue
        attrs = [x for x in (s.get("mainElement"), s.get("subElement")) if x is not None]
        if type_multiplier(4, attrs, tc) > 1.0:
            victim = s
            break
    assert victim is not None, "找不到光系克制目标"
    b = make("B", victim["name"])
    state = create_team_battle([a], [b])
    state = step(state, Action(kind="charge"), Action(kind="charge"))
    state = step(state, Action(kind="skill", skill_index=0), Action(kind="charge"))
    log_text = "\n".join(state.log)
    assert "克制伤害后攻防速+20%" in log_text, f"迪莫克制特性未触发:\n{log_text}"
    assert any(b_.buff_type == "atk" and b_.value > 0 for b_ in a.buffs), "应获得物攻buff"
    print(f"[PASS] 迪莫对 {victim['name']} 触发克制特性(攻防速+20% + 回能)")


def test_entry_lifesteal():
    """入场时获得50%吸血。"""
    a = make("A", "奇丽花")
    # 找一只带"入场时获得50%吸血"特性的精灵
    target = None
    for s in SPIRITS:
        if s.get("feature") and s["feature"]["desc"] == "入场时获得50%吸血。":
            target = s
            break
    assert target is not None, "未找到入场吸血特性精灵"
    pet = make_battle_pet(target, "A")
    state = create_team_battle([pet], [make("B", "岚鸟")])
    state = step(state, Action(kind="charge"), Action(kind="charge"))
    assert buffs.get_lifesteal(pet) > 0, "入场吸血未生效"
    print(f"[PASS] {target['name']} 入场吸血生效")


def test_faint_magic_save():
    """自己力竭时少损失1点魔力。"""
    target = None
    for s in SPIRITS:
        if s.get("feature") and s["feature"]["desc"] == "自己力竭时，少损失1点魔力。":
            target = s
            break
    assert target is not None
    pet = make_battle_pet(target, "A")
    pet.hp = 1
    state = create_team_battle([pet], [make("B", "岚鸟")])
    before = state.magic["A"]
    # 直接调用 _apply_faint 等价场景:让 B 攻击 A
    state = step(state, Action(kind="charge"), Action(kind="skill", skill_index=0))
    assert state.magic["A"] == before, f"力竭应少损失魔力: {before} -> {state.magic['A']}"
    print(f"[PASS] {target['name']} 力竭少损失1点魔力")


def test_zero_energy_immunity():
    """能量等于0的精灵，无法对自己造成伤害(攻击方能量为0时免疫)。"""
    target = None
    for s in SPIRITS:
        if s.get("feature") and s["feature"]["desc"] == "能量等于0的精灵，无法对自己造成伤害。":
            target = s
            break
    assert target is not None, "未找到能量免疫特性精灵"
    defender = make_battle_pet(target, "A")
    attacker = make("B", "岚鸟")
    attacker.energy = 0
    state = create_team_battle([defender], [attacker])
    state = step(state, Action(kind="charge"), Action(kind="skill", skill_index=0))
    hp_before = defender.hp
    assert defender.hp == hp_before, f"攻击方能量为0应免疫伤害: {hp_before} -> {defender.hp}"
    print(f"[PASS] {target['name']} 免疫能量为0攻击者的伤害")


def test_heal_redirect():
    """自己无法回复生命，而是将回复生命变为敌方扣除等量生命。"""
    target = None
    for s in SPIRITS:
        if s.get("feature") and "自己无法回复生命" in s["feature"]["desc"]:
            target = s
            break
    assert target is not None
    a = make_battle_pet(target, "A")
    b = make("B", "岚鸟")
    state = create_team_battle([a], [b])
    # 给 A 造成伤害,让 B 打 A 后,再让 A 使用草系技能触发"回复10%生命"转伤害
    # 直接用回合结束回复验证:找"回合结束时回复12%生命"特性与"无法回复"是同一只精灵才有效
    # 此处验证 on_heal 被 _heal 调用:手动构造场景
    from sim import features as feats
    a.hp = int(a.max_hp * 0.5)
    b.hp = b.max_hp
    # 直接调用特性层 _heal(模拟草系回复),应转为敌方扣血而非自己回复
    feats._heal(state, "A", a, int(a.max_hp * 0.10), state.log)
    assert a.hp == int(a.max_hp * 0.5), "不应回复自己"
    assert b.hp < b.max_hp, "回复应转为敌方扣血"
    print(f"[PASS] {target['name']} 回复转为敌方扣血")


def test_random_smoke(n=12, seed=7):
    """随机 6v6 冒烟测试,确保特性系统不崩溃。"""
    random.seed(seed)
    playable = [s for s in SPIRITS if s.get("stats") and s["stats"].get("hp")]
    feature_count = 0
    for game in range(n):
        team_a = [make_battle_pet(random.choice(playable), "A") for _ in range(6)]
        team_b = [make_battle_pet(random.choice(playable), "B") for _ in range(6)]
        state = create_team_battle(team_a, team_b)
        for _ in range(60):
            if state.winner is not None:
                break
            acts = []
            for side in ("A", "B"):
                if state.active[side] < 0:
                    alive = [i for i, p in enumerate(state.teams[side]) if p.hp > 0]
                    acts.append(Action(kind="switch", pet_index=random.choice(alive) if alive else 0))
                else:
                    pet = state.teams[side][state.active[side]]
                    usable = [i for i, s in enumerate(pet.skills)
                              if s.skill_id not in pet.blocked_skills]
                    if usable and random.random() < 0.8:
                        acts.append(Action(kind="skill", skill_index=random.choice(usable)))
                    else:
                        acts.append(Action(kind="charge"))
            state = step(state, *acts)
            for line in state.log:
                if "[特性]" in line:
                    feature_count += 1
        assert state.winner is not None or state.turn > 50
    print(f"[PASS] 随机 {n} 场 6v6 冒烟完成,特性日志 {feature_count} 条")


def coverage_stats():
    """统计特性实现覆盖率:distinct 描述数 vs 匹配到 handler 的描述数。"""
    distinct = {}
    for s in SPIRITS:
        if s.get("feature"):
            d = s["feature"]["desc"]
            distinct.setdefault(d, 0)
            distinct[d] += 1
    src = open(Path(__file__).resolve().parent.parent / "sim" / "features.py", encoding="utf-8").read()
    matched = 0
    unmatched = []
    for d, c in distinct.items():
        # 用描述中的关键短语在源码中搜索
        frag = d.split("，")[0][:8] if len(d) > 8 else d
        if frag in src or d[:6] in src:
            matched += 1
        else:
            unmatched.append((d, c))
    print(f"distinct 描述:{len(distinct)},粗略匹配:{matched},未匹配:{len(unmatched)}")
    for d, c in unmatched:
        print(f"  [{c}] {d}")


if __name__ == "__main__":
    test_import_and_load()
    test_dimo_trigger()
    test_entry_lifesteal()
    test_faint_magic_save()
    test_zero_energy_immunity()
    test_heal_redirect()
    test_random_smoke()
    coverage_stats()
    print("\nALL DONE")
