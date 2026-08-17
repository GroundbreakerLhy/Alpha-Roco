#!/usr/bin/env python3
"""对战仿真:特性系统实战演示。

  python tests/simulate_battle.py
    - 1v1 展示战:迪莫(克制特性) vs 锥尾羊(被光系克制)
    - 6v6 团队战:test_team(含迪莫) vs 随机带特性队伍
输出完整日志到 logs/simulation/,并打印特性触发摘要。
"""

import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

from sim.battle import create_team_battle, step
from sim.data_loader import find_spirit, load_spirits, make_battle_pet
from sim.models import Action

SPIRITS = load_spirits()
LOG_DIR = ROOT / "logs" / "simulation"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def choose_action(state, side):
    """简单启发式:优先使用可负担且预估伤害最高的攻击技能,否则聚能。"""
    if state.active[side] < 0:
        alive = [i for i, p in enumerate(state.teams[side]) if p.hp > 0]
        return Action(kind="switch", pet_index=alive[0] if alive else 0)
    pet = state.teams[side][state.active[side]]
    best = None
    for i, s in enumerate(pet.skills):
        if s.category in (0, 1) and s.power and s.energy_cost <= pet.energy \
                and s.skill_id not in pet.blocked_skills:
            score = s.power * (1.25 if s.element in pet.attributes else 1.0)
            if best is None or score > best[0]:
                best = (score, i)
    if best is not None:
        return Action(kind="skill", skill_index=best[1])
    return Action(kind="charge")


def run_battle(name, team_a, team_b, max_turns=60, seed=None):
    if seed is not None:
        random.seed(seed)
    state = create_team_battle(team_a, team_b)
    feature_lines = []
    turn_log = []
    for _ in range(max_turns):
        if state.winner is not None:
            break
        state = step(state, choose_action(state, "A"), choose_action(state, "B"))
        for line in state.log:
            if "[特性]" in line:
                feature_lines.append(f"  T{state.turn} {line.strip()}")
            turn_log.append(f"T{state.turn} {line}")
        turn_log.append("")
    out = LOG_DIR / f"{name}.txt"
    out.write_text(f"===== {name} =====\n"
                   + f"winner: {state.winner}  回合数: {state.turn}\n"
                   + f"A 剩余魔力 {state.magic['A']} / B 剩余魔力 {state.magic['B']}\n\n"
                   + "\n".join(turn_log), encoding="utf-8")
    print(f"[{name}] winner={state.winner} turn={state.turn} "
          f"魔力 {state.magic['A']}:{state.magic['B']} 特性触发 {len(feature_lines)} 条")
    return state, feature_lines, turn_log


def showcase_1v1():
    """迪莫(造成克制伤害后攻防速+20%回2能量) vs 被光系克制的锥尾羊。"""
    a = make_battle_pet(find_spirit("迪莫", SPIRITS), "A")
    b = make_battle_pet(find_spirit("锥尾羊", SPIRITS), "B")
    state, feats, log = run_battle("showcase_dimo_vs_zuiweiyang", [a], [b], seed=1)
    print("\n--- 迪莫特性触发片段 ---")
    for l in feats:
        print(l)
    return state


def team_6v6():
    """test_team(含迪莫) vs 随机带特性队伍。"""
    team_cfg = json.load(open(ROOT / "data" / "test_team.json", encoding="utf-8"))["team"]
    team_a = []
    for cfg in team_cfg:
        sp = find_spirit(cfg["spirit"], SPIRITS)
        team_a.append(make_battle_pet(sp, "A", ivs=cfg.get("ivs"), nature=cfg.get("nature", -1)))

    playable = [s for s in SPIRITS if s.get("stats") and s["stats"].get("hp") and s.get("feature")]
    random.seed(42)
    picks = random.sample(playable, 6)
    team_b = [make_battle_pet(sp, "B") for sp in picks]
    print("\nA 队:", [p.name for p in team_a])
    print("B 队:", [p.name for p in team_b])

    state, feats, log = run_battle("6v6_auto", team_a, team_b, seed=42)

    print("\n--- 特性触发统计 ---")
    counter = Counter()
    for l in feats:
        text = l.split("]")[1].strip() if "]" in l else l
        for key in ("克制伤害", "吸血", "回复", "中毒", "冻结", "灼烧", "连击",
                    "脱离", "偷取", "力竭", "击败", "入场", "萌化", "奉献",
                    "印记", "星陨", "免疫", "技能"):
            if key in text:
                counter[key] += 1
                break
        else:
            counter["其他"] += 1
    for k, v in counter.most_common():
        print(f"  {k}: {v} 次")
    print(f"\n完整日志: {LOG_DIR / '6v6_auto.txt'}")
    return state


if __name__ == "__main__":
    print("========== 1v1 展示战 ==========")
    showcase_1v1()
    print("\n========== 6v6 团队战 ==========")
    team_6v6()
