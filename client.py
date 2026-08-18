#!/usr/bin/env python3
"""Headless battle client.

Run:
  python client.py --port 5000
"""

import argparse
import json
import os
import random
import socket
from pathlib import Path


def send_line(conn, obj):
    conn.sendall((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))


def read_line(f):
    line = f.readline()
    if not line:
        return None
    return json.loads(line)


LOG_FILES = []
AUTO_MODE = False
MY_SIDE = None

# 元素 0-17 名称（与 sim/enums.Element 一致）
ELEMENT_NAMES = ["普通", "草", "火", "水", "光", "地", "冰", "龙", "电", "毒",
                 "虫", "武", "翼", "萌", "幽", "恶", "机械", "幻"]
# 首领血脉编号（与 sim/enums.LORD_BLOODLINE 一致）
LORD_BLOODLINE = 18


def log(msg):
    print(msg)
    for f in LOG_FILES:
        f.write(str(msg) + "\n")
        f.flush()


def load_team(team_name):
    """按名字加载 data/<team_name>.json 的队伍（如 --team 1abc）。"""
    path = Path(os.path.dirname(os.path.abspath(__file__))) / "data" / f"{team_name}.json"
    if not path.exists():
        raise SystemExit(f"找不到队伍文件：{path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)["team"]


def choose_lead(team):
    log("\n选择首发精灵：")
    for i, cfg in enumerate(team):
        log(f"  {i + 1}: {cfg['spirit']}")
    while True:
        value = input("请输入首发编号 (1-6): ").strip()
        if value.isdigit() and 1 <= int(value) <= len(team):
            return int(value) - 1
        log("无效编号，请重新输入")


def choose_resonance():
    while True:
        print("选择共鸣魔法 (0=愿力冲击, 1=进化之力, 2=光合治愈, 回车=无):")
        value = input("请输入: ").strip()
        if value == "":
            return None
        if value in ("0", "1", "2"):
            return int(value)
        print("无效选择，请输入 0/1/2 或直接回车")


def print_state(state):
    home = state.get("home_side", "?")
    if AUTO_MODE:
        log(f"回合 {state['turn']}  主场 {home}  魔力 A:{state['magic']['A']} B:{state['magic']['B']}")
        return
    log("\n" + "=" * 56)
    log(f"回合 {state['turn']}  主场 {home}  魔力 A:{state['magic']['A']} B:{state['magic']['B']}")
    for side in ("A", "B"):
        active_idx = state["active"][side]
        log(f"[{side}]")
        side_marks = state.get("marks", {}).get(side, {})
        pos = side_marks.get("positive")
        neg = side_marks.get("negative")
        pos_text = f"正:{pos['id']}x{pos['stacks']}" if pos else "正:无"
        neg_text = f"负:{neg['id']}x{neg['stacks']}" if neg else "负:无"
        log(f"  印记 {pos_text} {neg_text}")
        for i, pet in enumerate(state["teams"][side]):
            marker = ">" if i == active_idx else " "
            speed_text = str(pet['stats']['speed'])
            if side == MY_SIDE:
                # 服务端已下发真实速度（含特性修正），直接使用
                eff = pet.get('eff_speed')
                if eff is not None:
                    speed_text = str(eff)
                else:
                    speed = pet['stats']['speed']
                    speed_percent = 0
                    for buff in pet.get('buffs', []):
                        if buff['type'] == 'speed':
                            speed += buff['value'] * 10
                        elif buff['type'] == 'speed_percent':
                            speed_percent += buff['value']
                    neg_mark = state.get('marks', {}).get(side, {}).get('negative')
                    if neg_mark is not None and neg_mark['id'] == 3:
                        speed -= 10 * neg_mark['stacks']
                    speed = int(speed * (1.0 + speed_percent * 0.1))
                    speed_text = str(speed)
            elif pet.get('speed_range'):
                eff_range = pet.get('eff_speed_range')
                if eff_range is not None:
                    # 服务端已下发真实速度范围（含特性修正）
                    speed_text = f"{eff_range[0]}~{eff_range[1]}"
                else:
                    speed_mod = 0
                    speed_percent = 0
                    for buff in pet.get('buffs', []):
                        if buff['type'] == 'speed':
                            speed_mod += buff['value'] * 10
                        elif buff['type'] == 'speed_percent':
                            speed_percent += buff['value']
                    neg_mark = state.get('marks', {}).get(side, {}).get('negative')
                    if neg_mark is not None and neg_mark['id'] == 3:
                        speed_mod -= 10 * neg_mark['stacks']
                    lo = int((pet['speed_range'][0] + speed_mod) * (1.0 + speed_percent * 0.1))
                    hi = int((pet['speed_range'][1] + speed_mod) * (1.0 + speed_percent * 0.1))
                    speed_text = f"{lo}~{hi}"
            if side == MY_SIDE:
                hp_text = f"HP {pet['hp']}/{pet['max_hp']}"
            else:
                pct = round(pet['hp'] / pet['max_hp'] * 100) if pet['max_hp'] else 0
                hp_text = f"HP {pct}%"
            log(f" {marker} #{i + 1} {pet['name']} {hp_text} 能量 {pet['energy']} 速 {speed_text}")
            # 同类型同时长的 buff 合并显示（如助燃多次触发的 atk x2 合并为 atk x6）
            merged = {}
            for buff in pet.get("buffs", []):
                key = (buff['type'], buff['duration'])
                merged[key] = merged.get(key, 0) + buff['value']
            for (btype, dur), value in sorted(merged.items()):
                log(f"     buff {btype} x{value} ({dur})")
            for skill in pet["skills"]:
                elem = ELEMENT_NAMES[skill['element']] if skill.get('element') is not None else '?'
                if "desc" not in skill:
                    display = skill.get('display_power')
                    display_text = f" 显示威力 {display}" if display is not None else ""
                    log(f"     对方用过: {skill['name']} [{elem}] 能耗{skill['energy_cost']}{display_text}")
                    continue
                display = skill.get('display_power')
                display_text = f" 显示威力 {display}" if display is not None else ""
                log(f"     技能{skill['index'] + 1}: {skill['name']} [{elem}] 能耗{skill['energy_cost']}{display_text} "
                    f"{'物攻' if skill['category'] == 0 else '魔攻' if skill['category'] == 1 else '其他'} | {skill['desc']}")
            if not pet["skills"]:
                log("     （技能不可见）")
    log("=" * 56)


def can_use_resonance(state):
    magic_id = state.get("resonance_magic", {}).get(MY_SIDE)
    if magic_id is None:
        return False
    key = str(magic_id)
    used = state.get("resonance_usage", {}).get(MY_SIDE, {}).get(key, 0)
    limit = 2 if magic_id == 0 else 1
    if used >= limit:
        return False
    cooldown = state.get("resonance_cooldown", {}).get(MY_SIDE, {}).get(key, 0)
    return cooldown <= 0


def show_resonance_info(state):
    magic_id = state.get("resonance_magic", {}).get(MY_SIDE)
    if magic_id is None:
        log("当前未携带共鸣魔法")
        return
    names = {0: "愿力冲击", 1: "进化之力", 2: "光合治愈"}
    limits = {0: 2, 1: 1, 2: 1}
    key = str(magic_id)
    used = state.get("resonance_usage", {}).get(MY_SIDE, {}).get(key, 0)
    cooldown = state.get("resonance_cooldown", {}).get(MY_SIDE, {}).get(key, 0)
    log(f"共鸣魔法: {names.get(magic_id, magic_id)} 剩余 {max(0, limits.get(magic_id, 1) - used)} 次 冷却 {cooldown}")


def show_bench(state):
    log("场下精灵：")
    for i, pet in enumerate(state["teams"][MY_SIDE]):
        if i == state["active"][MY_SIDE]:
            continue
        status = "存活" if pet["hp"] > 0 else "无法战斗"
        log(f"  {i + 1}: {pet['name']} HP {pet['hp']}/{pet['max_hp']} {status}")


def choose_lord_branch(state):
    pet = state["teams"][MY_SIDE][state["active"][MY_SIDE]]
    spirits = json.load(open(Path(os.path.dirname(os.path.abspath(__file__))) / "data" / "spirits.json", encoding="utf-8"))["spirits"]
    spirit = next((x for x in spirits if x["id"] == pet["spirit_id"]), None)
    if spirit is None:
        return 0
    branches = []
    for chain in spirit.get("evolution") or []:
        for b in chain.get("lordBranches") or []:
            branches.append(b["name"])
    if not branches:
        return 0
    print("可选择的首领形态：")
    for i, name in enumerate(branches):
        print(f"  {i + 1}: {name}")
    while True:
        value = input("请输入编号: ").strip()
        if value.isdigit() and 1 <= int(value) <= len(branches):
            return int(value) - 1
        print("无效编号")


def prompt_action(conn, state):
    magic_id = None
    magic_branch = 0
    while True:
        action = input("输入 (W1=共鸣, X=聚能, 1-4=技能, E<0-9>=换人, esc=逃跑): ").strip()
        lower = action.lower()
        if lower == "w":
            show_resonance_info(state)
            continue
        if lower == "e":
            show_bench(state)
            continue
        if lower == "w1":
            if not can_use_resonance(state):
                log("共鸣魔法不可用")
                continue
            magic_id = state.get("resonance_magic", {}).get(MY_SIDE)
            if magic_id == 1:
                pet = state["teams"][MY_SIDE][state["active"][MY_SIDE]]
                if pet.get("bloodline") != LORD_BLOODLINE:
                    log("当前精灵不是首领血脉，无法首领化")
                    continue
                magic_branch = choose_lord_branch(state)
            log(f"操作: 共鸣魔法 {magic_id}")
            continue
        if lower == "x":
            log("操作: 聚能")
            send_line(conn, {"kind": "charge", "magic_id": magic_id, "magic_branch": magic_branch})
            return
        if action in ("1", "2", "3", "4"):
            idx = int(action) - 1
            pet = state["teams"][MY_SIDE][state["active"][MY_SIDE]]
            skill = pet["skills"][idx]
            # 能量不足不拦截也不预警：可能由特性兜底（如石头大餐），直接发送由服务端裁决
            if skill.get("category") == 2 and skill.get("skill_id") in pet.get("defense_cooldowns", []):
                log(f"防御技能冷却中：{skill['name']}")
                continue
            log(f"操作: 技能{int(action)}")
            send_line(conn, {"kind": "skill", "skill_index": idx, "magic_id": magic_id, "magic_branch": magic_branch})
            return
        if lower.startswith("e") and not lower.startswith("esc"):
            idx_text = action[1:].strip()
            if idx_text.isdigit() and 1 <= int(idx_text) <= 6:
                idx = int(idx_text) - 1
                target = state["teams"][MY_SIDE][idx]
                if idx == state["active"][MY_SIDE]:
                    log("当前精灵已经在场上")
                    continue
                if target["hp"] <= 0:
                    log(f"该精灵已无法战斗：{target['name']}")
                    continue
                log(f"操作: 换人 {idx_text}")
                send_line(conn, {"kind": "switch", "pet_index": idx, "magic_id": magic_id, "magic_branch": magic_branch})
                return
        if lower in ("esc", "escape"):
            log("操作: 逃跑")
            send_line(conn, {"kind": "flee", "magic_id": magic_id, "magic_branch": magic_branch})
            return
        log("无效输入，请输入 W1 / X / 1-4 / E<0-9> / esc")


def prompt_replacement(conn, state):
    while True:
        action = input("选择上场的精灵 (E <0-9>): ").strip()
        lower = action.lower()
        if lower.startswith("e"):
            idx_text = action[1:].strip()
        elif action.isdigit():
            idx_text = action
        else:
            idx_text = ""
        if idx_text.isdigit() and 1 <= int(idx_text) <= 6:
            idx = int(idx_text) - 1
            target = state["teams"][MY_SIDE][idx]
            if target["hp"] <= 0:
                log(f"该精灵已无法战斗：{target['name']}")
                continue
            log(f"操作: 换人 {idx_text}")
            send_line(conn, {"kind": "switch", "pet_index": idx})
            return
        log("请输入 E<0-9> 或直接输入编号")


def auto_action(conn, state, side):
    idx = state["active"][side]
    if idx < 0:
        return
    pet = state["teams"][side][idx]
    if pet["hp"] / pet["max_hp"] < 0.25:
        candidates = [
            i for i, p in enumerate(state["teams"][side])
            if i != idx and p["hp"] > 0 and p["hp"] / p["max_hp"] > 0.5
        ]
        if candidates and random.random() < 0.3:
            target = random.choice(candidates)
            log(f"操作: 换人 {target + 1}")
            send_line(conn, {"kind": "switch", "pet_index": target})
            return
    usable = [
        i for i, skill in enumerate(pet["skills"])
        if skill.get("power") is not None
        and skill.get("category") in (0, 1)
        and skill.get("energy_cost", 0) <= pet["energy"]
    ]
    if usable:
        best = max(usable, key=lambda i: pet["skills"][i].get("display_power") or pet["skills"][i].get("power") or 0)
        log(f"操作: 技能{best + 1}")
        send_line(conn, {"kind": "skill", "skill_index": best})
    else:
        log("操作: 聚能")
        send_line(conn, {"kind": "charge"})


def auto_replace(conn, state, side):
    alive = [i for i, pet in enumerate(state["teams"][side]) if pet["hp"] > 0]
    if alive:
        idx = random.choice(alive)
        log(f"操作: 换人 {idx}")
        send_line(conn, {"kind": "switch", "pet_index": idx})


def main():
    parser = argparse.ArgumentParser(description="Headless Roco 6v6 client")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--nature", type=int, default=-1)
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--team", default=None,
                        help="队伍文件名（data/<name>.json，如 1abc）；缺省则不发送队伍，走服务端默认 1v1")
    args = parser.parse_args()
    # random.seed(42)
    global AUTO_MODE
    AUTO_MODE = args.auto
    team = load_team(args.team) if args.team else None

    conn = socket.create_connection((args.host, args.port))
    f = conn.makefile("r", encoding="utf-8")
    awaiting_replacement = False
    current_state = None
    my_side = None
    last_printed_turn = None
    log(f"connected to {args.host}:{args.port}")

    while True:
        msg = read_line(f)
        if msg is None:
            log("server closed")
            break
        msg_type = msg.get("type")
        if msg_type == "welcome":
            side = msg["side"]
            global MY_SIDE
            MY_SIDE = side
            my_side = side
            logs_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "logs" / "battle"
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_name = "clientA.log" if side == "A" else "clientB.log"
            client_log = open(logs_dir / log_name, "w", encoding="utf-8")
            LOG_FILES.extend([client_log])

            log(f"你是 {side} 方")
            if team is None:
                lead = 0
                resonance = None
            elif args.auto:
                lead = random.randrange(len(team))
                log(f"自动选择首发：{lead + 1}")
                resonance = None
            else:
                lead = choose_lead(team)
                resonance = choose_resonance()
            send_line(conn, {"type": "config", "nature": args.nature, "team": team, "lead": lead, "resonance": resonance})
        elif msg_type == "choose_replacement":
            awaiting_replacement = True
            if args.auto and current_state is not None:
                auto_replace(conn, current_state, my_side)
            else:
                prompt_replacement(conn, current_state)
        elif msg_type == "state":
            awaiting_replacement = False
            current_state = msg["state"]
            if current_state.get("winner"):
                break
            if current_state["active"]["A"] < 0 or current_state["active"]["B"] < 0:
                continue
            if last_printed_turn != current_state["turn"]:
                print_state(current_state)
                last_printed_turn = current_state["turn"]
            if args.auto:
                auto_action(conn, current_state, my_side)
            else:
                prompt_action(conn, current_state)
        elif msg_type == "error":
            log(msg.get("message", "操作无效，请重新输入"))
            if args.auto:
                if awaiting_replacement and current_state is not None:
                    auto_replace(conn, current_state, my_side)
                else:
                    auto_action(conn, current_state, my_side)
            elif awaiting_replacement:
                prompt_replacement(conn, current_state)
            else:
                prompt_action(conn, current_state)
        elif msg_type == "game_over":
            log(f"游戏结束，胜者：{msg['winner']}")
            break

    conn.close()
    for f in LOG_FILES:
        f.close()


if __name__ == "__main__":
    main()
