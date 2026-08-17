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


def log(msg):
    print(msg)
    for f in LOG_FILES:
        f.write(str(msg) + "\n")
        f.flush()


def load_team(side):
    name = "test_team.json" if side == "A" else "test_team_b.json"
    path = Path(os.path.dirname(os.path.abspath(__file__))) / "data" / name
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
                speed = pet['stats']['speed']
                for buff in pet.get('buffs', []):
                    if buff['type'] == 'speed':
                        speed += buff['value'] * 10
                neg_mark = state.get('marks', {}).get(side, {}).get('negative')
                if neg_mark is not None and neg_mark['id'] == 3:
                    speed -= 10 * neg_mark['stacks']
                speed_text = str(speed)
            elif pet.get('speed_range'):
                speed_mod = 0
                for buff in pet.get('buffs', []):
                    if buff['type'] == 'speed':
                        speed_mod += buff['value'] * 10
                neg_mark = state.get('marks', {}).get(side, {}).get('negative')
                if neg_mark is not None and neg_mark['id'] == 3:
                    speed_mod -= 10 * neg_mark['stacks']
                speed_text = f"{pet['speed_range'][0] + speed_mod}~{pet['speed_range'][1] + speed_mod}"
            if side == MY_SIDE:
                hp_text = f"HP {pet['hp']}/{pet['max_hp']}"
            else:
                pct = round(pet['hp'] / pet['max_hp'] * 100) if pet['max_hp'] else 0
                hp_text = f"HP {pct}%"
            log(f" {marker} #{i + 1} {pet['name']} {hp_text} 能量 {pet['energy']} 速 {speed_text}")
            for buff in pet.get("buffs", []):
                log(f"     buff {buff['type']} x{buff['value']} ({buff['duration']})")
            for skill in pet["skills"]:
                if "desc" not in skill:
                    log(f"     对方用过: {skill['name']} 能耗{skill['energy_cost']}")
                    continue
                display = skill.get('display_power')
                display_text = f" 显示威力 {display}" if display is not None else ""
                log(f"     技能{skill['index'] + 1}: {skill['name']} 能耗{skill['energy_cost']}{display_text} "
                    f"{'物攻' if skill['category'] == 0 else '魔攻' if skill['category'] == 1 else '其他'} | {skill['desc']}")
            if not pet["skills"]:
                log("     （技能不可见）")
    log("=" * 56)


def prompt_action(conn, state):
    while True:
        action = input("输入动作 (X=聚能, 1-4=技能, E <0-9>=换人, esc=逃跑): ").strip()
        lower = action.lower()
        if lower == "x":
            log("操作: 聚能")
            send_line(conn, {"kind": "charge"})
            return
        if action in ("1", "2", "3", "4"):
            idx = int(action) - 1
            pet = state["teams"][MY_SIDE][state["active"][MY_SIDE]]
            skill = pet["skills"][idx]
            if skill["energy_cost"] > pet["energy"]:
                log(f"能量不足：{skill['name']} 需要 {skill['energy_cost']}，当前 {pet['energy']}")
                continue
            if skill.get("category") == 2 and skill.get("skill_id") in pet.get("defense_cooldowns", []):
                log(f"防御技能冷却中：{skill['name']}")
                continue
            log(f"操作: 技能{int(action)}")
            send_line(conn, {"kind": "skill", "skill_index": idx})
            return
        if lower.startswith("e") and not lower.startswith("esc"):
            idx_text = action[1:].strip()
            if idx_text.isdigit() and 1 <= int(idx_text) <= 6:
                idx = int(idx_text) - 1
                target = state["teams"][MY_SIDE][idx]
                if target["hp"] <= 0:
                    log(f"该精灵已无法战斗：{target['name']}")
                    continue
                log(f"操作: 换人 {idx_text}")
                send_line(conn, {"kind": "switch", "pet_index": idx})
                return
        if lower in ("esc", "escape"):
            log("操作: 逃跑")
            send_line(conn, {"kind": "flee"})
            return
        log("无效输入，请输入 X / 1-4 / E<0-9> / esc")


def prompt_replacement(conn, state):
    while True:
        action = input("选择上场的精灵 (E <0-9>): ").strip()
        lower = action.lower()
        if lower.startswith("e"):
            idx_text = action[1:].strip()
            if idx_text.isdigit() and 1 <= int(idx_text) <= 6:
                idx = int(idx_text) - 1
                target = state["teams"][MY_SIDE][idx]
                if target["hp"] <= 0:
                    log(f"该精灵已无法战斗：{target['name']}")
                    continue
                log(f"操作: 换人 {idx_text}")
                send_line(conn, {"kind": "switch", "pet_index": idx})
                return
        log("请输入 E<0-9>")


def auto_action(conn, state, side):
    idx = state["active"][side]
    if idx < 0:
        return
    pet = state["teams"][side][idx]
    usable = [
        i for i, skill in enumerate(pet["skills"])
        if skill.get("power") is not None
        and skill.get("category") in (0, 1)
        and skill.get("energy_cost", 0) <= pet["energy"]
    ]
    if usable:
        idx = random.choice(usable)
        log(f"操作: 技能{idx + 1}")
        send_line(conn, {"kind": "skill", "skill_index": idx})
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
    args = parser.parse_args()
    # random.seed(42)
    global AUTO_MODE
    AUTO_MODE = args.auto
    team = None

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
            team = load_team(side)
            logs_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "logs" / "battle"
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_name = "clientA.log" if side == "A" else "clientB.log"
            client_log = open(logs_dir / log_name, "w", encoding="utf-8")
            LOG_FILES.extend([client_log])

            log(f"你是 {side} 方")
            if args.auto:
                lead = random.randrange(len(team))
                log(f"自动选择首发：{lead + 1}")
            else:
                lead = choose_lead(team)
            send_line(conn, {"type": "config", "nature": args.nature, "team": team, "lead": lead})
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
