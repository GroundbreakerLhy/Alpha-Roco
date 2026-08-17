#!/usr/bin/env python3
"""Headless battle server.

Run:
  python server.py --port 5000

Then run two clients:
  python client.py --port 5000
  python client.py --port 5000
"""

import argparse
import json
import os
import random
import socket
from pathlib import Path

from sim import burst, features
from sim.battle import create_battle, create_team_battle, state_to_dict, step
from sim.data_loader import find_spirit, load_spirits, make_battle_pet
from sim.models import Action


def send_line(conn, obj):
    conn.sendall((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))


def recv_line(f):
    line = f.readline()
    if not line:
        return None
    return json.loads(line)


LOG_FILES = []


def log(msg):
    print(msg)
    for f in LOG_FILES:
        f.write(str(msg) + "\n")
        f.flush()


def is_valid_switch(raw, state, side):
    if raw is None or raw.get("kind") != "switch":
        return False
    idx = raw.get("pet_index")
    if idx is None:
        return False
    team = state.teams[side]
    if not (0 <= idx < len(team)):
        return False
    return team[idx].hp > 0


def read_action_with_switch(f, conn, state, side):
    while True:
        raw = recv_line(f)
        if raw is None:
            return None
        if state.active[side] >= 0 or is_valid_switch(raw, state, side):
            return raw
        send_line(conn, {"type": "error", "message": "请选择一只存活的上场精灵 (switch <0-5>)"})


def read_replacement(f, conn, state, side):
    while True:
        raw = recv_line(f)
        if raw is None:
            return None
        if is_valid_switch(raw, state, side):
            state.active[side] = raw["pet_index"]
            pet = state.teams[side][state.active[side]]
            pet.has_acted_since_entry = False
            positive_mark = state.marks[side]["positive"]
            if positive_mark is not None and positive_mark["id"] == 6:
                burst.add_burst(pet, "attack_power_flat", 10 * positive_mark["stacks"])
            state.log.append(f"{side} 换上 {pet.name}")
            features.apply_pending_entry(state, side, state.log)
            features.on_entry(state, side, state.log)
            log(f"{side} 换上 {pet.name}")
            return raw
        send_line(conn, {"type": "error", "message": "请选择一只存活的上场精灵 (switch <0-5>)"})


def is_valid_skill_energy(raw, state, side):
    if raw is None or raw.get("kind") != "skill":
        return True
    idx = raw.get("skill_index")
    if idx is None or state.active[side] < 0:
        return False
    team = state.teams[side]
    if not (0 <= idx < len(team[state.active[side]].skills)):
        return False
    pet = team[state.active[side]]
    skill = pet.skills[idx]
    if pet.energy < skill.energy_cost:
        return False
    if skill.category == 2 and skill.skill_id in pet.defense_cooldowns:
        return False
    return True


def read_action_with_energy(f, conn, state, side):
    while True:
        raw = recv_line(f)
        if raw is None:
            return None
        if is_valid_skill_energy(raw, state, side):
            return raw
        send_line(conn, {"type": "error", "message": "技能不可用，请重新选择"})


def make_pet(side, name, skill_names, nature, spirits):
    spirit = find_spirit(name, spirits)
    if spirit is None:
        raise SystemExit(f"找不到精灵：{name}")
    return make_battle_pet(spirit, side, skill_names=skill_names, nature=nature)


def make_team(side, team_config, spirits):
    team = []
    for cfg in team_config:
        spirit = find_spirit(cfg["spirit"], spirits)
        if spirit is None:
            raise SystemExit(f"找不到精灵：{cfg['spirit']}")
        team.append(make_battle_pet(
            spirit,
            side,
            ivs=cfg.get("ivs"),
            nature=cfg.get("nature", -1),
            skill_names=cfg.get("skills"),
        ))
    return team


def main():
    parser = argparse.ArgumentParser(description="Headless Roco 6v6 server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--spirit-a", default="岚鸟")
    parser.add_argument("--spirit-b", default="奇丽花")
    parser.add_argument("--skills-a", default="扇风,啄击,先发制人,俯冲猛击")
    parser.add_argument("--skills-b", default="棘突,叶绿光束,刺藤,仙人掌刺击")
    parser.add_argument("--day-of-week", type=int, default=0, help="0=周一 ... 6=周日(周末特性判定)")
    parser.add_argument("--no-night", action="store_true", help="关闭王国入夜环境")
    args = parser.parse_args()
    # random.seed(42)

    logs_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "logs" / "battle"
    logs_dir.mkdir(parents=True, exist_ok=True)
    server_log = open(logs_dir / "server.log", "w", encoding="utf-8")
    LOG_FILES.extend([server_log])

    spirits = load_spirits()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(2)
    log(f"server listening on {args.host}:{args.port}")

    log("waiting for client A...")
    conn_a, addr_a = srv.accept()
    f_a = conn_a.makefile("r", encoding="utf-8")
    send_line(conn_a, {"type": "welcome", "side": "A"})
    log(f"client A connected: {addr_a}")

    log("waiting for client B...")
    conn_b, addr_b = srv.accept()
    f_b = conn_b.makefile("r", encoding="utf-8")
    send_line(conn_b, {"type": "welcome", "side": "B"})
    log(f"client B connected: {addr_b}")

    log("waiting for client configs...")
    raw_config_a = recv_line(f_a)
    raw_config_b = recv_line(f_b)
    nature_a = raw_config_a.get("nature", -1) if raw_config_a else -1
    nature_b = raw_config_b.get("nature", -1) if raw_config_b else -1
    lead_a = raw_config_a.get("lead", 0) if raw_config_a else 0
    lead_b = raw_config_b.get("lead", 0) if raw_config_b else 0

    team_a = raw_config_a.get("team") if raw_config_a else None
    team_b = raw_config_b.get("team") if raw_config_b else None

    if team_a and team_b:
        pets_a = make_team("A", team_a, spirits)
        pets_b = make_team("B", team_b, spirits)
        state = create_team_battle(pets_a, pets_b)
        if 0 <= lead_a < len(pets_a):
            state.active["A"] = lead_a
        if 0 <= lead_b < len(pets_b):
            state.active["B"] = lead_b
    else:
        pets = {
            "A": make_pet("A", args.spirit_a, args.skills_a.split(","), nature_a, spirits),
            "B": make_pet("B", args.spirit_b, args.skills_b.split(","), nature_b, spirits),
        }
        state = create_battle(pets["A"], pets["B"])

    state.day_of_week = args.day_of_week
    state.is_night = not args.no_night

    send_line(conn_a, {"type": "state", "state": state_to_dict(state, view_side="A")})
    send_line(conn_b, {"type": "state", "state": state_to_dict(state, view_side="B")})

    while state.winner is None:
        # 死亡后的换人不占用回合：先处理所有需要上场的替换
        replaced = False
        for side, conn, f in (("A", conn_a, f_a), ("B", conn_b, f_b)):
            if state.active[side] < 0:
                send_line(conn, {"type": "choose_replacement"})
                raw = read_replacement(f, conn, state, side)
                if raw is None:
                    log("client disconnected")
                    break
                replaced = True
        else:
            if replaced:
                payload_a = {"type": "state", "state": state_to_dict(state, view_side="A")}
                payload_b = {"type": "state", "state": state_to_dict(state, view_side="B")}
                send_line(conn_a, payload_a)
                send_line(conn_b, payload_b)

            log(f"\n--- turn {state.turn} waiting actions ---")
            raw_a = read_action_with_energy(f_a, conn_a, state, "A")
            raw_b = read_action_with_energy(f_b, conn_b, state, "B")
            if raw_a is None or raw_b is None:
                log("client disconnected")
                break

            action_a = Action(
                kind=raw_a.get("kind", "charge"),
                skill_index=raw_a.get("skill_index"),
                pet_index=raw_a.get("pet_index"),
            )
            action_b = Action(
                kind=raw_b.get("kind", "charge"),
                skill_index=raw_b.get("skill_index"),
                pet_index=raw_b.get("pet_index"),
            )
            state = step(state, action_a, action_b)

            for line in state.log:
                log(line)

            payload_a = {"type": "state", "state": state_to_dict(state, view_side="A")}
            payload_b = {"type": "state", "state": state_to_dict(state, view_side="B")}
            send_line(conn_a, payload_a)
            send_line(conn_b, payload_b)

    if state.winner is not None:
        log(f"winner: {state.winner}")
        payload = {"type": "game_over", "winner": state.winner}
        send_line(conn_a, payload)
        send_line(conn_b, payload)

    conn_a.close()
    conn_b.close()
    srv.close()
    server_log.close()


if __name__ == "__main__":
    main()
