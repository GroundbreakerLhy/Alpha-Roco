from __future__ import annotations

import json
import multiprocessing
import random as _random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import torch

from server import make_team
from sim.actions import enumerate_candidates
from sim.battle import (
    apply_drive,
    create_team_battle,
    pet_to_dict,
    resume_after_leave,
    state_to_dict,
    step,
    switch_in,
    trait_leave_switch,
)
from sim.data_loader import load_spirits
from sim.models import Action

from .arena import choose, load_model, record_opponent_actions
from .features import ObservationEncoder
from .network import AlphaRoco, HistoryState
from .rng import THREAD_RANDOM

ROOT = Path(__file__).resolve().parents[1]
TEAM_DIR = ROOT / "data" / "teams"
SPIRITS = load_spirits()
TEAM_CONFIGS = {}


def _team_config(name: str) -> dict:
    if name not in TEAM_CONFIGS:
        TEAM_CONFIGS[name] = json.loads(
            (TEAM_DIR / f"{name}.json").read_text(encoding="utf-8")
        )
    return TEAM_CONFIGS[name]


def _lead_request(side, team_config, pets, enemy_pets):
    other = "B" if side == "A" else "A"
    own = [pet_to_dict(pet) for pet in pets]
    opponent_team = [pet.name for pet in enemy_pets]
    state_view = {
        "teams": {side: own, other: [{"name": name} for name in opponent_team]},
        "turn": 0,
        "winner": None,
        "active": {"A": -1, "B": -1},
        "magic": {"A": 4, "B": 4},
        "log": [],
    }
    return {
        "type": "decision",
        "side": side,
        "phase": "lead",
        "state": state_view,
        "public_state": {"teams": dict(state_view["teams"])},
        "team": team_config["team"],
        "opponent_team": opponent_team,
        "candidates": [
            {"kind": "lead", "lead": index, "pet_index": index}
            for index in range(len(pets))
        ],
        "events": [],
        "private_events": [],
    }


def _decision_request(state, side, phase, team_config, event_cursor, private_events):
    lines = state.log[event_cursor[side] :]
    event_cursor[side] = len(state.log)
    request = {
        "type": "decision",
        "side": side,
        "phase": phase,
        "state": state_to_dict(state, view_side=side),
        "public_state": None,
        "team": team_config["team"],
        "opponent_team": None,
        "candidates": enumerate_candidates(state, side, phase),
        "events": [{"kind": "log", "phase": "unknown", "line": line} for line in lines],
        "private_events": private_events.pop(side, []),
    }
    return request


def _action(raw):
    return Action(
        kind=raw["kind"],
        skill_index=raw.get("skill_index"),
        pet_index=raw.get("pet_index"),
        magic_id=raw.get("magic_id"),
        magic_branch=raw.get("magic_branch", 0),
        choice_branch=raw.get("choice_branch", 0),
    )


def _choose(model, encoder, request, history, device, sample, private_events):
    action, history = choose(model, encoder, request, history, device, sample)
    private_events[request["side"]] = [action]
    return action, history


def fast_play_game(
    candidate: AlphaRoco,
    reference: AlphaRoco,
    encoder: ObservationEncoder,
    team_a: str,
    team_b: str,
    seed: int,
    device: torch.device,
    sample: bool = False,
):
    config_a = _team_config(team_a)
    config_b = _team_config(team_b)
    pets_a = make_team("A", config_a["team"], SPIRITS)
    pets_b = make_team("B", config_b["team"], SPIRITS)
    models = {"A": candidate, "B": reference}
    histories = {"A": HistoryState(), "B": HistoryState()}
    private_events = {"A": [], "B": []}
    event_cursor = {"A": 0, "B": 0}
    rng = _random.Random(seed)
    THREAD_RANDOM.seed(rng.randrange(2**32))

    lead_requests = {
        "A": _lead_request("A", config_a, pets_a, pets_b),
        "B": _lead_request("B", config_b, pets_b, pets_a),
    }
    leads = {}
    for side in ("A", "B"):
        action, histories[side] = _choose(
            models[side],
            encoder,
            lead_requests[side],
            histories[side],
            device,
            sample,
            private_events,
        )
        leads[side] = action["lead"]
    state = create_team_battle(pets_a, pets_b, active_a=leads["A"], active_b=leads["B"])
    state.resonance_magic["A"] = config_a.get("resonance")
    state.resonance_magic["B"] = config_b.get("resonance")

    while state.winner is None:
        replaced = False
        for side in ("A", "B"):
            if state.active[side] >= 0:
                continue
            request = _decision_request(
                state,
                side,
                "replacement",
                config_a if side == "A" else config_b,
                event_cursor,
                private_events,
            )
            action, histories[side] = _choose(
                models[side],
                encoder,
                request,
                histories[side],
                device,
                sample,
                private_events,
            )
            index = action["pet_index"]
            state.active[side] = index
            pet = state.teams[side][index]
            state.log.append(f"{side} 换上 {pet.name}")
            switch_in(state, side, pet, state.log, thorn=False, active_switch=False)
            apply_drive(state, side)
            replaced = True
        if replaced:
            continue

        requests = {}
        actions = {}
        for side in ("A", "B"):
            requests[side] = _decision_request(
                state,
                side,
                "action",
                config_a if side == "A" else config_b,
                event_cursor,
                private_events,
            )
            action, histories[side] = _choose(
                models[side],
                encoder,
                requests[side],
                histories[side],
                device,
                sample,
                private_events,
            )
            actions[side] = action
        record_opponent_actions(histories, models, encoder, requests, actions)
        state = step(state, _action(actions["A"]), _action(actions["B"]))

        while state.paused_turn is not None and state.winner is None:
            side = state.paused_turn["leave_side"]
            request = _decision_request(
                state,
                side,
                "replacement",
                config_a if side == "A" else config_b,
                event_cursor,
                private_events,
            )
            action, histories[side] = _choose(
                models[side],
                encoder,
                request,
                histories[side],
                device,
                sample,
                private_events,
            )
            incoming = state.teams[side][action["pet_index"]]
            state = resume_after_leave(state, side, incoming)

        switched = False
        for side in ("A", "B"):
            if not state.pending_switch[side]:
                continue
            state.pending_switch[side] = False
            request = _decision_request(
                state,
                side,
                "replacement",
                config_a if side == "A" else config_b,
                event_cursor,
                private_events,
            )
            action, histories[side] = _choose(
                models[side],
                encoder,
                request,
                histories[side],
                device,
                sample,
                private_events,
            )
            incoming = state.teams[side][action["pet_index"]]
            trait_leave_switch(state, side, incoming, state.log)
            apply_drive(state, side)
            switched = True
        if switched:
            continue
    return state.winner


_WORKER = {}


def _init_worker(candidate_path, reference_path, seed, device, sample):
    torch.set_num_threads(1)
    encoder = ObservationEncoder()
    torch_device = torch.device(device)
    _WORKER.update(
        candidate=load_model(Path(candidate_path), encoder, torch_device),
        reference=load_model(Path(reference_path), encoder, torch_device),
        encoder=encoder,
        seed=seed,
        device=torch_device,
        sample=sample,
    )


def _run_worker_chunk(chunk):
    tally = {"win": 0, "loss": 0, "both_lose": 0}
    for index, team_a, team_b in chunk:
        winner = fast_play_game(
            _WORKER["candidate"],
            _WORKER["reference"],
            _WORKER["encoder"],
            team_a,
            team_b,
            _WORKER["seed"] + index * 0x9E3779B1,
            _WORKER["device"],
            _WORKER["sample"],
        )
        outcome = (
            "both_lose"
            if winner == "both_lose"
            else ("win" if winner == "A" else "loss")
        )
        tally[outcome] += 1
    return tally


def fast_play_match_paths(
    candidate_path: Path,
    reference_path: Path,
    games: int,
    seed: int,
    device: torch.device,
    sample: bool = False,
    workers: int = 16,
    team_pairs=None,
) -> dict[str, int]:
    worker_count = min(workers, games)
    pairs = team_pairs or [
        (f"team{index % 24}", f"team{(index + 1 + index // 24) % 24}")
        for index in range(games)
    ]
    indices = list(range(games))
    chunks = [
        [(index, *pairs[index]) for index in indices[offset::worker_count]]
        for offset in range(worker_count)
    ]
    context = multiprocessing.get_context("fork" if str(device) == "cpu" else "spawn")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=_init_worker,
        initargs=(str(candidate_path), str(reference_path), seed, str(device), sample),
    ) as executor:
        results = list(executor.map(_run_worker_chunk, chunks, chunksize=1))
    tally = {"win": 0, "loss": 0, "both_lose": 0}
    for result in results:
        for key in tally:
            tally[key] += result[key]
    return tally


def fast_play_match(
    candidate: AlphaRoco,
    reference: AlphaRoco,
    encoder: ObservationEncoder,
    games: int,
    seed: int,
    device: torch.device,
    sample: bool = False,
    workers: int = 16,
    team_pairs=None,
) -> dict[str, int]:
    worker_count = min(workers, games)
    pairs = team_pairs or [
        (f"team{index % 24}", f"team{(index + 1 + index // 24) % 24}")
        for index in range(games)
    ]
    chunks = [range(index, games, worker_count) for index in range(worker_count)]

    def run_chunk(chunk):
        tally = {"win": 0, "loss": 0, "both_lose": 0}
        for index in chunk:
            team_a, team_b = pairs[index]
            winner = fast_play_game(
                candidate,
                reference,
                encoder,
                team_a,
                team_b,
                seed + index * 0x9E3779B1,
                device,
                sample,
            )
            outcome = (
                "both_lose"
                if winner == "both_lose"
                else ("win" if winner == "A" else "loss")
            )
            tally[outcome] += 1
        return tally

    if worker_count == 1:
        return run_chunk(chunks[0])
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(run_chunk, chunks))
    tally = {"win": 0, "loss": 0, "both_lose": 0}
    for result in results:
        for key in tally:
            tally[key] += result[key]
    return tally
