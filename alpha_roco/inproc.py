from __future__ import annotations

import json
from pathlib import Path

from server import DecisionBuilder, make_team
from sim.battle import (
    apply_drive,
    create_team_battle,
    pet_to_dict,
    resume_after_leave,
    step,
    switch_in,
    trait_leave_switch,
)
from sim.data_loader import load_spirits
from sim.models import Action

from .rng import THREAD_RANDOM

ROOT = Path(__file__).resolve().parents[1]
TEAM_DIR = ROOT / "data" / "teams"
SPIRITS = load_spirits()


def _team_config(name: str) -> dict:
    path = Path(name)
    if not path.is_absolute():
        candidate = ROOT / path
        path = (
            candidate
            if candidate.is_file()
            else TEAM_DIR / (name if name.endswith(".json") else f"{name}.json")
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _wire_action(action: dict, phase: str) -> dict:
    if phase == "lead":
        return {"lead": int(action.get("lead", action.get("pet_index", 0)))}
    if phase == "replacement":
        return {"kind": "switch", "pet_index": action["pet_index"]}
    if action["kind"] == "skill":
        return {
            "kind": "skill",
            "skill_index": action["skill_index"],
            "magic_id": action.get("magic_id"),
            "magic_branch": action.get("magic_branch", 0),
            "choice_branch": action.get("choice_branch", 0),
        }
    if action["kind"] == "switch":
        return {
            "kind": "switch",
            "pet_index": action["pet_index"],
            "magic_id": action.get("magic_id"),
            "magic_branch": action.get("magic_branch", 0),
        }
    return {
        "kind": action["kind"],
        "magic_id": action.get("magic_id"),
        "magic_branch": action.get("magic_branch", 0),
    }


def _action(raw: dict) -> Action:
    return Action(
        kind=raw["kind"],
        skill_index=raw.get("skill_index"),
        pet_index=raw.get("pet_index"),
        magic_id=raw.get("magic_id"),
        magic_branch=raw.get("magic_branch", 0),
        choice_branch=raw.get("choice_branch", 0),
    )


class InProcessSession:
    def __init__(
        self,
        team_a="team4",
        team_b="team0",
        seed=0,
        timeout=None,
        *,
        game_index=None,
        log_dir=None,
        log_jsonl=None,
        host=None,
    ):
        self.team_a = team_a
        self.team_b = team_b
        self.seed = seed
        self.timeout = timeout
        self.game_index = game_index
        self.requested_log_dir = Path(log_dir) if log_dir is not None else None
        self.log_jsonl = Path(log_jsonl) if log_jsonl is not None else None
        self.host = host
        self.winner = None
        self._builder = DecisionBuilder()
        self._driver = None
        self._requests = []
        self._actions = {}
        self._log_file = None
        self._closed = False

    def __enter__(self):
        if self.requested_log_dir is not None:
            self.requested_log_dir.mkdir(parents=True, exist_ok=True)
        if self.log_jsonl is not None:
            self.log_jsonl.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = self.log_jsonl.open("w", encoding="utf-8")
        self._driver = self._run()
        self._requests = next(self._driver)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._driver is not None:
            self._driver.close()
        if self._log_file is not None:
            self._log_file.close()

    def requests(self) -> list[dict]:
        return [] if self.winner is not None else list(self._requests)

    def submit(self, side: str, action: dict) -> None:
        outstanding = {request["side"] for request in self._requests}
        if side not in outstanding or side in self._actions:
            raise RuntimeError(f"side {side} has no outstanding decision")
        self._actions[side] = action
        self._write_log(
            {
                "type": "submit",
                "side": side,
                "phase": self._requests[0]["phase"],
                "action": action,
            }
        )
        if len(self._actions) < len(outstanding):
            return
        actions = self._actions
        self._actions = {}
        self._requests = self._driver.send(actions)

    def _write_log(self, value):
        if self._log_file is not None:
            self._log_file.write(json.dumps(value, ensure_ascii=False) + "\n")
            self._log_file.flush()

    def _request(self, state, side, phase, team, **kwargs):
        request = json.loads(
            json.dumps(
                self._builder.request(state, side, phase, team, **kwargs),
                ensure_ascii=False,
            )
        )
        self._write_log({"type": "request", "request": request})
        return request

    def _run(self):
        THREAD_RANDOM.seed(self.seed)
        config_a = _team_config(self.team_a)
        config_b = _team_config(self.team_b)
        pets_a = make_team("A", config_a["team"], SPIRITS)
        pets_b = make_team("B", config_b["team"], SPIRITS)
        self._write_log({"type": "session", "seed": self.seed})
        raw_leads = yield [
            self._request(
                None,
                "A",
                "lead",
                config_a["team"],
                lead=True,
                opponent_team=[pet.name for pet in pets_b],
                own_pets=[pet_to_dict(pet) for pet in pets_a],
            ),
            self._request(
                None,
                "B",
                "lead",
                config_b["team"],
                lead=True,
                opponent_team=[pet.name for pet in pets_a],
                own_pets=[pet_to_dict(pet) for pet in pets_b],
            ),
        ]
        leads = {side: _wire_action(raw_leads[side], "lead") for side in ("A", "B")}
        state = create_team_battle(
            pets_a,
            pets_b,
            active_a=leads["A"].get("lead", 0),
            active_b=leads["B"].get("lead", 0),
        )
        state.resonance_magic["A"] = config_a.get("resonance")
        state.resonance_magic["B"] = config_b.get("resonance")
        teams = {"A": config_a["team"], "B": config_b["team"]}
        while state.winner is None:
            for side in ("A", "B"):
                if state.active[side] >= 0:
                    continue
                submitted = yield [self._request(state, side, "replacement", teams[side])]
                raw = _wire_action(submitted[side], "replacement")
                self._builder.record(side, raw)
                state.active[side] = raw["pet_index"]
                pet = state.teams[side][state.active[side]]
                state.log.append(f"{side} 换上 {pet.name}")
                switch_in(
                    state,
                    side,
                    pet,
                    state.log,
                    thorn=False,
                    active_switch=False,
                )
                apply_drive(state, side)
            submitted = yield [
                self._request(state, "A", "action", teams["A"]),
                self._request(state, "B", "action", teams["B"]),
            ]
            raw = {side: _wire_action(submitted[side], "action") for side in ("A", "B")}
            for side in ("A", "B"):
                self._builder.record(side, raw[side])
            state = step(state, _action(raw["A"]), _action(raw["B"]))
            while state.paused_turn is not None and state.winner is None:
                side = state.paused_turn["leave_side"]
                submitted = yield [self._request(state, side, "replacement", teams[side])]
                raw = _wire_action(submitted[side], "replacement")
                self._builder.record(side, raw)
                state = resume_after_leave(
                    state, side, state.teams[side][raw["pet_index"]]
                )
            switched = False
            for side in ("A", "B"):
                if not state.pending_switch[side]:
                    continue
                state.pending_switch[side] = False
                submitted = yield [self._request(state, side, "replacement", teams[side])]
                raw = _wire_action(submitted[side], "replacement")
                self._builder.record(side, raw)
                incoming = state.teams[side][raw["pet_index"]]
                trait_leave_switch(state, side, incoming, state.log)
                apply_drive(state, side)
                switched = True
            if switched:
                continue
        self.winner = state.winner
        self._write_log({"type": "game_over", "winner": self.winner})
        yield []
