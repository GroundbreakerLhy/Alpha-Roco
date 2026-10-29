from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .arena import choose, load_model, record_opponent_actions
from .config import EvaluationConfig, ModelConfig
from .features import ObservationEncoder
from .network import AlphaRoco, HistoryState
from .terminal import BattleSession


def main():
    defaults = EvaluationConfig()
    parser = argparse.ArgumentParser(description="使用现有对战服务运行 Alpha Roco")
    parser.add_argument("--team-a", default="team4")
    parser.add_argument("--team-b", default="team0")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--timeout", type=float, default=defaults.timeout)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--log-jsonl", type=Path)
    args = parser.parse_args()
    if args.checkpoint and args.random_init:
        parser.error("--checkpoint与--random-init不能同时使用")
    if not args.checkpoint and not args.random_init:
        parser.error("必须提供--checkpoint或显式指定--random-init")

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    encoder = ObservationEncoder()
    if args.checkpoint:
        model = load_model(args.checkpoint, encoder, device)
    else:
        model = AlphaRoco(ModelConfig(), encoder.vocab_sizes).to(device)
        model.eval()

    histories = {"A": HistoryState(), "B": HistoryState()}
    actions = 0
    with BattleSession(
        team_a=args.team_a,
        team_b=args.team_b,
        seed=args.seed,
        timeout=args.timeout,
        log_dir=args.log_dir,
        log_jsonl=args.log_jsonl,
    ) as session:
        while session.winner is None:
            chosen = {}
            request_by_side = {}
            for request in session.requests():
                side = request["side"]
                action, histories[side] = choose(
                    model, encoder, request, histories[side], device, args.sample
                )
                chosen[side] = action
                request_by_side[side] = request
                session.submit(side, action)
                actions += 1
            record_opponent_actions(
                histories,
                {"A": model, "B": model},
                encoder,
                request_by_side,
                chosen,
            )
        print(json.dumps({"winner": session.winner, "decisions": actions}))


if __name__ == "__main__":
    main()
