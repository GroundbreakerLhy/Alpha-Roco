from __future__ import annotations

import argparse
import atexit
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GPU_ORDER = (0, 1)
MIN_FREE_MIB = 16000


def _free_mib(index: int) -> int:
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
            "-i",
            str(index),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return int(out.splitlines()[0])


def _chunk_for(free: int) -> int | None:
    if free >= 65000:
        return 12
    if free >= 40000:
        return 8
    if free >= 22000:
        return 6
    if free >= MIN_FREE_MIB:
        return 4
    return None


def _claim(output: Path) -> None:
    path = output / "watch.pid"
    if path.exists():
        pid = int(path.read_text(encoding="utf-8").strip())
        if Path(f"/proc/{pid}").exists():
            raise SystemExit(f"已有 watcher 在运行：{pid}")
    path.write_text(str(os.getpid()), encoding="utf-8")
    atexit.register(lambda: path.unlink(missing_ok=True))


def _latest(output: Path) -> Path | None:
    paths = sorted(output.glob("checkpoint-batch-*.pt"))
    return paths[-1] if paths else None


def _strip(values: list[str], names: tuple[str, ...]) -> list[str]:
    result = []
    index = 0
    while index < len(values):
        if values[index] in names:
            index += 2
            continue
        result.append(values[index])
        index += 1
    return result


def _monitor(process, gpu, chunk, interval, switched):
    while process.poll() is None:
        time.sleep(interval)
        free_by_gpu = {index: _free_mib(index) for index in GPU_ORDER}
        options = {
            index: _chunk_for(free) for index, free in free_by_gpu.items()
        }
        best = max(
            options,
            key=lambda index: (options[index] or 0, free_by_gpu[index]),
        )
        if best == gpu or options[best] is None:
            continue
        if options[best] <= chunk:
            continue
        if free_by_gpu[best] - free_by_gpu[gpu] < 15000:
            continue
        switched.append((best, options[best]))
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        return



def main() -> None:
    parser = argparse.ArgumentParser(
        description="监督训练：自动选卡、按显存选 chunk、中断后从最新 checkpoint 续跑"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--batches", type=int, default=80)
    parser.add_argument("--interval", type=int, default=1800)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    extra = _strip(
        [value for value in args.command if value != "--"],
        ("--resume", "--batches", "--output", "--update-chunk"),
    )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _claim(output)
    log_path = output / "train.log"
    while True:
        checkpoint = _latest(output)
        done = int(checkpoint.stem.split("-")[-1]) if checkpoint is not None else 0
        if done >= args.batches:
            print(f"finished {done}", flush=True)
            return
        free_by_gpu = {index: _free_mib(index) for index in GPU_ORDER}
        gpu, free = max(free_by_gpu.items(), key=lambda item: item[1])
        chunk = _chunk_for(free)
        if chunk is None:
            time.sleep(args.interval)
            continue
        run_extra = (
            _strip(extra, ("--warm-start", "--champion"))
            if checkpoint is not None
            else extra
        )
        command = [
            sys.executable,
            "-u",
            "-m",
            "alpha_roco.train",
            "--batches",
            str(args.batches - done),
            "--output",
            str(output),
            "--update-chunk",
            str(chunk),
            *run_extra,
        ]
        if checkpoint is not None:
            command += ["--resume", str(checkpoint)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{time.strftime('%H:%M:%S')} gpu={gpu} chunk={chunk} "
                f"batches={args.batches - done} "
                f"resume={checkpoint.name if checkpoint is not None else '-'}\n"
            )
            handle.flush()
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            switched = []
            threading.Thread(
                target=_monitor,
                args=(process, gpu, chunk, args.interval, switched),
                daemon=True,
            ).start()
            for line in process.stdout:
                stamped = f"{time.strftime('%H:%M:%S')} {line}"
                handle.write(stamped)
                handle.flush()
                sys.stdout.write(stamped)
            process.wait()
        checkpoint = _latest(output)
        done = int(checkpoint.stem.split("-")[-1]) if checkpoint is not None else 0
        if done >= args.batches:
            print(f"finished {done}", flush=True)
            return
        if switched:
            note = f"{time.strftime('%H:%M:%S')} switch to gpu={switched[0][0]} chunk={switched[0][1]}\n"
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(note)
            sys.stdout.write(note)
            continue
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
