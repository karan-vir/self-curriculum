"""Run several seeds and arms one after another, restarting any run that hangs.

    uv run python scripts/run_queue.py --seeds 3-9 --arms a,b,bkey

Each run is `run_arm.py --arm X --seed N`. A watchdog checks the run's
log.jsonl: if no new round is written for --stall minutes, the run is
killed and restarted from its last saved round (--resume), up to
--retries times. Runs that already have all rounds are skipped, so the
queue can be restarted safely after a crash or reboot.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        seeds += range(int(lo), int(hi or lo) + 1)
    return seeds


def rounds_done(log: Path) -> int:
    """Highest round written to the log, or -1 if there is none."""
    if not log.exists():
        return -1
    rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    return max((r["round"] for r in rows), default=-1)


def say(msg: str) -> None:
    print(f"=== {time.strftime('%H:%M')} {msg}", flush=True)


def run_one(arm: str, seed: int, rounds: int, stall: float, retries: int, poll: float) -> bool:
    log = ROOT / "runs" / f"arm_{arm}" / f"seed{seed}" / "log.jsonl"
    for attempt in range(retries + 1):
        done = rounds_done(log)
        if done >= rounds:
            say(f"seed {seed} arm {arm} already complete")
            return True
        # resume only from a trained round; a log with just round 0 starts over
        cmd = [sys.executable, "scripts/run_arm.py", "--arm", arm, "--seed", str(seed), "--rounds", str(rounds)]
        if done >= 1:
            cmd.append("--resume")
        how = f"resuming after round {done}" if done >= 1 else "starting over"
        say(f"seed {seed} arm {arm} start" + (f" (attempt {attempt + 1}, {how})" if attempt else ""))
        proc = subprocess.Popen(cmd, cwd=ROOT, start_new_session=True)
        last, last_change = done, time.time()
        while proc.poll() is None:
            time.sleep(poll)
            now_done = rounds_done(log)
            if now_done != last:
                last, last_change = now_done, time.time()
            elif time.time() - last_change > stall:
                say(f"seed {seed} arm {arm} STALLED: no new round for {stall / 60:.0f} min after round {last}, restarting")
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                break
        if rounds_done(log) >= rounds:
            say(f"seed {seed} arm {arm} done")
            return True
        if proc.returncode not in (None, 0, -signal.SIGTERM, -signal.SIGKILL):
            say(f"seed {seed} arm {arm} exited with code {proc.returncode}")
    say(f"seed {seed} arm {arm} FAILED after {retries + 1} attempts")
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="3-9", help="e.g. 3-9 or 3,5,7")
    ap.add_argument("--arms", default="a,b,bkey")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--stall", type=float, default=15, help="minutes without a new round before restarting")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--poll", type=float, default=30, help="seconds between log checks")
    args = ap.parse_args()

    failed = []
    for seed in parse_seeds(args.seeds):
        for arm in args.arms.split(","):
            if not run_one(arm, seed, args.rounds, args.stall * 60, args.retries, args.poll):
                failed.append(f"seed {seed} arm {arm}")
    say("all done" + (f", failed: {', '.join(failed)}" if failed else ""))


if __name__ == "__main__":
    main()
