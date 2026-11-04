"""Terminal view of running arms: rounds done, held-out accuracy, edge, GPU memory.

    uv run python scripts/watch.py            # refresh every 10 s, Ctrl-C to stop
    uv run python scripts/watch.py --seed 1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARMS = [("a", "A  self-curriculum"), ("b", "B  fixed"), ("bkey", "B+key reference"), ("c", "C  hard-only")]


def swap() -> str:
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
        used = out.split("used = ")[1].split()[0]
        total = out.split("total = ")[1].split()[0]
        return f"{used} of {total}"
    except Exception:
        return "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=10)
    args = ap.parse_args()
    while True:
        lines = [f"seed {args.seed} · {time.strftime('%H:%M:%S')} · swap used {swap()}", ""]
        lines.append(f"{'arm':<18}{'rounds':>8}{'held-out':>10}{'change':>9}{'last round':>12}{'gpu GB':>8}  edge")
        for arm, name in ARMS:
            path = ROOT / "runs" / f"arm_{arm}" / f"seed{args.seed}" / "log.jsonl"
            rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
            if not rows:
                lines.append(f"{name:<18}{'waiting':>8}")
                continue
            last, first = rows[-1], rows[0]
            secs = sum(last["seconds"].values()) if isinstance(last["seconds"], dict) else last["seconds"]
            change = (last["eval_overall"] - first["eval_overall"]) * 100
            lines.append(
                f"{name:<18}{len(rows) - 1:>5}/10{last['eval_overall']:>10.1%}{change:>+8.1f}"
                f"{secs / 60:>10.1f} m{last.get('gpu_gb') or '':>8}  {last.get('edge') or ''}"
            )
        print("\033[2J\033[H" + "\n".join(lines), flush=True)
        time.sleep(args.every)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
