"""Greedy accuracy of the untrained model on the held-out set, per difficulty.

Writes results/baseline.json (accuracy + every completion) and prints a table.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from selfcurr.config import Config
from selfcurr.curriculum import select_edge
from selfcurr.model import evaluate, load_model, load_test_set, pick_device

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    cfg = Config()
    device = pick_device()
    problems = load_test_set(ROOT / cfg.test_path)
    print(f"model={cfg.model_name} device={device} problems={len(problems)}")

    model, tok = load_model(cfg.model_name, device)
    t0 = time.time()
    acc, records = evaluate(model, tok, problems)
    secs = time.time() - t0

    edge = select_edge(acc, cfg.edge_low, cfg.edge_high)
    print(f"\n{'digits':>6}  {'accuracy':>8}")
    for d, a in acc.items():
        print(f"{d:>6}  {a:>8.1%}{'  <- edge' if d in edge else ''}")
    print(f"\noverall {sum(r['correct'] for r in records) / len(records):.1%}  ({secs:.0f}s)")

    out = ROOT / "results" / "baseline.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(
        {"model": cfg.model_name, "device": device, "seconds": round(secs, 1),
         "accuracy": acc, "edge": edge, "records": records},
        indent=1,
    ))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
