"""Untrained-model accuracy on sums up to --max-digits, beyond the 2-10 experiment.

Continues the frozen set's random stream, so 2-10 digits are the same problems
as data/test_set.jsonl, which is not changed. Writes data/test_set_<N>.jsonl and
results/baseline_<N>.json, with greedy accuracy (as in baseline.py) and how often
any of 4 sampled tries is right (as in training collection).

16 new tokens would cut off answers beyond about 15 digits, so this allows 32.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from selfcurr.config import Config
from selfcurr.curriculum import select_edge
from selfcurr.model import evaluate, generate, load_model, load_test_set, pick_device
from selfcurr.tasks import check, make_problems

ROOT = Path(__file__).resolve().parent.parent
MAX_NEW_TOKENS = 32


def extended_set(cfg: Config, max_digits: int) -> list:
    rng = random.Random(cfg.test_seed)
    seen: set[tuple[int, int]] = set()
    problems = []
    for d in range(cfg.min_digits, max_digits + 1):
        for p in make_problems(d, cfg.test_per_difficulty, rng, exclude=seen):
            seen.add(p.key)
            problems.append(p)
    frozen = load_test_set(ROOT / cfg.test_path)
    if problems[: len(frozen)] != frozen:
        raise RuntimeError("extended set does not start with the frozen test set")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-digits", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0, help="for the sampled tries")
    args = ap.parse_args()

    cfg = Config()
    device = pick_device()
    problems = extended_set(cfg, args.max_digits)
    path = ROOT / "data" / f"test_set_{args.max_digits}.jsonl"
    path.write_text("".join(json.dumps(p.to_dict()) + "\n" for p in problems))
    print(f"model={cfg.model_name} device={device} problems={len(problems)}", flush=True)

    model, tok = load_model(cfg.model_name, device)
    t0 = time.time()
    acc, records = evaluate(model, tok, problems, max_new_tokens=MAX_NEW_TOKENS)
    greedy_secs = time.time() - t0
    print(f"greedy done in {greedy_secs:.0f}s", flush=True)

    torch.manual_seed(args.seed)
    t0 = time.time()
    tries = generate(
        model, tok, problems,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=cfg.sample_temperature,
        num_samples=cfg.samples_per_problem,
    )
    sampled_secs = time.time() - t0
    sampled: dict[int, dict[str, int]] = {}
    for p, r, ts in zip(problems, records, tries):
        hits = sum(check(p, t) for t in ts)
        r["tries"], r["tries_correct"] = ts, hits
        s = sampled.setdefault(p.digits, {"problems": 0, "solved": 0, "tries": 0, "correct_tries": 0})
        s["problems"] += 1
        s["solved"] += hits > 0
        s["tries"] += len(ts)
        s["correct_tries"] += hits

    edge = select_edge(acc, cfg.edge_low, cfg.edge_high)
    print(f"\n{'digits':>6}  {'greedy':>7}  {'4 tries':>7}")
    for d, a in acc.items():
        print(f"{d:>6}  {a:>7.1%}  {sampled[d]['solved'] / sampled[d]['problems']:>7.1%}{'  <- edge' if d in edge else ''}")
    print(f"\noverall {sum(r['correct'] for r in records) / len(records):.1%}  (greedy {greedy_secs:.0f}s, sampled {sampled_secs:.0f}s)")

    out = ROOT / "results" / f"baseline_{args.max_digits}.json"
    out.write_text(json.dumps(
        {"model": cfg.model_name, "device": device, "max_new_tokens": MAX_NEW_TOKENS,
         "seconds": {"greedy": round(greedy_secs, 1), "sampled": round(sampled_secs, 1)},
         "accuracy": acc, "edge": edge, "sampled": sampled, "records": records},
        indent=1,
    ))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
