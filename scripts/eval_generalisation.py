"""Score the saved round-10 adapters on sums longer than they trained on. No training.

    uv run --with scipy python scripts/eval_generalisation.py
    uv run python scripts/eval_generalisation.py --collect-only --only a:0

Each checkpoint is scored greedily on 11-20 digits from data/test_set_20.jsonl and
re-scored on 2-10 to check it against its logged round-10 result. The untrained
model is scored in float16 (to match results/baseline_20.json) and float32 (the
precision the adapters ran in). Replies go to runs/generalisation/, so an
interrupted run continues where it stopped; the summary goes to
results/generalisation_11_20.json.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import time
from pathlib import Path

import torch
from peft import PeftModel

from selfcurr.config import Config
from selfcurr.model import evaluate, load_model, load_test_set, pick_device
from selfcurr.tasks import parse_answer

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "runs" / "generalisation"
ARMS = ["a", "b", "bkey", "c"]
SEEDS = range(10)
LONG = range(11, 21)
MAIN = (11, 12, 13)
PAIRS = [("a", "c"), ("a", "b"), ("c", "b")]
MAX_NEW_TOKENS = 32


def common_prefix(x: str, y: str) -> int:
    n = 0
    for p, q in zip(x, y):
        if p != q:
            break
        n += 1
    return n


def score(model, tok, problems) -> tuple[dict[int, int], list[dict]]:
    _, records = evaluate(model, tok, problems, max_new_tokens=MAX_NEW_TOKENS)
    counts: dict[int, int] = {}
    for r in records:
        counts[r["digits"]] = counts.get(r["digits"], 0) + r["correct"]
    return counts, records


def collect_one(name: str, device: str, short, long, adapter: Path | None = None, dtype=torch.float32, together=False) -> None:
    out = RAW / f"{name}.json"
    if out.exists():
        return
    t0 = time.time()
    model, tok = load_model(Config().model_name, device, dtype=dtype)
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    if together:  # same batches as baseline_long.py
        counts, records = score(model, tok, short + long)
        short_counts = {d: c for d, c in counts.items() if d not in LONG}
        long_counts = {d: c for d, c in counts.items() if d in LONG}
        long_records = records[len(short):]
    else:  # same batches as the logged round-10 evaluation
        short_counts, _ = score(model, tok, short)
        long_counts, long_records = score(model, tok, long)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"short": short_counts, "long": long_counts, "records": long_records}))
    tmp.rename(out)
    print(f"{name}: 11-13 {sum(long_counts[d] for d in MAIN) / 6:.1f}%  ({time.time() - t0:.0f}s)", flush=True)
    del model
    if device == "mps":
        torch.mps.empty_cache()


def error_stats(records: list[dict]) -> dict:
    wrong = [r for r in records if r["digits"] in MAIN and not r["correct"]]
    right_len, prefix, prefix_right_len, no_number = 0, [], [], 0
    for r in wrong:
        ans = parse_answer(r["completion"])
        if ans is None:
            no_number += 1
            continue
        text, truth = str(ans), str(r["answer"])
        n = common_prefix(text, truth)
        prefix.append(n)
        if len(text) == len(truth):
            right_len += 1
            prefix_right_len.append(n)
    return {
        "wrong": len(wrong),
        "no_number": no_number,
        "correct_length_share": right_len / len(wrong) if wrong else None,
        "leading_digits_correct": st.mean(prefix) if prefix else None,
        "leading_digits_correct_when_right_length": st.mean(prefix_right_len) if prefix_right_len else None,
    }


def paired(d: list[float]) -> dict:
    from scipy import stats as ss

    m, se = st.mean(d), st.stdev(d) / math.sqrt(len(d))
    h = ss.t.ppf(0.975, len(d) - 1) * se
    return {
        "mean_gap_points": m, "ci95_low": m - h, "ci95_high": m + h,
        "p": float(ss.ttest_1samp(d, 0).pvalue), "seeds_first_arm_ahead": sum(x > 0 for x in d),
        "seeds_tied": sum(x == 0 for x in d), "gaps_by_seed": d,
    }


def logged_round10(arm: str, seed: int) -> dict:
    rows = [json.loads(l) for l in (ROOT / f"runs/arm_{arm}/seed{seed}/log.jsonl").read_text().splitlines() if l.strip()]
    return next(r for r in rows if r["round"] == 10)["eval"]


def summarise(per: int) -> dict:
    raw = {(a, s): json.loads((RAW / f"{a}_seed{s}.json").read_text()) for a in ARMS for s in SEEDS}
    pct = lambda c: c / per * 100
    main_pct = lambda x: sum(x["long"][str(d)] for d in MAIN) / (per * len(MAIN)) * 100

    arms = {}
    for a in ARMS:
        by_length = {}
        for d in LONG:
            vals = [pct(raw[a, s]["long"][str(d)]) for s in SEEDS]
            by_length[d] = {"mean": st.mean(vals), "min": min(vals), "max": max(vals), "by_seed": vals}
        vals = [main_pct(raw[a, s]) for s in SEEDS]
        errs = [error_stats(raw[a, s]["records"]) for s in SEEDS]
        arms[a] = {
            "by_length": by_length,
            "digits_11_13": {"mean": st.mean(vals), "min": min(vals), "max": max(vals), "by_seed": vals},
            "wrong_answers_11_13": {
                k: st.mean(e[k] for e in errs)
                for k in ("correct_length_share", "leading_digits_correct", "leading_digits_correct_when_right_length")
            } | {"no_number": sum(e["no_number"] for e in errs), "wrong": sum(e["wrong"] for e in errs)},
        }

    # sanity: the re-scored 2-10 against each logged round-10 result
    mismatches = []
    for (a, s), x in raw.items():
        logged = logged_round10(a, s)
        for d, c in x["short"].items():
            if abs(c / per - logged[d]) > 1e-9:
                mismatches.append({"arm": a, "seed": s, "digits": int(d), "logged": logged[d], "rescored": c / per})

    # sanity: the untrained model against results/baseline_20.json
    base16 = json.loads((RAW / "base_float16.json").read_text())
    base32 = json.loads((RAW / "base_float32.json").read_text())
    earlier = json.loads((ROOT / "results/baseline_20.json").read_text())
    old = [r for r in earlier["records"] if r["digits"] in LONG]
    differ = sum(p["completion"] != q["completion"] for p, q in zip(old, base16["records"]))
    length_diff = {str(d): base16["long"][str(d)] / per - earlier["accuracy"][str(d)] for d in LONG}

    return {
        "model": Config().model_name,
        "checkpoint": "round 10, saved LoRA adapter",
        "decoding": {"greedy": True, "max_new_tokens": MAX_NEW_TOKENS, "precision": "float32, untrained also float16"},
        "problems": f"data/test_set_20.jsonl, {per} per length",
        "seeds": list(SEEDS),
        "units": "percent; gaps in points",
        "arms": arms,
        "untrained": {
            "float32": {"by_length": {d: pct(base32["long"][str(d)]) for d in LONG}, "digits_11_13": main_pct(base32), "wrong_answers_11_13": error_stats(base32["records"])},
            "float16": {"by_length": {d: pct(base16["long"][str(d)]) for d in LONG}, "digits_11_13": main_pct(base16), "wrong_answers_11_13": error_stats(base16["records"])},
        },
        "paired_digits_11_13": {
            f"{x}-{y}": paired([main_pct(raw[x, s]) - main_pct(raw[y, s]) for s in SEEDS]) for x, y in PAIRS
        },
        "sanity": {
            "rescored_2_10_vs_logged": {"checked": len(raw) * 9, "mismatches": mismatches},
            "untrained_float16_vs_baseline_20": {"accuracy_differences": length_diff, "completions_differing": differ, "completions_checked": len(old)},
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="one checkpoint, e.g. a:0")
    ap.add_argument("--collect-only", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    device = pick_device()
    problems = load_test_set(ROOT / "data/test_set_20.jsonl")
    short = [p for p in problems if p.digits not in LONG]
    long = [p for p in problems if p.digits in LONG]
    RAW.mkdir(parents=True, exist_ok=True)

    if args.only:
        a, s = args.only.split(":")
        todo = [(a, int(s))]
    else:
        collect_one("base_float16", device, short, long, dtype=torch.float16, together=True)
        collect_one("base_float32", device, short, long)
        todo = [(a, s) for s in SEEDS for a in ARMS]
    for a, s in todo:
        collect_one(f"{a}_seed{s}", device, short, long, adapter=ROOT / f"runs/arm_{a}/seed{s}/round10")

    if args.collect_only or args.only:
        return
    out = ROOT / "results/generalisation_11_20.json"
    out.write_text(json.dumps(summarise(cfg.test_per_difficulty), indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
