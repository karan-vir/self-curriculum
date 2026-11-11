"""Exploratory, after the fact: held-out accuracy against cumulative generated tokens.

    uv run --with scipy python scripts/compute_vs_tokens.py

Reads runs/arm_<x>/seed<N>/log.jsonl for Arms A, B, B+key and C, seeds 0-9. No
model is run. For each run, the curve is (cumulative generated tokens after
round r, held-out accuracy after round r) for r = 0..10, starting at (0, round 0).
`generated_tokens` counts every token the model generated for collection in that
round, Arm A's probe included, end-of-turn included, padding excluded; held-out
evaluation is not counted.

Between round ends the curve is joined linearly (the primary reading, as
requested). Because a model only changes when a round's training finishes, a
step reading (accuracy of the last finished round) is also given as a
sensitivity check.

Comparisons are paired across seeds, two-sided t-tests, uncorrected:
- Arm A's final accuracy against Arm C's accuracy at the same seed's Arm A
  final token total;
- every pair of arms at 200k and 300k cumulative tokens (an arm whose run used
  fewer tokens than that on some seed is left out of that comparison).

Writes results/compute_vs_tokens.json.
"""

from __future__ import annotations

import json
import math
import statistics as st
from bisect import bisect_right
from pathlib import Path

from scipy import stats as ss

ROOT = Path(__file__).resolve().parent.parent
ARMS = ["a", "b", "bkey", "c"]
SEEDS = range(10)
MARKS = [200_000, 300_000]
PAIRS = [("a", "c"), ("a", "b"), ("c", "b"), ("a", "bkey"), ("c", "bkey"), ("b", "bkey")]


def curve(arm: str, seed: int) -> tuple[list[int], list[float]]:
    rows = [json.loads(l) for l in (ROOT / f"runs/arm_{arm}/seed{seed}/log.jsonl").read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: r["round"])
    xs, ys, total = [], [], 0
    for r in rows:
        total += r.get("generated_tokens") or 0
        xs.append(total)
        ys.append(r["eval_overall"])
    assert [r["round"] for r in rows] == list(range(11)), (arm, seed)
    return xs, ys


def linear(xs, ys, x) -> float | None:
    if x > xs[-1]:
        return None
    i = bisect_right(xs, x) - 1
    if i == len(xs) - 1:
        return ys[i]
    return ys[i] + (ys[i + 1] - ys[i]) * (x - xs[i]) / (xs[i + 1] - xs[i])


def step(xs, ys, x) -> float | None:
    if x > xs[-1]:
        return None
    return ys[bisect_right(xs, x) - 1]


def paired(d: list[float]) -> dict:
    m, se = st.mean(d), st.stdev(d) / math.sqrt(len(d))
    h = ss.t.ppf(0.975, len(d) - 1) * se
    return {
        "mean_gap_points": m, "ci95_low": m - h, "ci95_high": m + h, "p": float(ss.ttest_1samp(d, 0).pvalue),
        "seeds_first_arm_ahead": sum(x > 0 for x in d), "n": len(d), "gaps_by_seed": d,
    }


def main() -> None:
    C = {(a, s): curve(a, s) for a in ARMS for s in SEEDS}

    per_round = {}
    for a in ARMS:
        rounds = []
        for r in range(11):
            tok = [C[a, s][0][r] for s in SEEDS]
            acc = [C[a, s][1][r] * 100 for s in SEEDS]
            rounds.append({"round": r, "tokens_mean": st.mean(tok), "tokens_min": min(tok), "tokens_max": max(tok),
                           "acc_mean": st.mean(acc), "acc_min": min(acc), "acc_max": max(acc)})
        cost = [[C[a, s][0][r] - C[a, s][0][r - 1] for s in SEEDS] for r in range(1, 11)]
        per_round[a] = {
            "rounds": rounds,
            "round1_tokens_mean": st.mean(cost[0]),
            "rounds2_10_tokens_per_round_mean": st.mean(x for c in cost[1:] for x in c),
            "rounds2_10_tokens_per_round_range": [min(x for c in cost[1:] for x in c), max(x for c in cost[1:] for x in c)],
            "total_tokens_mean": st.mean(C[a, s][0][-1] for s in SEEDS),
            "round1_share_of_total": st.mean(cost[0]) / st.mean(C[a, s][0][-1] for s in SEEDS),
        }

    at_a_total = {}
    for name, f in (("linear", linear), ("step", step)):
        d, c_at = [], []
        for s in SEEDS:
            xa, ya = C["a", s]
            c = f(*C["c", s], xa[-1])
            c_at.append(c * 100)
            d.append((ya[-1] - c) * 100)
        at_a_total[name] = paired(d) | {"arm_c_accuracy_at_a_total_by_seed": c_at,
                                        "arm_a_total_tokens_by_seed": [C["a", s][0][-1] for s in SEEDS]}

    marks = {}
    for x in MARKS:
        out = {"accuracy": {}, "paired": {}}
        for name, f in (("linear", linear), ("step", step)):
            acc = {a: [f(*C[a, s], x) for s in SEEDS] for a in ARMS}
            out["accuracy"][name] = {
                a: ({"mean": st.mean(v) * 100, "min": min(v) * 100, "max": max(v) * 100} if None not in v
                    else {"not_reached_on_seeds": [s for s, y in zip(SEEDS, v) if y is None]})
                for a, v in acc.items()
            }
            out["paired"][name] = {
                f"{p}-{q}": paired([(acc[p][s] - acc[q][s]) * 100 for s in SEEDS])
                for p, q in PAIRS if None not in acc[p] and None not in acc[q]
            }
        marks[str(x)] = out

    res = {
        "status": "exploratory, defined after the results were seen; not part of the original plan",
        "tokens": "cumulative generated tokens for training-data collection (Arm A's probe included); evaluation excluded",
        "interpolation": "linear between round ends (primary); step = last finished round (sensitivity)",
        "tests": "paired across seeds 0-9, two-sided t-test, not corrected for multiple comparisons",
        "per_round": per_round,
        "a_final_vs_c_at_a_total": at_a_total,
        "at_tokens": marks,
    }
    out = ROOT / "results/compute_vs_tokens.json"
    out.write_text(json.dumps(res, indent=1))

    for name, r in at_a_total.items():
        print(f"A final - C at A's total ({name}): {r['mean_gap_points']:+.2f} [{r['ci95_low']:+.2f}, {r['ci95_high']:+.2f}] p={r['p']:.3f} A ahead {r['seeds_first_arm_ahead']}/10")
    for x, m in marks.items():
        for name in ("linear", "step"):
            print(x, name, {a: round(v["mean"], 1) if "mean" in v else v for a, v in m["accuracy"][name].items()})
            for k, r in m["paired"][name].items():
                print(f"   {k}: {r['mean_gap_points']:+.2f} [{r['ci95_low']:+.2f}, {r['ci95_high']:+.2f}] p={r['p']:.4f} ahead {r['seeds_first_arm_ahead']}/10")
    for a in ARMS:
        p = per_round[a]
        print(a, "r1", round(p["round1_tokens_mean"]), "r2-10/round", round(p["rounds2_10_tokens_per_round_mean"]), p["rounds2_10_tokens_per_round_range"], "total", round(p["total_tokens_mean"]), "r1 share", round(p["round1_share_of_total"], 3))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
