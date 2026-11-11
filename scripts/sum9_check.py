"""Exploratory, after the fact: are columns that sum to 9 a hard case for the untrained model?

    uv run --with scipy --with statsmodels python scripts/sum9_check.py

Reads results/baseline.json (the untrained model's 1,800 greedy replies). No
model is run.

Definitions. Columns are numbered from the units, i = 0. Column i holds the
digits a_i and b_i of the two operands. The carry into column 0 is 0, and the
carry into column i+1 is 1 when a_i + b_i + carry_i >= 10.

- A "sum-9 column, no carry in" is a column with a_i + b_i = 9 and carry_i = 0.
  Its true digit is 9 and it passes no carry on.
- A "sum-9 column, carry in" (the cascading case) has a_i + b_i = 9 and
  carry_i = 1. Its digit is 0 and it passes the carry on.

1. For 2-6-digit problems, accuracy on problems with at least one sum-9 column
   with no carry in, against problems with none; per length, and pooled with a
   Mantel-Haenszel test stratified by length (lengths differ in how often the
   case occurs and in base accuracy).
2. Among wrong answers that are too high by exactly 10^k (k >= 0), the digit
   sum a_{k-1} + b_{k-1} of the column immediately to the right of column k,
   and whether that column receives a carry. k = 0 has no such column.

Writes results/sum9_check.json.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats as ss
from statsmodels.stats.contingency_tables import StratifiedTable

from selfcurr.tasks import parse_answer

ROOT = Path(__file__).resolve().parent.parent
LENGTHS = range(2, 7)


def columns(a: int, b: int, n: int) -> list[tuple[int, int]]:
    """(digit sum, carry in) per column from the units up, n columns."""
    out, carry = [], 0
    for i in range(n):
        s = a // 10**i % 10 + b // 10**i % 10
        out.append((s, carry))
        carry = int(s + carry >= 10)
    return out


def main() -> None:
    recs = json.loads((ROOT / "results/baseline.json").read_text())["records"]

    by_len, tables = {}, []
    for d in LENGTHS:
        rs = [r for r in recs if r["digits"] == d]
        groups = {"with": [], "without": []}
        casc = {"with": [], "without": []}
        for r in rs:
            cols = columns(r["a"], r["b"], d)
            groups["with" if any(s == 9 and c == 0 for s, c in cols) else "without"].append(r["correct"])
            casc["with" if any(s == 9 and c == 1 for s, c in cols) else "without"].append(r["correct"])
        w, wo = groups["with"], groups["without"]
        by_len[d] = {
            "with_sum9_no_carry": {"n": len(w), "correct": sum(w), "accuracy": sum(w) / len(w) if w else None},
            "without": {"n": len(wo), "correct": sum(wo), "accuracy": sum(wo) / len(wo) if wo else None},
            "fisher_p": float(ss.fisher_exact([[sum(w), len(w) - sum(w)], [sum(wo), len(wo) - sum(wo)]]).pvalue),
            "with_sum9_carry_in_cascade": {"n": len(casc["with"]), "accuracy": sum(casc["with"]) / len(casc["with"]) if casc["with"] else None},
            "without_cascade": {"n": len(casc["without"]), "accuracy": sum(casc["without"]) / len(casc["without"]) if casc["without"] else None},
        }
        if w and wo:
            tables.append(np.array([[sum(w), len(w) - sum(w)], [sum(wo), len(wo) - sum(wo)]]))

    st = StratifiedTable(tables)
    pooled_w = [x for d in LENGTHS for x in [by_len[d]["with_sum9_no_carry"]]]
    pooled = {
        "with_sum9_no_carry": {"n": sum(x["n"] for x in pooled_w), "correct": sum(x["correct"] for x in pooled_w)},
        "without": {"n": sum(by_len[d]["without"]["n"] for d in LENGTHS), "correct": sum(by_len[d]["without"]["correct"] for d in LENGTHS)},
        "mantel_haenszel_odds_ratio_correct": float(st.oddsratio_pooled),
        "mantel_haenszel_p": float(st.test_null_odds().pvalue),
        "length_strata_used": [d for d in LENGTHS if by_len[d]["with_sum9_no_carry"]["n"] and by_len[d]["without"]["n"]],
    }
    for k in ("with_sum9_no_carry", "without"):
        pooled[k]["accuracy"] = pooled[k]["correct"] / pooled[k]["n"]

    def pow10(lengths) -> dict:
        sums, carry_in, units, rows = Counter(), Counter(), 0, []
        for r in recs:
            if r["digits"] not in lengths or r["correct"]:
                continue
            got = parse_answer(r["completion"])
            if got is None:
                continue
            diff = got - r["answer"]
            if diff <= 0 or 10 ** round(np.log10(diff)) != diff:
                continue
            k = round(np.log10(diff))
            if k == 0:
                units += 1
                continue
            cols = columns(r["a"], r["b"], max(k, r["digits"] + 1))
            s, c = cols[k - 1]
            sums[s] += 1
            carry_in[(s, c)] += 1
            rows.append({"a": r["a"], "b": r["b"], "k": k, "right_digit_sum": s, "right_carry_in": c})
        n = sum(sums.values())
        return {
            "answers_too_high_by_power_of_10": n + units,
            "error_in_units_column_no_right_neighbour": units,
            "right_neighbour_digit_sum_counts": {str(s): sums[s] for s in sorted(sums)},
            "right_neighbour_sum9_no_carry_in": carry_in[(9, 0)],
            "right_neighbour_sum9_carry_in": carry_in[(9, 1)],
            "share_right_neighbour_sum9": sums[9] / n if n else None,
            "examples": rows[:12],
        }

    # how often a random non-leading column sums to 9: 10 of 100 digit pairs
    res = {
        "status": "exploratory, defined after the results were seen; not part of the original plan",
        "source": "results/baseline.json, untrained model, greedy, float16 on MPS",
        "definitions": __doc__.split("Definitions.")[1].split("1. For")[0].strip(),
        "accuracy_by_length": by_len,
        "pooled_2_6": pooled,
        "too_high_by_power_of_10": {"2-6 digits": pow10(set(LENGTHS)), "2-10 digits": pow10(set(range(2, 11)))},
        "reference": "a column of two uniform random digits sums to 9 with probability 0.10 (leading column of two non-zero digits: 8/81 = 0.099)",
    }
    out = ROOT / "results/sum9_check.json"
    out.write_text(json.dumps(res, indent=1))
    for d in LENGTHS:
        x = by_len[d]
        print(d, x["with_sum9_no_carry"], x["without"], f"p={x['fisher_p']:.3g}", "cascade", x["with_sum9_carry_in_cascade"], x["without_cascade"])
    print("pooled", {k: v for k, v in pooled.items()})
    for k, v in res["too_high_by_power_of_10"].items():
        print(k, {kk: vv for kk, vv in v.items() if kk != "examples"})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
