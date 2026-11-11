# Archived: token-budget sweep (planned, dropped after a pilot)

This folder keeps a follow-up experiment that was planned and then dropped.
Nothing here feeds the results in `reports/baseline.html` or the main README.

## What was planned

The main runs match every arm on **training examples** (512 per round), which
lets the arms spend very different amounts of generation to collect them. The
sweep was meant to match them on **generated tokens** instead: each round, an
arm samples until a fixed token budget is spent (Arm A's probe included), then
trains on whatever it collected.

- Arms A (self-curriculum), B (uniform 2–10 digits) and C (hard-only 8–10).
- Five budgets per round, log-spaced from 4,125 to 225,764 tokens.
- Seeds 0–4, 10 rounds: 75 runs.
- Paired t-tests of A − C and A − B at each budget, plus an area-under-curve summary.

The full plan is in [`PREREGISTRATION.md`](PREREGISTRATION.md). It was a draft
and was never committed or finalised; it is kept here as written.

## Why it was dropped

- **Arm C's extra cost is almost entirely in round 1.** In the main runs Arm C
  spends about 213k tokens in round 1, when the untrained model rarely solves
  8–10-digit sums, and then about 30k per round, close to the other arms. A
  per-round budget would mostly confirm that one fact.
- **The comparison would rest on two budgets.** At the top budgets the budget
  hardly binds (arms fill their plan first), so those cells repeat the
  equal-examples comparison already run. At the bottom budget Arm A cannot
  train at all. Only the two middle budgets would compare the arms at equal,
  binding token budgets.
- **Arm A can't afford its probe at the lowest budget.** The probe alone costs
  4,402–4,440 tokens per round, above the 4,125 budget, so Arm A collected no
  examples in any round. A − B and A − C there measure the probe's cost, not
  the curriculum.

The question the sweep asked is instead looked at, exploratorily and after the
fact, in the "Compute" section of the report (accuracy against cumulative
generated tokens from the existing runs).

## What actually ran

One pilot, **seed 0 only, at two budgets** (4,125 and 213,093 tokens per round),
for Arms A, B and C, 10 rounds each. Arm C was then rerun at 225,764 after it
collected 510 of 512 examples in round 1 at 213,093. Before the pilot, one
default-mode round (Arm A, seed 1, no budget) was run to check that the budget
code leaves default runs unchanged: it reproduced the original log exactly
(round-1 held-out 55.94%, 34,060 tokens, edge 5–6 digits).

Tokens spent per round and examples collected, from the pilot logs:

| Budget | Arm | Tokens per round | Examples per round |
|---|---|---|---|
| 4,125 | A | 4,402–4,440 (probe only) | 0 in every round |
| 4,125 | B | 4,090–4,125 | 66–116 |
| 4,125 | C | 4,075–4,125 | 3–30 |
| 213,093 | A | 22k–44k | 512 |
| 213,093 | B | 87k in round 1, then 20k–28k | 512 |
| 213,093 | C | 213k in round 1 (budget hit), then 26k–36k | 510, then 512 |
| 225,764 | C | 213k in round 1, then 25k–35k | 512 |

The pre-registration said pilot held-out accuracy would not be analysed, and it
is not reported here. One qualitative note, from re-running the pilot's final
Arm C model at 4,125 tokens (seed 0, round 10) greedily on the held-out set:
with so few training examples, every one of the 1,800 replies began with prose
("The total is:" followed by markdown) instead of the bare number. The parser
takes the first integer in the reply, so a reply that restates an operand
before the answer scores as wrong whatever the token cap. In the
main runs no reply lost any part of its number at the 16-token cap (see the cap
check in the report's Limitations section).

## Files

| Path | What it is |
|---|---|
| `PREREGISTRATION.md` | The draft plan, unchanged. |
| `pilot.sh` | The pilot script. Its `cd` line was changed from an absolute local path to the repo root; it still writes to `runs/budget_pilot/`, where the outputs were before they were moved here. |
| `pilot_outputs/*.out` | Console output of each pilot run (local paths replaced by `<repo>`). |
| `pilot_outputs/b<budget>/arm_<x>/seed0/log.jsonl` | Per-round logs, including a `budget` block with tokens, examples and stop reason. |
| `pilot_outputs/default_check/` | The one-round default-mode check described above. |

The LoRA adapter checkpoints (`pilot_outputs/**/round*/`, about 2.3 GB) are
kept locally but gitignored.

## Code that stays

Budget mode remains in the code, off by default: `Config.token_budget = None`,
`scripts/run_arm.py --token-budget N`, `collect_budget` in
`src/selfcurr/collect.py`, tested in `tests/test_collect_budget.py`. Without
`--token-budget`, runs follow the original fill-the-plan path.
