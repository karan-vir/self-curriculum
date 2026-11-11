# Pre-registration: token-budget sweep, 2-10 digit addition

Status: DRAFT, not committed, main sweep not started.

## Question

When every arm gets the same number of generated tokens per round, which matters
more: choosing difficulty adaptively (Arm A), or simply what is trained on (Arms
B and C)? Earlier runs matched arms on 512 examples per round, which let arms
spend very different amounts of model output to get them.

## Arms

All arms sample at temperature 0.8 with 4 tries per problem, keep at most one
verified example per problem (its first correct try), and share the same model,
LoRA settings and held-out set.

- **A, self-curriculum.** Each round, a greedy probe of 64 problems at each of
  the lengths 2-10, then a plan over the lengths with probe accuracy in [0.3, 0.7].
- **B, fixed curriculum.** Plan spread evenly over lengths 2-10.
- **C, hard-only.** Plan spread evenly over lengths 8-10.

Arm B+key is not part of this experiment.

## Design

- **Budgets (tokens per round):** 4125, 11220, 30517, 83003, 225764.
- **Seeds:** 0, 1, 2, 3, 4.
- **Rounds:** 10.
- **Cells:** 5 budgets x 3 arms x 5 seeds = 75 runs, each from a fresh model.
- **Held-out set:** the frozen `data/test_set.jsonl` (200 problems at each of
  lengths 2-10, 1800 in all), which is never trained on.
- **Pilot data are not used.** The pilot ran seed 0, and seed 0 is rerun in the
  sweep. Pilot held-out accuracy was not read and will not be analysed.

## How the budgets were derived

From token counts only, using round 1 of the existing logs, seeds 0-9. Round 1 is
the same starting model for all arms. `generated_tokens` includes Arm A's probe
and the end-of-turn token, and excludes padding.

| Arm | Mean tokens to collect 512 in round 1 | SD | Min | Max |
|---|---|---|---|---|
| A | 32,999.5 | 4,268 | 26,433 | 40,565 |
| B | 96,479.1 | 6,617 | 86,618 | 104,535 |
| C | 213,092.8 | 7,580 | 201,828 | 225,764 |

- Bottom = smallest mean / 8 = 32,999.5 / 8 = 4,124.94, used as 4125.
- Top as first specified = largest mean = 213,093.
- Top as adjusted = 225,764, see below.
- Five log-spaced values: bottom x (top / bottom)^(i/4), i = 0..4. The ratio is
  54.73, so the factor per step is 2.72.

### Endpoint adjustment after the pilot

The pilot checks were: at the bottom budget every arm stays at or below 256
examples (half the cap) in every round, and at the top budget every arm collects
512 in every round.

At 213,093, Arm C seed 0 collected 510 of 512 in round 1, which fails the top
check. The top was raised to the largest round-1 cost of any arm in any logged
seed, 225,764, so that it covers the most expensive logged seed and not only the
average one. Nothing but token and example counts was used. The bottom passed and
was not changed. Arm C seed 0 rerun at 225,764 collected 512 in all 10 rounds,
spending 213,471 tokens in round 1. Arms A and B stopped on a full plan in every
round at 213,093, so the higher budget does not change their runs, and they were
not rerun.

## Budget mode rules

Implemented in `collect_budget` (`src/selfcurr/collect.py`) and `scripts/run_arm.py
--token-budget`. The rules are the same for every arm.

1. **Counting.** Every generated token except padding, end-of-turn included. A
   problem is charged for all 4 tries, right or wrong. Held-out evaluation and
   training are not charged.
2. **Probe.** Arm A's full probe is always run and charged first. Sampling gets
   what remains. The probe is not shrunk, skipped or exempted.
3. **Probe over budget.** If the probe uses the whole budget, Arm A draws nothing
   that round and trains on nothing. This is the only way a round can exceed its
   budget, and the excess buys nothing, since no samples follow.
4. **Hard limit.** Sampling never overshoots. The first problem that does not fit
   ends the round and is not read. Problems drawn in the batch after it are
   discarded unread and logged as `uncharged_tokens`.
5. **Order of lengths.** Lengths take turns problem by problem, so the budget
   reaches all planned lengths and is not used up on the first ones.
6. **Per-length caps.** The plan, totalling 512, caps the examples kept at each
   length, as in the original arms. A full length stops being drawn and its share
   goes to the other lengths.
7. **Early stop.** When the plan is full the round stops. Unspent tokens are not
   kept and do not carry over.
8. **Training.** One LoRA epoch on all examples collected, however few, with a
   fresh optimiser. Zero examples means no training that round. There is no
   minimum.
9. **Stopping rule.** The stop depends only on token counts and on the same
   per-length caps for all arms, never on held-out accuracy.

Logged per round: tokens spent, probe tokens, examples, problems, tries, uncharged
tokens, stop reason, wall-clock seconds and held-out accuracy.

## Metrics

- **Primary:** final held-out accuracy, `eval_overall` at round 10, per budget.
- **Secondary:** area under the accuracy-vs-budget curve, per arm and seed: the
  trapezoid rule on final accuracy against log10(budget), divided by the
  log10 range so that it reads in accuracy units. The x axis is the nominal budget,
  not tokens actually spent.
- **Descriptive:** tokens actually spent and examples collected per round.

## Analysis

- At each budget, paired t-tests across the 5 seeds for A - C and A - B on final
  accuracy: two-sided, df = 4, with the mean difference and its 95% interval
  (t = 2.776).
- The same paired t-tests and 95% intervals for A - C and A - B on the area.
- That is 10 tests on the primary metric, with no multiplicity correction. Single
  results near the threshold are to be read with that in mind.
- With 5 seeds, a non-significant result is not evidence that arms are equal. The
  interval is the result to report.
- **Sensitivity (reported separately, not replacing the above):** the area
  recomputed without 4125, for the reason given below.
- Accuracy is not looked at, per cell or overall, until all 75 runs are complete.

## Known in advance

These follow from the pilot's token counts and are stated before any result.

- **Arm A does not train at 4125.** Its probe cost 4402-4440 tokens per round in
  the pilot, above the budget, so it collected 0 examples in all 10 rounds. Its
  final accuracy there will equal its starting accuracy. A - B and A - C at 4125
  therefore measure what the probe costs when tokens are this scarce, and say
  nothing about whether the curriculum is good.
- **High budgets rarely bind.** In the pilot at the top budget, rounds cost A
  22-44k, B 20-28k after round 1 (87k in round 1), and C 25-36k after round 1
  (213k in round 1). So A is expected to run the same process at 83003 and 225764,
  and B and C differ between those two mainly in round 1.
- **At high budgets, the comparison is at equal examples more than equal tokens,**
  since arms stop early at different costs.

## What each outcome would mean

An effect below means the 95% interval excludes zero.

| Outcome | Meaning |
|---|---|
| A > C and A > B at low-to-middle budgets (11220, 30517), converging at the top | Adapting pays when tokens are scarce, after paying for the probe. |
| A > B but A and C not distinguishable anywhere | The gain comes from spending on hard sums, not from adapting. |
| C > A at most budgets | Hard-only training is the better use of tokens. The probe and adaptive plan do not earn their cost at these settings. |
| C > A only at low budgets, equal at the top | The probe overhead matters when tokens are scarce and fades when they are not. |
| A > C only at the top two budgets | Not a token-efficiency result: at those budgets the budget hardly binds on A, so this repeats the equal-examples comparison. |
| B >= A and B >= C at low budgets | Easy examples are the cheapest verified data, and when tokens are scarce the number of examples matters more than their difficulty. |
| Arm gaps not distinguishable at any budget | No difference shown at this sample size. Report the interval widths, and make no claim that arms are equal. |
| A < others at 4125 only | Expected, see above. Not evidence against the curriculum. |

## Deviations

Any change after this document is committed, and any run that crashes and is
resumed (the log marks it with `resumed`, and a resumed run reseeds, so its later
draws differ), will be listed in the report. If an arm falls a few examples short
of 512 on some seed at the top budget, that is reported and not corrected: the
budget will not be changed once the sweep starts.
