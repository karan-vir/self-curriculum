# Self-Curriculum Experiment

Does a small language model learn faster when it **chooses its own training problems** at the edge of its ability than when it trains on a fixed curriculum, for the same training budget?

## Problem statement

Most fine-tuning curricula are fixed ahead of time: a person decides the mix of easy and hard examples, and the model sees that mix whatever it already knows. That wastes budget. Problems the model already solves teach it little, and problems far beyond it give no correct solutions to learn from.

A **self-curriculum** closes the loop. Each round, the model is tested across difficulty levels, and training focuses on the levels where it is partly right, roughly 30–70% accuracy (its *edge*). The training data is the model's own solutions at that edge, kept only when a programmatic checker confirms they are correct (rejection sampling, as in [STaR](https://arxiv.org/abs/2203.14465)).

**Question:** for the same number of training examples, does the self-curriculum reach higher held-out accuracy, or solve longer problems, than a fixed curriculum?

**What is measured:** *learning efficiency*, meaning held-out accuracy as a function of training examples seen. Reasoning ability in general is not measured.

**Hypothesis:** Arm A (self-curriculum) climbs faster per round and reaches more digits than Arm B (fixed curriculum).

**Result so far:** Arm A beats the uniform mix (84.7% against 78.9%, ten seeds), but a fixed mix of only 8–10-digit sums does at least as well (87.1%). Spending the budget on long sums, rather than adapting, looks like what matters on this task. See [Arm C](#arm-c-hard-only-seeds-09).

## Setup

| Part | Choice |
| --- | --- |
| Model | `Qwen/Qwen2.5-0.5B-Instruct`, LoRA fine-tuning |
| Task | Addition of two N-digit integers, difficulty = N, from 2 to 10 |
| Data | Generated in Python; answers checked exactly, so no labels needed |
| Rounds | ~10 per arm |
| Budget | Same number of training examples per round in every arm |

## The arms

**Arm A: self-curriculum.** Each round:

1. Evaluate the model on 64 fresh probe problems at every difficulty.
2. Pick the edge: difficulties with accuracy in [0.3, 0.7]. If none qualify, take the one difficulty closest to 0.5.
3. Sample problems at the edge and let the model attempt each one several times.
4. Keep only answers the checker confirms are correct.
5. LoRA fine-tune on the kept examples (exactly `examples_per_round` of them).

**Arm B: fixed curriculum (control).** Same model, same `examples_per_round`, same training settings. Difficulties are drawn uniformly from 2–10 every round, with no adaptation. To isolate *who picks the difficulty*, Arm B also trains only on the model's own checker-verified solutions, drawing more problems at a length until enough are solved.

**Arm B+key: reference, not a control.** As Arm B, but a problem the model never solves is trained on the true answer from the generator. It measures what the answer key adds on top of self-training, so it answers a different question from A vs B.

**Arm C: hard-only.** As Arm B, but difficulties are drawn evenly from 8, 9 and 10 digits only. No probe, no adaptation, and no answer key. It was added after the ten-seed results to test whether Arm A wins by adapting or simply by spending its budget on long sums. Its 8–10-digit range was chosen knowing where Arm A's gain appeared. It may draw up to 200 batches of problems per length, against 50 for the other arms; it never needed more than 38.

## Fairness rules

- **Matched training budget:** all arms train on the same number of examples per round, with the same LoRA settings and steps.
- **Inference is logged, not matched:** Arm A spends extra generation on probing and sampling. Every arm logs generated tokens so the cost difference is reported.
- **No test leakage:** the held-out test set is generated once with a fixed seed. Training problems that appear in it are rejected.
- **Seeds:** each arm runs with 10 seeds (planned as 3, extended after the first results); plots show the mean and range.

## Measurement

- Held-out test set: 200 problems per difficulty, evaluated with greedy decoding after every round.
- Main plot: held-out accuracy against round (equivalently, training examples seen) for every arm, overall and per difficulty.
- Secondary: the highest difficulty with at least 50% accuracy, per round.
- Success signal: Arm A's curve sits above Arm B's at matched rounds, or it reaches more digits.

## Baseline

Untrained `Qwen2.5-0.5B-Instruct`, greedy decoding, 200 held-out problems per difficulty (`uv run python scripts/baseline.py`, 30 s on an M4 Pro via MPS):

| Digits | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Accuracy | 91.5% | 87.5% | 74.0% | **58.5%** | **39.5%** | 23.5% | 8.5% | 10.0% | 3.0% |

Overall 44.0%. The starting edge (30–70%) is **5–6 digits**. Every wrong completion parsed as a number, so the errors are arithmetic. Accuracy does not depend on how many carries a problem needs, only on its length. Short sums mostly fail with one column off by 1 (phantom carries outnumber dropped ones 107 to 68); long sums fail across every column, including the units. Full completions are in `results/baseline.json`; the full analysis is in `reports/baseline.html`.

## Repository layout

```
src/selfcurr/
  tasks.py        problem generator, prompt format, answer parser, checker
  curriculum.py   edge selection (Arm A) and fixed mix (Arms B and C)
  config.py       experiment settings
  model.py        model loading, batched generation, evaluation
  collect.py      draws problems per plan, keeps checker-verified answers
  train.py        LoRA setup and one training round (loss on the answer only)
scripts/
  make_test_set.py   writes the frozen held-out set to data/test_set.jsonl
  baseline.py        untrained-model accuracy per difficulty
  run_arm.py         round loop for any arm (--arm a|b|bkey|c)
  run_queue.py       runs seeds and arms in turn, restarting hung runs
  watch.py           terminal view of running arms
results/          committed experiment results
tests/            unit tests for the generator, checker and curriculum
```

## Local setup

Requires [uv](https://docs.astral.sh/uv/). Runs on CPU, Apple Silicon (MPS) or CUDA.

```bash
uv sync                          # core + dev tools
uv sync --extra train            # adds torch, transformers, peft
uv run pytest                    # tests
uv run python scripts/make_test_set.py
uv run python scripts/baseline.py
uv run python scripts/run_arm.py --arm a --rounds 1 --seed 0   # one Arm A round
uv run python scripts/run_arm.py --arm b --rounds 1 --seed 0   # one Arm B round
uv run python scripts/run_arm.py --arm bkey --rounds 1 --seed 0   # one Arm B+key round
uv run python scripts/run_arm.py --arm c --rounds 1 --seed 0   # one Arm C round
```

`run_arm.py` writes `runs/arm_<a|b|bkey|c>/seed<N>/log.jsonl` (plan, probe accuracy and edge for Arm A, collection stats, fallback count, generated tokens, losses and held-out accuracy per round) and a LoRA adapter per round. Training runs in float32 on MPS; round 0 re-evaluates the untrained model in the same precision.

## First round, all arms (seed 0)

| | Overall | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Round 0 (all arms) | 43.9% | 92% | 88% | 74% | 58% | 40% | 24% | 8% | 10% | 3% |
| Arm A · self-curriculum | 58.9% | 94% | 89% | 80% | 74% | 64% | 58% | 24% | 30% | 16% |
| Arm B · fixed curriculum | **67.6%** | 91% | 86% | 84% | 78% | 69% | 68% | 50% | 44% | 38% |
| Arm B+key · reference | 70.3% | 93% | 86% | 82% | 70% | 72% | 66% | 61% | 60% | 43% |

| | Problems drawn | Answer-key examples | Generated tokens | Collect time |
| --- | --- | --- | --- | --- |
| Arm A | 1,088 | 0 | 35,897 | 77 s |
| Arm B | 2,688 | 0 | 104,535 | 211 s |
| Arm B+key | 576 | 258 | 17,739 | 42 s |

After one round, the fixed curriculum beats the self-curriculum by 8.7 points at the same 512 training examples. Arm A trained only at 5–6 digits and gained most at 7; Arm B trained a little at every length and gained most at 8–10. The answer key adds only 2.7 points on top of Arm B, mostly at 8–10 digits. Arm B pays for its lead in generation: it sampled 2.9× Arm A's tokens to find rare correct answers at long lengths (7% solve rate at 10 digits). One seed and one round: Arm A's edge should move up in later rounds, so the 10-round curves decide the result.

## Ten rounds, all arms (seeds 0–9)

Mean held-out accuracy (%) over ten seeds:

| Round | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Arm A · self-curriculum | 43.9 | 63.9 | 74.9 | 78.6 | 80.4 | 79.9 | 81.0 | 82.7 | 83.8 | 84.7 | **84.7** |
| Arm B · fixed curriculum | 43.9 | 71.6 | 74.8 | 74.5 | 76.9 | 76.6 | 77.6 | 77.3 | 78.0 | 78.7 | 78.9 |
| Arm B+key · reference | 43.9 | **73.8** | 74.1 | 74.0 | 75.9 | 77.1 | 78.3 | 79.7 | 80.7 | 80.7 | 81.4 |

| Seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Arm A | 84.2 | 87.3 | 87.1 | 77.7 | 85.3 | 86.5 | 85.9 | 84.4 | 83.4 | 85.6 |
| Arm B | 83.8 | 80.1 | 76.9 | 81.3 | 78.6 | 81.1 | 67.7 | 80.7 | 79.0 | 79.9 |
| Arm B+key | 83.6 | 78.0 | 79.5 | 83.2 | 81.4 | 76.0 | 84.6 | 84.0 | 80.6 | 82.8 |

| | First round ≥ 80%, median (range) | 10 digits, final | Worst one-round drop, mean | Tokens per run | Answer-key examples |
| --- | --- | --- | --- | --- | --- |
| Arm A | 4 (3–9) | 73.4% | 2.2 pts | 356k | 0 |
| Arm B | 7 (3–10, never on 1 seed) | 58.7% | 5.8 pts | 371k | 0 |
| Arm B+key | 7 (5–10, never on 1 seed) | 64.8% | 5.5 pts | 176k | 880–1,221 |

The self-curriculum finishes ahead of the fixed mix on 9 of 10 seeds, by 5.8 points on average (paired t-test across seeds: 95% interval 1.7 to 10.0, two-sided p = 0.011). The result holds up when averaging the last three rounds (5.9 points, p = 0.003) and without seed 6, where Arm B had an unusually bad last round (4.5 points, p = 0.010). The seven seeds added after the first three show a similar gap on their own (5.8 points, p = 0.055). Seed 3 goes the other way (−3.6 points), and single-seed gaps range up to +18.2, so one seed is not enough to compare curricula.

Arm B leads after round 1 on 9 of 10 seeds, by 7.7 points on average, because it practises all lengths at once. Arm A is ahead by round 3 on 9 of 10 seeds and keeps climbing while Arm B stays near 77–79%. The arms are level on 2–5-digit sums, and Arm A's lead grows with length to 14.7 points at 10 digits. It is also steadier between rounds and generates slightly fewer tokens.

The answer key shows no clear benefit: B+key's 2.5-point lead over Arm B is not distinguishable from zero (p = 0.21), and falls to 0.9 points without seed 6. Arm A finishes above B+key on 9 of 10 seeds by 3.4 points, which is borderline (p = 0.051).

Arm A's edge reaches 10 digits by round 2–4 on every seed, so a longer range is the natural next test.

Three Arm A runs were resumed from a saved round after a power problem or hang (seed 0 after round 9, seed 4 after round 1, seed 7 after round 3), and Arm B on seed 2 was rerun from the start. Each round starts a fresh optimizer, so resuming loses no training state, though the sampling sequence changes. The plan was three seeds; seven more were added after the first three looked promising but inconclusive.

## Arm C: hard-only (seeds 0–9)

Same budget and settings as the other arms, training only on the model's own correct answers at 8, 9 and 10 digits. All ten runs finished without a restart, with no shortfall in any round.

Mean held-out accuracy (%) over ten seeds:

| Round | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Arm A · self-curriculum | 43.9 | 63.9 | 74.9 | 78.6 | 80.4 | 79.9 | 81.0 | 82.7 | 83.8 | 84.7 | 84.7 |
| Arm B · fixed curriculum | 43.9 | 71.6 | 74.8 | 74.5 | 76.9 | 76.6 | 77.6 | 77.3 | 78.0 | 78.7 | 78.9 |
| Arm C · hard-only | 43.9 | 74.3 | 75.9 | 78.7 | 79.3 | 79.9 | 82.2 | 81.4 | 82.5 | 84.7 | **87.1** |

| Seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Arm C | 84.3 | 88.8 | 87.6 | 88.9 | 86.2 | 90.3 | 85.1 | 85.3 | 86.6 | 87.9 |

Paired across seeds, final round (t-test, two-sided, not corrected for multiple comparisons):

| Comparison | Mean gap (points) | 95% interval | p | Seeds first arm ahead | Gap, last 3 rounds (p) |
| --- | --- | --- | --- | --- | --- |
| Arm A − Arm C | −2.4 | −4.8 to +0.1 | 0.056 | 1 of 10 | −0.3 (0.78) |
| Arm B − Arm C | −8.2 | −11.3 to −5.1 | <0.001 | 0 of 10 | −6.2 (<0.001) |
| Arm A − Arm B | +5.8 | +1.7 to +10.0 | 0.011 | 9 of 10 | +5.9 (0.003) |

| Final accuracy by length (%) | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Arm A | 97.1 | 94.8 | 90.9 | 85.7 | 80.4 | 85.5 | 77.2 | 77.7 | 73.4 |
| Arm B | 97.2 | 94.6 | 89.2 | 82.8 | 75.4 | 78.0 | 67.5 | 66.8 | 58.7 |
| Arm C | 97.0 | 95.8 | 92.9 | 88.3 | 83.5 | 86.8 | 83.5 | 80.0 | 76.3 |

| Per run, mean | Tokens | Of which round 1 | Problems drawn | Tries | Shortfall | Wall time |
| --- | --- | --- | --- | --- | --- | --- |
| Arm A | 356k | 33k | 7,680 | 30,720 | 0 | 24 min |
| Arm B | 371k | 96k | 10,778 | 43,110 | 0 | 26 min |
| Arm C | 509k | 213k | 11,802 | 47,206 | 0 | 28 min |

**Spending on long sums appears to account for Arm A's win over Arm B.** Arm C beats Arm B on all 10 seeds by more than Arm A does, with no probe and no adaptation. Arm A ends up doing nearly the same thing: it trained only on 8–10 digits in 67 of 100 rounds.

**Adapting showed no accuracy gain, but Arm C isn't shown to be better either.** Arm C finishes ahead of Arm A on 9 of 10 seeds, but the gap is borderline and rests on one round. From round 2 to round 9 the two are level (+0.2 points, p = 0.90), and the gap falls to 0.3 over the last three rounds. Seed 3, where Arm A fell to 77.7%, is a large part of it: without that seed the gap is 1.4 points (p = 0.022).

**Adapting may still have value this run can't show.** Arm C's range was chosen after seeing where Arm A gained, and Arm A found those lengths with no prior knowledge. Here a good fixed range exists and is easy to guess. A longer difficulty range would test whether that still holds.

Arm C never trains below 8 digits, yet it beats Arm B on 2–7-digit sums by 4.5 points (p = 0.005). It also leads Arm A by 10.4 points after round 1 on all 10 seeds, which argues against the earlier reading that Arm B starts fast because it practises every length; Arm A's first edge of 5–7 digits is the likelier cause. Arm C's cost is its first round: 213k tokens while long sums are rarely solved (15%, 12% and 8% of 8-, 9- and 10-digit problems), then about 33k per round, close to Arm A.

Charts are in the Ten rounds tab of `reports/baseline.html`. `reports/status.html` shows runs live when the repo is served locally (`python3 -m http.server 8765`), and `scripts/watch.py` shows the same in a terminal.

## Roadmap

- [x] Week 1, days 1–2: generator, checker, frozen test set
- [x] Baseline accuracy per difficulty for the untrained model
- [x] Days 3–4: Arm A loop, one round end to end
- [x] Days 5–6: Arm B with a matched budget (plus Arm B+key reference)
- [x] Day 7: 10 rounds of the first three arms, plots
- [x] Seeds 1 and 2
- [x] Seeds 3–9, with a watchdog that restarts hung runs (`scripts/run_queue.py`)
- [x] Test why Arm A wins: fixed curriculum on long sums only (Arm C)
- [ ] A longer difficulty range (e.g. 2–20 digits), where a fixed range is harder to choose
- [ ] Run past round 10, to see whether Arm C's final-round lead over Arm A lasts
- [ ] Steadier training: keep optimizer state between rounds or lower the learning rate
- [ ] v2: the model writes its own problems, not only picks the difficulty
