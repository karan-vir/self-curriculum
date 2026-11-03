# Self-Curriculum Experiment

Does a small language model learn faster when it **chooses its own training problems** at the edge of its ability than when it trains on a fixed curriculum, for the same training budget?

## Problem statement

Most fine-tuning curricula are fixed ahead of time: a person decides the mix of easy and hard examples, and the model sees that mix whatever it already knows. That wastes budget. Problems the model already solves teach it little, and problems far beyond it give no correct solutions to learn from.

A **self-curriculum** closes the loop. Each round, the model is tested across difficulty levels, and training focuses on the levels where it is partly right, roughly 30–70% accuracy (its *edge*). The training data is the model's own solutions at that edge, kept only when a programmatic checker confirms they are correct (rejection sampling, as in [STaR](https://arxiv.org/abs/2203.14465)).

**Question:** for the same number of training examples, does the self-curriculum reach higher held-out accuracy, or solve longer problems, than a fixed curriculum?

**What is measured:** *learning efficiency*, meaning held-out accuracy as a function of training examples seen. Reasoning ability in general is not measured.

**Hypothesis:** Arm A (self-curriculum) climbs faster per round and reaches more digits than Arm B (fixed curriculum).

## Setup

| Part | Choice |
| --- | --- |
| Model | `Qwen/Qwen2.5-0.5B-Instruct`, LoRA fine-tuning |
| Task | Addition of two N-digit integers, difficulty = N, from 2 to 10 |
| Data | Generated in Python; answers checked exactly, so no labels needed |
| Rounds | ~10 per arm |
| Budget | Same number of training examples per round in both arms |

## The two arms

**Arm A: self-curriculum.** Each round:

1. Evaluate the model on 64 fresh probe problems at every difficulty.
2. Pick the edge: difficulties with accuracy in [0.3, 0.7]. If none qualify, take the one difficulty closest to 0.5.
3. Sample problems at the edge and let the model attempt each one several times.
4. Keep only answers the checker confirms are correct.
5. LoRA fine-tune on the kept examples (exactly `examples_per_round` of them).

**Arm B: fixed curriculum (control).** Same model, same `examples_per_round`, same training settings. Difficulties are drawn uniformly from 2–10 every round, with no adaptation. To isolate *who picks the difficulty*, Arm B also trains only on the model's own checker-verified solutions, drawing more problems at a length until enough are solved.

**Arm B+key: reference, not a control.** As Arm B, but a problem the model never solves is trained on the true answer from the generator. It measures what the answer key adds on top of self-training, so it answers a different question from A vs B.

## Fairness rules

- **Matched training budget:** both arms train on the same number of examples per round, with the same LoRA settings and steps.
- **Inference is logged, not matched:** Arm A spends extra generation on probing and sampling. Both arms log generated tokens so the cost difference is reported.
- **No test leakage:** the held-out test set is generated once with a fixed seed. Training problems that appear in it are rejected.
- **Seeds:** each arm runs with 10 seeds (planned as 3, extended after the first results); plots show the mean and range.

## Measurement

- Held-out test set: 200 problems per difficulty, evaluated with greedy decoding after every round.
- Main plot: held-out accuracy against round (equivalently, training examples seen) for both arms, overall and per difficulty.
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
  curriculum.py   edge selection (Arm A) and fixed mix (Arm B)
  config.py       experiment settings
  model.py        model loading, batched generation, evaluation
  collect.py      draws problems per plan, keeps checker-verified answers
  train.py        LoRA setup and one training round (loss on the answer only)
scripts/
  make_test_set.py   writes the frozen held-out set to data/test_set.jsonl
  baseline.py        untrained-model accuracy per difficulty
  run_arm.py         round loop for any arm (--arm a|b|bkey)
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
```

`run_arm.py` writes `runs/arm_<a|b|bkey>/seed<N>/log.jsonl` (plan, probe accuracy and edge for Arm A, collection stats, fallback count, generated tokens, losses and held-out accuracy per round) and a LoRA adapter per round. Training runs in float32 on MPS; round 0 re-evaluates the untrained model in the same precision.

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

Arm A's edge reaches 10 digits by round 2–4 on every seed, so a longer range is the natural next test. These results don't yet show why Arm A wins: it may be the adaptation, or simply that it spends its budget on long sums.

Three Arm A runs were resumed from a saved round after a power problem or hang (seed 0 after round 9, seed 4 after round 1, seed 7 after round 3), and Arm B on seed 2 was rerun from the start. Each round starts a fresh optimizer, so resuming loses no training state, though the sampling sequence changes. The plan was three seeds; seven more were added after the first three looked promising but inconclusive.

Charts are in the Ten rounds tab of `reports/baseline.html`. `reports/status.html` shows runs live when the repo is served locally (`python3 -m http.server 8765`), and `scripts/watch.py` shows the same in a terminal.

## Roadmap

- [x] Week 1, days 1–2: generator, checker, frozen test set
- [x] Baseline accuracy per difficulty for the untrained model
- [x] Days 3–4: Arm A loop, one round end to end
- [x] Days 5–6: Arm B with a matched budget (plus Arm B+key reference)
- [x] Day 7: 10 rounds of all three arms, plots
- [x] Seeds 1 and 2
- [x] Seeds 3–9, with a watchdog that restarts hung runs (`scripts/run_queue.py`)
- [ ] A longer difficulty range (e.g. 2–20 digits)
- [ ] Test why Arm A wins: compare it with a fixed curriculum on long sums only
- [ ] Steadier training: keep optimizer state between rounds or lower the learning rate
- [ ] v2: the model writes its own problems, not only picks the difficulty
