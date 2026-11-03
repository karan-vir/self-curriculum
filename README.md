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
- **Seeds:** each arm runs with 3 seeds; plots show the mean and range.

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

## Ten rounds, all arms (seed 0)

| Round | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Arm A · self-curriculum | 43.9 | 58.9 | 68.9 | **80.9** | 80.7 | 75.1 | 77.3 | 80.2 | 80.9 | 82.8 | **84.2** |
| Arm B · fixed curriculum | 43.9 | 67.6 | 74.1 | 73.1 | **80.8** | 78.9 | 71.2 | 77.8 | 81.7 | 84.9 | **83.8** |
| Arm B+key · reference | 43.9 | 70.3 | 75.3 | 74.9 | 77.5 | **80.1** | 78.1 | 82.1 | 81.8 | 79.2 | **83.6** |

Held-out accuracy (%); bold marks the first round at 80% or above and the final. Over 10 rounds Arm A generated 353k tokens, Arm B 371k and Arm B+key 175k (with 923 answer-key examples).

The arms finish effectively tied: the finals are within 0.6 points, inside the ±1.7-point 95% interval of one score and smaller than the 2–8-point swings between rounds. Arm A reached 80% first (round 3, against 4 and 5), but its edge climbed from 5–6 to 8–10 digits by round 4 and stayed within 7–10 after that: the task stops at 10 digits, so the self-curriculum had nowhere further to go and the fixed mix caught up. Each arm ends slightly ahead where it practised (A at 8–10 digits, B at 4–5), within per-length noise. The answer key led early and gave no final benefit. Next: seeds 1 and 2, longer sums (e.g. 2–20 digits), and steadier training (keep optimizer state between rounds or lower the learning rate). Charts are in the Ten rounds tab of `reports/baseline.html`; `reports/status.html` shows runs live when the repo is served locally (`python3 -m http.server 8765`), and `scripts/watch.py` shows the same in a terminal.

## Roadmap

- [x] Week 1, days 1–2: generator, checker, frozen test set
- [x] Baseline accuracy per difficulty for the untrained model
- [x] Days 3–4: Arm A loop, one round end to end
- [x] Days 5–6: Arm B with a matched budget (plus Arm B+key reference)
- [x] Day 7: 10 rounds of all three arms, plots (seed 0)
- [ ] Seeds 1 and 2; a longer difficulty range
- [ ] v2: the model writes its own problems, not only picks the difficulty
