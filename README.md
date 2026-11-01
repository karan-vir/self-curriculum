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

1. Evaluate the model on a small probe batch at every difficulty.
2. Pick the edge: difficulties with accuracy in [0.3, 0.7]. If none qualify, take the one difficulty closest to 0.5.
3. Sample problems at the edge and let the model attempt each one several times.
4. Keep only answers the checker confirms are correct.
5. LoRA fine-tune on the kept examples (exactly `examples_per_round` of them).

**Arm B: fixed curriculum (control).** Same model, same `examples_per_round`, same training settings. Difficulties are drawn uniformly from 2–10 every round, with no adaptation. To isolate *who picks the difficulty*, Arm B also trains on the model's own checker-verified solutions, not on generator-supplied answers. Where the model gets no correct solution at a difficulty, Arm B falls back to the true answer, and this rate is logged.

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

Overall 44.0%. The starting edge (30–70%) is **5–6 digits**. Every wrong completion parsed as a number; the errors are arithmetic, mostly dropped or extra carries (e.g. 759 → 1759). Full completions are in `results/baseline.json`.

## Repository layout

```
src/selfcurr/
  tasks.py        problem generator, prompt format, answer parser, checker
  curriculum.py   edge selection (Arm A) and fixed mix (Arm B)
  config.py       experiment settings
  model.py        model loading, batched generation, evaluation
scripts/
  make_test_set.py   writes the frozen held-out set to data/test_set.jsonl
  baseline.py        untrained-model accuracy per difficulty
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
```

## Roadmap

- [x] Week 1, days 1–2: generator, checker, frozen test set
- [x] Baseline accuracy per difficulty for the untrained model
- [ ] Days 3–4: Arm A loop, one round end to end
- [ ] Days 5–6: Arm B with a matched budget
- [ ] Day 7: several rounds of both arms, plots
- [ ] v2: the model writes its own problems, not only picks the difficulty
