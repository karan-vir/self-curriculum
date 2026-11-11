# Self-Curriculum Experiment

Does a small language model learn more, measured as accuracy on held-out problems, when it **chooses its own training problems** at the edge of its ability than when it trains on a fixed curriculum, for the same number of training examples?

> **Key result:** a fixed mix of only 8–10-digit sums, with no adaptation at all, did at least as well as the self-curriculum (87.1% against 84.7%; both beat the uniform mix at 78.9%). Counting the budget in training examples, spending it on long sums looks like what mattered; this run gives no evidence that adapting adds anything on top.
>
> Caveats: Arm C's range was chosen after seeing the results, and long sums give each example more answer tokens to learn from. Counted in generated tokens instead of examples, the self-curriculum comes out ahead (+3.8 points, an exploratory, after-the-fact comparison).

The full write-up, with charts, is [`reports/baseline.html`](reports/baseline.html) (open it in a browser; the "Ten rounds" tab has the trained results).

## Main findings

Qwen2.5-0.5B-Instruct with LoRA, learning to add two N-digit integers (N = 2–10), 10 rounds of 512 self-generated, checker-verified examples, 10 seeds per arm. Gaps are paired across seeds (two-sided t-test, not corrected for multiple comparisons).

| Finding | Number | Confidence |
| --- | --- | --- |
| The self-curriculum (A) beats a uniform 2–10-digit mix (B) | A − B +5.8 points at round 10 (95% interval +1.7 to +10.0, p = 0.011), A ahead on 9 of 10 seeds; +5.9 over the last three rounds (p = 0.003). A trails at round 1 (−7.7, behind on 9 of 10) and is level at round 2. Seeds 3–9 alone, added after an interim look: +5.8 (p = 0.055, 6 of 7) | Planned arms; seed count, test and final-round readout chosen after an interim look at 3 seeds; moderate |
| A fixed mix of only 8–10-digit sums (C) beats the uniform mix by more | C − B +8.2 points (+5.1 to +11.3, p < 0.001), 10 of 10 seeds | High for the gap; Arm C was added after the results and its range chosen with hindsight |
| Adapting showed no accuracy gain at equal training examples | A − C −2.4 points (−4.8 to +0.1, p = 0.056), A ahead on 1 of 10; level over rounds 2–9; −0.3 over the last three rounds (p = 0.78) | Inconclusive: unlike A − B, the gap does not hold over the last three rounds |
| At equal generated tokens, A is ahead of C, because C spends 42% of its tokens in round 1 | +3.8 points at A's final token total (+0.4 to +7.1, p = 0.031), 8 of 10 seeds | Exploratory, after the fact; low |
| Gains carry past the trained range | 11–13 digits: A 56.5%, B 43.9%, B+key 43.2%, C 61.1%, untrained 1.8% | Added after the results; A and C not distinguishable |

In short: a fixed mix of long sums matches or beats the self-curriculum, so this run gives no evidence that adapting adds accuracy beyond spending the budget on the hardest problems the model can still sometimes solve. Final mean held-out accuracy: A 84.7%, B 78.9%, B+key 81.4%, C 87.1%, untrained 43.9%.

## Problem statement

Most fine-tuning curricula are fixed ahead of time: a person decides the mix of easy and hard examples, and the model sees that mix whatever it already knows. Problems the model already solves teach it little, and problems far beyond it give no correct solutions to learn from.

A **self-curriculum** closes the loop. Each round the model is tested across difficulty levels, and training focuses on the levels where it is partly right, 30–70% accuracy (its *edge*). The training data is the model's own solutions at that edge, kept only when a programmatic checker confirms them (rejection sampling, as in [STaR](https://arxiv.org/abs/2203.14465)).

**What is measured:** learning efficiency, meaning held-out accuracy as a function of training examples seen. General reasoning is not measured.

## The arms

| Arm | Who picks the difficulty | Training answers | Status |
| --- | --- | --- | --- |
| **A, self-curriculum** | The model: each round, a greedy probe of 64 fresh problems per length (2–10), then train on lengths with 30–70% probe accuracy (or the one closest to 50% if none) | Model's own, checker-verified | Planned (3 seeds) |
| **B, fixed curriculum** | Nobody: uniform over 2–10 digits | Model's own, checker-verified | Planned (control) |
| **B+key, reference** | As B | Model's own; the true answer when no try is right | Reference, not a control; split off from B before any B result |
| **C, hard-only** | Nobody: uniform over 8–10 digits | Model's own, checker-verified | Added after the ten-seed A vs B result |

Shared: Qwen/Qwen2.5-0.5B-Instruct, LoRA rank 16, alpha 32, dropout 0.05, learning rate 2e-4, batch 16, one epoch per round with a fresh AdamW optimizer, 512 examples per round (one per problem), 4 tries per problem at temperature 0.8, 10 rounds, seeds 0–9. Every round is scored greedily (16 new tokens) on a frozen held-out set of 1,800 problems (`data/test_set.jsonl`, 200 per length), which training never draws from. Arm C may draw up to 200 batches of 64 problems per length against 50 for the others; it never needed more than 38.

## Results

Mean held-out accuracy (%) over ten seeds:

| Round | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Arm A · self-curriculum | 43.9 | 63.9 | 74.9 | 78.6 | 80.4 | 79.9 | 81.0 | 82.7 | 83.8 | 84.7 | 84.7 |
| Arm B · fixed curriculum | 43.9 | 71.6 | 74.8 | 74.5 | 76.9 | 76.6 | 77.6 | 77.3 | 78.0 | 78.7 | 78.9 |
| Arm B+key · reference | 43.9 | 73.8 | 74.1 | 74.0 | 75.9 | 77.1 | 78.3 | 79.7 | 80.7 | 80.7 | 81.4 |
| Arm C · hard-only | 43.9 | 74.3 | 75.9 | 78.7 | 79.3 | 79.9 | 82.2 | 81.4 | 82.5 | 84.7 | **87.1** |

Paired across seeds, final round:

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

Cost per run, mean over seeds (tokens are generated for training-data collection, Arm A's probe included):

| Arm | Tokens | Round 1 | Per round after round 1 | Wall time (M4 Pro) |
| --- | --- | --- | --- | --- |
| A | 356k | 33k | 36k | 24 min |
| B | 371k | 96k | 30k | 26 min |
| B+key | 176k | 18k | 18k | 18 min |
| C | 509k | 213k | 33k | 28 min |

What the results suggest, in more detail (each point is argued, with its caveats, in the report):

- **Arm C, with no probe and no adaptation, beats B by more than A does**, so this run gives no evidence that adapting adds accuracy beyond choosing long sums. Arm A itself trained only on 8–10 digits in 67 of 100 rounds.
- **A against C is close either way.** At equal examples C finishes 2.4 points ahead, a gap that rests on the last round (level over rounds 2–9, −0.3 over the last three rounds). At equal tokens A is ahead, because C pays for round 1 up front. Both comparisons are a few points and depend on how compute is counted; the token comparison was defined after the fact.
- **Arm A's slow start is not explained by focusing on few lengths.** After round 1, C leads A by 10.4 points on all 10 seeds despite practising only three lengths, and A's round-1 score ranges from 55.9% to 71.6% across seeds. A's first edge of 5–7 digits is one possible cause, not tested here.
- **Training only on long sums did not cost accuracy on shorter ones.** C never trains below 8 digits, yet finishes above B at 4–7 digits (3.7–8.8 points, each p < 0.05) and level at 2–3.
- **The answer key shows no clear benefit:** B+key − B is +2.5 points (p = 0.21).
- **The untrained model trips on columns that sum to 9** (exploratory): on 2–6 digits it gets 50.6% of problems with such a column (and no incoming carry) right, against 76.4% without, and its phantom carries concentrate where the column to the right sums to 7–9, most at 9.

## What was planned and what was added

Arms A and B, the edge band, the held-out set and the per-round readout (mean and range; success if A's curve sits above B's at matched rounds) were fixed before any result, with 3 seeds. The final-round paired test over 10 seeds that the A − B result now rests on was chosen after an interim look at those 3 seeds; on the planned criterion the picture is mixed (A behind at round 1, level at round 2, ahead from round 3). Changed before any training result: Arm B's answer-key fallback was split off into Arm B+key, and the probe was raised from 32 to 64 problems per length. Added after seeing results: seeds 3–9, the paired tests and sensitivity checks, Arm C (range chosen with hindsight), the 11–20-digit evaluation, the accuracy-against-tokens comparison, the sum-9 analysis and the 16-token-cap check. A token-budget sweep was drafted and dropped after a one-seed pilot; it is archived in [`archive/budget_sweep/`](archive/budget_sweep/README.md). Three Arm A runs were resumed after interruptions (seeds 0, 4, 7) and Arm B seed 2 was rerun from the start. The full table is in section R8 of the report.

## Limitations

Equal examples are not equal supervision: the loss is on answer tokens, and the mean operand length of training examples is 8.6 digits for A, 9.0 for C and 6.0 for B and B+key. Overall accuracy weights the nine lengths equally, and 2–4 digits are near the ceiling for every arm, so the gaps come from the longer lengths where A and C train. The planned secondary outcome (longest length with at least 50% accuracy) reaches the 10-digit ceiling by round 1–6 on every run, so it can't separate the arms. Also: one model (Qwen2.5-0.5B-Instruct), one task (addition), LoRA only, float16 (baseline) and float32 (training) on Apple MPS, one held-out test set, a 10-digit ceiling that Arm A's edge reached by round 2–4, Arm C's range chosen with hindsight and its higher draw limit, uncorrected multiple comparisons, a 16-token evaluation cap (checked after the fact for round 10 only: of 72,000 replies, 271 reached the cap and none lost any part of the number), interrupted runs, and ten seeds. Details in section R9 of the report.

## Reproducing

**Environment.** Python 3.12 with [uv](https://docs.astral.sh/uv/); versions are pinned in `uv.lock` (torch 2.15.0, transformers 5.22.1, peft 0.21.2 at the time of the runs). Runs on CPU, Apple Silicon (MPS) or CUDA. The model weights (Qwen/Qwen2.5-0.5B-Instruct) download from Hugging Face on first use. No API keys are needed.

**Hardware used.** Apple M4 Pro, 24 GB unified memory, MPS. Training and every round's evaluation run in float32; the untrained baseline in `scripts/baseline.py` runs in float16. Greedy replies can differ slightly on other hardware or precisions.

```bash
uv sync --extra train                                  # torch, transformers, peft
uv run pytest                                          # unit tests

uv run python scripts/make_test_set.py                 # frozen held-out set (already in data/)
uv run python scripts/baseline.py                      # untrained model, ~30 s

uv run python scripts/run_arm.py --arm a --seed 0      # one arm, one seed, 10 rounds
uv run python scripts/run_queue.py --seeds 0-9 --arms a,b,bkey,c   # everything, restarting hung runs

# analyses on the finished runs (no training)
uv run python scripts/baseline_long.py --max-digits 20                # untrained model on 2-20 digits
uv run --with scipy python scripts/eval_generalisation.py            # round-10 adapters on 11-20 digits
uv run python scripts/check_token_cap.py                              # 16-token cap check
uv run --with scipy python scripts/compute_vs_tokens.py              # accuracy against tokens
uv run --with scipy --with statsmodels python scripts/sum9_check.py  # sum-9 columns
uv run python scripts/build_report.py                                 # rebuild reports/baseline.html
```

**Runtime on the M4 Pro.** One 10-round run takes about 18–28 minutes (B+key 18, A 24, B 26, C 28), so all 40 runs take about 16 hours. The 11–20-digit evaluation takes about 1.5 minutes per adapter (about an hour in all), and the token-cap check about 35 seconds per adapter.

Budget mode (`run_arm.py --token-budget N`, from the archived sweep) is off by default; without the flag, runs follow the original fill-the-plan path.

## Where things live

| Path | Contents |
| --- | --- |
| `reports/baseline.html` | The report: untrained baseline (Report tab), trained results (Ten rounds tab), every baseline reply (Samples tab). Built from `reports/baseline.template.html` and `results/report_data.json` by `scripts/build_report.py`. |
| `results/baseline.json` | Every untrained-model reply on the held-out set. |
| `results/baseline_20.json`, `data/test_set_20.jsonl` | Untrained model on 2–20 digits, and the extended test set (2–10 identical to `data/test_set.jsonl`). |
| `results/generalisation_11_20.json` | Round-10 adapters on 11–20 digits. |
| `results/compute_vs_tokens.json` | Accuracy against cumulative tokens (exploratory). |
| `results/sum9_check.json` | Sum-9 column analysis of the baseline (exploratory). |
| `results/token_cap_check.json` | 16-token cap check. |
| `results/report_data.json` | Per-round, per-seed results and baseline data that the report's charts read. |
| `runs/` (gitignored) | Per-run logs (`runs/arm_<x>/seed<N>/log.jsonl`) and LoRA adapters for every round. Too large to commit; regenerate with the commands above. |
| `archive/budget_sweep/` | The dropped token-budget sweep: draft plan, pilot script and pilot logs. |

```
src/selfcurr/
  tasks.py        problem generator, prompt format, answer parser, checker
  curriculum.py   edge selection (Arm A) and fixed mixes (Arms B and C)
  config.py       experiment settings
  model.py        model loading, batched generation, evaluation
  collect.py      draws problems per plan, keeps checker-verified answers (and budget mode)
  train.py        LoRA setup and one training round (loss on the answer only)
scripts/          the commands above, plus watch.py (terminal view of running arms)
tests/            unit tests
```

## License

MIT, see [LICENSE](LICENSE).
