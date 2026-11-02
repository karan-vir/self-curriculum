"""Arm A: self-curriculum. Probe -> pick edge -> sample -> keep correct -> LoRA -> evaluate.

Round 0 is the held-out evaluation before any training, run in the same
precision as training so every round is compared like for like.

    uv run python scripts/run_arm_a.py --rounds 1 --seed 0

Writes runs/arm_a/seed<N>/log.jsonl (one line per round) and the LoRA
adapter after each round.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from selfcurr.collect import collect
from selfcurr.config import Config
from selfcurr.curriculum import select_edge, self_curriculum_plan
from selfcurr.model import evaluate, generate, load_model, load_test_set, pick_device
from selfcurr.tasks import make_problems
from selfcurr.train import add_lora, train_round

ROOT = Path(__file__).resolve().parent.parent


def fmt_acc(acc: dict[int, float]) -> str:
    return " ".join(f"{d}:{a:.0%}" for d, a in acc.items())


def overall(records: list[dict]) -> float:
    return sum(r["correct"] for r in records) / len(records)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=None, help="default: Config.rounds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = Config()
    rounds = args.rounds or cfg.rounds
    out = args.out or ROOT / "runs" / "arm_a" / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.jsonl"
    log_path.write_text("")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    device = pick_device()
    test = load_test_set(ROOT / cfg.test_path)
    test_keys = {p.key for p in test}
    model, tok = load_model(cfg.model_name, device, dtype=torch.float32)
    model = add_lora(model, cfg)
    print(f"arm A · seed {args.seed} · {rounds} rounds · device {device} · out {out}")

    def log(row: dict) -> None:
        with log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    t0 = time.time()
    acc, records = evaluate(model, tok, test)
    log({"round": 0, "eval": acc, "eval_overall": overall(records), "seconds": round(time.time() - t0, 1)})
    print(f"round 0 · held-out {overall(records):.1%} · {fmt_acc(acc)}")

    for r in range(1, rounds + 1):
        usage: dict = {}
        times: dict = {}

        t = time.time()
        probe = [p for d in cfg.difficulties for p in make_problems(d, cfg.probe_per_difficulty, rng, exclude=test_keys)]
        probe_acc, _ = evaluate(model, tok, probe, usage=usage)
        edge = select_edge(probe_acc, cfg.edge_low, cfg.edge_high)
        plan = self_curriculum_plan(probe_acc, cfg.examples_per_round, rng, cfg.edge_low, cfg.edge_high)
        times["probe"] = round(time.time() - t, 1)

        t = time.time()
        sample_fn = lambda ps: generate(  # noqa: E731
            model, tok, ps, temperature=cfg.sample_temperature, num_samples=cfg.samples_per_problem, usage=usage
        )
        col = collect(sample_fn, plan, rng, exclude=test_keys, batch_size=cfg.collect_batch, max_batches=cfg.collect_max_batches)
        times["collect"] = round(time.time() - t, 1)

        t = time.time()
        losses = train_round(model, tok, col.examples, cfg, rng)
        times["train"] = round(time.time() - t, 1)

        t = time.time()
        acc, records = evaluate(model, tok, test)
        times["eval"] = round(time.time() - t, 1)
        model.save_pretrained(out / f"round{r}")

        row = {
            "round": r,
            "probe": probe_acc,
            "edge": edge,
            "plan": plan,
            "collect": {d: s.to_dict() for d, s in col.stats.items()},
            "kept": len(col.examples),
            "shortfall": col.shortfall,
            "generated_tokens": usage.get("generated_tokens", 0),
            "loss_first": round(losses[0], 4) if losses else None,
            "loss_last": round(losses[-1], 4) if losses else None,
            "losses": [round(x, 4) for x in losses],
            "eval": acc,
            "eval_overall": overall(records),
            "seconds": times,
        }
        log(row)
        print(
            f"round {r} · probe {fmt_acc(probe_acc)} · edge {edge} · kept {len(col.examples)}"
            f" (shortfall {col.shortfall}) · loss {row['loss_first']}→{row['loss_last']}"
            f" · held-out {row['eval_overall']:.1%} · {fmt_acc(acc)} · {times}"
        )

    print(f"done in {time.time() - t0:.0f}s · log {log_path}")


if __name__ == "__main__":
    main()
