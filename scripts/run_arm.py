"""Run one arm of the experiment for N rounds.

Arm A (self-curriculum): probe -> pick edge -> sample -> keep correct -> LoRA -> evaluate.
Arm B (fixed curriculum): uniform over all lengths -> sample -> keep correct -> LoRA -> evaluate.
Arm B+key (reference): as Arm B, but a problem the model never solves is
trained on the true answer. Not a control for A; it measures what the answer
key is worth on top of self-training.
Arm C (hard-only): as Arm B, but only over Config.hard_digits. Tests whether
Arm A's lead comes from adapting or just from spending its budget on long sums.

Everything after the plan is shared, with the same settings and budget.
Round 0 is the held-out evaluation before any training, run in the same
precision as training so every round is compared like for like.

    uv run python scripts/run_arm.py --arm a|b|bkey|c --rounds 1 --seed 0

Writes runs/arm_<x>/seed<N>/log.jsonl (one line per round) and the LoRA
adapter after each round.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path

import torch

from selfcurr.collect import collect
from selfcurr.config import Config
from selfcurr.curriculum import fixed_curriculum_plan, select_edge, self_curriculum_plan
from selfcurr.model import evaluate, generate, load_model, load_test_set, pick_device
from selfcurr.tasks import make_problems
from selfcurr.train import add_lora, train_round
from peft import PeftModel

ROOT = Path(__file__).resolve().parent.parent


def fmt_acc(acc: dict[int, float]) -> str:
    return " ".join(f"{d}:{a:.0%}" for d, a in acc.items())


def overall(records: list[dict]) -> float:
    return sum(r["correct"] for r in records) / len(records)


def free_memory() -> None:
    """Return cached GPU memory between phases.

    Without this, the MPS allocator keeps every buffer it has handed out,
    and over 9 rounds Arm A grew to 21 GB on a 24 GB Mac and stalled in swap.
    """
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def gpu_gb() -> float | None:
    if torch.backends.mps.is_available():
        return round(torch.mps.driver_allocated_memory() / 2**30, 2)
    if torch.cuda.is_available():
        return round(torch.cuda.memory_reserved() / 2**30, 2)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["a", "b", "bkey", "c"], required=True)
    ap.add_argument("--rounds", type=int, default=None, help="default: Config.rounds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--resume", action="store_true", help="continue from the last round saved in --out")
    args = ap.parse_args()

    cfg = Config()
    rounds = args.rounds or cfg.rounds
    out = args.out or ROOT / "runs" / f"arm_{args.arm}" / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.jsonl"
    done = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()] if args.resume and log_path.exists() else []
    start = max((r["round"] for r in done), default=0)
    if not args.resume:
        log_path.write_text("")

    # a resumed run reseeds from (seed, round), so its later draws differ from an uninterrupted run
    random.seed(args.seed)
    torch.manual_seed(args.seed + start)
    rng = random.Random(f"{args.seed}-{start}" if start else args.seed)

    device = pick_device()
    test = load_test_set(ROOT / cfg.test_path)
    test_keys = {p.key for p in test}
    model, tok = load_model(cfg.model_name, device, dtype=torch.float32)
    if start:
        model = PeftModel.from_pretrained(model, out / f"round{start}", is_trainable=True)
    else:
        model = add_lora(model, cfg)
    print(f"arm {args.arm.upper()} · seed {args.seed} · {rounds} rounds · device {device} · out {out}"
          + (f" · resuming after round {start}" if start else ""))

    def log(row: dict) -> None:
        with log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    t0 = time.time()
    if not start:
        acc, records = evaluate(model, tok, test)
        log({"arm": args.arm, "round": 0, "eval": acc, "eval_overall": overall(records), "seconds": round(time.time() - t0, 1)})
        print(f"round 0 · held-out {overall(records):.1%} · {fmt_acc(acc)}")
        free_memory()

    for r in range(start + 1, rounds + 1):
        usage: dict = {}
        times: dict = {}
        probe_acc = edge = None

        t = time.time()
        if args.arm == "a":
            probe = [p for d in cfg.difficulties for p in make_problems(d, cfg.probe_per_difficulty, rng, exclude=test_keys)]
            probe_acc, _ = evaluate(model, tok, probe, usage=usage)
            edge = select_edge(probe_acc, cfg.edge_low, cfg.edge_high)
            plan = self_curriculum_plan(probe_acc, cfg.examples_per_round, rng, cfg.edge_low, cfg.edge_high)
        else:
            plan = fixed_curriculum_plan(cfg.hard_digits if args.arm == "c" else cfg.difficulties, cfg.examples_per_round, rng)
        times["plan"] = round(time.time() - t, 1)

        t = time.time()
        sample_fn = lambda ps: generate(  # noqa: E731
            model, tok, ps, temperature=cfg.sample_temperature, num_samples=cfg.samples_per_problem, usage=usage
        )
        col = collect(
            sample_fn, plan, rng, exclude=test_keys,
            batch_size=cfg.collect_batch, fallback=args.arm == "bkey",
            max_batches=cfg.hard_max_batches if args.arm == "c" else cfg.collect_max_batches,
        )
        times["collect"] = round(time.time() - t, 1)
        free_memory()

        t = time.time()
        losses = train_round(model, tok, col.examples, cfg, rng)
        times["train"] = round(time.time() - t, 1)
        free_memory()

        t = time.time()
        acc, records = evaluate(model, tok, test)
        times["eval"] = round(time.time() - t, 1)
        model.save_pretrained(out / f"round{r}")
        free_memory()

        row = {
            "arm": args.arm,
            "round": r,
            "probe": probe_acc,
            "edge": edge,
            "plan": plan,
            "collect": {d: s.to_dict() for d, s in col.stats.items()},
            "kept": len(col.examples),
            "shortfall": col.shortfall,
            "fallback": col.fallback,
            "generated_tokens": usage.get("generated_tokens", 0),
            "loss_first": round(losses[0], 4) if losses else None,
            "loss_last": round(losses[-1], 4) if losses else None,
            "losses": [round(x, 4) for x in losses],
            "eval": acc,
            "eval_overall": overall(records),
            "seconds": times,
            "gpu_gb": gpu_gb(),
            "resumed": bool(start),
        }
        log(row)
        chosen = f"probe {fmt_acc(probe_acc)} · edge {edge}" if probe_acc else f"plan {plan}"
        print(
            f"round {r} · {chosen} · kept {len(col.examples)} (shortfall {col.shortfall}, fallback {col.fallback})"
            f" · loss {row['loss_first']}→{row['loss_last']} · held-out {row['eval_overall']:.1%} · {fmt_acc(acc)} · {times}"
        )

    print(f"done in {time.time() - t0:.0f}s · log {log_path}")


if __name__ == "__main__":
    main()
