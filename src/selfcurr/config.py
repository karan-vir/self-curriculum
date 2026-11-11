"""Experiment settings shared by both arms."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # task
    min_digits: int = 2
    max_digits: int = 10

    # held-out evaluation (frozen, see scripts/make_test_set.py)
    test_per_difficulty: int = 200
    test_seed: int = 1234
    test_path: str = "data/test_set.jsonl"

    # rounds and budget, matched across arms
    rounds: int = 10
    examples_per_round: int = 512
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])

    # Arm A (64 gives about ±12 points at 50%; see reports/baseline.html)
    probe_per_difficulty: int = 64
    edge_low: float = 0.3
    edge_high: float = 0.7
    samples_per_problem: int = 4
    sample_temperature: float = 0.8
    collect_batch: int = 64
    collect_max_batches: int = 50

    # Arm C (long sums are rarely solved early on, so it draws for longer before giving up)
    hard_digits: list[int] = field(default_factory=lambda: [8, 9, 10])
    hard_max_batches: int = 200

    # Token-budget mode: when set, every round samples until this many generated
    # tokens are spent (probes included) instead of until the plan is filled.
    # None keeps the original fill-the-plan behaviour. See collect_budget.
    token_budget: int | None = None

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    batch_size: int = 16
    epochs_per_round: int = 1

    @property
    def difficulties(self) -> list[int]:
        return list(range(self.min_digits, self.max_digits + 1))
