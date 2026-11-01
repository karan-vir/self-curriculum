"""How each arm picks the difficulties it trains on."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence


def select_edge(
    accuracy: Mapping[int, float],
    low: float = 0.3,
    high: float = 0.7,
) -> list[int]:
    """Arm A: difficulties whose accuracy lies in [low, high].

    If none qualify (e.g. a sharp cliff from 100% to 0%), fall back to the
    single difficulty closest to the band's midpoint; ties go to the easier one.
    """
    if not accuracy:
        raise ValueError("accuracy is empty")
    edge = sorted(d for d, acc in accuracy.items() if low <= acc <= high)
    if edge:
        return edge
    mid = (low + high) / 2
    # round so float noise (|0.8-0.5| vs |0.2-0.5|) doesn't break ties
    return [min(accuracy, key=lambda d: (round(abs(accuracy[d] - mid), 9), d))]


def allocate(
    difficulties: Sequence[int],
    n: int,
    rng: random.Random,
) -> dict[int, int]:
    """Spread `n` examples as evenly as possible over `difficulties`.

    The remainder goes to randomly chosen difficulties, so no level is
    systematically favoured across rounds.
    """
    if not difficulties:
        raise ValueError("difficulties is empty")
    base, rem = divmod(n, len(difficulties))
    counts = {d: base for d in difficulties}
    for d in rng.sample(list(difficulties), rem):
        counts[d] += 1
    return counts


def self_curriculum_plan(
    accuracy: Mapping[int, float],
    n: int,
    rng: random.Random,
    low: float = 0.3,
    high: float = 0.7,
) -> dict[int, int]:
    """Arm A: how many training examples to draw at each edge difficulty."""
    return allocate(select_edge(accuracy, low, high), n, rng)


def fixed_curriculum_plan(
    difficulties: Sequence[int],
    n: int,
    rng: random.Random,
) -> dict[int, int]:
    """Arm B: the same `n`, spread uniformly over every difficulty, every round."""
    return allocate(difficulties, n, rng)
