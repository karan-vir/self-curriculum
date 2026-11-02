"""Turn a difficulty plan into checker-verified training examples.

Model-free on purpose: `sample_fn` maps problems to lists of completions, so
the logic can be tested with a fake model.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from selfcurr.tasks import Problem, check, make_problems

SampleFn = Callable[[Sequence[Problem]], list[list[str]]]


@dataclass
class LengthStats:
    needed: int
    kept: int = 0
    problems: int = 0
    tries: int = 0
    correct_tries: int = 0
    solved_problems: int = 0

    @property
    def try_accuracy(self) -> float:
        return self.correct_tries / self.tries if self.tries else 0.0

    @property
    def solve_rate(self) -> float:
        return self.solved_problems / self.problems if self.problems else 0.0

    def to_dict(self) -> dict:
        return {**self.__dict__, "try_accuracy": round(self.try_accuracy, 4), "solve_rate": round(self.solve_rate, 4)}


@dataclass
class Collection:
    examples: list[tuple[Problem, str]] = field(default_factory=list)
    stats: dict[int, LengthStats] = field(default_factory=dict)

    @property
    def shortfall(self) -> int:
        return sum(s.needed - s.kept for s in self.stats.values())


def collect(
    sample_fn: SampleFn,
    plan: dict[int, int],
    rng: random.Random,
    exclude: set[tuple[int, int]],
    batch_size: int = 64,
    max_batches: int = 50,
) -> Collection:
    """Draw fresh problems per length until `plan[d]` verified examples are kept.

    One example per problem (its first correct try), so every kept example is
    a different problem. Problems in `exclude` (the held-out set) are never
    drawn. Stops a length after `max_batches` and records any shortfall.
    """
    out = Collection()
    seen = set(exclude)
    for d, need in sorted(plan.items()):
        st = out.stats[d] = LengthStats(needed=need)
        batches = 0
        while st.kept < need and batches < max_batches:
            problems = make_problems(d, batch_size, rng, exclude=seen)
            seen.update(p.key for p in problems)
            for p, tries in zip(problems, sample_fn(problems)):
                ok = [c for c in tries if check(p, c)]
                st.problems += 1
                st.tries += len(tries)
                st.correct_tries += len(ok)
                st.solved_problems += bool(ok)
                if ok and st.kept < need:
                    out.examples.append((p, ok[0]))
                    st.kept += 1
            batches += 1
    return out
