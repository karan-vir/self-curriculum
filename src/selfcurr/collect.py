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
# completions per problem, and the tokens generated for each problem
CountedSampleFn = Callable[[Sequence[Problem]], tuple[list[list[str]], list[int]]]


@dataclass
class LengthStats:
    needed: int
    kept: int = 0
    problems: int = 0
    tries: int = 0
    correct_tries: int = 0
    solved_problems: int = 0
    fallback: int = 0

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

    @property
    def fallback(self) -> int:
        return sum(s.fallback for s in self.stats.values())


def collect(
    sample_fn: SampleFn,
    plan: dict[int, int],
    rng: random.Random,
    exclude: set[tuple[int, int]],
    batch_size: int = 64,
    max_batches: int = 50,
    fallback: bool = False,
) -> Collection:
    """Draw fresh problems per length until `plan[d]` examples are kept.

    One example per problem (its first correct try), so every kept example is
    a different problem. Problems in `exclude` (the held-out set) are never
    drawn. Stops a length after `max_batches` and records any shortfall.

    With `fallback` (Arm B), a problem the model never solves contributes the
    true answer instead, and `LengthStats.fallback` counts how often.
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
                if st.kept >= need:
                    continue
                if ok:
                    out.examples.append((p, ok[0]))
                    st.kept += 1
                elif fallback:
                    out.examples.append((p, str(p.answer)))
                    st.kept += 1
                    st.fallback += 1
            batches += 1
    return out


@dataclass
class BudgetCollection(Collection):
    budget: int = 0
    spent: int = 0  # tokens charged to the budget
    uncharged: int = 0  # tokens generated in a batch but thrown away unread
    uncharged_problems: int = 0
    stop: str = ""  # "budget", "plan_full" or "max_batches"
    tokens_by_length: dict[int, int] = field(default_factory=dict)


def collect_budget(
    sample_fn: CountedSampleFn,
    plan: dict[int, int],
    rng: random.Random,
    exclude: set[tuple[int, int]],
    budget: int,
    batch_size: int = 64,
    max_batches: int = 100_000,
) -> BudgetCollection:
    """Draw problems until `budget` tokens are spent or `plan` is filled.

    The result is the same as sampling one problem at a time; batches only make
    it faster. The rules are the same for every arm:

    - Lengths take turns, problem by problem, so a small budget is shared evenly
      across the plan rather than used up on the first lengths.
    - `plan[d]` caps the examples kept at length d. A full length stops being
      drawn, and its share of the budget goes to the others.
    - A problem is charged for all its tries, correct or not. The budget is a
      hard limit: the first problem that does not fit ends the round.
    - Problems after that one, or at a length that filled earlier in the batch,
      would not have been drawn one at a time. They are neither charged nor
      read: correctness is not looked at, and the tokens go in `uncharged`.
    - Once the plan is full the round stops, and unspent tokens are not carried
      over to the next round.

    `max_batches` is only a guard against a model that generates no tokens.
    """
    out = BudgetCollection(budget=budget)
    seen = set(exclude)
    lengths = sorted(plan)
    for d in lengths:
        out.stats[d] = LengthStats(needed=plan[d])
        out.tokens_by_length[d] = 0
    remaining = budget
    turn = 0
    batches = 0
    while not out.stop:
        active = [d for d in lengths if out.stats[d].kept < out.stats[d].needed]
        if not active:
            out.stop = "plan_full"
        elif remaining <= 0:
            out.stop = "budget"
        elif batches >= max_batches:
            out.stop = "max_batches"
        if out.stop:
            break
        problems = []
        for _ in range(batch_size):
            d = active[turn % len(active)]
            turn += 1
            problems.extend(make_problems(d, 1, rng, exclude=seen))
            seen.add(problems[-1].key)
        completions, token_counts = sample_fn(problems)
        for p, tries, tokens in zip(problems, completions, token_counts):
            st = out.stats[p.digits]
            if not out.stop and tokens > remaining:
                out.stop = "budget"
            if out.stop or st.kept >= st.needed:
                out.uncharged += tokens
                out.uncharged_problems += 1
                continue
            remaining -= tokens
            out.spent += tokens
            out.tokens_by_length[p.digits] += tokens
            ok = [c for c in tries if check(p, c)]
            st.problems += 1
            st.tries += len(tries)
            st.correct_tries += len(ok)
            st.solved_problems += bool(ok)
            if ok:
                out.examples.append((p, ok[0]))
                st.kept += 1
        batches += 1
    return out
