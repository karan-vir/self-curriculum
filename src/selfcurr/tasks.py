"""Addition task: problem generator, prompt format, answer parser and checker.

Difficulty is the number of digits in each operand. Answers are checked
exactly, so no human labels are needed.
"""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass

PROMPT_TEMPLATE = "Compute the sum. Reply with the number only.\n{a} + {b} ="

_INT_RE = re.compile(r"-?\d[\d,]*")


@dataclass(frozen=True)
class Problem:
    a: int
    b: int
    digits: int

    @property
    def answer(self) -> int:
        return self.a + self.b

    @property
    def key(self) -> tuple[int, int]:
        """Order-independent identity, used to keep training out of the test set."""
        return (min(self.a, self.b), max(self.a, self.b))

    def prompt(self) -> str:
        return PROMPT_TEMPLATE.format(a=self.a, b=self.b)

    def to_dict(self) -> dict:
        return {**asdict(self), "answer": self.answer}


def make_problem(digits: int, rng: random.Random) -> Problem:
    """Two random integers with exactly `digits` digits each."""
    if digits < 1:
        raise ValueError(f"digits must be >= 1, got {digits}")
    lo, hi = 10 ** (digits - 1), 10**digits - 1
    return Problem(rng.randint(lo, hi), rng.randint(lo, hi), digits)


def make_problems(
    digits: int,
    n: int,
    rng: random.Random,
    exclude: set[tuple[int, int]] | None = None,
) -> list[Problem]:
    """`n` distinct problems at one difficulty, skipping any key in `exclude`."""
    exclude = exclude or set()
    seen: set[tuple[int, int]] = set()
    out: list[Problem] = []
    attempts = 0
    while len(out) < n:
        attempts += 1
        if attempts > 100 * n:
            raise RuntimeError(f"could not draw {n} distinct {digits}-digit problems")
        p = make_problem(digits, rng)
        if p.key in exclude or p.key in seen:
            continue
        seen.add(p.key)
        out.append(p)
    return out


def parse_answer(completion: str) -> int | None:
    """First integer in the model's completion (commas allowed), or None."""
    m = _INT_RE.search(completion)
    if m is None:
        return None
    try:
        return int(m.group().replace(",", ""))
    except ValueError:
        return None


def check(problem: Problem, completion: str) -> bool:
    return parse_answer(completion) == problem.answer
