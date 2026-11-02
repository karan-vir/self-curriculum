import random

from selfcurr.collect import collect
from selfcurr.tasks import make_problems


def oracle(n_right: int, n_tries: int = 4):
    """Fake model: the first `n_right` of `n_tries` tries are correct."""
    def fn(problems):
        return [[str(p.answer)] * n_right + ["0"] * (n_tries - n_right) for p in problems]
    return fn


def test_collects_exactly_the_plan():
    col = collect(oracle(2), {5: 100, 6: 37}, random.Random(0), exclude=set(), batch_size=16)
    assert len(col.examples) == 137
    assert col.shortfall == 0
    assert sum(p.digits == 5 for p, _ in col.examples) == 100
    assert col.stats[6].kept == 37


def test_one_example_per_problem_and_all_correct():
    col = collect(oracle(4), {4: 50}, random.Random(0), exclude=set())
    keys = [p.key for p, _ in col.examples]
    assert len(set(keys)) == len(keys)
    assert all(int(c) == p.answer for p, c in col.examples)


def test_never_draws_held_out_problems():
    rng = random.Random(1)
    held_out = {p.key for p in make_problems(2, 500, rng)}
    col = collect(oracle(1), {2: 200}, random.Random(2), exclude=held_out)
    assert not held_out & {p.key for p, _ in col.examples}


def test_stats_count_tries():
    col = collect(oracle(1), {3: 10}, random.Random(0), exclude=set(), batch_size=10)
    st = col.stats[3]
    assert (st.problems, st.tries, st.correct_tries, st.solved_problems) == (10, 40, 10, 10)
    assert st.try_accuracy == 0.25 and st.solve_rate == 1.0


def test_shortfall_when_model_never_right():
    col = collect(oracle(0), {9: 5}, random.Random(0), exclude=set(), batch_size=8, max_batches=3)
    assert col.examples == []
    assert col.shortfall == 5
    assert col.stats[9].problems == 24
