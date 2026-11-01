import random

import pytest

from selfcurr.curriculum import allocate, fixed_curriculum_plan, select_edge, self_curriculum_plan


def test_edge_picks_band():
    acc = {2: 1.0, 3: 0.95, 4: 0.6, 5: 0.35, 6: 0.1, 7: 0.0}
    assert select_edge(acc) == [4, 5]


def test_edge_band_is_inclusive():
    assert select_edge({2: 0.3, 3: 0.7, 4: 0.71}) == [2, 3]


def test_edge_fallback_on_cliff():
    # jumps from 95% straight to 20%: nothing in [0.3, 0.7]
    acc = {2: 1.0, 3: 0.95, 4: 0.2, 5: 0.0}
    assert select_edge(acc) == [4]


def test_edge_fallback_tie_goes_easier():
    assert select_edge({3: 0.8, 4: 0.2}) == [3]


def test_edge_empty_raises():
    with pytest.raises(ValueError):
        select_edge({})


@pytest.mark.parametrize("n", [0, 1, 7, 512, 513])
def test_allocate_sums_to_n_and_is_even(n):
    counts = allocate([2, 3, 4, 5, 6, 7, 8, 9, 10], n, random.Random(0))
    assert sum(counts.values()) == n
    assert max(counts.values()) - min(counts.values()) <= 1


def test_both_arms_get_same_budget():
    acc = {d: max(0.0, 1.0 - 0.2 * (d - 2)) for d in range(2, 11)}
    a = self_curriculum_plan(acc, 512, random.Random(0))
    b = fixed_curriculum_plan(list(range(2, 11)), 512, random.Random(0))
    assert sum(a.values()) == sum(b.values()) == 512
    assert set(b) == set(range(2, 11))
    assert set(a) <= set(range(2, 11))
