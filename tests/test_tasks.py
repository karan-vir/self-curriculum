import random

import pytest

from selfcurr.tasks import Problem, check, make_problem, make_problems, parse_answer


@pytest.mark.parametrize("digits", range(1, 11))
def test_operands_have_exact_digit_count(digits):
    rng = random.Random(0)
    for _ in range(50):
        p = make_problem(digits, rng)
        assert len(str(p.a)) == digits
        assert len(str(p.b)) == digits


def test_same_seed_same_problems():
    a = make_problems(5, 20, random.Random(7))
    b = make_problems(5, 20, random.Random(7))
    assert a == b


def test_make_problems_distinct_and_excluded():
    rng = random.Random(0)
    first = make_problems(3, 100, rng)
    keys = {p.key for p in first}
    assert len(keys) == 100
    second = make_problems(3, 100, rng, exclude=keys)
    assert keys.isdisjoint(p.key for p in second)


def test_key_ignores_operand_order():
    assert Problem(12, 34, 2).key == Problem(34, 12, 2).key


@pytest.mark.parametrize(
    "completion, expected",
    [
        ("5793", 5793),
        (" 5793\n", 5793),
        ("The answer is 5,793.", 5793),
        ("5793 because 1234 + 4559", 5793),
        ("I don't know", None),
        ("", None),
    ],
)
def test_parse_answer(completion, expected):
    assert parse_answer(completion) == expected


def test_check():
    p = Problem(1234, 4559, 4)
    assert check(p, " 5793")
    assert not check(p, " 5794")
    assert not check(p, "no idea")


def test_prompt_contains_operands():
    p = Problem(1234, 4559, 4)
    assert "1234 + 4559 =" in p.prompt()
