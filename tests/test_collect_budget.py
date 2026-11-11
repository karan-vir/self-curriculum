import random

import pytest
import torch

from selfcurr.collect import collect_budget
from selfcurr.config import Config
from selfcurr.curriculum import fixed_curriculum_plan
from selfcurr.model import generate
from selfcurr.tasks import Problem
from selfcurr.train import train_round

TOKENS_PER_TRY = 5


def counted(right=lambda p: True, n_tries: int = 4, tokens_per_try=lambda p: TOKENS_PER_TRY, calls: list | None = None):
    """Fake model: every try is right if `right(problem)`; cost by `tokens_per_try`."""
    def fn(problems):
        if calls is not None:
            calls.append(list(problems))
        completions = [[str(p.answer) if right(p) else "0"] * n_tries for p in problems]
        return completions, [n_tries * tokens_per_try(p) for p in problems]
    return fn


def test_stops_at_budget_and_never_goes_over():
    # 20 tokens a problem, so 1010 tokens buy exactly 50 problems and 10 are left over
    col = collect_budget(counted(), {5: 10_000}, random.Random(0), set(), budget=1010, batch_size=64)
    assert col.spent == 1000 <= col.budget
    assert col.stop == "budget"
    assert col.stats[5].problems == 50 and col.stats[5].tries == 200
    assert len(col.examples) == 50


def test_problem_that_does_not_fit_is_not_read_or_charged():
    col = collect_budget(counted(), {5: 10_000}, random.Random(0), set(), budget=30, batch_size=8)
    assert col.spent == 20 and len(col.examples) == 1
    assert col.uncharged == 7 * 20 and col.uncharged_problems == 7


def test_wrong_tries_cost_the_same_as_right_ones():
    col = collect_budget(counted(right=lambda p: False), {5: 10}, random.Random(0), set(), budget=400)
    assert col.spent == 400 and col.examples == [] and col.shortfall == 10
    assert col.stats[5].problems == 20


def test_stop_does_not_depend_on_which_tries_are_right():
    # the same draws and token costs must spend the same, whatever the model gets right
    a = collect_budget(counted(right=lambda p: False), {5: 10**6, 6: 10**6}, random.Random(3), set(), budget=500)
    b = collect_budget(counted(right=lambda p: p.a % 2 == 0), {5: 10**6, 6: 10**6}, random.Random(3), set(), budget=500)
    assert a.spent == b.spent
    assert {d: s.problems for d, s in a.stats.items()} == {d: s.problems for d, s in b.stats.items()}


def test_stops_when_plan_is_full_and_leaves_tokens_unspent():
    col = collect_budget(counted(), {5: 30, 6: 30}, random.Random(0), set(), budget=10**6)
    assert col.stop == "plan_full"
    assert len(col.examples) == 60
    assert col.spent == 60 * 20
    assert col.stats[5].kept == col.stats[6].kept == 30


def test_full_lengths_are_not_charged_for_the_rest_of_their_batch():
    # everything is solved, so one-at-a-time sampling would cost exactly one problem per example
    col = collect_budget(counted(), {4: 3}, random.Random(0), set(), budget=10**6, batch_size=64)
    assert col.stats[4].problems == 3
    assert col.spent == 60
    assert col.uncharged_problems == 61


def test_budget_runs_out_evenly_across_lengths():
    # a small budget must reach the long lengths, not stop in the first ones
    plan = {d: 57 for d in range(2, 11)}
    col = collect_budget(counted(), plan, random.Random(0), set(), budget=20 * 90, batch_size=64)
    assert [col.stats[d].problems for d in plan] == [10] * 9


def test_every_length_reached_at_tiny_budget_with_three_lengths():
    col = collect_budget(counted(), {8: 170, 9: 171, 10: 171}, random.Random(0), set(), budget=60, batch_size=64)
    assert [col.stats[d].problems for d in (8, 9, 10)] == [1, 1, 1]


def test_full_length_gives_its_share_to_the_others():
    col = collect_budget(
        counted(right=lambda p: p.digits == 2), {2: 5, 9: 1000}, random.Random(0), set(), budget=20 * 100, batch_size=10
    )
    assert col.stats[2].kept == 5
    assert col.stats[2].problems == 5 and col.stats[9].problems == 95


def test_cost_is_charged_per_problem_at_its_own_length():
    col = collect_budget(
        counted(tokens_per_try=lambda p: p.digits + 2), {2: 1000, 10: 1000}, random.Random(0), set(), budget=4 * (4 + 12) * 5
    )
    assert col.tokens_by_length == {2: 4 * 4 * 5, 10: 4 * 12 * 5}


def test_no_budget_left_means_no_sampling():
    # Arm A when the probe has used everything: the fake model must never be called
    calls: list = []
    for left in (0, -500):
        col = collect_budget(counted(calls=calls), {5: 10}, random.Random(0), set(), budget=left)
        assert col.examples == [] and col.spent == 0 and col.stop == "budget"
        assert col.shortfall == 10
    assert calls == []


def test_never_draws_held_out_or_repeats():
    rng = random.Random(1)
    from selfcurr.tasks import make_problems

    held_out = {p.key for p in make_problems(2, 1000, rng)}
    col = collect_budget(counted(), {2: 300}, random.Random(2), held_out, budget=10**6)
    keys = [p.key for p, _ in col.examples]
    assert len(set(keys)) == 300 and not held_out & set(keys)


def test_keeps_first_correct_try_only():
    def fn(problems):
        return [["0", str(p.answer), str(p.answer), "0"] for p in problems], [20] * len(problems)
    col = collect_budget(fn, {5: 4}, random.Random(0), set(), budget=10**6, batch_size=4)
    assert len(col.examples) == 4
    st = col.stats[5]
    assert (st.tries, st.correct_tries, st.solved_problems) == (16, 8, 4)


def test_model_generating_nothing_cannot_loop_forever():
    fn = lambda ps: ([["0"] * 4 for _ in ps], [0] * len(ps))  # noqa: E731
    col = collect_budget(fn, {5: 10}, random.Random(0), set(), budget=100, batch_size=4, max_batches=3)
    assert col.stop == "max_batches" and col.stats[5].problems == 12


def test_plan_sums_to_cap_for_every_arm():
    cfg = Config()
    for lengths in (cfg.difficulties, cfg.hard_digits, [5, 6]):
        assert sum(fixed_curriculum_plan(lengths, cfg.examples_per_round, random.Random(0)).values()) == 512


def test_budget_is_off_by_default():
    assert Config().token_budget is None


def test_training_on_nothing_changes_nothing():
    model = torch.nn.Linear(2, 2)
    before = [p.clone() for p in model.parameters()]
    assert train_round(model, None, [], Config(), random.Random(0)) == []
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeTok:
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def __call__(self, texts, **kwargs):
        return FakeBatch(input_ids=torch.ones((len(texts), 3), dtype=torch.long))

    def batch_decode(self, gen, **kwargs):
        return [str(int((row != 0).sum())) for row in gen]


class FakeModel:
    device = "cpu"

    def generate(self, input_ids, num_return_sequences, **kwargs):
        # try number k of every problem generates k+1 tokens, then padding
        n = input_ids.shape[0] * num_return_sequences
        new = torch.zeros((n, 6), dtype=torch.long)
        for i in range(n):
            new[i, : i % num_return_sequences + 1] = 7
        return torch.cat([input_ids.repeat_interleave(num_return_sequences, 0), new], dim=1)


@pytest.mark.parametrize("batch_size", [2, 64])
def test_token_counts_per_problem_match_usage(batch_size):
    problems = [Problem(10 + i, 20, 2) for i in range(5)]
    usage: dict = {}
    counts: list[int] = []
    out = generate(FakeModel(), FakeTok(), problems, batch_size=batch_size, num_samples=4, usage=usage, token_counts=counts)
    assert counts == [1 + 2 + 3 + 4] * 5
    assert usage["generated_tokens"] == sum(counts)
    assert [len(tries) for tries in out] == [4] * 5


def test_generate_leaves_token_counts_alone_by_default():
    usage: dict = {}
    generate(FakeModel(), FakeTok(), [Problem(10, 20, 2)], num_samples=4, usage=usage)
    assert usage == {"generated_tokens": 10}


def _run_arm():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("run_arm", Path(__file__).parent.parent / "scripts" / "run_arm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("arm,budget,sub", [("a", 100, "runs/arm_a/seed0"), ("c", 100, "runs/arm_bkey/x/y"), ("bkey", 100, "runs/other"), ("a", 0, "runs/other")])
def test_budget_run_refuses_unsafe_settings(arm, budget, sub):
    import argparse

    run_arm = _run_arm()
    with pytest.raises(SystemExit):
        run_arm.check_budget_run(argparse.ArgumentParser(), arm, budget, run_arm.ROOT / sub, resume=False)


def test_budget_run_will_not_overwrite_a_log(tmp_path):
    import argparse

    run_arm = _run_arm()
    run_arm.check_budget_run(argparse.ArgumentParser(), "a", 100, tmp_path, resume=False)  # empty dir is fine
    (tmp_path / "log.jsonl").write_text('{"round": 0}\n')
    with pytest.raises(SystemExit):
        run_arm.check_budget_run(argparse.ArgumentParser(), "a", 100, tmp_path, resume=False)
    run_arm.check_budget_run(argparse.ArgumentParser(), "a", 100, tmp_path, resume=True)
