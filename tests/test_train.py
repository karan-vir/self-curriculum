import pytest

from selfcurr.tasks import Problem

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tok():
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    except OSError:
        pytest.skip("Qwen tokenizer not cached; run scripts/baseline.py once")


def test_loss_only_on_answer_and_end_token(tok):
    from selfcurr.train import IGNORE, build_example

    ids, labels = build_example(tok, Problem(23821, 28970, 5), " 52791\n")
    assert len(ids) == len(labels)
    target = [t for t in labels if t != IGNORE]
    assert tok.decode(target) == "52791<|im_end|>"
    # everything before the answer is masked, and the answer sits at the end
    n = len(target)
    assert all(t == IGNORE for t in labels[:-n])
    assert ids[-n:] == target
