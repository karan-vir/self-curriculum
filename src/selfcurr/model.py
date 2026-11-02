"""Model loading and batched generation, shared by evaluation and both arms."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from selfcurr.tasks import Problem, check


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name: str, device: str | None = None, dtype: torch.dtype | None = None):
    """Half precision by default for inference; pass float32 to train on MPS."""
    device = device or pick_device()
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if dtype is None:
        dtype = torch.bfloat16 if device == "cuda" else torch.float16 if device == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device)
    model.eval()
    return model, tok


def chat_prompt(tok, problem: Problem) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": problem.prompt()}],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.no_grad()
def generate(
    model,
    tok,
    problems: Sequence[Problem],
    batch_size: int = 64,
    max_new_tokens: int = 16,
    temperature: float = 0.0,
    num_samples: int = 1,
    usage: dict | None = None,
) -> list[list[str]]:
    """Completions per problem: `num_samples` each, greedy when temperature is 0.

    If `usage` is given, its "generated_tokens" count is increased in place.
    """
    out: list[list[str]] = []
    for i in range(0, len(problems), batch_size):
        batch = problems[i : i + batch_size]
        enc = tok([chat_prompt(tok, p) for p in batch], return_tensors="pt", padding=True).to(model.device)
        kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, num_return_sequences=num_samples)
        if temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
        else:
            kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        gen = model.generate(**enc, **kwargs)[:, enc["input_ids"].shape[1] :]
        if usage is not None:
            usage["generated_tokens"] = usage.get("generated_tokens", 0) + int((gen != tok.pad_token_id).sum())
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        out.extend(texts[j : j + num_samples] for j in range(0, len(texts), num_samples))
    return out


def evaluate(
    model, tok, problems: Sequence[Problem], batch_size: int = 64, usage: dict | None = None
) -> tuple[dict[int, float], list[dict]]:
    """Greedy accuracy per difficulty, plus one record per problem."""
    completions = generate(model, tok, problems, batch_size=batch_size, usage=usage)
    correct: dict[int, list[bool]] = defaultdict(list)
    records = []
    for p, (c,) in zip(problems, completions):
        ok = check(p, c)
        correct[p.digits].append(ok)
        records.append({**p.to_dict(), "completion": c, "correct": ok})
    acc = {d: sum(v) / len(v) for d, v in sorted(correct.items())}
    return acc, records


def load_test_set(path: str | Path) -> list[Problem]:
    with Path(path).open() as f:
        return [Problem(r["a"], r["b"], r["digits"]) for r in map(json.loads, f)]
