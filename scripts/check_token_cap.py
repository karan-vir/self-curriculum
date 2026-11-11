"""Does the 16-token cap on held-out replies cut off any answers? No training.

    uv run python scripts/check_token_cap.py

Every round-10 adapter (Arms A, B, B+key and C, seeds 0-9) is re-run greedily on
the 1,800 held-out problems in data/test_set.jsonl, in the same batches as the
logged evaluation, but with 32 new tokens instead of 16. Greedy decoding is
step by step, so the first 16 tokens are the reply the logged evaluation saw.

A reply "reaches the cap" when the model has not ended its turn within the
first 16 generated tokens. Such a reply "loses part of the number" when the
first integer parsed from its first 16 tokens differs from the first integer
parsed from the 32-token reply (the number was cut, or not reached at all).
As a sanity check, accuracy from the first 16 tokens must equal the logged
round-10 accuracy for every arm, seed and length.

Raw replies go to runs/token_cap/ (gitignored); the summary goes to
results/token_cap_check.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from peft import PeftModel

from selfcurr.config import Config
from selfcurr.model import chat_prompt, load_model, load_test_set, pick_device
from selfcurr.tasks import parse_answer

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "runs" / "token_cap"
ARMS = ["a", "b", "bkey", "c"]
SEEDS = range(10)
CAP, LONGER, BATCH = 16, 32, 64


@torch.no_grad()
def replies(model, tok, problems) -> list[list[int]]:
    """Generated token ids per problem, without padding, end-of-turn kept."""
    out = []
    for i in range(0, len(problems), BATCH):
        batch = problems[i : i + BATCH]
        enc = tok([chat_prompt(tok, p) for p in batch], return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(
            **enc, max_new_tokens=LONGER, pad_token_id=tok.pad_token_id,
            do_sample=False, temperature=None, top_p=None, top_k=None,
        )[:, enc["input_ids"].shape[1] :]
        for row in gen.tolist():
            out.append([t for t in row if t != tok.pad_token_id])
    return out


def collect_one(arm: str, seed: int, device: str, problems) -> None:
    out = RAW / f"{arm}_seed{seed}.json"
    if out.exists():
        return
    t0 = time.time()
    model, tok = load_model(Config().model_name, device, dtype=torch.float32)
    model = PeftModel.from_pretrained(model, ROOT / f"runs/arm_{arm}/seed{seed}/round10")
    model.eval()
    ids = replies(model, tok, problems)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"ids": ids}))
    tmp.rename(out)
    print(f"{arm} seed {seed}: {time.time() - t0:.0f}s", flush=True)
    del model
    if device == "mps":
        torch.mps.empty_cache()


def logged_round10(arm: str, seed: int) -> dict:
    rows = [json.loads(l) for l in (ROOT / f"runs/arm_{arm}/seed{seed}/log.jsonl").read_text().splitlines() if l.strip()]
    return next(r for r in rows if r["round"] == 10)["eval"]


def summarise(problems) -> dict:
    _, tok = load_model(Config().model_name, "cpu", dtype=torch.float32)
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    arms, mismatches, examples = {}, [], []
    for arm in ARMS:
        tot = {"replies": 0, "reach_cap": 0, "lose_number": 0, "correctness_changes": 0, "by_seed": {}}
        for seed in SEEDS:
            ids = json.loads((RAW / f"{arm}_seed{seed}.json").read_text())["ids"]
            reach = lose = flips = 0
            right16: dict[int, int] = {}
            for p, r in zip(problems, ids):
                first16 = r[:CAP]
                text16 = tok.decode(first16, skip_special_tokens=True)
                text32 = tok.decode(r, skip_special_tokens=True)
                ok16 = parse_answer(text16) == p.answer
                right16[p.digits] = right16.get(p.digits, 0) + ok16
                if not any(t in eos for t in first16):
                    reach += 1
                    if parse_answer(text16) != parse_answer(text32):
                        lose += 1
                        if len(examples) < 20:
                            examples.append({"arm": arm, "seed": seed, "a": p.a, "b": p.b, "first_16_tokens": text16, "first_32_tokens": text32})
                    flips += ok16 != (parse_answer(text32) == p.answer)
            logged = logged_round10(arm, seed)
            for d, k in right16.items():
                if abs(k / 200 - logged[str(d)]) > 1e-9:
                    mismatches.append({"arm": arm, "seed": seed, "digits": d, "logged": logged[str(d)], "first_16": k / 200})
            tot["by_seed"][seed] = {"reach_cap": reach, "lose_number": lose}
            tot["replies"] += len(ids)
            tot["reach_cap"] += reach
            tot["lose_number"] += lose
            tot["correctness_changes"] += flips
        arms[arm] = tot
    return {
        "what": "round-10 held-out replies (data/test_set.jsonl, 1,800 problems) re-generated greedily with 32 new tokens",
        "reach_cap": "no end-of-turn token within the first 16 generated tokens",
        "lose_number": "first integer parsed from the first 16 tokens differs from the one parsed from 32 tokens",
        "correctness_changes": "replies whose correctness differs between 16 and 32 tokens",
        "arms": arms,
        "sanity_first16_vs_logged": {"checked": len(ARMS) * len(SEEDS) * 9, "mismatches": mismatches},
        "examples_losing_number": examples,
    }


def main() -> None:
    device = pick_device()
    problems = load_test_set(ROOT / "data/test_set.jsonl")
    RAW.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        for arm in ARMS:
            collect_one(arm, seed, device, problems)
    out = ROOT / "results/token_cap_check.json"
    out.write_text(json.dumps(summarise(problems), indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
