"""LoRA fine-tuning on (problem, verified completion) pairs."""

from __future__ import annotations

import random
from collections.abc import Sequence

import torch
from peft import LoraConfig, get_peft_model

from selfcurr.config import Config
from selfcurr.model import chat_prompt
from selfcurr.tasks import Problem

END = "<|im_end|>"
IGNORE = -100
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def add_lora(model, cfg: Config):
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=LORA_TARGETS,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora)


def build_example(tok, problem: Problem, completion: str) -> tuple[list[int], list[int]]:
    """Token ids for prompt + answer, with loss only on the answer and end token."""
    prompt_ids = tok(chat_prompt(tok, problem), add_special_tokens=False)["input_ids"]
    target_ids = tok(completion.strip() + END, add_special_tokens=False)["input_ids"]
    return prompt_ids + target_ids, [IGNORE] * len(prompt_ids) + target_ids


def _collate(batch: Sequence[tuple[list[int], list[int]]], pad_id: int, device) -> dict[str, torch.Tensor]:
    width = max(len(ids) for ids, _ in batch)
    ids = torch.full((len(batch), width), pad_id)
    labels = torch.full((len(batch), width), IGNORE)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, : len(x)] = torch.tensor(x)
        labels[i, : len(y)] = torch.tensor(y)
        mask[i, : len(x)] = 1
    return {"input_ids": ids.to(device), "labels": labels.to(device), "attention_mask": mask.to(device)}


def train_round(
    model,
    tok,
    examples: Sequence[tuple[Problem, str]],
    cfg: Config,
    rng: random.Random,
) -> list[float]:
    """`cfg.epochs_per_round` passes over `examples`; returns the loss per step.

    A fresh AdamW each round, so rounds don't share optimizer state.
    """
    data = [build_example(tok, p, c) for p, c in examples]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.learning_rate)
    model.train()
    losses = []
    for _ in range(cfg.epochs_per_round):
        rng.shuffle(data)
        for i in range(0, len(data), cfg.batch_size):
            batch = _collate(data[i : i + cfg.batch_size], tok.pad_token_id, model.device)
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
    model.eval()
    return losses
