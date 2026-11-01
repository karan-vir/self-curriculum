"""Write the frozen held-out test set to data/test_set.jsonl.

Run once. Training code loads these keys and rejects any training problem
that matches, so the test set never leaks into training.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from selfcurr.config import Config
from selfcurr.tasks import make_problems


def main() -> None:
    cfg = Config()
    rng = random.Random(cfg.test_seed)
    path = Path(__file__).resolve().parent.parent / cfg.test_path
    path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[tuple[int, int]] = set()
    with path.open("w") as f:
        for d in cfg.difficulties:
            for p in make_problems(d, cfg.test_per_difficulty, rng, exclude=seen):
                seen.add(p.key)
                f.write(json.dumps(p.to_dict()) + "\n")

    print(f"wrote {len(seen)} problems ({cfg.test_per_difficulty} x {len(cfg.difficulties)} difficulties) to {path}")


if __name__ == "__main__":
    main()
