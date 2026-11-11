#!/bin/zsh
# Token-budget pilot: seed 0, Arms A, B, C, bottom and top budgets. Output only under runs/budget_pilot/.
cd "$(git rev-parse --show-toplevel)" || exit 1
for b in 4125 213093; do
  for arm in a b c; do
    echo "=== $(date +%H:%M) budget $b arm $arm"
    uv run python scripts/run_arm.py --arm $arm --seed 0 --token-budget $b --out runs/budget_pilot/b$b/arm_$arm/seed0 \
      > runs/budget_pilot/b${b}_arm_$arm.out 2>&1
    echo "=== $(date +%H:%M) exit $? budget $b arm $arm"
  done
done
echo "=== $(date +%H:%M) pilot done"
