#!/bin/bash
# 50k Experiment: Vanilla ACCEL baseline
# 3 seeds x 50k updates, WandB project JAXUED_50K, group accel-baseline
set -e

# Use conda jax_env python (TPU: /home/gmaralla, GPU: use LD_LIBRARY_PATH version)
PYTHON=/home/gmaralla/miniconda3/envs/jax_env/bin/python

COMMON="--project JAXUED_50K \
        --num_updates 50000 --eval_freq 250 \
        --skip_video --skip_post_eval"

export WANDB_DIR=/tmp/wandb
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
mkdir -p /tmp/jax_cache
mkdir -p logs

for seed in 0 1 2; do
  echo "=== [$(date)] Seed $seed starting ==="

  $PYTHON examples/maze_plr.py $COMMON \
    --use_accel \
    --seed $seed \
    --run_name "accel-baseline" \
    2>&1 | tee logs/50k_accel_baseline_seed${seed}.log

  echo "=== [$(date)] Seed $seed complete ==="
done

echo "All ACCEL baseline experiments complete."
