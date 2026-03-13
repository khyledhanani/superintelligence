#!/bin/bash
# PCA-space CMA-ES comparison run (Phase 6)
# 5 seeds x 30k updates, logging to JAXUED_COMPARISON group pca-cmaes-accel
# Mirrors examples/launch_comparison.sh — adds --use_pca_search flags only
set -e

COMMON="--project JAXUED_COMPARISON \
        --num_updates 30000 --eval_freq 250 \
        --skip_video --skip_post_eval"

# Redirect wandb to /tmp to avoid local disk bloat
export WANDB_DIR=/tmp/wandb

# Cache XLA compilations across seeds/runs (critical for 5-seed run)
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
mkdir -p /tmp/jax_cache
mkdir -p logs

for seed in 0 1 2 3 4; do
  echo "=== [$(date)] Seed $seed starting ==="

  python3 examples/maze_plr.py $COMMON \
    --use_cmaes --use_accel \
    --use_pca_search \
    --pca_stage2_step 10000 \
    --pca_components 20 \
    --pca_sigma_init 0.5 \
    --seed $seed \
    --run_name "pca-cmaes-accel" \
    2>&1 | tee logs/pca_comparison_seed${seed}.log

  echo "=== [$(date)] Seed $seed complete ==="
done

echo "All PCA experiments complete."
