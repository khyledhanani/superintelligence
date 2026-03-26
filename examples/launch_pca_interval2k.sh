#!/bin/bash
# PCA-space CMA-ES ABLATION: longer reset interval (2000 vs default 500)
# 5 seeds x 30k updates, logging to JAXUED_COMPARISON group pca-cmaes-accel-interval2k
# Same as launch_pca_comparison.sh except --cmaes_reset_interval 2000
set -e

# Use conda jax_env python + force conda CUDA libs (system CUDA 13.1 cuSOLVER is incompatible)
export LD_LIBRARY_PATH=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/lib:${LD_LIBRARY_PATH:-}
PYTHON=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python

COMMON="--project JAXUED_COMPARISON \
        --num_updates 30000 --eval_freq 250 \
        --skip_video --skip_post_eval"

# Redirect wandb to /tmp to avoid local disk bloat
export WANDB_DIR=/tmp/wandb

# Cache XLA compilations across seeds/runs
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
mkdir -p /tmp/jax_cache
mkdir -p logs

for seed in 0 1 2 3 4; do
  echo "=== [$(date)] Seed $seed starting ==="

  $PYTHON examples/maze_plr.py $COMMON \
    --use_cmaes --use_accel \
    --use_pca_search \
    --pca_stage2_step 10000 \
    --pca_components 20 \
    --pca_sigma_init 0.5 \
    --cmaes_reset_interval 2000 \
    --seed $seed \
    --run_name "pca-cmaes-accel-interval2k" \
    2>&1 | tee logs/pca_interval2k_seed${seed}.log

  echo "=== [$(date)] Seed $seed complete ==="
done

echo "All PCA interval-2k experiments complete."
