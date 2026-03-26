#!/bin/bash
# 50k Experiment: PCA-CMA-ES with staged reset interval, NO ACCEL mutation
# Same as pca-staged-accel but without --use_accel
# Stage 1 (0-10k): CMA-ES reset every 500
# Stage 2 (10k+): CMA-ES reset every 2k
# PCA refit every 10k
# 3 seeds x 50k updates, WandB project JAXUED_50K, group pca-staged-no-accel
set -e

# Force conda CUDA libs (system CUDA 13.1 cuSOLVER is incompatible)
export LD_LIBRARY_PATH=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/lib:${LD_LIBRARY_PATH:-}
PYTHON=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python

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
    --use_cmaes \
    --use_pca_search \
    --pca_stage2_step 10000 \
    --pca_components 20 \
    --pca_sigma_init 0.5 \
    --cmaes_reset_interval 500 \
    --cmaes_reset_interval_post_pca 2000 \
    --pca_refit_interval 10000 \
    --seed $seed \
    --run_name "pca-staged-no-accel" \
    2>&1 | tee logs/50k_pca_staged_no_accel_seed${seed}.log

  echo "=== [$(date)] Seed $seed complete ==="
done

echo "All PCA staged (no ACCEL) experiments complete."
