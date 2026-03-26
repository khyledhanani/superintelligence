#!/bin/bash
# 50k Experiment: NEW CNN-VAE (1M steps) + staged PCA-CMA-ES + ACCEL
# Seed 0 only — run on one GPU, same WandB group as seeds 1-2
set -e

export LD_LIBRARY_PATH=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/lib:${LD_LIBRARY_PATH:-}
PYTHON=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python

COMMON="--project JAXUED_50K \
        --num_updates 50000 --eval_freq 250 \
        --skip_video --skip_post_eval"

CNN_CKPT="vae/checkpoints/cnn_vae/run11_1M/default"

export WANDB_DIR=/tmp/wandb
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
mkdir -p /tmp/jax_cache
mkdir -p logs

echo "=== [$(date)] Seed 0 starting ==="

$PYTHON examples/maze_plr.py $COMMON \
  --use_cmaes --use_accel \
  --vae_checkpoint_path "$CNN_CKPT" \
  --use_pca_search \
  --pca_stage2_step 10000 \
  --pca_components 20 \
  --pca_sigma_init 0.5 \
  --cmaes_reset_interval 500 \
  --cmaes_reset_interval_post_pca 2000 \
  --pca_refit_interval 10000 \
  --seed 0 \
  --run_name "cnn1M-staged-accel" \
  2>&1 | tee logs/50k_cnn1M_staged_seed0.log

echo "=== [$(date)] Seed 0 complete ==="
