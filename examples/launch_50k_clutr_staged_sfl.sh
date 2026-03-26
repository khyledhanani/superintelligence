#!/bin/bash
# 50k Experiment: CluttrVAE + staged PCA-CMA-ES + ACCEL + SFL learnability
# Same staged params as clutr-staged-accel, but with SFL scoring instead of MaxMC
# 3 seeds x 50k updates, WandB project JAXUED_LEARNABILITY, group clutr-staged-sfl
set -e

export LD_LIBRARY_PATH=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/lib:${LD_LIBRARY_PATH:-}
PYTHON=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python

COMMON="--project JAXUED_LEARNABILITY \
        --num_updates 50000 --eval_freq 250 \
        --skip_video --skip_post_eval"

VAE_CKPT="vae/runs/clutr_aligned/checkpoint_500000.pkl"
VAE_CONFIG="vae/runs/clutr_aligned/config.yaml"

export WANDB_DIR=/tmp/wandb
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
mkdir -p /tmp/jax_cache
mkdir -p logs

for seed in 0 1 2; do
  echo "=== [$(date)] Seed $seed starting ==="

  $PYTHON examples/maze_plr.py $COMMON \
    --use_cmaes --use_accel \
    --use_clutr_vae \
    --vae_checkpoint_path "$VAE_CKPT" \
    --vae_config_path "$VAE_CONFIG" \
    --use_pca_search \
    --pca_stage2_step 10000 \
    --pca_components 20 \
    --pca_sigma_init 0.5 \
    --cmaes_reset_interval 500 \
    --cmaes_reset_interval_post_pca 2000 \
    --pca_refit_interval 10000 \
    --score_function sfl \
    --seed $seed \
    --run_name "clutr-staged-sfl" \
    2>&1 | tee logs/50k_clutr_staged_sfl_seed${seed}.log

  echo "=== [$(date)] Seed $seed complete ==="
done

echo "All CluttrVAE staged + SFL experiments complete."
