#!/bin/bash
# Three-way comparison: CMA-ES+ACCEL vs CMA-ES+PLR vs Vanilla ACCEL
# Run sequentially on single TPU (one seed at a time)
set -e

BUCKET="ucl-ued-project-bucket"
PREFIX="accel"

# === UPDATE THESE PATHS ===
VAE_CKPT="vae/runs/<YOUR_RUN>/checkpoints/checkpoint_260000.pkl"
VAE_CFG="vae/runs/<YOUR_RUN>/config.yaml"

# Common args matching ACCEL paper (facebookresearch/dcd)
COMMON="--project JAXUED_COMPARISON \
        --num_updates 30000 --eval_freq 250 \
        --entropy_coeff 0.0 --gae_lambda 0.95 --score_function pvl \
        --gcs_bucket $BUCKET --gcs_prefix $PREFIX"

# Redirect wandb to /tmp to avoid local disk bloat
export WANDB_DIR=/tmp/wandb

for seed in 0 1 2 3 4; do
  echo "=== Seed $seed ==="

  # Condition 1: CMA-ES + ACCEL (your method)
  echo "[Seed $seed] Running CMA-ES + ACCEL..."
  python3 examples/maze_plr.py $COMMON --use_cmaes --use_accel \
    --vae_checkpoint_path "$VAE_CKPT" --vae_config_path "$VAE_CFG" \
    --seed $seed --run_name cmaes_accel

  # Condition 2: CMA-ES + PLR only (no mutation)
  echo "[Seed $seed] Running CMA-ES + PLR only..."
  python3 examples/maze_plr.py $COMMON --use_cmaes \
    --vae_checkpoint_path "$VAE_CKPT" --vae_config_path "$VAE_CFG" \
    --seed $seed --run_name cmaes_plr_only

  # Condition 3: Vanilla ACCEL (baseline)
  echo "[Seed $seed] Running Vanilla ACCEL..."
  python3 examples/maze_plr.py $COMMON --use_accel \
    --seed $seed --run_name vanilla_accel

  echo "=== Seed $seed complete ==="
done

echo "All experiments complete."
