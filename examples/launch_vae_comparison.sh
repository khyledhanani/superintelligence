#!/bin/bash
# ==============================================================================
# VAE Comparison: Plain ACCEL vs CMA-ES+ACCEL with multiple VAEs
#
# Runs 4 conditions × 3 seeds = 12 runs sequentially on a single TPU.
#   1. Plain ACCEL (baseline, no VAE)
#   2. CMA-ES + ACCEL with VAE #1
#   3. CMA-ES + ACCEL with VAE #2
#   4. CMA-ES + ACCEL with VAE #3
#
# Each run: 50k PPO updates (~410M env steps), eval every 250 updates.
# All checkpoints, buffer dumps, and plots saved to GCS.
# Stats visible per-seed on wandb (grouped by run_name).
#
# Usage:
#   bash examples/launch_vae_comparison.sh
#
# Before running, fill in the VAE arrays below.
# ==============================================================================
set -e

# === GCS CONFIG ===
BUCKET="ucl-ued-project-bucket"
PREFIX="accel"

# === WANDB CONFIG ===
PROJECT="JAXUED_VAE_COMPARISON"
export WANDB_DIR=/tmp/wandb

# === TRAINING CONFIG ===
NUM_UPDATES=50000
EVAL_FREQ=250
SEEDS=(0 1 2)

# === VAE CONFIGURATIONS ===
# Fill in your VAE run names, checkpoint paths, and config paths.
# Arrays must have the same length. Add or remove entries as needed.

VAE_NAMES=(
    "vae_beta1.0"
    "vae_beta1.5"
    "vae_beta2.0"
)

VAE_CKPTS=(
    "vae/runs/runs/20260227_185835_lr5e-05_lat64_baseline_weighted_recon_model_beta1.0_beta1.0/checkpoints/checkpoint_260000.pkl"
    "vae/runs/runs/20260227_202754_lr5e-05_lat64_baseline_weighted_recon_model_beta1.0_beta1.5/checkpoints/checkpoint_120000.pkl"
    "vae/runs/runs/20260227_215731_lr5e-05_lat64_baseline_weighted_recon_model_beta1.0_beta2.0/checkpoints/checkpoint_80000.pkl"
)

VAE_CFGS=(
    "vae/runs/runs/20260227_185835_lr5e-05_lat64_baseline_weighted_recon_model_beta1.0_beta1.0/config.yaml"
    "vae/runs/runs/20260227_202754_lr5e-05_lat64_baseline_weighted_recon_model_beta1.0_beta1.5/config.yaml"
    "vae/runs/runs/20260227_215731_lr5e-05_lat64_baseline_weighted_recon_model_beta1.0_beta2.0/config.yaml"
)

# ==============================================================================
# Common training args
# ==============================================================================
COMMON="--project $PROJECT \
        --num_updates $NUM_UPDATES --eval_freq $EVAL_FREQ \
        --gcs_bucket $BUCKET --gcs_prefix $PREFIX \
        --buffer_dump_interval 10000"

NUM_VAES=${#VAE_NAMES[@]}
NUM_SEEDS=${#SEEDS[@]}
TOTAL_RUNS=$(( (1 + NUM_VAES) * NUM_SEEDS ))
RUN_NUM=0

echo "============================================"
echo "  VAE Comparison Experiment"
echo "  Conditions: 1 (plain ACCEL) + $NUM_VAES (CMA-ES VAEs)"
echo "  Seeds: ${SEEDS[*]}"
echo "  Total runs: $TOTAL_RUNS"
echo "  Updates per run: $NUM_UPDATES"
echo "  GCS: gs://$BUCKET/$PREFIX/"
echo "  wandb project: $PROJECT"
echo "============================================"

# ==============================================================================
# Condition 1: Plain ACCEL (baseline)
# ==============================================================================
# for seed in 1 2; do  # seed 0 already completed
#     RUN_NUM=$((RUN_NUM + 1))
#     echo ""
#     echo "[$RUN_NUM/$TOTAL_RUNS] Plain ACCEL | seed=$seed"
#     echo "----------------------------------------------"
#
#     PYTHONUNBUFFERED=1 python3 examples/maze_plr.py $COMMON \
#         --use_accel \
#         --seed $seed \
#         --run_name plain_accel
#
#     echo "[DONE] Plain ACCEL seed=$seed"
# done

# ==============================================================================
# Conditions 2-N: CMA-ES + ACCEL with each VAE
# ==============================================================================
for ((i=0; i<NUM_VAES; i++)); do
    VAE_NAME="${VAE_NAMES[$i]}"
    VAE_CKPT="${VAE_CKPTS[$i]}"
    VAE_CFG="${VAE_CFGS[$i]}"

    # Verify files exist before starting
    if [ ! -f "$VAE_CKPT" ]; then
        echo "[ERROR] VAE checkpoint not found: $VAE_CKPT"
        echo "        Skipping $VAE_NAME"
        RUN_NUM=$((RUN_NUM + NUM_SEEDS))
        continue
    fi
    if [ ! -f "$VAE_CFG" ]; then
        echo "[ERROR] VAE config not found: $VAE_CFG"
        echo "        Skipping $VAE_NAME"
        RUN_NUM=$((RUN_NUM + NUM_SEEDS))
        continue
    fi

    for seed in "${SEEDS[@]}"; do
        RUN_NUM=$((RUN_NUM + 1))
        echo ""
        echo "[$RUN_NUM/$TOTAL_RUNS] CMA-ES + ACCEL ($VAE_NAME) | seed=$seed"
        echo "----------------------------------------------"

        PYTHONUNBUFFERED=1 python3 examples/maze_plr.py $COMMON \
            --use_cmaes --use_accel \
            --vae_checkpoint_path "$VAE_CKPT" \
            --vae_config_path "$VAE_CFG" \
            --seed $seed \
            --run_name "cmaes_${VAE_NAME}"

        echo "[DONE] CMA-ES ($VAE_NAME) seed=$seed"
    done
done

echo ""
echo "============================================"
echo "  All $TOTAL_RUNS runs complete!"
echo "  View results: https://wandb.ai/ (project: $PROJECT)"
echo "  GCS artifacts: gs://$BUCKET/$PREFIX/"
echo "============================================"
