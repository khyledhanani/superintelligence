#!/bin/bash
# Launch training runs with configurable scoring method and generation mode.
#
# Usage:
#   bash scripts/launch_sfl_cenie.sh --method sfl              # ACCEL+SFL + CMA-ES+VAE+SFL
#   bash scripts/launch_sfl_cenie.sh --method cenie             # ACCEL+CENIE + CMA-ES+VAE+CENIE
#   bash scripts/launch_sfl_cenie.sh --method both              # All 8 runs
#   bash scripts/launch_sfl_cenie.sh --method cenie --no-vae    # ACCEL+CENIE only (no VAE runs)
#   bash scripts/launch_sfl_cenie.sh --method sfl --vae-only    # CMA-ES+VAE+SFL only

set -e

# --- Parse arguments ---
METHOD="both"
RUN_ACCEL=true
RUN_VAE=true
SEEDS="1 2"

while [[ $# -gt 0 ]]; do
    case $1 in
        --method) METHOD="$2"; shift 2 ;;
        --no-vae) RUN_VAE=false; shift ;;
        --vae-only) RUN_ACCEL=false; shift ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

VAE_DIR="vae/runs/beta2.0"
VAE_GCS="gs://ucl-ued-project-bucket/vae/runs/20260227_215731_lr5e-05_lat64_baseline_weighted_recon_model_beta1.0_beta2.0"
VAE_CKPT="${VAE_DIR}/checkpoint_80000.pkl"
VAE_CONFIG="${VAE_DIR}/config.yaml"

PROJECT="JAXUED_VAE_COMPARISON"
NUM_UPDATES=30000
GCS_BUCKET="ucl-ued-project-bucket"
GCS_PREFIX="accel"

COMMON_ARGS="--project $PROJECT --gcs_bucket $GCS_BUCKET --gcs_prefix $GCS_PREFIX --buffer_dump_interval 10000 --num_updates $NUM_UPDATES"
VAE_ARGS="--use_cmaes --cmaes_sigma_init 1.0 --cmaes_reset_interval 500 --save_cmaes_populations --vae_checkpoint_path $VAE_CKPT --vae_config_path $VAE_CONFIG"

# --- Download VAE if needed ---
if [ "$RUN_VAE" = true ]; then
    mkdir -p "${VAE_DIR}"
    if [ ! -f "${VAE_CKPT}" ]; then
        echo "[Setup] Downloading VAE checkpoint..."
        gcloud storage cp "${VAE_GCS}/checkpoints/checkpoint_80000.pkl" "${VAE_CKPT}"
    fi
    if [ ! -f "${VAE_CONFIG}" ]; then
        echo "[Setup] Downloading VAE config..."
        gcloud storage cp "${VAE_GCS}/config.yaml" "${VAE_CONFIG}"
    fi
fi

echo "============================================"
echo "  Method: ${METHOD} | ACCEL: ${RUN_ACCEL} | VAE: ${RUN_VAE}"
echo "  Seeds: ${SEEDS} | Updates: ${NUM_UPDATES}"
echo "============================================"

run_sfl_accel() {
    for SEED in $SEEDS; do
        echo -e "\n>>> ACCEL+SFL seed=${SEED} starting..."
        python examples/maze_plr.py \
            --seed $SEED \
            --run_name accel_sfl \
            --use_accel \
            --score_function sfl \
            --num_sfl_rollouts 10 \
            $COMMON_ARGS
        echo ">>> ACCEL+SFL seed=${SEED} done."
    done
}

run_cenie_accel() {
    for SEED in $SEEDS; do
        echo -e "\n>>> ACCEL+CENIE seed=${SEED} starting..."
        python examples/maze_plr.py \
            --seed $SEED \
            --run_name accel_cenie \
            --use_accel \
            --score_function cenie \
            --cenie_alpha 0.5 \
            --cenie_buffer_size 50000 \
            --cenie_num_components 10 \
            --cenie_refit_interval 5 \
            $COMMON_ARGS
        echo ">>> ACCEL+CENIE seed=${SEED} done."
    done
}

run_sfl_vae() {
    for SEED in $SEEDS; do
        echo -e "\n>>> CMA-ES+VAE+SFL seed=${SEED} starting..."
        python examples/maze_plr.py \
            --seed $SEED \
            --run_name cmaes_vae_beta2.0_sfl \
            --score_function sfl \
            --num_sfl_rollouts 10 \
            $VAE_ARGS \
            $COMMON_ARGS
        echo ">>> CMA-ES+VAE+SFL seed=${SEED} done."
    done
}

run_cenie_vae() {
    for SEED in $SEEDS; do
        echo -e "\n>>> CMA-ES+VAE+CENIE seed=${SEED} starting..."
        python examples/maze_plr.py \
            --seed $SEED \
            --run_name cmaes_vae_beta2.0_cenie \
            --score_function cenie \
            --cenie_alpha 0.5 \
            --cenie_buffer_size 50000 \
            --cenie_num_components 10 \
            --cenie_refit_interval 5 \
            $VAE_ARGS \
            $COMMON_ARGS
        echo ">>> CMA-ES+VAE+CENIE seed=${SEED} done."
    done
}

# --- Dispatch ---
if [ "$METHOD" = "sfl" ] || [ "$METHOD" = "both" ]; then
    [ "$RUN_ACCEL" = true ] && run_sfl_accel
    [ "$RUN_VAE" = true ] && run_sfl_vae
fi

if [ "$METHOD" = "cenie" ] || [ "$METHOD" = "both" ]; then
    [ "$RUN_ACCEL" = true ] && run_cenie_accel
    [ "$RUN_VAE" = true ] && run_cenie_vae
fi

echo -e "\n============================================"
echo "  All runs complete!"
echo "============================================"
