#!/bin/bash
# VAE interpolation mutation experiment
#
# Full pipeline: fetch data from GCS → generate eligible pool → create merged
# buffers at each injection % → resume training from 10k checkpoint for 10k
# more updates.
#
# Usage: bash experiments/gpu_scripts/run_vae_interp.sh [target_eligible] [inject_pcts] [seeds...]
# Example: bash experiments/gpu_scripts/run_vae_interp.sh 1000 5,10,15,20,25 0 1 2

set -e
export WANDB_ENTITY=romain-hautier-university-college-london-ucl-

TARGET_ELIGIBLE=${1:-1000}; shift || true
INJECT_PCTS=${1:-5,10,15,20,25}; shift || true
SEEDS=("${@}")
if [ ${#SEEDS[@]} -eq 0 ]; then SEEDS=(0 1 2); fi

PY=/cs/student/msc/csml/2025/rhautier/miniforge3/envs/jaxued_env/bin/python
SCRIPT=/cs/student/msc/csml/2025/rhautier/Documents/jaxued/jaxued/examples/maze_plr.py
REPO_ROOT=/cs/student/msc/csml/2025/rhautier/Documents/jaxued/jaxued

GCS_BUCKET="ucl-ued-project-bucket"
GCS_PROJECT="ucl-ued-project"
GCS_PREFIX="llm-exp/injection"
LOCAL_DATA="/tmp/injection_data"
TRAIN_UPDATES=10000
# WANDB_GROUP is set per injection % inside the loop

VAE_CKPT="${REPO_ROOT}/vae/runs/runs/20260309_091933_lr5e-05_lat64_clutr_aligned_rw1000_beta10.0/checkpoints/checkpoint_500000.pkl"
VAE_CONFIG="${REPO_ROOT}/vae/runs/runs/20260309_091933_lr5e-05_lat64_clutr_aligned_rw1000_beta10.0/config.yaml"

# --- Step 0: Fetch data from GCS ---
echo "=== Fetching data from GCS ==="
$PY "${REPO_ROOT}/experiments/fetch_gcs_data.py" \
    --gcs_bucket "$GCS_BUCKET" \
    --gcs_project "$GCS_PROJECT" \
    --local_dir "$LOCAL_DATA" \
    --checkpoint_step 39 \
    --buffer_update 10000

AGENT_CKPT="${LOCAL_DATA}/checkpoint"
BUFFER_NPZ="${LOCAL_DATA}/buffer/buffer_dump_10000.npz"
SEEDS_DIR="${LOCAL_DATA}/seeds"
BASE_OUTPUT="${LOCAL_DATA}/results"

echo ""
echo "=== VAE Interpolation Mutation ==="
echo "  target_eligible=$TARGET_ELIGIBLE  inject_pcts=$INJECT_PCTS"
echo "  seeds=${SEEDS[*]}  train_updates=$TRAIN_UPDATES"

for SEED in "${SEEDS[@]}"; do
    RUN_NAME="vae_interp_t${TARGET_ELIGIBLE}_seed${SEED}"
    OUTPUT_DIR="${BASE_OUTPUT}/${RUN_NAME}"

    # --- Step 1: Generate eligible pool + merged buffers ---
    if [ ! -f "$OUTPUT_DIR/experiment_log.json" ]; then
        echo ""
        echo "--- [Step 1] Generating mutations: seed=$SEED ---"
        $PY "${REPO_ROOT}/experiments/run_injection_experiment.py" \
            --agent_checkpoint_dir "$AGENT_CKPT" \
            --buffer_npz "$BUFFER_NPZ" \
            --seeds_dir "$SEEDS_DIR" \
            --mutation_strategy vae_interpolation \
            --target_eligible "$TARGET_ELIGIBLE" \
            --inject_pcts "$INJECT_PCTS" \
            --vae_checkpoint_path "$VAE_CKPT" \
            --vae_config_path "$VAE_CONFIG" \
            --n_scoring_rollouts 10 \
            --output_dir "$OUTPUT_DIR" \
            --seed "$SEED" \
            --gcs_bucket "$GCS_BUCKET" \
            --gcs_prefix "$GCS_PREFIX"
    else
        echo "--- [Step 1] SKIP (already done): $OUTPUT_DIR ---"
    fi

    # --- Step 2: Train from each injection % ---
    for PCT_FILE in "$OUTPUT_DIR"/merged_buffer_*pct.npz; do
        [ -f "$PCT_FILE" ] || continue
        PCT_TAG=$(basename "$PCT_FILE" .npz | sed 's/merged_buffer_//')
        TRAIN_RUN="inject_vaeinterp_${PCT_TAG}_seed${SEED}"
        TRAIN_OUT="${OUTPUT_DIR}/training_${PCT_TAG}"
        WANDB_GROUP="vaeinterp_${PCT_TAG}"

        echo ""
        echo "--- [Step 2] Training: $TRAIN_RUN ---"
        $PY $SCRIPT \
            --n_walls 25 \
            --use_accel \
            --num_updates "$TRAIN_UPDATES" \
            --preload_buffer_npz "$PCT_FILE" \
            --resume_checkpoint_dir "$AGENT_CKPT" \
            --seed "$SEED" \
            --run_name "$TRAIN_RUN" \
            --wandb_group "$WANDB_GROUP" \
            --project JAXUED_LLM_INJECTION \
            --gcs_bucket "$GCS_BUCKET" \
            --gcs_prefix "llm-exp/training/$TRAIN_RUN" \
            --buffer_dump_interval 250 \
            --output_dir "$TRAIN_OUT" \
            --score_function sfl
    done
done
