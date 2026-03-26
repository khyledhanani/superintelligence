#!/bin/bash
# Wall-flip mutation experiment — no VAE needed
#
# Full pipeline: generate eligible pool → create merged buffers at each
# injection % → resume training from 10k checkpoint for 10k more updates.
#
# Usage: bash experiments/gpu_scripts/run_wallflip.sh [target_eligible] [inject_pcts] [num_edits] [seeds...]
# Example: bash experiments/gpu_scripts/run_wallflip.sh 1000 5,10,15,20,25 3 0 1 2

set -e
export WANDB_ENTITY=romain-hautier-university-college-london-ucl-

TARGET_ELIGIBLE=${1:-1000}; shift || true
INJECT_PCTS=${1:-5,10,15,20,25}; shift || true
NUM_EDITS=${1:-3}; shift || true
SEEDS=("${@}")
if [ ${#SEEDS[@]} -eq 0 ]; then SEEDS=(0 1 2); fi

AGENT_CKPT="/cs/student/project_msc/2025/csml/rhautier/llm-exp-ued/checkpoint_10k"
BUFFER_NPZ="/cs/student/project_msc/2025/csml/rhautier/llm-exp-ued/buffer/buffer_dump_10000.npz"
SEEDS_DIR="/cs/student/project_msc/2025/csml/rhautier/llm-exp-ued/seeds_10k_gated"
BASE_OUTPUT="/cs/student/project_msc/2025/csml/rhautier/llm-exp-ued/results"
GCS_BUCKET="ucl-ued-project-bucket"
GCS_PREFIX="llm-exp/injection"
PY=/cs/student/msc/csml/2025/rhautier/miniforge3/envs/jaxued_env/bin/python
SCRIPT=/cs/student/msc/csml/2025/rhautier/Documents/jaxued/jaxued/examples/maze_plr.py
TRAIN_UPDATES=10000

echo "=== Wall-Flip Mutation ==="
echo "  target_eligible=$TARGET_ELIGIBLE  inject_pcts=$INJECT_PCTS  num_edits=$NUM_EDITS"
echo "  seeds=${SEEDS[*]}  train_updates=$TRAIN_UPDATES"

for SEED in "${SEEDS[@]}"; do
    RUN_NAME="wall_flip_e${NUM_EDITS}_t${TARGET_ELIGIBLE}_seed${SEED}"
    OUTPUT_DIR="${BASE_OUTPUT}/${RUN_NAME}"

    # --- Step 1: Generate eligible pool + merged buffers ---
    if [ ! -f "$OUTPUT_DIR/experiment_log.json" ]; then
        echo ""
        echo "--- [Step 1] Generating mutations: seed=$SEED ---"
        $PY experiments/run_injection_experiment.py \
            --agent_checkpoint_dir "$AGENT_CKPT" \
            --buffer_npz "$BUFFER_NPZ" \
            --seeds_dir "$SEEDS_DIR" \
            --mutation_strategy wall_flip \
            --target_eligible "$TARGET_ELIGIBLE" \
            --inject_pcts "$INJECT_PCTS" \
            --num_edits "$NUM_EDITS" \
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
        TRAIN_RUN="inject_wallflip_e${NUM_EDITS}_${PCT_TAG}_seed${SEED}"
        TRAIN_OUT="${OUTPUT_DIR}/training_${PCT_TAG}"

        echo ""
        echo "--- [Step 2] Training: $TRAIN_RUN ---"
        $PY $SCRIPT \
            --maze_height 13 --maze_width 13 --n_walls 25 \
            --use_accel \
            --num_updates "$TRAIN_UPDATES" \
            --preload_buffer_npz "$PCT_FILE" \
            --resume_checkpoint_dir "$AGENT_CKPT" \
            --seed "$SEED" \
            --run_name "$TRAIN_RUN" \
            --project JAXUED_LLM_INJECTION \
            --gcs_bucket "$GCS_BUCKET" \
            --gcs_prefix "llm-exp/training/$TRAIN_RUN" \
            --buffer_dump_interval 250 \
            --output_dir "$TRAIN_OUT" \
            --score_function sfl
    done
done
