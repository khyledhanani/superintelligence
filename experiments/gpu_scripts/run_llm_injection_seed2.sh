#!/bin/bash
# LLM Injection Experiment — Seed 0
#
# Pipeline: prepare LLM seeds → mutate + create merged buffers at 5-25% →
#           resume training from 10k checkpoint for each %.
#
set -e
SEED=2
export WANDB_ENTITY=romain-hautier-university-college-london-ucl-

PY=/cs/student/msc/csml/2025/rhautier/miniforge3/envs/jaxued_env/bin/python
SCRIPT=/cs/student/msc/csml/2025/rhautier/Documents/jaxued/jaxued/examples/maze_plr.py
REPO_ROOT=/cs/student/msc/csml/2025/rhautier/Documents/jaxued/jaxued

GCS_BUCKET="ucl-ued-project-bucket"
GCS_PREFIX="llm-exp/injection"
LOCAL_DATA="/cs/student/project_msc/2025/csml/rhautier/injection_data"
INJECT_PCTS="5,10,15,20,25"
TRAIN_UPDATES=10000
NUM_EDITS=3
TARGET_ELIGIBLE=1000

AGENT_CKPT="${REPO_ROOT}/gcs_artifacts/agent/39"
BUFFER_NPZ="${REPO_ROOT}/gcs_artifacts/buffer/buffer_dump_final.npz"
SEEDS_DIR="${LOCAL_DATA}/seeds"
BASE_OUTPUT="${LOCAL_DATA}/results/llm_inject_seed${SEED}"

CACHE_DIR="/cs/student/project_msc/2025/csml/rhautier/jax_llm_inject"
export WANDB_DIR="${CACHE_DIR}/wandb"
export JAX_COMPILATION_CACHE_DIR="${CACHE_DIR}/jax_cache"
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"
mkdir -p "${CACHE_DIR}/jax_cache" "${CACHE_DIR}/wandb"

echo "=== [$(date)] LLM Injection Experiment — Seed $SEED ==="

# --- Step 0: Prepare LLM seeds from generated mazes ---
echo ""
echo "--- [Step 0] Preparing LLM seeds ---"
mkdir -p "$SEEDS_DIR"
$PY "${REPO_ROOT}/experiments/prepare_llm_seeds.py" \
    --input_dir "${REPO_ROOT}/llm/generated_levels" \
    --output_dir "$SEEDS_DIR"

# --- Step 1: Mutate seeds + create merged buffers at each % ---
if [ ! -f "$BASE_OUTPUT/experiment_log.json" ]; then
    echo ""
    echo "--- [Step 1] Generating mutations + merged buffers ---"
    $PY "${REPO_ROOT}/experiments/run_injection_experiment.py" \
        --agent_checkpoint_dir "$AGENT_CKPT" \
        --buffer_npz "$BUFFER_NPZ" \
        --seeds_dir "$SEEDS_DIR" \
        --mutation_strategy wall_flip \
        --target_eligible "$TARGET_ELIGIBLE" \
        --inject_pcts "$INJECT_PCTS" \
        --num_edits "$NUM_EDITS" \
        --n_scoring_rollouts 10 \
        --output_dir "$BASE_OUTPUT" \
        --seed "$SEED" \
        --gcs_bucket "$GCS_BUCKET" \
        --gcs_prefix "$GCS_PREFIX"
else
    echo "--- [Step 1] SKIP (already done): $BASE_OUTPUT ---"
fi

# --- Step 2: Train from each injection % ---
for PCT_FILE in "$BASE_OUTPUT"/merged_buffer_*pct.npz; do
    [ -f "$PCT_FILE" ] || continue
    PCT_TAG=$(basename "$PCT_FILE" .npz | sed 's/merged_buffer_//')
    TRAIN_RUN="inject_llm_${PCT_TAG}_seed${SEED}"
    TRAIN_OUT="${BASE_OUTPUT}/training_${PCT_TAG}"
    WANDB_GROUP="llm_inject_${PCT_TAG}"

    # Skip if training already ran for this %
    if [ -d "$TRAIN_OUT/buffer_dumps" ]; then
        echo "--- [Step 2] SKIP (already done): $TRAIN_RUN ---"
        continue
    fi

    echo ""
    echo "--- [Step 2] Training: $TRAIN_RUN (group: $WANDB_GROUP) ---"
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
        --score_function sfl --num_sfl_rollouts 10
done

echo "=== [$(date)] Seed $SEED complete ==="
