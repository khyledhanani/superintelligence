#!/bin/bash
# LLM Injection Experiment — Re-run failed seeds (0 and 2)
#
# Steps 0+1 (seed prep + mutation buffers) already completed for both seeds.
# Only Step 2 (training) needs to re-run.
#
# Fixes applied vs original scripts:
#   1. Permission error (seed 0): /tmp/buffer_dumps/ owned by another user.
#      Fix: export TMPDIR to user-owned dir + clean stale /tmp dirs.
#   2. OOM error (seed 2): vmapped eval allocates too much memory.
#      Fix: --eval_num_attempts 5 (down from default 10).
#
# Usage:
#   bash experiments/gpu_scripts/run_remaining_seeds.sh
#
set -e

export WANDB_ENTITY=romain-hautier-university-college-london-ucl-

PY=/cs/student/msc/csml/2025/rhautier/miniforge3/envs/jaxued_env/bin/python
SCRIPT=/cs/student/msc/csml/2025/rhautier/Documents/jaxued/jaxued/examples/maze_plr.py
REPO_ROOT=/cs/student/msc/csml/2025/rhautier/Documents/jaxued/jaxued

GCS_BUCKET="ucl-ued-project-bucket"
LOCAL_DATA="/cs/student/project_msc/2025/csml/rhautier/injection_data"
TRAIN_UPDATES=10000

AGENT_CKPT="${REPO_ROOT}/gcs_artifacts/agent/39"

# --- Fix 1: Redirect TMPDIR to avoid /tmp permission errors ---
CACHE_DIR="/cs/student/project_msc/2025/csml/rhautier/jax_llm_inject"
export TMPDIR="${CACHE_DIR}/tmp"
export WANDB_DIR="${CACHE_DIR}/wandb"
export JAX_COMPILATION_CACHE_DIR="${CACHE_DIR}/jax_cache"
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"
mkdir -p "${CACHE_DIR}/jax_cache" "${CACHE_DIR}/wandb" "${CACHE_DIR}/tmp"

# Clean stale /tmp dirs if we can
rm -rf /tmp/buffer_dumps/inject_llm_*_seed0 /tmp/buffer_dumps/inject_llm_*_seed2 2>/dev/null || true

# --- Fix 2: Reduce eval attempts to avoid OOM ---
EVAL_NUM_ATTEMPTS=5

SEEDS="0 2"
PCTS="10 15 20 25"

echo "=== [$(date)] Re-running failed training — Seeds: $SEEDS, Pcts: $PCTS ==="
echo "  Fixes: TMPDIR=${TMPDIR}, eval_num_attempts=${EVAL_NUM_ATTEMPTS}"

for SEED in $SEEDS; do
    BASE_OUTPUT="${LOCAL_DATA}/results/llm_inject_seed${SEED}"

    echo ""
    echo "=========================================="
    echo "  Seed $SEED"
    echo "=========================================="

    for PCT in $PCTS; do
        PCT_TAG="${PCT}pct"
        PCT_FILE="${BASE_OUTPUT}/merged_buffer_${PCT_TAG}.npz"

        if [ ! -f "$PCT_FILE" ]; then
            echo "  [SKIP] Missing merged buffer: $PCT_FILE"
            continue
        fi

        TRAIN_RUN="inject_llm_${PCT_TAG}_seed${SEED}"
        TRAIN_OUT="${BASE_OUTPUT}/training_${PCT_TAG}"
        WANDB_GROUP="llm_inject_${PCT_TAG}"

        echo ""
        echo "--- Training: $TRAIN_RUN (group: $WANDB_GROUP) ---"

        # Ensure output_dir exists
        mkdir -p "$TRAIN_OUT"

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
            --score_function sfl --num_sfl_rollouts 10 \
            --eval_num_attempts "$EVAL_NUM_ATTEMPTS"

        echo "  [$(date)] $TRAIN_RUN done."
    done
done

echo ""
echo "=== [$(date)] All remaining seeds finished ==="
