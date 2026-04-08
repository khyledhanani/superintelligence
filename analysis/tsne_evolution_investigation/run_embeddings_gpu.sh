#!/bin/bash
# Run compute_embeddings.py across 4 GPUs in parallel.
# Usage: called per-GPU with GPU_ID as argument, or launch all 4 via tmux.

set -euo pipefail

BASE="/cs/student/project_msc/2025/csml/sruppage/superintelligence"
SCRIPT="utils/compute_embeddings.py"
OUT_DIR="analysis/tsne_perplexity_investigation/cache_solved"

export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="${BASE}:${BASE}/examples:${PYTHONPATH:-}"

GPU_ID="${1:?Usage: $0 <gpu_id>  (0-3)}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# GPU -> timestep range mapping
case "$GPU_ID" in
    0) START=2250;  END=7000  ;;
    1) START=4750;  END=9500  ;;
    2) START=9750;  END=14500 ;;
    3) START=14750; END=19500 ;;
    *) echo "Invalid GPU_ID: $GPU_ID"; exit 1 ;;
esac

# Per-seed paths (absolute for orbax)
declare -A BUF_DIRS CKPT_DIRS
BUF_DIRS[1]="${BASE}/buffer_dumps/llm_injection_fresh/seed1/buffer_dumps"
CKPT_DIRS[1]="${BASE}/buffer_dumps/llm_injection_fresh/seed1/checkpoints"
BUF_DIRS[2]="${BASE}/buffer_dumps/llm_injection_fresh/seed2/seed1/buffer_dumps"
CKPT_DIRS[2]="${BASE}/buffer_dumps/llm_injection_fresh/seed2/seed1/checkpoints"
BUF_DIRS[3]="${BASE}/buffer_dumps/llm_injection_fresh/seed3/seed1/buffer_dumps"
CKPT_DIRS[3]="${BASE}/buffer_dumps/llm_injection_fresh/seed3/seed1/checkpoints"

cd "$BASE"

TOTAL=0
SKIPPED=0
DONE=0
FAILED=0

# Count total work
for SEED in 1 2 3; do
    for TS in $(seq "$START" 250 "$END"); do
        TOTAL=$((TOTAL + 2))  # buf + eval
    done
done

echo "=== GPU $GPU_ID: timesteps ${START}–${END}, 3 seeds, $TOTAL total jobs ==="

for SEED in 1 2 3; do
    BUF_DIR="${BUF_DIRS[$SEED]}"
    CKPT_DIR="${CKPT_DIRS[$SEED]}"

    for TS in $(seq "$START" 250 "$END"); do
        CKPT=$(( TS / 250 - 1 ))

        EMB_OUT="${OUT_DIR}/emb_solved_s${SEED}_t${TS}.npz"
        EVAL_OUT="${OUT_DIR}/eval_solved_s${SEED}_t${TS}.npz"

        # --- Buffer embedding ---
        if [ -f "$EMB_OUT" ]; then
            SKIPPED=$((SKIPPED + 1))
            echo "[GPU $GPU_ID] SKIP (exists): $EMB_OUT"
        else
            echo "[GPU $GPU_ID] RUN s${SEED} t${TS} ckpt${CKPT} — buffer embedding"
            if python "$SCRIPT" \
                --buffer "${BUF_DIR}/buffer_dump_${TS}.npz" \
                --checkpoint_dir "$CKPT_DIR" \
                --checkpoint_step "$CKPT" \
                --output "$EMB_OUT"; then
                DONE=$((DONE + 1))
            else
                FAILED=$((FAILED + 1))
                echo "[GPU $GPU_ID] FAILED: $EMB_OUT"
            fi
        fi

        # --- Eval embedding ---
        if [ -f "$EVAL_OUT" ]; then
            SKIPPED=$((SKIPPED + 1))
            echo "[GPU $GPU_ID] SKIP (exists): $EVAL_OUT"
        else
            echo "[GPU $GPU_ID] RUN s${SEED} t${TS} ckpt${CKPT} — eval embedding"
            if python "$SCRIPT" \
                --eval_only \
                --checkpoint_dir "$CKPT_DIR" \
                --checkpoint_step "$CKPT" \
                --output "$EVAL_OUT"; then
                DONE=$((DONE + 1))
            else
                FAILED=$((FAILED + 1))
                echo "[GPU $GPU_ID] FAILED: $EVAL_OUT"
            fi
        fi
    done
done

echo "=== GPU $GPU_ID FINISHED: $DONE completed, $SKIPPED skipped, $FAILED failed (of $TOTAL total) ==="
