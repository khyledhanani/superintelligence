#!/bin/bash
# Run trajectory feature analysis across 5 ACCEL training stages.
# Designed to run on TPU where JAX rollouts work.
#
# Usage (on TPU):
#   bash trajectory/run_accel_analysis.sh

set -euo pipefail

PYTHON="${PYTHON:-python}"
GCS_CP="gcloud storage cp"
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

GCS_BUCKET="gs://ucl-ued-project-bucket"
BUFFER_PREFIX="${GCS_BUCKET}/accel/buffer_dumps/plain_accel/0"
CKPT_PREFIX="${GCS_BUCKET}/accel/checkpoints/plain_accel/0/models"

LOCAL_DIR="/tmp/traj_analysis"
BUFFER_DIR="${LOCAL_DIR}/buffers"
CKPT_DIR="${LOCAL_DIR}/checkpoints"
DATA_DIR="${LOCAL_DIR}/data"
ANALYSIS_DIR="${LOCAL_DIR}/analysis"

mkdir -p "$BUFFER_DIR" "$CKPT_DIR" "$DATA_DIR" "$ANALYSIS_DIR"

# Buffer → approximate checkpoint mapping
declare -A CKPT_MAP
# eval_freq=250, save_interval=2 → ckpt at even eval_steps
# buffer Nk → eval_step = N*1000/250, max_to_keep=60 evicts < step 80
CKPT_MAP[20k]=80
CKPT_MAP[30k]=118
CKPT_MAP[40k]=158
CKPT_MAP[50k]=198

STAGES="20k 30k 40k 50k"

echo "=== Trajectory Feature Analysis: Plain ACCEL ==="
echo "  Stages: $STAGES"
echo "  Output: $ANALYSIS_DIR"
echo ""

# --- Step 1: Pull buffers and checkpoints ---
echo "[Step 1] Pulling data from GCS..."
for STAGE in $STAGES; do
    STEP=${CKPT_MAP[$STAGE]}

    # Pull buffer
    BUFF_LOCAL="${BUFFER_DIR}/buffer_dump_${STAGE}.npz"
    if [ ! -f "$BUFF_LOCAL" ]; then
        echo "  Pulling buffer_dump_${STAGE}.npz..."
        $GCS_CP "${BUFFER_PREFIX}/buffer_dump_${STAGE}.npz" "$BUFF_LOCAL"
    else
        echo "  Buffer ${STAGE} already cached."
    fi

    # Pull checkpoint — orbax expects parent/step/ structure
    CKPT_PARENT="${CKPT_DIR}/stage_${STAGE}"
    CKPT_LOCAL="${CKPT_PARENT}/${STEP}"
    if [ ! -d "$CKPT_LOCAL" ]; then
        echo "  Pulling checkpoint step ${STEP}..."
        mkdir -p "$CKPT_LOCAL"
        $GCS_CP --recursive "${CKPT_PREFIX}/${STEP}/*" "$CKPT_LOCAL/"
    else
        echo "  Checkpoint ${STEP} already cached."
    fi
done
echo ""

# --- Step 2: Collect trajectories for each stage ---
echo "[Step 2] Collecting trajectories..."
for STAGE in $STAGES; do
    STEP=${CKPT_MAP[$STAGE]}
    TRAJ_PATH="${DATA_DIR}/trajectories_${STAGE}.npz"

    if [ -f "$TRAJ_PATH" ]; then
        echo "  Stage ${STAGE}: already collected, skipping."
        continue
    fi

    echo "  Stage ${STAGE}: rolling out agent (ckpt ${STEP}) on buffer levels..."
    $PYTHON "$BASE_DIR/collect_trajectories.py" \
        --agent_checkpoint "${CKPT_DIR}/stage_${STAGE}" \
        --buffer_path "${BUFFER_DIR}/buffer_dump_${STAGE}.npz" \
        --source buffer \
        --output_path "$TRAJ_PATH" \
        --batch_size 256
    echo ""
done

# --- Step 3: Run raw feature analysis on each stage ---
echo "[Step 3] Analyzing trajectory features vs regret..."
for STAGE in $STAGES; do
    TRAJ_PATH="${DATA_DIR}/trajectories_${STAGE}.npz"
    STAGE_DIR="${ANALYSIS_DIR}/${STAGE}"
    mkdir -p "$STAGE_DIR"

    echo "  Stage ${STAGE}:"
    $PYTHON "$BASE_DIR/analyze_correlation.py" \
        --data_path "$TRAJ_PATH" \
        --output_dir "$STAGE_DIR"
    echo ""
done

# --- Step 4: Summary across stages ---
echo "[Step 4] Cross-stage summary..."
$PYTHON -c "
import numpy as np
from scipy import stats
import os

stages = '$STAGES'.split()
data_dir = '$DATA_DIR'
print()
print(f\"{'Stage':>6s} | {'N':>5s} | {'Regret mean':>11s} | {'Ep len mean':>11s} | {'Goal %':>6s} | {'len↔regret ρ':>12s} | {'unique↔regret ρ':>15s} | {'revisit↔regret ρ':>16s}\")
print('-' * 110)

for stage in stages:
    d = np.load(os.path.join(data_dir, f'trajectories_{stage}.npz'), allow_pickle=True)
    traj = d['trajectories']
    regret = d['regret']
    ep_len = d['episode_lengths']
    goal = d['reached_goal']

    traj_len = (traj > 0).sum(axis=1)
    unique = np.array([len(np.unique(t[t > 0])) for t in traj])
    revisit = traj_len / np.maximum(unique, 1)

    rho_len, _ = stats.spearmanr(traj_len, regret)
    rho_uniq, _ = stats.spearmanr(unique, regret)
    rho_rev, _ = stats.spearmanr(revisit, regret)

    print(f'{stage:>6s} | {len(traj):5d} | {regret.mean():11.4f} | {ep_len.mean():11.1f} | {goal.mean()*100:5.1f}% | {rho_len:+12.4f} | {rho_uniq:+15.4f} | {rho_rev:+16.4f}')
print()
"

echo ""
echo "=== Done ==="
echo "  Per-stage plots: ${ANALYSIS_DIR}/<stage>/"
echo "  Run on TPU. Copy results back with:"
echo "    gcloud storage cp --recursive ${ANALYSIS_DIR} ${GCS_BUCKET}/accel/trajectory_analysis/"
