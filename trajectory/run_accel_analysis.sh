#!/bin/bash
# Run trajectory behavior analysis across ACCEL training stages.
# Extracts spatial/behavioral features from agent trajectories and
# tests their correlation with regret — no VAE needed.
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

echo "=== Trajectory Behavior Analysis: Plain ACCEL ==="
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

# --- Step 3: Analyze trajectory behavior vs regret ---
echo "[Step 3] Analyzing trajectory behavior vs regret..."
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

# --- Step 4: Cross-stage summary ---
echo "[Step 4] Cross-stage summary..."
$PYTHON -c "
import numpy as np
from scipy import stats
import os, pickle

stages = '$STAGES'.split()
data_dir = '$DATA_DIR'
analysis_dir = '$ANALYSIS_DIR'

print()
print('=== Cross-Stage Overview ===')
print(f\"{'Stage':>6s} | {'N':>5s} | {'Regret mean':>11s} | {'Goal %':>6s} | {'All R²':>8s} | {'Solved R²':>10s}\")
print('-' * 65)

for stage in stages:
    d = np.load(os.path.join(data_dir, f'trajectories_{stage}.npz'), allow_pickle=True)
    n = len(d['trajectories'])
    regret = d['regret']
    goal = d['reached_goal']

    # Load analysis results
    results_path = os.path.join(analysis_dir, stage, 'results.pkl')
    r2_all = r2_solved = '   N/A'
    if os.path.exists(results_path):
        with open(results_path, 'rb') as f:
            res = pickle.load(f)
        if res.get('all'):
            r2_all = f'{res[\"all\"][\"r2_mean\"]:8.4f}'
        if res.get('solved') and res['solved'] is not None:
            r2_solved = f'{res[\"solved\"][\"r2_mean\"]:10.4f}'

    print(f'{stage:>6s} | {n:5d} | {regret.mean():11.4f} | {goal.mean()*100:5.1f}% | {r2_all} | {r2_solved}')

print()

# Print top features across stages
print('=== Top Correlated Features (All Levels) ===')
for stage in stages:
    results_path = os.path.join(analysis_dir, stage, 'results.pkl')
    if not os.path.exists(results_path):
        continue
    with open(results_path, 'rb') as f:
        res = pickle.load(f)
    if not res.get('all'):
        continue
    spearman = res['all']['spearman']
    top3 = spearman[:3]
    top_str = ', '.join([f'{name}: {rho:+.3f}' for name, rho, _, _ in top3])
    print(f'  {stage:>5s}: {top_str}')

print()
print('=== Top Correlated Features (Solved Only) ===')
for stage in stages:
    results_path = os.path.join(analysis_dir, stage, 'results.pkl')
    if not os.path.exists(results_path):
        continue
    with open(results_path, 'rb') as f:
        res = pickle.load(f)
    if not res.get('solved') or res['solved'] is None:
        continue
    spearman = res['solved']['spearman']
    top3 = spearman[:3]
    top_str = ', '.join([f'{name}: {rho:+.3f}' for name, rho, _, _ in top3])
    print(f'  {stage:>5s}: {top_str}')
print()
"

echo ""
echo "=== Done ==="
echo "  Per-stage results: ${ANALYSIS_DIR}/<stage>/"
echo "  Copy results back with:"
echo "    gcloud storage cp --recursive ${ANALYSIS_DIR} ${GCS_BUCKET}/accel/trajectory_analysis/"
