#!/bin/bash
# Phase 5: Four-Way Comparison — sequential (single GPU).
# Smoke tests already passed. Runs CMA-ES -> NS-ES -> SV-CMA-ES -> ACCEL.
# Total: ~10-12 hours. Run inside tmux.
# Usage: bash scripts/run_phase5_sequential.sh |& tee logs/phase5.log

set -e

cd /cs/student/project_msc/2025/csml/gmaralla/superintelligence
unset XLA_FLAGS

PYTHON=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python
CONFIG=accel_training/config.yml
SEED=42
GROUP=phase5
UPDATES=20000

mkdir -p logs

echo "=========================================="
echo "Phase 5: Four-Way Comparison (sequential)"
echo "Seed=$SEED | Updates=$UPDATES | Group=$GROUP"
echo "Started: $(date)"
echo "=========================================="
echo ""

# 1. CMA-ES with restarts
echo "=== [1/4] CMA-ES (20k) ==="
$PYTHON accel_training/train.py --config $CONFIG \
  --es_strategy cma_es \
  --run_name phase5-cma-es \
  --group $GROUP \
  --seed $SEED \
  --num_updates $UPDATES \
  --log_dir runs/phase5-cma-es
echo "CMA-ES done: $(date)"
echo ""

# 2. NS-ES
echo "=== [2/4] NS-ES (20k) ==="
$PYTHON accel_training/train.py --config $CONFIG \
  --es_strategy ns_es \
  --run_name phase5-ns-es \
  --group $GROUP \
  --seed $SEED \
  --num_updates $UPDATES \
  --log_dir runs/phase5-ns-es
echo "NS-ES done: $(date)"
echo ""

# 3. SV-CMA-ES
echo "=== [3/4] SV-CMA-ES (20k) ==="
$PYTHON accel_training/train.py --config $CONFIG \
  --es_strategy sv_cma_es \
  --run_name phase5-sv-cma-es \
  --group $GROUP \
  --seed $SEED \
  --num_updates $UPDATES \
  --log_dir runs/phase5-sv-cma-es
echo "SV-CMA-ES done: $(date)"
echo ""

# 4. ACCEL baseline (vanilla maze_plr.py)
echo "=== [4/4] ACCEL baseline (20k) ==="
$PYTHON examples/maze_plr.py \
  --run_name phase5-accel \
  --project es-accel \
  --seed $SEED \
  --num_updates $UPDATES
echo "ACCEL done: $(date)"
echo ""

echo "=========================================="
echo "All 4 runs complete: $(date)"
echo "Next: python scripts/evaluate_checkpoints.py --all"
echo "=========================================="
