#!/bin/bash
# Phase 5.1: CMA-ES with restarts (IPOP-style, threshold=0.05).
# Uses eval fix: stochastic + multi-attempt benchmark eval matching maze_plr.py.
#
# Usage:
#   bash scripts/run_phase5_cma_es_v2.sh |& tee logs/phase5-cma-es-v2.log

set -e

cd /cs/student/project_msc/2025/csml/gmaralla/superintelligence
unset XLA_FLAGS

CONDA_ENV=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env
PYTHON=$CONDA_ENV/bin/python

export LD_LIBRARY_PATH=$CONDA_ENV/lib:$LD_LIBRARY_PATH

CONFIG=accel_training/config.yml
SEED=42
UPDATES=20000
GROUP=phase5-comparison
RUN_NAME=phase5-cma-es-v2
LOG_DIR=runs/phase5-cma-es-v2

mkdir -p logs "$LOG_DIR"

echo "=========================================="
echo "Phase 5.1: CMA-ES v2 with restarts"
echo "Seed=$SEED | Updates=$UPDATES | Group=$GROUP"
echo "Run name: $RUN_NAME | Log: $LOG_DIR"
echo "Started: $(date)"
echo "GPU state:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
echo "=========================================="
echo ""

$PYTHON accel_training/train.py \
  --config $CONFIG \
  --es_strategy cma_es \
  --run_name $RUN_NAME \
  --group $GROUP \
  --seed $SEED \
  --num_updates $UPDATES \
  --log_dir $LOG_DIR

echo ""
echo "CMA-ES v2 done: $(date)"
echo "Checkpoints in: $LOG_DIR"
echo "WandB group: $GROUP"
