#!/bin/bash
# CMA-ES with anti-clustering restarts.
#
# Key features:
#   - IPOP-style restart when sigma collapses below threshold
#   - Anti-clustering: new mean maximally distant from all previous convergence points
#   - Logs es/std, es/n_restarts, es/mean_norm to WandB for sawtooth monitoring
#
# Usage:
#   bash scripts/run_cma_es_restart.sh |& tee logs/cma-es-restart.log
#
# Or in tmux:
#   tmux new -s cma_es
#   bash scripts/run_cma_es_restart.sh |& tee logs/cma-es-restart.log

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
RUN_NAME=cma-es-restart
LOG_DIR=runs/cma-es-restart

mkdir -p logs "$LOG_DIR"

echo "=========================================="
echo "CMA-ES with Anti-Clustering Restarts"
echo "Seed=$SEED | Updates=$UPDATES | Group=$GROUP"
echo "Run name: $RUN_NAME | Log: $LOG_DIR"
echo "Restart threshold: 0.05 (sigma_init fraction)"
echo "Started: $(date)"
echo "GPU state:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "(no GPU info)"
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
echo "CMA-ES restart run done: $(date)"
echo "Checkpoints in: $LOG_DIR"
echo "WandB group: $GROUP"
