#!/bin/bash
# Phase 5.1: SV-CMA-ES v2 re-run (with mean clamping + new VAE checkpoint).
# Fixes: (1) mean clamping in svcmaes_strategy.py prevents latent drift
#         (2) new VAE checkpoint in config.yml has more uniform valid manifold
#         (3) cma_restart_threshold=0.05 in config.yml
#
# Run in a FRESH tmux window AFTER ACCEL run has started (or on prowl if sideswipe is busy).
# Do NOT run while the ACCEL run is active on the same GPU — they will OOM.
#
# Usage:
#   bash scripts/run_phase5_sv_cma_es_v2.sh |& tee logs/phase5-sv-cma-es-v2.log
#
# Monitor collapse risk:
#   Watch WandB: valid_fraction must stay > 0.3 past update 5399

set -e

cd /cs/student/project_msc/2025/csml/gmaralla/superintelligence
unset XLA_FLAGS

CONDA_ENV=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env
PYTHON=$CONDA_ENV/bin/python

# Ensure conda env libs are in LD_LIBRARY_PATH so cuSolver can find libcublas.so.12
# (required when invoking Python by full path without `conda activate`)
export LD_LIBRARY_PATH=$CONDA_ENV/lib:$LD_LIBRARY_PATH

CONFIG=accel_training/config.yml
SEED=42
UPDATES=20000
GROUP=phase5-comparison
RUN_NAME=phase5-sv-cma-es-v2
LOG_DIR=runs/phase5-sv-cma-es-v2

mkdir -p logs "$LOG_DIR"

echo "=========================================="
echo "Phase 5.1: SV-CMA-ES v2 (standalone)"
echo "Seed=$SEED | Updates=$UPDATES | Group=$GROUP"
echo "Run name: $RUN_NAME | Log: $LOG_DIR"
echo "Started: $(date)"
echo "GPU state:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
echo "=========================================="
echo ""

$PYTHON accel_training/train.py \
  --config $CONFIG \
  --es_strategy sv_cma_es \
  --run_name $RUN_NAME \
  --group $GROUP \
  --seed $SEED \
  --num_updates $UPDATES \
  --log_dir $LOG_DIR

echo ""
echo "SV-CMA-ES v2 done: $(date)"
echo "Checkpoints in: $LOG_DIR"
echo "WandB group: $GROUP"
