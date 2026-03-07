#!/bin/bash
# Phase 5.1: ACCEL baseline — standalone script.
# Run in a FRESH tmux window AFTER all ES runs have exited.
# cuSolver requires a clean CUDA context — do NOT chain after ES jobs.
#
# Usage:
#   bash scripts/run_phase5_accel_only.sh |& tee logs/phase5-accel.log
#
# Prerequisites:
#   - All three ES runs (CMA-ES, NS-ES, SV-CMA-ES) must have EXITED
#   - GPU must be free (check: nvidia-smi)
#   - No other JAX sessions on sideswipe/prowl
#
# Note: maze_plr.py does not have a --group argparse flag.
# WandB group is set to config["run_name"] inside main().
# To group with the ES runs in WandB, set run_name=phase5-accel
# and manually move the run into the phase5-comparison group in the UI,
# OR use WANDB_RUN_GROUP env var below (overrides wandb.init group).

set -e

cd /cs/student/project_msc/2025/csml/gmaralla/superintelligence
unset XLA_FLAGS

CONDA_ENV=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env
PYTHON=$CONDA_ENV/bin/python

# Ensure conda env libs are in LD_LIBRARY_PATH so cuSolver can find libcublas.so.12
# (required when invoking Python by full path without `conda activate`)
export LD_LIBRARY_PATH=$CONDA_ENV/lib:$LD_LIBRARY_PATH
SEED=42
UPDATES=20000
GROUP=phase5-comparison

mkdir -p logs

# Set WandB group via environment variable (maze_plr.py uses group=config["run_name"],
# but WANDB_RUN_GROUP env var takes precedence and groups with the ES runs).
export WANDB_RUN_GROUP=$GROUP

echo "=========================================="
echo "Phase 5.1: ACCEL baseline (standalone)"
echo "Seed=$SEED | Updates=$UPDATES | Group=$GROUP"
echo "Started: $(date)"
echo "GPU state before launch:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
echo "=========================================="
echo ""

$PYTHON examples/maze_plr.py \
  --run_name phase5-accel \
  --project es-accel \
  --seed $SEED \
  --num_updates $UPDATES \
  --use_accel \
  --checkpoint_save_interval 2000 \
  --max_number_of_checkpoints 11 \
  --eval_freq 500

echo ""
echo "ACCEL done: $(date)"
echo "Checkpoints saved to: checkpoints/phase5-accel/$SEED/models/"
echo "WandB group: $GROUP"
