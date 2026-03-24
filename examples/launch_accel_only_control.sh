#!/bin/bash
# ACCEL-only Control Experiment
# Control condition for ACCEL+LLM comparison.
# Run this script on a GPU node or TPU.
# Matching seeds, buffer sizes, and non-injection hyperparameters with launch_llm_injection.sh.
set -e

# Force conda CUDA libs (system CUDA 13.1 cuSOLVER is incompatible)
export LD_LIBRARY_PATH=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/lib:${LD_LIBRARY_PATH:-}
PYTHON=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python

COMMON="--project JAXUED_LLM \
        --num_updates 50000 --eval_freq 250 \
        --skip_video --skip_post_eval"

export WANDB_DIR=/tmp/wandb
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
mkdir -p /tmp/jax_cache
mkdir -p logs

for seed in 0 1 2; do
  echo "=== [$(date)] Seed $seed starting ==="

  $PYTHON examples/maze_plr.py $COMMON \
    --use_accel \
    --seed $seed \
    --run_name "accel-only" \
    2>&1 | tee logs/accel_only_control_seed${seed}.log

  echo "=== [$(date)] Seed $seed complete ==="
done

echo "All ACCEL-only control experiments complete."
