#!/bin/bash
# ACCEL+LLM Injection Experiment
# Run this script on a GPU node (albacore, smew, or canada).
# One seed per machine for parallel execution, or all 3 seeds sequentially on one machine.
#
# === Ablation parameters (change these for ablation experiments) ===
INJECT_START=5000       # --llm_inject_start_step
INJECT_INTERVAL=3000    # --llm_inject_interval
BATCH_SIZE=25           # --llm_batch_size

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
    --use_accel --use_llm \
    --llm_provider claude-code --llm_config llm/config.yaml \
    --llm_inject_start_step ${INJECT_START} \
    --llm_inject_interval ${INJECT_INTERVAL} \
    --llm_batch_size ${BATCH_SIZE} \
    --llm_gate \
    --seed $seed \
    --run_name "accel-llm" \
    2>&1 | tee logs/llm_injection_seed${seed}.log

  echo "=== [$(date)] Seed $seed complete ==="
done

echo "All ACCEL+LLM injection experiments complete."
