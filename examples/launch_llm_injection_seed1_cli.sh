#!/bin/bash
# ACCEL+LLM Injection — Seed 1 via OpenRouter (Claude Sonnet)
# Run on a DIFFERENT GPU node than seed 0.
#
SEED=1

# === Ablation parameters ===
INJECT_START=3000
INJECT_INTERVAL=3000
BATCH_SIZE=10

set -e

# Force conda CUDA libs (system CUDA 13.1 cuSOLVER is incompatible)
export LD_LIBRARY_PATH=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/lib:${LD_LIBRARY_PATH:-}
PYTHON=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python

COMMON="--project JAXUED_LLM \
        --num_updates 50000 --eval_freq 250 \
        --score_function sfl --num_sfl_rollouts 10 \
        --skip_post_eval"

export WANDB_DIR=/tmp/wandb
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
mkdir -p /tmp/jax_cache

echo "=== [$(date)] Seed $SEED (openrouter claude-sonnet) starting ==="

$PYTHON examples/maze_plr.py $COMMON \
  --use_accel --use_llm \
  --llm_provider openrouter --llm_model anthropic/claude-sonnet-4 --llm_config llm/config.yaml \
  --llm_inject_start_step ${INJECT_START} \
  --llm_inject_interval ${INJECT_INTERVAL} \
  --llm_batch_size ${BATCH_SIZE} \
  --llm_gate \
  --llm_difficulty_gate_mode buffer_mean \
  --llm_diversity_gate_mode buffer_median \
  --seed $SEED \
  --run_name "accel-llm-v2" \
  2>&1 | tee /tmp/llm_injection_seed${SEED}.log

echo "=== [$(date)] Seed $SEED complete ==="
