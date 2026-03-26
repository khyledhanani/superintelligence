#!/bin/bash
# ACCEL+LLM Injection — Claude Sonnet via OpenRouter
# Includes: seed retries, normalized TD EMD, auto-visualization
#
SEED=0

# === Ablation parameters ===
INJECT_START=5000
INJECT_INTERVAL=5000
BATCH_SIZE=20           # min accepted seeds
MUTATIONS=50            # mutations per seed
MAX_SEED_RETRIES=5      # cap = BATCH_SIZE * MAX_SEED_RETRIES = 100

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

echo "=== [$(date)] Seed $SEED (Sonnet via OpenRouter) starting ==="

$PYTHON examples/maze_plr.py $COMMON \
  --use_accel --use_llm \
  --llm_provider openrouter --llm_model anthropic/claude-sonnet-4-6 --llm_config llm/config.yaml \
  --llm_inject_start_step ${INJECT_START} \
  --llm_inject_interval ${INJECT_INTERVAL} \
  --llm_batch_size ${BATCH_SIZE} \
  --llm_mutations_per_seed ${MUTATIONS} \
  --llm_max_seed_retries ${MAX_SEED_RETRIES} \
  --llm_gate \
  --llm_difficulty_gate_mode buffer_mean \
  --llm_diversity_gate_mode buffer_median \
  --seed $SEED \
  --run_name "accel-llm-sonnet-v4" \
  2>&1 | tee /tmp/llm_sonnet_v3_seed${SEED}.log

echo "=== [$(date)] Seed $SEED complete ==="
