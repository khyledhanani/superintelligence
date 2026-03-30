#!/bin/bash
# 50k Experiment: ACCEL + LLM injection (OpenRouter Sonnet, CENIE diversity)
# SFL scoring, buffer_mean/buffer_median gates, feedback loop
# Force conda CUDA libs (system CUDA 13.1 cuSOLVER is incompatible)
export LD_LIBRARY_PATH=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/lib:${LD_LIBRARY_PATH:-}
PYTHON=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python

COMMON="--project JAXUED_LLM \
        --num_updates 50000 --eval_freq 250 \
        --score_function sfl \
        --num_sfl_rollouts 10 \
        --diversity_log_interval 3000 \
        --skip_post_eval"

LLM="--use_llm \
     --llm_provider openrouter \
     --llm_model anthropic/claude-sonnet-4 \
     --llm_config llm/config.yaml \
     --llm_inject_start_step 3000 \
     --llm_inject_interval 3000 \
     --llm_batch_size 10 \
     --llm_n_references 6 \
     --llm_gate \
     --llm_difficulty_metric sfl \
     --llm_difficulty_gate_mode buffer_mean \
     --llm_diversity_metric cenie \
     --llm_diversity_gate_mode buffer_median \
     --llm_max_seed_retries 3 \
     --llm_n_rollouts 100 \
     --llm_amplification \
     --llm_mutations_per_seed 50"

# Set OPENROUTER_API_KEY in your shell before running, or in ~/.bashrc
# export OPENROUTER_API_KEY="sk-or-v1-..."
if [ -z "$OPENROUTER_API_KEY" ]; then echo "ERROR: OPENROUTER_API_KEY not set"; exit 1; fi

OUTDIR="$(cd "$(dirname "$0")/.." && pwd)/results/accel-llm-sonnet-v6"
mkdir -p "$OUTDIR"

export WANDB_DIR=/tmp/wandb
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
mkdir -p /tmp/jax_cache
mkdir -p logs

seed=0
echo "=== [$(date)] LLM+ACCEL (Sonnet) seed $seed starting ==="

$PYTHON examples/maze_plr.py $COMMON $LLM \
  --use_accel \
  --seed $seed \
  --run_name "llm-accel-sonnet" \
  --output_dir "$OUTDIR" \
  2>&1 | tee logs/50k_llm_sonnet_seed${seed}.log

echo "=== [$(date)] Seed $seed complete ==="
