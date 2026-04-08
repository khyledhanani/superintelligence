#!/bin/bash
# ============================================================================
# FINAL GPT-4.1-mini (ACCEL + LLM) re-run — SEED 6 (smew, 3090 Ti)
# Same config as examples/launch_final_gpt41mini_all.sh, but keeping ALL 120
# checkpoints (not just the latest 60). Same wandb group as seeds 0-4 so
# everything aggregates together.
#
# Usage:
#   export OPENROUTER_API_KEY=...
#   bash examples/launch_final_gpt41mini_fullckpt_seed6_smew.sh
# ============================================================================

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/lib:${LD_LIBRARY_PATH:-}
PYTHON=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python

COMMON="--project JAXUED_LLM_FINAL \
        --num_updates 30000 --eval_freq 250 \
        --score_function sfl \
        --num_sfl_rollouts 10 \
        --buffer_dump_interval 250 \
        --checkpoint_save_interval 1 \
        --max_number_of_checkpoints 121 \
        --skip_post_eval"

LLM="--use_llm \
     --llm_provider openrouter \
     --llm_model openai/gpt-4.1-mini \
     --llm_config llm/config.yaml \
     --llm_inject_start_step 3000 \
     --llm_inject_interval 3000 \
     --llm_batch_size 20 \
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

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "ERROR: OPENROUTER_API_KEY not set"; exit 1
fi

SEED=6
OUTDIR="$(cd "$(dirname "$0")/.." && pwd)/results/final-gpt41mini-fullckpt"
mkdir -p "$OUTDIR"

export WANDB_DIR=/tmp/wandb
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache_${USER}_seed${SEED}
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
mkdir -p logs

echo "=== [$(date)] LLM+ACCEL (GPT-4.1-mini, full ckpt) seed $SEED on smew starting ==="

$PYTHON examples/maze_plr.py $COMMON $LLM \
  --use_accel \
  --seed $SEED \
  --wandb_group "llm-accel-gpt41mini" \
  --run_name "llm-accel-gpt41mini-seed${SEED}" \
  --output_dir "$OUTDIR" \
  2>&1 | tee logs/final_gpt41mini_fullckpt_seed${SEED}.log

echo "=== [$(date)] Seed $SEED complete ==="
