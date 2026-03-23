#!/bin/bash
# Smoke test: LLM injection with decision gate enabled
# Verifies: (1) gate filters, (2) levels appear in buffer, (3) WandB metrics log,
#           (4) no crashes over 5k steps, (5) solve rate curve looks sane
#
# Usage: bash scripts/smoke_test_llm_gate.sh [provider] [model]
#   Default provider: claude-code (free via Max plan)
#   Default model: (empty, uses config.yaml default)
#
# Check results in WandB: project=JAXUED_SMOKE, group=llm-gate-smoke

set -euo pipefail

PROVIDER="${1:-claude-code}"
MODEL="${2:-}"

# CUDA 13.1 fix for GPU nodes
export LD_LIBRARY_PATH=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/lib:${LD_LIBRARY_PATH:-}

PYTHON=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== LLM Gate Smoke Test ==="
echo "Provider: ${PROVIDER}"
echo "Model: ${MODEL:-'(config.yaml default)'}"
echo "Steps: 5000 (num_env_steps)"
echo ""

MODEL_FLAG=""
if [ -n "$MODEL" ]; then
    MODEL_FLAG="--llm_model ${MODEL}"
fi

cd "${SCRIPT_DIR}"

${PYTHON} examples/maze_plr.py \
    --num_env_steps 5000 \
    --eval_freq 10 \
    --num_train_envs 32 \
    --num_test_envs 32 \
    --use_llm \
    --llm_provider "${PROVIDER}" \
    ${MODEL_FLAG} \
    --llm_config llm/config.yaml \
    --llm_inject_interval 50 \
    --llm_warmup_steps 100 \
    --llm_batch_size 4 \
    --llm_gate \
    --llm_difficulty_threshold 0.6 \
    --llm_min_diversity 0.02 \
    --llm_n_rollouts 100 \
    --llm_max_diversity_retries 2 \
    --wandb_project JAXUED_SMOKE \
    --wandb_group llm-gate-smoke \
    --seed 0

echo ""
echo "=== Smoke test complete ==="
echo "Check WandB JAXUED_SMOKE project for:"
echo "  - llm/acceptance_rate (should be between 0.2 and 0.8)"
echo "  - llm/diversity_score_mean (should be > 0)"
echo "  - llm/batch_all_rejected_count (ideally 0)"
echo "  - llm/injected_count (should be > 0 at some point)"
echo "  - solve rate curve (should not crash or flatline)"
