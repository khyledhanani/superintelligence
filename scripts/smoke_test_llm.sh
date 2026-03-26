#!/bin/bash
# Quick smoke test for LLM injection pipeline (~200 steps, injects at step 50)
# Run from repo root: bash scripts/smoke_test_llm.sh

set -e
export LD_LIBRARY_PATH=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/lib:${LD_LIBRARY_PATH:-}
export WANDB_DIR=/tmp/wandb
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
export PATH=/cs/student/project_msc/2025/csml/gmaralla/home/.vscode-extensions/anthropic.claude-code-2.1.59-linux-x64/resources/native-binary:${PATH}
mkdir -p /tmp/jax_cache

PYTHON=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python

$PYTHON -c "
import logging, sys
logging.basicConfig(level=logging.INFO, format='%(name)s %(levelname)s %(message)s', stream=sys.stderr)
import runpy
runpy.run_path('examples/maze_plr.py', run_name='__main__')
" \
  --project JAXUED_LLM_SMOKE \
  --num_updates 200 --eval_freq 50 \
  --skip_post_eval \
  --use_accel --use_llm \
  --llm_provider claude-code --llm_config llm/config.yaml \
  --llm_inject_start_step 50 --llm_inject_interval 50 --llm_batch_size 5 \
  --llm_gate \
  --seed 0 --run_name smoke-llm \
  2>&1 | tee /tmp/smoke_llm.log
