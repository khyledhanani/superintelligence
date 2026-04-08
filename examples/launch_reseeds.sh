#!/bin/bash
# Re-seed experiments: 2 more seeds each for xindi-seven-65 (LLM injection)
# and cardassian-data-59 (ACCEL baseline).
#
# GPUs 2-3 only. Runs in tmux with 4 windows (batch 1: seed1, batch 2: seed2).
# Each batch runs 2 jobs concurrently, then moves to the next batch.
set -e

SESSION="reseeds"
ROOT=/cs/student/project_msc/2025/csml/sruppage/superintelligence
PYTHON=$ROOT/.venv/bin/python

ENV_SETUP="export WANDB_DIR=/tmp/wandb; export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache; export XLA_PYTHON_CLIENT_PREALLOCATE=false; mkdir -p /tmp/jax_cache /tmp/wandb; cd $ROOT"

COMMON="--project JAXUED_LLM_INJECTION --score_function sfl --num_updates 30000 --eval_freq 250 --eval_num_attempts 10 --buffer_dump_steps 2500,5000,10000,30000"

LLM_FLAGS="--use_accel --use_llm --llm_provider claude-code --llm_config llm/config.yaml --llm_inject_interval 5000 --llm_inject_start_step 2500 --llm_batch_size 10 --llm_target_buffer_pct 0.05 --llm_amplification --llm_mutation_retries 5"

BASELINE_FLAGS="--use_accel"

# Kill existing session if any
tmux kill-session -t $SESSION 2>/dev/null || true

# === BATCH 1: seed 1 ===
# Window 0: LLM injection seed 1 on GPU 2
tmux new-session -d -s $SESSION -n "llm_s1"
tmux send-keys -t $SESSION:llm_s1 "$ENV_SETUP && CUDA_VISIBLE_DEVICES=2 $PYTHON examples/maze_plr.py $COMMON $LLM_FLAGS --seed 1 --run_name llm_injection_seed1 --wandb_group llm_injection_fresh --output_dir $ROOT/buffer_dumps/llm_injection_fresh/seed1 2>&1 | tee /tmp/llm_injection_seed1.log" Enter

# Window 1: ACCEL baseline seed 1 on GPU 3
tmux new-window -t $SESSION -n "base_s1"
tmux send-keys -t $SESSION:base_s1 "$ENV_SETUP && CUDA_VISIBLE_DEVICES=3 $PYTHON examples/maze_plr.py $COMMON $BASELINE_FLAGS --seed 1 --run_name accel_baseline_seed1 --wandb_group accel_baseline --output_dir $ROOT/buffer_dumps/accel_baseline/seed1 2>&1 | tee /tmp/accel_baseline_seed1.log" Enter

echo "Batch 1 (seed 1) launched in tmux session '$SESSION'"
echo "  Window llm_s1:  LLM injection seed 1 on GPU 2"
echo "  Window base_s1: ACCEL baseline seed 1 on GPU 3"
echo ""
echo "When batch 1 finishes, launch batch 2 with:"
echo "  bash examples/launch_reseeds_batch2.sh"
echo ""
echo "Monitor: tmux attach -t $SESSION"
