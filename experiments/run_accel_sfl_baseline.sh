#!/bin/bash
# Plain ACCEL + SFL baseline on 13x13 grid — 30k updates
# Checkpoints + buffer dumps saved to gs://ucl-ued-project-bucket/llm-exp/
cd /cs/student/msc/csml/2025/rhautier/Documents/jaxued/jaxued

PY=/cs/student/msc/csml/2025/rhautier/miniforge3/envs/jaxued_env/bin/python
SCRIPT=examples/maze_plr.py

SEED=${1:-0}
RUN_NAME="accel_sfl_baseline_13x13"

$PY $SCRIPT \
    --use_accel \
    --score_function sfl \
    --num_updates 30000 \
    --eval_freq 250 \
    --checkpoint_save_interval 1 \
    --max_number_of_checkpoints 120 \
    --buffer_dump_interval 250 \
    --diversity_log_interval 1000 \
    --diversity_sample_size 20 \
    --gcs_bucket ucl-ued-project-bucket \
    --gcs_prefix llm-exp \
    --project JAXUED_LLM_EXPERIMENTS \
    --run_name "$RUN_NAME" \
    --seed "$SEED"
