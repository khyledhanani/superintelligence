#!/bin/bash
export WANDB_ENTITY=romain-hautier-university-college-london-ucl-
export GOOGLE_CLOUD_PROJECT=open-endedness-ued-project
cd /cs/student/project_msc/2025/csml/rhautier/oe_proj
PY=/cs/student/msc/csml/2025/rhautier/miniforge3/envs/jaxued_env/bin/python
SCRIPT=/cs/student/project_msc/2025/csml/rhautier/oe_proj/examples/maze_plr.py
COMMON="--maze_height 13 --maze_width 13 --n_walls 25 --score_function sfl --num_sfl_rollouts 10 --project JAXUED_VAE_COMPARISON --run_name plr_sfl_13x13 --gcs_bucket ucl-ued-project-bucket --gcs_prefix accel/plr_sfl_13x13 --buffer_dump_interval 10000 --num_updates 30000"
$PY $SCRIPT $COMMON --seed 0
$PY $SCRIPT $COMMON --seed 1
$PY $SCRIPT $COMMON --seed 2
