#!/bin/bash
cd /cs/student/project_msc/2025/csml/rhautier/oe_proj
PY=/cs/student/msc/csml/2025/rhautier/miniforge3/envs/jaxued_env/bin/python
SCRIPT=/cs/student/project_msc/2025/csml/rhautier/oe_proj/examples/maze_plr.py
COMMON="--maze_height 21 --maze_width 21 --n_walls 150 --use_accel --score_function sfl --project JAXUED_VAE_COMPARISON --run_name accel_sfl_baseline --gcs_bucket ucl-ued-project-bucket --gcs_prefix accel/accel_sfl_baseline"
$PY $SCRIPT $COMMON --seed 3
$PY $SCRIPT $COMMON --seed 4
$PY $SCRIPT $COMMON --seed 5
