#!/bin/bash
cd /cs/student/msc/csml/2025/rhautier/Documents/jaxued/jaxued
PY=/cs/student/msc/csml/2025/rhautier/miniforge3/envs/jaxued_env/bin/python
SCRIPT=/cs/student/msc/csml/2025/rhautier/Documents/jaxued/jaxued/examples/maze_plr.py
COMMON="--maze_height 21 --maze_width 21 --n_walls 150 --use_accel --use_cmaes --vae_checkpoint /tmp/vae_21x21_final.pkl --vae_config /tmp/vae_21x21_config/config.yaml --cmaes_kl_threshold 0.1 --cmaes_kl_data /tmp/val_21x21_20k.npy --cmaes_reset_to_buffer_mean --cmaes_stop_after 2000 --cmaes_interp_until 5000 --cmaes_interp_source buffer --cmaes_score_decay 0.999 --score_function sfl --project JAXUED_VAE_COMPARISON --run_name 3phase_decay999_sfl --gcs_bucket ucl-ued-project-bucket --gcs_prefix accel/3phase_decay999_sfl"
$PY $SCRIPT $COMMON --seed 1
$PY $SCRIPT $COMMON --seed 2
