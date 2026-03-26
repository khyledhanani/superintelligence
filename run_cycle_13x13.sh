#!/bin/bash
export WANDB_ENTITY=romain-hautier-university-college-london-ucl-
cd /cs/student/msc/csml/2025/rhautier/Documents/jaxued/jaxued
PY=/cs/student/msc/csml/2025/rhautier/miniforge3/envs/jaxued_env/bin/python
SCRIPT=/cs/student/msc/csml/2025/rhautier/Documents/jaxued/jaxued/examples/maze_plr.py
COMMON="--maze_height 13 --maze_width 13 --n_walls 25 --use_accel --use_cmaes --vae_checkpoint /tmp/vae_13x13_checkpoint.pkl --vae_config /tmp/vae_13x13_config.yaml --cmaes_kl_threshold 0.1 --cmaes_kl_data /tmp/val_13x13_20k.npy --cmaes_reset_to_buffer_mean --cmaes_phase_on 2000 --cmaes_phase_interp 3000 --cmaes_phase_off 5000 --cmaes_interp_source buffer --cmaes_score_decay 0.999 --score_function sfl --project JAXUED_VAE_COMPARISON --run_name cycle_2k3k5k_decay999_sfl_13x13 --gcs_bucket ucl-ued-project-bucket --gcs_prefix accel/cycle_2k3k5k_decay999_sfl_13x13"
$PY $SCRIPT $COMMON --seed 1
$PY $SCRIPT $COMMON --seed 2
