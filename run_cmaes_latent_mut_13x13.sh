#!/bin/bash
export WANDB_ENTITY=romain-hautier-university-college-london-ucl-
export GOOGLE_CLOUD_PROJECT=open-endedness-ued-project
cd /cs/student/project_msc/2025/csml/rhautier/oe_proj
PY=/cs/student/msc/csml/2025/rhautier/miniforge3/envs/jaxued_env/bin/python
SCRIPT=/cs/student/project_msc/2025/csml/rhautier/oe_proj/examples/maze_plr.py
COMMON="--maze_height 13 --maze_width 13 --n_walls 25 --use_accel --use_cmaes --use_latent_mutations --vae_checkpoint /tmp/vae_13x13_checkpoint.pkl --vae_config /tmp/vae_13x13_config.yaml --cmaes_kl_threshold 0.1 --cmaes_sigma_min 0.1 --cmaes_kl_data /tmp/val_13x13_20k.npy --latent_mutation_alpha 0.5 --latent_mutation_top_k 64 --score_function sfl --num_sfl_rollouts 10 --checkpoint_save_interval 1 --project JAXUED_VAE_COMPARISON --run_name cmaes_latent_mut_sfl_13x13 --gcs_bucket ucl-ued-project-bucket --gcs_prefix accel/cmaes_latent_mut_sfl_13x13 --num_updates 30000"
$PY $SCRIPT $COMMON --seed 0
$PY $SCRIPT $COMMON --seed 1
$PY $SCRIPT $COMMON --seed 2
