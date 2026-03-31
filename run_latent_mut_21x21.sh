#!/bin/bash
export WANDB_ENTITY=romain-hautier-university-college-london-ucl-
export GOOGLE_CLOUD_PROJECT=open-endedness-ued-project
cd /cs/student/project_msc/2025/csml/rhautier/oe_proj
PY=/cs/student/msc/csml/2025/rhautier/miniforge3/envs/jaxued_env/bin/python
SCRIPT=/cs/student/project_msc/2025/csml/rhautier/oe_proj/examples/maze_plr.py
COMMON="--maze_height 21 --maze_width 21 --n_walls 150 --use_accel --use_latent_mutations --vae_checkpoint /tmp/vae_21x21_final.pkl --vae_config /tmp/vae_21x21_config/config.yaml --latent_mutation_alpha 0.5 --latent_mutation_top_k 64 --score_function sfl --num_sfl_rollouts 10 --project JAXUED_VAE_COMPARISON --run_name latent_mut_sfl_21x21 --gcs_bucket ucl-ued-project-bucket --gcs_prefix accel/latent_mut_sfl_21x21 --num_updates 30000"
$PY $SCRIPT $COMMON --seed 0
$PY $SCRIPT $COMMON --seed 1
$PY $SCRIPT $COMMON --seed 2
