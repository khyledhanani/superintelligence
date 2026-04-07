#!/bin/bash
# CMA-ES + PLR (no mutations) with --exploratory_grad_updates on 21x21.
# 3 seeds sequential.

export WANDB_ENTITY=romain-hautier-university-college-london-ucl-
export GOOGLE_CLOUD_PROJECT=open-endedness-ued-project
export WANDB_DIR=/cs/student/project_msc/2025/csml/rhautier/wandb_logs
mkdir -p $WANDB_DIR

cd /cs/student/project_msc/2025/csml/rhautier/oe_proj

PY=/cs/student/msc/csml/2025/rhautier/miniforge3/envs/jaxued_env/bin/python
SCRIPT=examples/maze_plr.py

VAE_CKPT=/cs/student/project_msc/2025/csml/rhautier/vae_checkpoints/21x21/checkpoint_final.pkl
VAE_CFG=/cs/student/project_msc/2025/csml/rhautier/vae_checkpoints/21x21/config.yaml
KL_DATA=/cs/student/project_msc/2025/csml/rhautier/vae_datasets/val_21x21_20k.npy

if [ ! -f "$KL_DATA" ]; then
    echo "Pulling 21x21 validation data from GCS..."
    mkdir -p $(dirname $KL_DATA)
    python3 -c "
from google.cloud import storage
client = storage.Client(project='open-endedness-ued-project')
bucket = client.bucket('ucl-ued-project-bucket')
bucket.blob('vae/datasets/val_21x21_20k.npy').download_to_filename('$KL_DATA')
print('Downloaded.')
"
fi

RUN=cmaes_plr_nomut_sfl_21x21_v3

SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then SEEDS=(0 1 2); fi

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "=== CMA-ES + PLR (no mut, explore_grad) 21x21 seed $SEED ==="
    $PY $SCRIPT \
        --maze_height 21 --maze_width 21 --n_walls 150 \
        --use_cmaes \
        --exploratory_grad_updates \
        --vae_checkpoint_path $VAE_CKPT \
        --vae_config_path $VAE_CFG \
        --cmaes_kl_threshold 0.1 --cmaes_sigma_min 0.1 \
        --cmaes_kl_data $KL_DATA \
        --score_function sfl --num_sfl_rollouts 10 \
        --checkpoint_save_interval 1 --max_number_of_checkpoints 120 \
        --buffer_dump_interval 250 \
        --project JAXUED_VAE_COMPARISON \
        --run_name ${RUN} \
        --gcs_bucket ucl-ued-project-bucket \
        --gcs_prefix accel/${RUN} \
        --num_updates 30000 --seed $SEED
done

echo ""
echo "CMA-ES + PLR (no mut, explore_grad) 21x21 done."
