#!/bin/bash
# Replicate the original 21x21 "cycle_2k3k5k_decay999_sfl" experiment
# (CMA-ES always on + score decay 0.999 cycling, no actual interpolation)
# on commit d0bfbfe, but add provenance tracking + checkpoint retention
# for embedding analysis.
#
# 3 seeds sequential (0, 1, 2).

export WANDB_ENTITY=romain-hautier-university-college-london-ucl-
export GOOGLE_CLOUD_PROJECT=open-endedness-ued-project
export WANDB_DIR=/cs/student/project_msc/2025/csml/rhautier/wandb_logs
mkdir -p $WANDB_DIR

OE_PROJ=/cs/student/project_msc/2025/csml/rhautier/oe_proj
cd $OE_PROJ

# Checkout the commit where cycling/decay flags were added (actual code that ran).
# wandb logged d0bfbfe (parent) but the flags only exist in ba0d807.
ORIG_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $ORIG_BRANCH"
echo "Checking out ba0d807 for maze_plr.py..."
git checkout ba0d807 -- examples/maze_plr.py

# Restore on exit (success or failure)
trap "cd $OE_PROJ && git checkout $ORIG_BRANCH -- examples/maze_plr.py && echo 'Restored maze_plr.py'" EXIT

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

RUN=cycle_2k3k5k_decay999_sfl_21x21_v2

SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then SEEDS=(0 1 2); fi

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "=== Cycle 21x21 (d0bfbfe) seed $SEED ==="
    $PY $SCRIPT \
        --maze_height 21 --maze_width 21 --n_walls 150 \
        --use_accel --use_cmaes \
        --vae_checkpoint_path $VAE_CKPT \
        --vae_config_path $VAE_CFG \
        --cmaes_kl_threshold 0.1 \
        --cmaes_kl_data $KL_DATA \
        --cmaes_reset_to_buffer_mean \
        --cmaes_phase_on 2000 --cmaes_phase_interp 3000 --cmaes_phase_off 5000 \
        --cmaes_interp_source buffer \
        --cmaes_score_decay 0.999 \
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
echo "Cycle 21x21 done."
