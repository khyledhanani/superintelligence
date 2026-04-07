#!/bin/bash
# CMA-ES + PLR + Latent-Noise Mutations on 21x21.
#
# Three branches:
#  - on_new_levels (~11%): CMA-ES searches VAE latent space, decodes, scores, inserts
#  - on_replay_levels (~44%): PLR replays high-scoring buffer levels, grad update
#  - on_mutate_levels (~44%): top-8 buffer levels encoded, each perturbed 4x with
#                             isotropic N(0, sigma^2 * I) in latent space, decoded,
#                             scored, inserted, grad update. NO wall-flip mutations.
#
# 3 seeds sequentially. Launch in parallel with the 13x13 script on a second GPU.

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

RUN=cmaes_plr_latentnoise_21x21_v1
export WANDB_RUN_GROUP=cmaes_plr_latentnoise_21x21_v1

# Allow overriding seeds from CLI, e.g. `bash run_cmaes_plr_latentnoise_21x21.sh 0 1 2`
SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then SEEDS=(0 1 2); fi

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "=== CMA-ES + PLR + latent-noise mut 21x21 seed $SEED ==="
    $PY $SCRIPT \
        --maze_height 21 --maze_width 21 --n_walls 150 \
        --use_accel --use_cmaes --use_latent_noise_mut \
        --latent_noise_top_k 8 --latent_noise_per_parent 4 --latent_noise_sigma 0.3 \
        --replay_prob 0.8 \
        --vae_checkpoint_path $VAE_CKPT \
        --vae_config_path $VAE_CFG \
        --cmaes_kl_threshold 0.1 --cmaes_sigma_min 0.1 \
        --cmaes_kl_data $KL_DATA \
        --score_function sfl --num_sfl_rollouts 10 \
        --checkpoint_save_interval 1 --max_number_of_checkpoints 999999 \
        --buffer_dump_interval 250 \
        --project JAXUED_VAE_COMPARISON \
        --run_name ${RUN} \
        --gcs_bucket ucl-ued-project-bucket \
        --gcs_prefix accel/${RUN} \
        --num_updates 30000 --seed $SEED
done

echo ""
echo "CMA-ES + PLR + latent-noise mut 21x21 done."
