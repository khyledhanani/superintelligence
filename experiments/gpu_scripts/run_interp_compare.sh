#!/bin/bash
# Run ACCEL training with shadow interp-vs-ACCEL comparison logging.
# 3 seeds for both 13x13 and 21x21.
#
# Usage:
#   bash run_interp_compare.sh          # runs all 6 (2 sizes x 3 seeds)
#   bash run_interp_compare.sh 13       # runs 3 seeds for 13x13 only
#   bash run_interp_compare.sh 21       # runs 3 seeds for 21x21 only
#   bash run_interp_compare.sh 13 1     # runs single seed

PY=/cs/student/msc/csml/2025/rhautier/miniforge3/envs/jaxued_env/bin/python
SCRIPT=/cs/student/msc/csml/2025/rhautier/Documents/jaxued/jaxued/examples/maze_plr.py
VAE_BASE=/cs/student/project_msc/2025/csml/rhautier/vae_checkpoints

export WANDB_ENTITY=romain-hautier-university-college-london-ucl-
export GOOGLE_CLOUD_PROJECT=open-endedness-ued-project

run_one() {
    local SIZE=$1
    local SEED=$2

    if [ "$SIZE" = "13" ]; then
        WALLS=25
        VAE_CKPT=${VAE_BASE}/13x13/checkpoint_500000.pkl
        VAE_CFG=${VAE_BASE}/13x13/config.yaml
    elif [ "$SIZE" = "21" ]; then
        WALLS=150
        VAE_CKPT=${VAE_BASE}/21x21/checkpoint_final.pkl
        VAE_CFG=${VAE_BASE}/21x21/config.yaml
    else
        echo "Unknown size: $SIZE"; return 1
    fi

    if [ ! -f "$VAE_CKPT" ]; then echo "VAE checkpoint not found: $VAE_CKPT"; return 1; fi
    if [ ! -f "$VAE_CFG" ]; then echo "VAE config not found: $VAE_CFG"; return 1; fi

    local GROUP=interp_compare_${SIZE}x${SIZE}
    local RUN_NAME=${GROUP}_s${SEED}

    echo "============================================"
    echo "  ${SIZE}x${SIZE} seed=${SEED}  (group=${GROUP})"
    echo "============================================"

    $PY $SCRIPT \
        --maze_height $SIZE --maze_width $SIZE --n_walls $WALLS \
        --use_accel \
        --vae_checkpoint_path $VAE_CKPT \
        --vae_config_path $VAE_CFG \
        --score_function sfl --num_sfl_rollouts 10 \
        --checkpoint_save_interval 1 \
        --project JAXUED_VAE_COMPARISON \
        --wandb_group $GROUP \
        --run_name $RUN_NAME \
        --gcs_bucket ucl-ued-project-bucket \
        --gcs_prefix accel/$RUN_NAME \
        --interp_compare_interval 250 \
        --num_updates 30000 --seed $SEED
}

# Parse args
SIZE_ARG=${1:-all}
SEED_ARG=${2:-all}

if [ "$SIZE_ARG" = "all" ]; then
    SIZES="13 21"
else
    SIZES="$SIZE_ARG"
fi

for S in $SIZES; do
    if [ "$SEED_ARG" = "all" ]; then
        for SEED in 0 1 2; do
            run_one $S $SEED
        done
    else
        run_one $S $SEED_ARG
    fi
done

echo ""
echo "All done."
