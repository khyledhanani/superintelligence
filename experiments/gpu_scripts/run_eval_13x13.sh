#!/bin/bash
# Evaluate final checkpoints (step 119 = 30k updates) on 13x13 ACCEL test levels
# 100 rollouts per level, logs to wandb.
#
# Usage:
#   bash experiments/gpu_scripts/run_eval_13x13.sh              # all methods
#   bash experiments/gpu_scripts/run_eval_13x13.sh accel        # single method

export WANDB_ENTITY=romain-hautier-university-college-london-ucl-
export GOOGLE_CLOUD_PROJECT=open-endedness-ued-project
export WANDB_DIR=/cs/student/project_msc/2025/csml/rhautier/wandb_logs
mkdir -p $WANDB_DIR

cd /cs/student/project_msc/2025/csml/rhautier/oe_proj

PY=/cs/student/msc/csml/2025/rhautier/miniforge3/envs/jaxued_env/bin/python
EVAL_SCRIPT=examples/eval_test_levels.py
GCS_BUCKET=ucl-ued-project-bucket
LOCAL_BASE=/cs/student/project_msc/2025/csml/rhautier/eval_checkpoints_13x13
RESULTS_DIR=/cs/student/project_msc/2025/csml/rhautier/eval_results_13x13
CKPT_STEP=${2:-119}

mkdir -p $LOCAL_BASE $RESULTS_DIR

DL_SCRIPT=/cs/student/msc/csml/2025/rhautier/Documents/jaxued/jaxued/scripts/download_gcs_checkpoint.py

download_ckpt() {
    local GCS_DIR=$1
    local LOCAL_DIR=$2
    if [ -f "$LOCAL_DIR/config.json" ]; then
        echo "  Already downloaded: $LOCAL_DIR"
        return
    fi
    echo "  Downloading $GCS_DIR ..."
    python3 $DL_SCRIPT --gcs_dir "$GCS_DIR" --local_dir "$LOCAL_DIR" --step $CKPT_STEP
}

# GCS paths: accel/{gcs_prefix}/checkpoints/{run_id}/{seed}/
declare -A METHODS

# PLR with uniform DR (seeds 0,1,2)
METHODS[plr]="plr_uniformdr_sfl_13x13_v1:plr_uniformdr_sfl_13x13_v1/0,plr_uniformdr_sfl_13x13_v1/1,plr_uniformdr_sfl_13x13_v1/2"

# ACCEL with uniform DR (seeds 0,1,2)
METHODS[accel]="accel_uniformdr_sfl_13x13_v1:accel_uniformdr_sfl_13x13_v1/0,accel_uniformdr_sfl_13x13_v1/1,accel_uniformdr_sfl_13x13_v1/2"

# CMA-ES + PLR (seeds 0,1,2)
METHODS[cmaes_nomut]="cmaes_plr_nomut_sfl_13x13_v3:cmaes_plr_nomut_sfl_13x13_v3/0,cmaes_plr_nomut_sfl_13x13_v3/1,cmaes_plr_nomut_sfl_13x13_v3/2"

# CMA-ES + ACCEL (seeds 3,4,5)
METHODS[cmaes_accel]="cmaes_accel_sfl_13x13_v2:cmaes_accel_sfl_13x13_v2_s3/3,cmaes_accel_sfl_13x13_v2_s4/4,cmaes_accel_sfl_13x13_v2_s5/5"

# CMA-ES + Latent Interp (seeds 3,4,5)
METHODS[cmaes_latent_interp]="cmaes_latent_mut_sfl_13x13_v2:cmaes_latent_mut_sfl_13x13_v2_s3/3,cmaes_latent_mut_sfl_13x13_v2_s4/4,cmaes_latent_mut_sfl_13x13_v2_s5/5"

# CMA-ES + Latent Noise (seeds 0,1,2)
METHODS[cmaes_latentnoise]="cmaes_plr_latentnoise_13x13_v1:cmaes_plr_latentnoise_13x13_v1/0,cmaes_plr_latentnoise_13x13_v1/1,cmaes_plr_latentnoise_13x13_v1/2"

# CMA-ES only (seeds 0,1,2)
METHODS[cmaes_only]="cmaes_only_sfl_13x13_v3:cmaes_only_sfl_13x13_v3/0,cmaes_only_sfl_13x13_v3/1,cmaes_only_sfl_13x13_v3/2"

FILTER=${1:-all}

for METHOD in "${!METHODS[@]}"; do
    if [ "$FILTER" != "all" ] && [ "$FILTER" != "$METHOD" ]; then
        continue
    fi

    IFS=: read -r GCS_PREFIX SEEDS_STR <<< "${METHODS[$METHOD]}"
    IFS=, read -ra SEEDS <<< "$SEEDS_STR"

    echo ""
    echo "============================================"
    echo "  Evaluating: $METHOD ($GCS_PREFIX)"
    echo "============================================"

    CKPT_DIRS=""
    for SEED_PATH in "${SEEDS[@]}"; do
        GCS_CKPT="accel/$GCS_PREFIX/checkpoints/$SEED_PATH"
        LOCAL_CKPT="$LOCAL_BASE/$METHOD/$(echo $SEED_PATH | tr '/' '_')"
        download_ckpt "$GCS_CKPT" "$LOCAL_CKPT"
        CKPT_DIRS="$CKPT_DIRS $LOCAL_CKPT"
    done

    echo "  Running eval_test_levels.py..."
    $PY $EVAL_SCRIPT --batch \
        --checkpoint_dirs $CKPT_DIRS \
        --checkpoint_steps $CKPT_STEP \
        --num_attempts 100 \
        --output_dir $RESULTS_DIR/$METHOD \
        --wandb_project vae_13x13_results \
        --wandb_group "${METHOD}_eval100"

done

echo ""
echo "All evals done. Results in $RESULTS_DIR/"
echo "Run plot_eval_comparison.py to generate plots:"
echo "  python vae/plot_eval_comparison.py --results_dir $RESULTS_DIR"
