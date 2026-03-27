#!/bin/bash
# ==============================================================================
# ACCEL + SFL baseline on 13x13, with mean_embeddings in buffer dumps.
#
# Replicates accel_sfl_baseline_13x13 but saves mean LSTM state-action
# embeddings (shape N x 257) in every buffer dump .npz.
#
# Usage:
#   bash run_accel_sfl_emb_13x13.sh
# ==============================================================================
set -e

BUCKET="ucl-ued-project-bucket"
PREFIX="llm-exp"
PROJECT="OEGI"
RUN_NAME="accel_sfl_emb_13x13"

export WANDB_DIR=/tmp/wandb
export WANDB_ENTITY=shr1ramrg
export WANDB_PROJECT=OEGI
export PYTHONUNBUFFERED=1
export PYTHONPATH="src:."
export GOOGLE_CLOUD_PROJECT=open-endedness-ued-project

COMMON="--project $PROJECT \
        --run_name $RUN_NAME \
        --use_accel \
        --score_function sfl \
        --num_updates 10000 \
        --eval_freq 250 \
        --buffer_dump_interval 10000 \
        --gcs_bucket $BUCKET \
        --gcs_prefix $PREFIX"

python3 examples/maze_plr.py $COMMON --seed 0

echo ""
echo "============================================"
echo "  Done: $RUN_NAME"
echo "  GCS buffer: gs://$BUCKET/$PREFIX/buffer_dumps/$RUN_NAME/0/"
echo "  GCS checkpoint: gs://$BUCKET/$PREFIX/checkpoints/$RUN_NAME/0/"
echo "============================================"
