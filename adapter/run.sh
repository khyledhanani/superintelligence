#!/bin/bash
# ============================================================
# Edit parameters below, then run:  bash adapter/run.sh
# ============================================================
set -e

# -- Paths --
VAE_CKPT="/tmp/vae_beta10/checkpoint_500000.pkl"
VAE_CFG="/tmp/vae_beta10/config.yaml"
BUFFER="/tmp/buffer_dump_30k.npz"
AGENT_CKPT="/tmp/agent_checkpoint/120/120"
OUT_DIR="/tmp/adapter_data_maxmc_v2"

# -- Data preparation --
KL_THRESHOLD=0.1
PRIOR_MULT=3
TEST_SPLIT=0.2
SCORE_FUNCTION="maxmc"
NUM_ROLLOUTS=5
ROLLOUT_BATCH=256

# -- Training hyperparameters --
EPOCHS=300
LR=1e-4
BATCH_SIZE=256
HIDDEN_DIM=128
N_LAYERS=2
WEIGHT_DECAY=1e-5

# -- Loss weights --
LAMBDA_PRED=1.0       # predictor MSE (z -> score)
LAMBDA_REGRET=5.0     # regret-maximising direction for adapter
LAMBDA_REG=0.1        # ||delta_z||^2 regularisation
LAMBDA_KL=0.0         # KL penalty on adapted z

# -- Logging --
export WANDB_DIR="${WANDB_DIR:-/tmp/wandb_logs}"
PROJECT="ADAPTER_TRAINING"
RUN_NAME="maxmc_lr${LAMBDA_REGRET}_reg${LAMBDA_REG}"

# -- GCS upload --
GCS_BUCKET="gs://ucl-ued-project-bucket"
GCS_PREFIX="adapter_results"

# -- Steps to run (comment out to skip) --
RUN_STEP1=1   # data preparation
RUN_STEP2=1   # training
RUN_STEP3=1   # diagnostics
RUN_STEP4=1   # GCS upload

# ============================================================
# DO NOT EDIT BELOW THIS LINE
# ============================================================
mkdir -p "${OUT_DIR}"

echo "=========================================="
echo "Config:"
echo "  VAE:      ${VAE_CKPT}"
echo "  Buffer:   ${BUFFER}"
echo "  Agent:    ${AGENT_CKPT:-NONE}"
echo "  Score:    ${SCORE_FUNCTION} (${NUM_ROLLOUTS} rollouts)"
echo "  Output:   ${OUT_DIR}"
echo "  Lambdas:  pred=${LAMBDA_PRED} regret=${LAMBDA_REGRET} reg=${LAMBDA_REG} kl=${LAMBDA_KL}"
echo "  Model:    hidden=${HIDDEN_DIM} layers=${N_LAYERS}"
echo "  Training: epochs=${EPOCHS} lr=${LR} batch=${BATCH_SIZE}"
echo "=========================================="

if [ "${RUN_STEP1}" = "1" ]; then
    echo ""
    echo "=========================================="
    echo "Step 1: Prepare data"
    echo "=========================================="
    AGENT_ARGS=""
    if [ -n "${AGENT_CKPT}" ]; then
        AGENT_ARGS="--agent_checkpoint_path ${AGENT_CKPT} --num_rollouts ${NUM_ROLLOUTS} --rollout_batch_size ${ROLLOUT_BATCH} --score_function ${SCORE_FUNCTION}"
    fi
    python3 adapter/prepare_data_from_buffer.py \
        --buffer_path "${BUFFER}" \
        --vae_checkpoint_path "${VAE_CKPT}" \
        --vae_config_path "${VAE_CFG}" \
        --kl_threshold ${KL_THRESHOLD} \
        --prior_multiplier ${PRIOR_MULT} \
        --test_split ${TEST_SPLIT} \
        --output_path "${OUT_DIR}/train_data.npz" \
        ${AGENT_ARGS}
fi

if [ "${RUN_STEP2}" = "1" ]; then
    echo ""
    echo "=========================================="
    echo "Step 2: Training (${EPOCHS} epochs)"
    echo "=========================================="
    python3 adapter/train_joint.py \
        --data_path "${OUT_DIR}/train_data.npz" \
        --test_data_path "${OUT_DIR}/train_data_test.npz" \
        --vae_checkpoint_path "${VAE_CKPT}" \
        --vae_config_path "${VAE_CFG}" \
        --output_dir "${OUT_DIR}" \
        --hidden_dim ${HIDDEN_DIM} --n_layers ${N_LAYERS} \
        --lambda_regret ${LAMBDA_REGRET} --lambda_reg ${LAMBDA_REG} --lambda_pred ${LAMBDA_PRED} --lambda_kl ${LAMBDA_KL} \
        --epochs ${EPOCHS} --lr ${LR} --batch_size ${BATCH_SIZE} --weight_decay ${WEIGHT_DECAY} \
        --project "${PROJECT}" \
        --run_name "${RUN_NAME}"
fi

if [ "${RUN_STEP3}" = "1" ]; then
    echo ""
    echo "=========================================="
    echo "Step 3: Diagnostics"
    echo "=========================================="
    python3 adapter/diagnostics.py \
        --data_path "${OUT_DIR}/train_data.npz" \
        --adapter_path "${OUT_DIR}/adapter.pkl" \
        --vae_checkpoint_path "${VAE_CKPT}" \
        --vae_config_path "${VAE_CFG}" \
        --output_dir "${OUT_DIR}/diagnostics/"
fi

if [ "${RUN_STEP4}" = "1" ]; then
    echo ""
    echo "=========================================="
    echo "Step 4: Upload to GCS"
    echo "=========================================="
    GCS_DEST="${GCS_BUCKET}/${GCS_PREFIX}/${RUN_NAME}/"
    echo "  Uploading ${OUT_DIR} → ${GCS_DEST}"
    gcloud storage cp -r "${OUT_DIR}/" "${GCS_DEST}"
    echo "  Upload complete: ${GCS_DEST}"
fi

echo ""
echo "=========================================="
echo "Done! Output: ${OUT_DIR}"
echo "=========================================="
