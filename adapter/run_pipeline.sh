#!/bin/bash
set -e
export WANDB_DIR=${WANDB_DIR:-/cs/student/project_msc/2025/csml/rhautier/tmp/wandb_logs}

# === CONFIG (all overridable via env vars) ===
# -- Paths --
VAE_CKPT="${VAE_CKPT:-/tmp/vae_beta10/checkpoint_500000.pkl}"
VAE_CFG="${VAE_CFG:-/tmp/vae_beta10/config.yaml}"
BUFFER="${BUFFER:-/cs/student/project_msc/2025/csml/rhautier/tmp/buffer_dumps/cmaes_pca30_start6k_refit1halfk_kl_s/1/buffer_dump_10k.npz}"
OUT_DIR="${OUT_DIR:-/cs/student/project_msc/2025/csml/rhautier/tmp/adapter_data}"
AGENT_CKPT="${AGENT_CKPT:-}"          # orbax checkpoint dir (e.g. /tmp/agent_checkpoint/120)

# -- Data preparation --
KL_THRESHOLD="${KL_THRESHOLD:-0.1}"
PRIOR_MULT="${PRIOR_MULT:-3}"         # sample N * buffer_size from prior N(0,I)
TEST_SPLIT="${TEST_SPLIT:-0.2}"       # fraction held out for test
SCORE_FUNCTION="${SCORE_FUNCTION:-maxmc}" # maxmc (regret) or sfl (learnability)
NUM_ROLLOUTS="${NUM_ROLLOUTS:-5}"     # rollouts per level for scoring
ROLLOUT_BATCH="${ROLLOUT_BATCH:-256}" # batch size for rollouts

# -- Training hyperparameters --
EPOCHS="${EPOCHS:-300}"
LR="${LR:-1e-3}"
BATCH_SIZE="${BATCH_SIZE:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
N_LAYERS="${N_LAYERS:-2}"
LAMBDA_REGRET="${LAMBDA_REGRET:-0.1}"
LAMBDA_REG="${LAMBDA_REG:-0.01}"
LAMBDA_PRED="${LAMBDA_PRED:-1.0}"

# -- Logging --
PROJECT="${PROJECT:-ADAPTER_TRAINING}"
RUN_NAME="${RUN_NAME:-joint_beta10_kl${KL_THRESHOLD}_prior${PRIOR_MULT}x_${SCORE_FUNCTION}}"
GCS_BUCKET="${GCS_BUCKET:-gs://ucl-ued-project-bucket}"
GCS_PREFIX="${GCS_PREFIX:-adapter_results}"

# -- Steps to run (set to 0 to skip) --
RUN_STEP1="${RUN_STEP1:-1}"  # data preparation
RUN_STEP2="${RUN_STEP2:-1}"  # training
RUN_STEP3="${RUN_STEP3:-1}"  # diagnostics
RUN_STEP4="${RUN_STEP4:-1}"  # GCS upload

mkdir -p "${OUT_DIR}"

echo "=========================================="
echo "Config:"
echo "  VAE:          ${VAE_CKPT}"
echo "  Buffer:       ${BUFFER}"
echo "  Agent:        ${AGENT_CKPT:-NONE}"
echo "  Score:        ${SCORE_FUNCTION} (${NUM_ROLLOUTS} rollouts)"
echo "  Output:       ${OUT_DIR}"
echo "  Lambdas:      regret=${LAMBDA_REGRET} reg=${LAMBDA_REG} pred=${LAMBDA_PRED}"
echo "  Model:        hidden=${HIDDEN_DIM} layers=${N_LAYERS}"
echo "  Training:     epochs=${EPOCHS} lr=${LR} batch=${BATCH_SIZE}"
echo "=========================================="

# === Step 1: Prepare data ===
if [ "${RUN_STEP1}" = "1" ]; then
    echo ""
    echo "=========================================="
    echo "Step 1: Prepare data (buffer + ${PRIOR_MULT}x prior + KL eviction + train/test split)"
    echo "=========================================="
    AGENT_ARGS=""
    if [ -n "${AGENT_CKPT}" ]; then
        AGENT_ARGS="--agent_checkpoint_path ${AGENT_CKPT} --num_rollouts ${NUM_ROLLOUTS} --rollout_batch_size ${ROLLOUT_BATCH} --score_function ${SCORE_FUNCTION}"
        echo "  Agent checkpoint: ${AGENT_CKPT} (${NUM_ROLLOUTS} rollouts/level, score=${SCORE_FUNCTION})"
    else
        echo "  WARNING: No AGENT_CKPT set — prior samples will get score=0"
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
else
    echo "Step 1: SKIPPED (RUN_STEP1=0)"
fi

# === Step 2: Joint training ===
if [ "${RUN_STEP2}" = "1" ]; then
    echo ""
    echo "=========================================="
    echo "Step 2: Joint training of predictor + adapter (${EPOCHS} epochs)"
    echo "=========================================="
    python3 adapter/train_joint.py \
        --data_path "${OUT_DIR}/train_data.npz" \
        --test_data_path "${OUT_DIR}/train_data_test.npz" \
        --vae_checkpoint_path "${VAE_CKPT}" \
        --vae_config_path "${VAE_CFG}" \
        --output_dir "${OUT_DIR}" \
        --hidden_dim ${HIDDEN_DIM} --n_layers ${N_LAYERS} \
        --lambda_regret ${LAMBDA_REGRET} --lambda_reg ${LAMBDA_REG} --lambda_pred ${LAMBDA_PRED} \
        --epochs ${EPOCHS} --lr ${LR} --batch_size ${BATCH_SIZE} \
        --project "${PROJECT}" \
        --run_name "${RUN_NAME}"
else
    echo "Step 2: SKIPPED (RUN_STEP2=0)"
fi

# === Step 3: Diagnostics ===
if [ "${RUN_STEP3}" = "1" ]; then
    echo ""
    echo "=========================================="
    echo "Step 3: Diagnostics (PCA viz, reconstruction, delta_z analysis)"
    echo "=========================================="
    python3 adapter/diagnostics.py \
        --data_path "${OUT_DIR}/train_data.npz" \
        --adapter_path "${OUT_DIR}/adapter.pkl" \
        --vae_checkpoint_path "${VAE_CKPT}" \
        --vae_config_path "${VAE_CFG}" \
        --output_dir "${OUT_DIR}/diagnostics/"
else
    echo "Step 3: SKIPPED (RUN_STEP3=0)"
fi

# === Step 4: Upload to GCS ===
if [ "${RUN_STEP4}" = "1" ]; then
    echo ""
    echo "=========================================="
    echo "Step 4: Upload results to GCS"
    echo "=========================================="
    GCS_DEST="${GCS_BUCKET}/${GCS_PREFIX}/${RUN_NAME}/"
    echo "  Uploading ${OUT_DIR} → ${GCS_DEST}"
    gcloud storage cp -r "${OUT_DIR}/" "${GCS_DEST}"
    echo "  Upload complete: ${GCS_DEST}"
else
    echo "Step 4: SKIPPED (RUN_STEP4=0)"
fi

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "  Train data:  ${OUT_DIR}/train_data.npz"
echo "  Test data:   ${OUT_DIR}/train_data_test.npz"
echo "  Predictor:   ${OUT_DIR}/predictor.pkl"
echo "  Adapter:     ${OUT_DIR}/adapter.pkl"
echo "  Diagnostics: ${OUT_DIR}/diagnostics/"
if [ "${RUN_STEP4}" = "1" ]; then
    echo "  GCS:         ${GCS_DEST}"
fi
echo "=========================================="
