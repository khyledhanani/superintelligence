#!/bin/bash
set -e
export WANDB_DIR=${WANDB_DIR:-/cs/student/project_msc/2025/csml/rhautier/tmp/wandb_logs}

# === CONFIG (all overridable via env vars) ===
VAE_CKPT="${VAE_CKPT:-/tmp/vae_beta10/checkpoint_500000.pkl}"
VAE_CFG="${VAE_CFG:-/tmp/vae_beta10/config.yaml}"
BUFFER="${BUFFER:-/cs/student/project_msc/2025/csml/rhautier/tmp/buffer_dumps/cmaes_pca30_start6k_refit1halfk_kl_s/1/buffer_dump_10k.npz}"
OUT_DIR="${OUT_DIR:-/cs/student/project_msc/2025/csml/rhautier/tmp/adapter_data}"
KL_THRESHOLD="${KL_THRESHOLD:-0.1}"
PRIOR_MULT="${PRIOR_MULT:-3}"        # sample 3x buffer_size from prior N(0,I)
TEST_SPLIT="${TEST_SPLIT:-0.2}"      # 20% held out for test
AGENT_CKPT="${AGENT_CKPT:-}"          # orbax checkpoint dir (e.g. /tmp/agent_checkpoint/40)
SCORE_FUNCTION="${SCORE_FUNCTION:-maxmc}" # scoring: maxmc (regret) or sfl (learnability)
NUM_ROLLOUTS="${NUM_ROLLOUTS:-5}"     # rollouts per level for scoring
ROLLOUT_BATCH="${ROLLOUT_BATCH:-256}" # batch size for rollouts
EPOCHS="${EPOCHS:-300}"
PROJECT="${PROJECT:-ADAPTER_TRAINING}"

mkdir -p "${OUT_DIR}"

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
    --hidden_dim 128 --n_layers 2 \
    --lambda_regret 0.1 --lambda_reg 0.01 --lambda_pred 1.0 \
    --epochs ${EPOCHS} --lr 1e-3 --batch_size 256 \
    --project "${PROJECT}" \
    --run_name "joint_beta10_kl${KL_THRESHOLD}_prior${PRIOR_MULT}x"

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

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "  Train data:  ${OUT_DIR}/train_data.npz"
echo "  Test data:   ${OUT_DIR}/train_data_test.npz"
echo "  Predictor:   ${OUT_DIR}/predictor.pkl"
echo "  Adapter:     ${OUT_DIR}/adapter.pkl"
echo "  Diagnostics: ${OUT_DIR}/diagnostics/"
echo "=========================================="
