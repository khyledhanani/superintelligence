#!/bin/bash
# Trajectory VAE prototype pipeline
#
# Tests the hypothesis: does encoding agent trajectories (behavior)
# produce a latent space that correlates with regret better than
# encoding levels (structure)?
#
# Steps:
#   1. Collect trajectories by rolling out agent on levels
#   2. Train trajectory VAE on position sequences
#   3. Analyze latent-regret correlation (vs level VAE baseline)

set -euo pipefail

PYTHON="/cs/student/msc/csml/2025/rhautier/miniforge3/envs/jaxued_env/bin/python"
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$BASE_DIR")"

# --- Configuration ---
AGENT_CHECKPOINT="${1:?Usage: $0 <agent_checkpoint> [vae_checkpoint] [vae_config]}"
VAE_CHECKPOINT="${2:-}"
VAE_CONFIG="${3:-}"

N_LEVELS=10000
SOURCE="random"  # "random" needs no VAE, "prior" needs VAE
LATENT_DIM=32
EMBED_DIM=128
EPOCHS=200
BATCH_SIZE=128
KL_WARMUP=20

# If VAE paths provided, use prior sampling
if [ -n "$VAE_CHECKPOINT" ] && [ -n "$VAE_CONFIG" ]; then
    SOURCE="prior"
fi

RUN_NAME="traj_vae_${SOURCE}_${N_LEVELS}_d${LATENT_DIM}"
DATA_DIR="${BASE_DIR}/data"
CKPT_DIR="${BASE_DIR}/checkpoints/${RUN_NAME}"
ANALYSIS_DIR="${BASE_DIR}/analysis/${RUN_NAME}"

mkdir -p "$DATA_DIR" "$CKPT_DIR" "$ANALYSIS_DIR"

echo "=== Trajectory VAE Pipeline ==="
echo "  Agent:     $AGENT_CHECKPOINT"
echo "  Source:    $SOURCE"
echo "  N levels:  $N_LEVELS"
echo "  Latent dim: $LATENT_DIM"
echo "  Run name:  $RUN_NAME"
echo ""

# --- Step 1: Collect trajectories ---
echo "[Step 1/3] Collecting trajectories..."
DATA_PATH="${DATA_DIR}/trajectories_${SOURCE}_${N_LEVELS}.npz"

if [ -f "$DATA_PATH" ]; then
    echo "  Data already exists at $DATA_PATH, skipping collection."
else
    COLLECT_ARGS=(
        "$BASE_DIR/collect_trajectories.py"
        --agent_checkpoint "$AGENT_CHECKPOINT"
        --n_levels "$N_LEVELS"
        --source "$SOURCE"
        --output_path "$DATA_PATH"
        --batch_size 256
    )
    if [ "$SOURCE" = "prior" ]; then
        COLLECT_ARGS+=(--vae_checkpoint_path "$VAE_CHECKPOINT" --vae_config_path "$VAE_CONFIG")
    fi
    $PYTHON "${COLLECT_ARGS[@]}"
fi
echo ""

# --- Step 2: Train trajectory VAE ---
echo "[Step 2/3] Training trajectory VAE..."
$PYTHON "$BASE_DIR/train_trajectory_vae.py" \
    --data_path "$DATA_PATH" \
    --output_dir "$CKPT_DIR" \
    --latent_dim "$LATENT_DIM" \
    --embed_dim "$EMBED_DIM" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --kl_warmup_epochs "$KL_WARMUP"
echo ""

# --- Step 3: Analyze correlation ---
echo "[Step 3/3] Analyzing latent-regret correlation..."
ANALYSIS_ARGS=(
    "$BASE_DIR/analyze_correlation.py"
    --data_path "$DATA_PATH"
    --traj_vae_path "${CKPT_DIR}/best_model.pkl"
    --output_dir "$ANALYSIS_DIR"
)
if [ -n "$VAE_CHECKPOINT" ] && [ -n "$VAE_CONFIG" ]; then
    ANALYSIS_ARGS+=(--level_vae_path "$VAE_CHECKPOINT" --level_vae_config "$VAE_CONFIG")
fi
$PYTHON "${ANALYSIS_ARGS[@]}"

echo ""
echo "=== Pipeline complete ==="
echo "  Data:      $DATA_PATH"
echo "  Model:     ${CKPT_DIR}/best_model.pkl"
echo "  Analysis:  $ANALYSIS_DIR/"
