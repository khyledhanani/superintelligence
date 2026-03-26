#!/bin/bash
# Experiment grid: mutation strategy × flood ratio
#
# Edit the variables below to match your setup, then run:
#   bash experiments/run_grid.sh
#
# Prerequisites:
#   1. Run harvest_llm_seeds.py first to generate seeds
#   2. Have agent checkpoint + buffer dump ready

set -e

# === EDIT THESE ===
AGENT_CKPT="checkpoints/cmaes_vae_beta2.0_seed0_198"
BUFFER_NPZ="/tmp/buffer_dumps/run_name/seed/buffer_dump_30k.npz"
SEEDS_DIR="experiments/seeds/run1"
VAE_CKPT="vae/runs/best/checkpoint.pkl"
VAE_CONFIG="vae/runs/best/config.yaml"
BASE_OUTPUT="experiments/results"
SEED=0
N_MUTATIONS=30
N_SCORING_ROLLOUTS=10

# === EXPERIMENT GRID ===
STRATEGIES=("wall_flip" "vae_noise" "vae_interpolation")
FLOOD_RATIOS=("0.05" "0.1" "0.2" "0.5")
VAE_SIGMAS=("0.3" "0.5" "1.0")  # only for vae_noise

echo "========================================"
echo "LLM Injection Experiment Grid"
echo "========================================"
echo "Agent:     $AGENT_CKPT"
echo "Buffer:    $BUFFER_NPZ"
echo "Seeds:     $SEEDS_DIR"
echo "Strategies: ${STRATEGIES[*]}"
echo "Flood ratios: ${FLOOD_RATIOS[*]}"
echo ""

# --- Stage 1: Harvest seeds (if not already done) ---
if [ ! -f "$SEEDS_DIR/seeds_levels.pkl" ]; then
    echo "[Stage 1] Harvesting LLM seeds..."
    python experiments/harvest_llm_seeds.py \
        --agent_checkpoint_dir "$AGENT_CKPT" \
        --buffer_npz "$BUFFER_NPZ" \
        --n_seeds 10 \
        --llm_provider claude-code \
        --llm_model claude-sonnet-4-20250514 \
        --output_dir "$SEEDS_DIR"
    echo ""
fi

# --- Stage 2: Run mutation × flood ratio grid ---
for strategy in "${STRATEGIES[@]}"; do
    for ratio in "${FLOOD_RATIOS[@]}"; do

        # For vae_noise, also sweep sigma
        if [ "$strategy" = "vae_noise" ]; then
            for sigma in "${VAE_SIGMAS[@]}"; do
                run_name="${strategy}_s${sigma}_f${ratio}_seed${SEED}"
                output_dir="${BASE_OUTPUT}/${run_name}"

                if [ -f "$output_dir/experiment_log.json" ]; then
                    echo "[SKIP] $run_name (already done)"
                    continue
                fi

                echo "[RUN] $run_name"
                python experiments/run_injection_experiment.py \
                    --agent_checkpoint_dir "$AGENT_CKPT" \
                    --buffer_npz "$BUFFER_NPZ" \
                    --seeds_dir "$SEEDS_DIR" \
                    --mutation_strategy "$strategy" \
                    --flood_ratio "$ratio" \
                    --n_mutations_per_seed "$N_MUTATIONS" \
                    --n_scoring_rollouts "$N_SCORING_ROLLOUTS" \
                    --vae_checkpoint_path "$VAE_CKPT" \
                    --vae_config_path "$VAE_CONFIG" \
                    --vae_sigma "$sigma" \
                    --output_dir "$output_dir" \
                    --seed "$SEED"
                echo ""
            done
        else
            run_name="${strategy}_f${ratio}_seed${SEED}"
            output_dir="${BASE_OUTPUT}/${run_name}"

            if [ -f "$output_dir/experiment_log.json" ]; then
                echo "[SKIP] $run_name (already done)"
                continue
            fi

            echo "[RUN] $run_name"

            extra_args=""
            if [ "$strategy" = "vae_interpolation" ]; then
                extra_args="--vae_checkpoint_path $VAE_CKPT --vae_config_path $VAE_CONFIG"
            fi

            python experiments/run_injection_experiment.py \
                --agent_checkpoint_dir "$AGENT_CKPT" \
                --buffer_npz "$BUFFER_NPZ" \
                --seeds_dir "$SEEDS_DIR" \
                --mutation_strategy "$strategy" \
                --flood_ratio "$ratio" \
                --n_mutations_per_seed "$N_MUTATIONS" \
                --n_scoring_rollouts "$N_SCORING_ROLLOUTS" \
                $extra_args \
                --output_dir "$output_dir" \
                --seed "$SEED"
            echo ""
        fi
    done
done

echo "========================================"
echo "All experiments complete!"
echo "Results in: $BASE_OUTPUT/"
echo "========================================"

# --- Summary ---
echo ""
echo "Experiment summary:"
for d in "$BASE_OUTPUT"/*/; do
    if [ -f "$d/experiment_log.json" ]; then
        name=$(basename "$d")
        score_mean=$(python -c "import json; d=json.load(open('$d/experiment_log.json')); print(f'{d[\"score_mean\"]:.4f}')" 2>/dev/null || echo "N/A")
        n_inj=$(python -c "import json; d=json.load(open('$d/experiment_log.json')); print(d['n_injected'])" 2>/dev/null || echo "N/A")
        echo "  $name: injected=$n_inj, mean_sfl=$score_mean"
    fi
done
