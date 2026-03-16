#!/bin/bash
# 50k Experiment: NEW CNN-VAE (1M steps) + staged PCA-CMA-ES + ACCEL + SFL learnability
# HARD EVAL: 12 levels (8 original + 4 harder mazes)
# 3 seeds x 50k updates, WandB project JAXUED_LEARNABILITY, group cnn1M-staged-sfl-hard
set -e

export LD_LIBRARY_PATH=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/lib:${LD_LIBRARY_PATH:-}
PYTHON=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python

COMMON="--project JAXUED_LEARNABILITY \
        --num_updates 50000 --eval_freq 250 \
        --skip_video --skip_post_eval"

CNN_CKPT="vae/checkpoints/cnn_vae/run11_1M/default"

EVAL_LEVELS="--eval_levels SixteenRooms SixteenRooms2 Labyrinth LabyrinthFlipped \
        Labyrinth2 StandardMaze StandardMaze2 StandardMaze3 \
        Zigzag ZigzagTeeth HardLabyrinth NarrowGapZigzag"

export WANDB_DIR=/tmp/wandb
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
mkdir -p /tmp/jax_cache
mkdir -p logs

for seed in 0 1 2; do
  echo "=== [$(date)] Seed $seed starting ==="

  $PYTHON examples/maze_plr.py $COMMON $EVAL_LEVELS \
    --use_cmaes --use_accel \
    --vae_checkpoint_path "$CNN_CKPT" \
    --use_pca_search \
    --pca_stage2_step 10000 \
    --pca_components 20 \
    --pca_sigma_init 0.5 \
    --cmaes_reset_interval 500 \
    --cmaes_reset_interval_post_pca 2000 \
    --pca_refit_interval 10000 \
    --score_function sfl \
    --seed $seed \
    --run_name "cnn1M-staged-sfl-hard" \
    2>&1 | tee logs/50k_cnn1M_staged_sfl_hard_seed${seed}.log

  echo "=== [$(date)] Seed $seed complete ==="
done

echo "All CNN-1M staged + SFL (hard eval) experiments complete."
