#!/bin/bash
# 50k Experiment: Vanilla ACCEL + SFL learnability (baseline)
# HARD EVAL: 12 levels (8 original + 4 harder mazes)
# 3 seeds x 50k updates, WandB project JAXUED_LEARNABILITY, group accel-sfl-hard
set -e

export LD_LIBRARY_PATH=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/lib:${LD_LIBRARY_PATH:-}
PYTHON=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python

COMMON="--project JAXUED_LEARNABILITY \
        --num_updates 50000 --eval_freq 250 \
        --skip_video --skip_post_eval"

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
    --use_accel \
    --score_function sfl \
    --seed $seed \
    --run_name "accel-sfl-hard" \
    2>&1 | tee logs/50k_accel_sfl_hard_seed${seed}.log

  echo "=== [$(date)] Seed $seed complete ==="
done

echo "All ACCEL + SFL (hard eval) experiments complete."
