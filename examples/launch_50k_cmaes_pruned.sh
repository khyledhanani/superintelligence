#!/bin/bash
# 50k Experiment: CMA-ES with weight-norm pruning only (no PCA, no ACCEL mutation)
# 3 seeds x 50k updates, WandB project JAXUED_50K, group cmaes-pruned
# Stage 1 pruning stays active forever (pca_stage2_step=999999 prevents PCA transition)
set -e

# Force conda CUDA libs (system CUDA 13.1 cuSOLVER is incompatible)
export LD_LIBRARY_PATH=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/lib:${LD_LIBRARY_PATH:-}
PYTHON=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python

COMMON="--project JAXUED_50K \
        --num_updates 50000 --eval_freq 250 \
        --skip_video --skip_post_eval"

export WANDB_DIR=/tmp/wandb
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
mkdir -p /tmp/jax_cache
mkdir -p logs

for seed in 0 1 2; do
  echo "=== [$(date)] Seed $seed starting ==="

  $PYTHON examples/maze_plr.py $COMMON \
    --use_cmaes \
    --use_pca_search \
    --pca_stage2_step 999999 \
    --seed $seed \
    --run_name "cmaes-pruned" \
    2>&1 | tee logs/50k_cmaes_pruned_seed${seed}.log

  echo "=== [$(date)] Seed $seed complete ==="
done

echo "All CMA-ES pruned experiments complete."
