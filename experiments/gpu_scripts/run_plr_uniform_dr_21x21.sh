#!/bin/bash
# PLR with uniform wall count DR (0-150 walls) on 21x21.
# No ACCEL mutations. exploratory_grad_updates=False (RPLR).
# Matches the ACCEL paper's DR setup where levels have varied complexity.
# 3 seeds sequential.

export WANDB_ENTITY=romain-hautier-university-college-london-ucl-
export GOOGLE_CLOUD_PROJECT=open-endedness-ued-project
export WANDB_DIR=/cs/student/project_msc/2025/csml/rhautier/wandb_logs
mkdir -p $WANDB_DIR

cd /cs/student/project_msc/2025/csml/rhautier/oe_proj

PY=/cs/student/msc/csml/2025/rhautier/miniforge3/envs/jaxued_env/bin/python
SCRIPT=examples/maze_plr.py

RUN=plr_uniformdr_sfl_21x21_v1
export WANDB_RUN_GROUP=$RUN

SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then SEEDS=(0 1 2); fi

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "=== PLR + uniform DR 21x21 seed $SEED ==="
    $PY $SCRIPT \
        --maze_height 21 --maze_width 21 --n_walls 150 \
        --no-use_accel \
        --uniform_wall_count \
        --score_function sfl --num_sfl_rollouts 10 \
        --checkpoint_save_interval 1 --max_number_of_checkpoints 999999 \
        --buffer_dump_interval 250 \
        --project JAXUED_VAE_COMPARISON \
        --run_name ${RUN} \
        --gcs_bucket ucl-ued-project-bucket \
        --gcs_prefix accel/${RUN} \
        --num_updates 30000 --seed $SEED
done

echo ""
echo "PLR + uniform DR 21x21 done."
