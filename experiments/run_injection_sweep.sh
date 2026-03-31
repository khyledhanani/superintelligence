#!/bin/bash
# Injection experiment sweep
# Step 1: Generate mutations (sequential, needs GPU for agent rollouts)
# Step 2: Launch 4 training scripts in parallel (one per GPU)
#         Each script runs 3 seeds sequentially
set -e

ENTITY="romain-hautier-university-college-london-ucl-"
PROJECT="JAXUED_LLM_INJECTION"
GROUP="injection_train_v2"
AGENT_DIR="/cs/student/msc/csml/2025/rhautier/Documents/jaxued/jaxued/gcs_artifacts/agent/39"
BUFFER_NPZ="gcs_artifacts/buffer/buffer_dump_final.npz"
SEEDS_DIR="/cs/student/project_msc/2025/csml/rhautier/injection_data/seeds"
RESULTS_BASE="/cs/student/project_msc/2025/csml/rhautier/injection_data/results"

echo "=========================================="
echo "Step 1: Mutation generation"
echo "=========================================="

echo "--- div=0.20 ---"
python experiments/run_injection_experiment.py \
    --agent_checkpoint_dir gcs_artifacts/agent/39 \
    --buffer_npz $BUFFER_NPZ \
    --seeds_dir $SEEDS_DIR \
    --mutation_strategy wall_flip --num_edits 5 \
    --target_eligible 1000 --inject_pcts 5,10 \
    --n_scoring_rollouts 10 --min_embedding_diversity 0.20 \
    --wandb_group injection_mutation_v2 \
    --output_dir ${RESULTS_BASE}/llm_inject_div020_seed0 \
    --seed 0

echo "--- no div gate ---"
python experiments/run_injection_experiment.py \
    --agent_checkpoint_dir gcs_artifacts/agent/39 \
    --buffer_npz $BUFFER_NPZ \
    --seeds_dir $SEEDS_DIR \
    --mutation_strategy wall_flip --num_edits 5 \
    --target_eligible 1000 --inject_pcts 5,10 \
    --n_scoring_rollouts 10 --min_embedding_diversity 0 \
    --wandb_group injection_mutation_v2 \
    --output_dir ${RESULTS_BASE}/llm_inject_nodiv_seed0 \
    --seed 0

echo "=========================================="
echo "Step 2: Training (4 conditions in parallel)"
echo "=========================================="
echo "Launch each of these on a separate GPU:"
echo "  bash experiments/train_div020_5pct.sh"
echo "  bash experiments/train_div020_10pct.sh"
echo "  bash experiments/train_nodiv_5pct.sh"
echo "  bash experiments/train_nodiv_10pct.sh"
