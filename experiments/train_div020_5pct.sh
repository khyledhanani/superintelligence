#!/bin/bash
set -e
ENTITY="romain-hautier-university-college-london-ucl-"
PROJECT="JAXUED_LLM_INJECTION"
AGENT="/cs/student/msc/csml/2025/rhautier/Documents/jaxued/jaxued/gcs_artifacts/agent/39"
BUF="/cs/student/project_msc/2025/csml/rhautier/injection_data/results/llm_inject_div020_seed0/merged_buffer_5pct.npz"

for s in 0 1 2; do
  echo "=== div020 5pct seed${s} ==="
  python examples/maze_plr.py \
    --project $PROJECT --wandb_entity $ENTITY \
    --wandb_group div020_5pct \
    --run_name ws_5pct_div020_seed${s} \
    --wandb_update_offset 10000 \
    --score_function sfl \
    --use_accel --num_edits 5 --num_updates 10000 \
    --preload_buffer_npz $BUF \
    --resume_checkpoint_dir $AGENT \
    --seed $s
done
