#!/bin/bash
# Phase 5: Four-Way Comparison Experiments
# Run in tmux/screen: experiments take several hours total.
# Usage: bash scripts/run_phase5_comparison.sh
set -e

PYTHON=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python
CONFIG=accel_training/config.yml
SEED=42
GROUP=phase5-comparison
UPDATES=20000

echo "=== Phase 5: Four-Way Comparison ==="
echo "Seed: $SEED | Updates: $UPDATES | Group: $GROUP"
echo ""

# Step 0: Install jupyter if not present (for notebook later)
echo "--- Installing jupyter (if needed) ---"
/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/pip install -q notebook 2>/dev/null || true
echo ""

# Step 1: Pre-launch smoke test (SV-CMA-ES, 1000 updates)
echo "=== Step 1: Pre-launch Smoke Test (SV-CMA-ES, 1000 updates) ==="
$PYTHON accel_training/train.py --config $CONFIG \
  --es_strategy sv_cma_es \
  --run_name sv-cma-es-smoke \
  --group $GROUP \
  --seed $SEED \
  --num_updates 1000 \
  --log_dir runs/phase5-smoke/
echo "Smoke test complete. Check WandB for buf_score > 0.004 before continuing."
echo "Press Enter to continue with full runs, or Ctrl+C to abort..."
read -r
echo ""

# Step 2: CMA-ES baseline (20k updates)
echo "=== Step 2: CMA-ES (20k updates) ==="
$PYTHON accel_training/train.py --config $CONFIG \
  --es_strategy cma_es \
  --run_name cma-es \
  --group $GROUP \
  --seed $SEED \
  --num_updates $UPDATES \
  --log_dir runs/phase5-cma-es/
echo "CMA-ES complete."
echo ""

# Step 3: NS-ES (20k updates)
echo "=== Step 3: NS-ES (20k updates) ==="
$PYTHON accel_training/train.py --config $CONFIG \
  --es_strategy ns_es \
  --run_name ns-es \
  --group $GROUP \
  --seed $SEED \
  --num_updates $UPDATES \
  --log_dir runs/phase5-ns-es/
echo "NS-ES complete."
echo ""

# Step 4: SV-CMA-ES (20k updates)
echo "=== Step 4: SV-CMA-ES (20k updates) ==="
$PYTHON accel_training/train.py --config $CONFIG \
  --es_strategy sv_cma_es \
  --run_name sv-cma-es \
  --group $GROUP \
  --seed $SEED \
  --num_updates $UPDATES \
  --log_dir runs/phase5-sv-cma-es/
echo "SV-CMA-ES complete."
echo ""

# Step 5: ACCEL baseline (vanilla maze_plr.py, matched hyperparameters)
echo "=== Step 5: ACCEL Baseline (20k updates) ==="
$PYTHON examples/maze_plr.py \
  --run_name accel-baseline \
  --project es-accel \
  --seed $SEED \
  --num_updates $UPDATES
echo "ACCEL baseline complete."
echo ""

echo "=== All experiments complete! ==="
echo "Next: Run notebooks/phase5_comparison.ipynb to generate thesis figures."
