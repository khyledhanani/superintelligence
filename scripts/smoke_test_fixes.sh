#!/bin/bash
# Smoke test: 500-update runs for all 3 strategies to verify bug fixes.
#
# Expected healthy signals after fixes:
#   - valid_fraction stays > 0.5 for ES steps (was 0.037 for SV-CMA-ES before fix)
#   - No NaN in ES state checkpoints
#   - NS-ES es/std stays < 100 (was 1e8 before fix)
#   - SV-CMA-ES particle_distance stays > 0 (was collapsing to 0)
#
# Each run takes ~3-5 minutes. Total: ~10-15 minutes.

set -e

cd /cs/student/project_msc/2025/csml/gmaralla/superintelligence

# Ensure XLA_FLAGS is unset — JAX 0.5.3 finds libdevice automatically; setting
# XLA_FLAGS overrides internal discovery and breaks it.
unset XLA_FLAGS

PYTHON=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python
CONFIG=accel_training/config.yml
SEED=123
GROUP=smoke-fixes
UPDATES=500

echo "=============================================="
echo "SMOKE TEST: Verifying bug fixes (500 updates)"
echo "=============================================="
echo ""

# 1. CMA-ES
echo "=== CMA-ES (500 updates) ==="
$PYTHON accel_training/train.py --config $CONFIG \
  --es_strategy cma_es \
  --run_name smoke-cma-es \
  --group $GROUP \
  --log_dir runs/smoke-cma-es \
  --seed $SEED \
  --num_updates $UPDATES
echo "CMA-ES smoke done."
echo ""

# 2. NS-ES
echo "=== NS-ES (500 updates) ==="
$PYTHON accel_training/train.py --config $CONFIG \
  --es_strategy ns_es \
  --run_name smoke-ns-es \
  --group $GROUP \
  --log_dir runs/smoke-ns-es \
  --seed $SEED \
  --num_updates $UPDATES
echo "NS-ES smoke done."
echo ""

# 3. SV-CMA-ES
echo "=== SV-CMA-ES (500 updates) ==="
$PYTHON accel_training/train.py --config $CONFIG \
  --es_strategy sv_cma_es \
  --run_name smoke-sv-cma-es \
  --group $GROUP \
  --log_dir runs/smoke-sv-cma-es \
  --seed $SEED \
  --num_updates $UPDATES
echo "SV-CMA-ES smoke done."
echo ""

# 4. Quick health check on results
echo "=== Health Check ==="
$PYTHON -c "
import os, pickle, numpy as np

strategies = ['cma-es', 'ns-es', 'sv-cma-es']
for strat in strategies:
    run_dir = f'runs/smoke-{strat}'
    # Find latest checkpoint
    ckpts = sorted([d for d in os.listdir(run_dir) if d.startswith('checkpoint_')])
    if not ckpts:
        print(f'  {strat}: NO CHECKPOINTS')
        continue
    ckpt = os.path.join(run_dir, ckpts[-1])
    es_path = os.path.join(ckpt, 'es_state.pkl')
    if not os.path.exists(es_path):
        print(f'  {strat}: no es_state.pkl')
        continue
    with open(es_path, 'rb') as f:
        es = pickle.load(f)

    if 'particles' in es:
        for i, p in enumerate(es['particles']):
            has_nan = np.any(np.isnan(p['mean'])) or np.isnan(p['std'])
            print(f'  {strat} particle {i}: std={p[\"std\"]:.4f}, mean_norm={np.linalg.norm(p[\"mean\"]):.4f}, nan={has_nan}')
        print(f'  {strat} particle_distance: {es[\"particle_mean_distance\"]:.4f}')
    else:
        has_nan = np.any(np.isnan(es['mean'])) or np.isnan(es['std'])
        print(f'  {strat}: std={es[\"std\"]:.4f}, mean_norm={np.linalg.norm(es[\"mean\"]):.4f}, nan={has_nan}')

    # Check metrics CSV for valid_fraction
    csv_path = os.path.join(run_dir, 'metrics.csv')
    if os.path.exists(csv_path):
        import csv
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        es_rows = [r for r in rows if r['mode'] == 'es_step']
        if es_rows:
            vf = [float(r['valid_fraction']) for r in es_rows]
            print(f'  {strat} ES valid_frac: min={min(vf):.3f}, mean={np.mean(vf):.3f}, max={max(vf):.3f}')
    print()
"

echo "=== Smoke tests complete! ==="
echo "Check WandB group 'smoke-fixes' for plots."
echo "Look for: valid_frac > 0.5, no NaN, stable es/std"
