#!/bin/bash
# Test script to verify the evolution pipeline works correctly.
# This will:
#   1. Generate a small batch of random environments
#   2. Run evolution for 20 generations
#   3. Visualize random vs evolved environments
#   4. Print statistics comparison

set -e  # Exit on any error

PYTHON=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python
cd /cs/student/msc/csml/2025/gmaralla/superintelligence/es

echo "=========================================="
echo "CLUTTR Evolution Pipeline Test"
echo "=========================================="
echo ""

echo "Step 1: Running evolution (20 generations, pop_size=16)..."
$PYTHON evolve_envs.py \
    --num_generations 20 \
    --pop_size 16 \
    --seed 12345 \
    --log_freq 5

echo ""
echo "Step 2: Generating comparison visualization..."
$PYTHON visualize_envs.py \
    --compare \
    --num_envs 8 \
    --seed 12345 \
    --output evolved/comparison.png

echo ""
echo "Step 3: Statistics summary..."
$PYTHON -c "
import numpy as np

evolved = np.load('evolved/evolved_envs.npy')
history = np.load('evolved/fitness_history.npy')

print(f'Evolved {len(evolved)} environments')
print(f'Fitness improvement: {history[0]:.4f} -> {history[-1]:.4f} (delta: {history[-1] - history[0]:.4f})')
print(f'')

# Compute detailed stats
obstacles = evolved[:, :50]
goals = evolved[:, 50]
agents = evolved[:, 51]

obs_counts = (obstacles > 0).sum(axis=1)
print(f'Obstacle counts: min={obs_counts.min()}, max={obs_counts.max()}, mean={obs_counts.mean():.1f}')

# Manhattan distances
inner_dim = 13
agent_rows = (agents - 1) // inner_dim
agent_cols = (agents - 1) % inner_dim
goal_rows = (goals - 1) // inner_dim
goal_cols = (goals - 1) % inner_dim
dists = np.abs(agent_rows - goal_rows) + np.abs(agent_cols - goal_cols)
print(f'Manhattan distances: min={dists.min()}, max={dists.max()}, mean={dists.mean():.1f}')

# Validity
valid = ((goals >= 1) & (goals <= 169) & (agents >= 1) & (agents <= 169) & (goals != agents))
print(f'All valid: {valid.all()}')
"

echo ""
echo "=========================================="
echo "Test complete!"
echo "Check evolved/comparison.png to see results"
echo "=========================================="
