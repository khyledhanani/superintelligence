# Observation Sequence DTW

**Function:** `observation_dtw(obs_a, dones_a, obs_b, dones_b)`

## What it measures

How different the raw visual experiences are between two trajectories. This is the ground-truth measure of agent experience diversity -- if two levels produce identical observation sequences, they are functionally identical from the agent's perspective regardless of their underlying structure.

## How it works

1. Truncate each observation sequence at the first `done=True` flag (compare only the active episode).
2. Flatten each observation from `(H, W, C)` to a 1D vector (e.g., `5x5x3 = 75` floats for the agent's egocentric view).
3. Compute Dynamic Time Warping (DTW) between the two flattened sequences.
4. Normalize by dividing total warping path cost by path length.

### DTW overview

DTW finds the optimal alignment between two time series that may differ in length or speed. It constructs a cost matrix of pairwise Euclidean distances between all pairs of frames, then finds the minimum-cost path through this matrix from `(0,0)` to `(T1,T2)`. The path can stretch or compress time to align similar frames.

The algorithm:
1. Build cost matrix `C[i,j] = ||obs_a[i] - obs_b[j]||` (Euclidean distance between flattened observations).
2. Fill accumulated cost matrix via `D[i,j] = C[i,j] + min(D[i-1,j], D[i,j-1], D[i-1,j-1])`.
3. Backtrack from `D[T1,T2]` to `D[0,0]` to recover the optimal warping path.
4. Normalize: `distance = D[T1,T2] / path_length`.

Complexity: `O(T1 * T2 * D)` where `D` is the flattened observation dimension. This is the slowest of the four metrics.

## Output

| Field | Type | Description |
|-------|------|-------------|
| `distance` | float | Normalized DTW distance (lower = more similar observations) |
| `path` | `(K, 2)` array | Aligned frame index pairs along the optimal warping path |
| `local_costs` | `(K,)` array | Per-step costs along the warping path (the "similarity profile") |

## Interpretation

| Value Range | Meaning |
|-------------|---------|
| ~1.0-1.2 | Levels produce very similar observation sequences -- the agent sees similar things in similar order |
| ~1.3-1.5 | Moderate visual difference -- typical for diverse buffer levels |
| ~1.6+ | Levels produce visually distinct experiences |

The observation DTW tends to have **lower variance** than position DTW because the 5x5 egocentric view normalizes away absolute position -- two mazes with similar local wall structure will score similarly even if the global layout differs.

## Relationship to other metrics

- **vs Position DTW:** Observation DTW operates in the agent's egocentric frame while Position DTW operates in absolute grid coordinates. Two levels can have high position DTW (agent goes to different places) but low observation DTW (the local wall patterns look similar).
- **vs Spatial Jaccard:** Observation DTW preserves temporal ordering; Jaccard discards it entirely.
- **vs Value Correlation:** Observation DTW captures what the agent *sees*; Value Correlation captures what the agent *thinks* about its progress.

## Typical ranges (trained agent, buffer levels)

- Mean: ~1.5
- Std: ~0.09
- Range: [1.07, 1.67]

## Source

Implemented in `metrics/trajectory_metrics.py` as `observation_dtw()`, built on top of `dtw_with_path()`.
