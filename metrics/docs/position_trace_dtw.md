# Position Trace DTW

**Function:** `position_trace_dtw(pos_a, dones_a, pos_b, dones_b)`

## What it measures

How different the agent's navigation paths are in absolute grid coordinates. This captures the structural diversity of the agent's movement patterns -- whether it takes similar routes, visits similar areas, or follows fundamentally different paths.

## How it works

1. Truncate each position trace `(T, 2)` at the first `done=True` flag.
2. Compute DTW between the two 2D position sequences using Euclidean distance as the pointwise cost.
3. Normalize by dividing total warping path cost by path length.

The cost between two positions is simply `||pos_a[i] - pos_b[j]||_2`, the Euclidean distance in grid-cell units. A cost of 1.0 means the positions are one cell apart; a cost of 5.0 means they are 5 cells apart.

Complexity: `O(T1 * T2)` -- much faster than Observation DTW since positions are only 2D.

## Output

| Field | Type | Description |
|-------|------|-------------|
| `distance` | float | Normalized DTW distance (in grid-cell units) |
| `path` | `(K, 2)` array | Aligned position index pairs |
| `local_costs` | `(K,)` array | Per-step Euclidean distances along the warping path |

## Interpretation

| Value Range | Meaning |
|-------------|---------|
| < 1.0 | Very similar paths -- the agent moves through roughly the same cells in the same order |
| 1.0 - 3.0 | Moderate difference -- different paths but in the same general region or with similar structure |
| 3.0 - 5.0 | Substantially different navigation patterns |
| 5.0+ | Fundamentally different -- the agent traverses completely different parts of the grid |

## Relationship to other metrics

- **vs Observation DTW:** Position DTW operates in absolute coordinates; Observation DTW operates in the agent's egocentric frame. Two levels can have high position DTW (agent goes to different places) but low observation DTW (the local wall patterns look similar). Conversely, the same position path through two different mazes would have low position DTW but potentially high observation DTW.
- **vs Spatial Jaccard:** Position DTW preserves the temporal ordering and speed of traversal. Jaccard only cares about *which* cells were visited, not *when* or *in what order*. Position DTW will distinguish two trajectories that visit the same cells in reverse order; Jaccard will not.
- **vs Value Correlation:** Position DTW captures the physical path; Value Correlation captures the agent's internal difficulty assessment along that path.

## Use cases

- **Detecting redundant levels:** Low position DTW between many level pairs suggests the buffer contains structurally similar mazes that produce near-identical agent paths.
- **Identifying outliers:** Levels with consistently high position DTW against all others are structurally unique -- they force the agent into unusual navigation patterns.
- **Deep-dive visualization:** The `local_costs` array (similarity profile) shows where two trajectories diverge and converge over time, useful for understanding which parts of the paths differ.

## Typical ranges (trained agent, buffer levels)

- Mean: ~2.7
- Std: ~1.1
- Range: [0.4, 5.7]

## Source

Implemented in `metrics/trajectory_metrics.py` as `position_trace_dtw()`, built on top of `dtw_with_path()`.
