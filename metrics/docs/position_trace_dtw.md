# Position Trace DTW

**Function:** `position_trace_dtw(pos_a, dones_a, pos_b, dones_b)`

## What it measures

How different the agent's navigation paths are in terms of movement pattern. Positions are made relative to the agent's start position, making this metric **translation invariant** -- two agents taking the same shaped path from different starting positions will score as identical.

## How it works

1. Truncate each position trace `(T, 2)` at the first `done=True` flag.
2. Subtract the start position from each trace (`pos -= pos[0]`), converting absolute coordinates to relative displacements from the agent's starting cell.
3. Compute DTW between the two relative 2D position sequences using Euclidean distance as the pointwise cost.
4. Normalize by dividing total warping path cost by path length.

The cost between two relative positions is `||pos_a[i] - pos_b[j]||_2`, the Euclidean distance in grid-cell units. A cost of 1.0 means the relative positions are one cell apart; a cost of 5.0 means they are 5 cells apart.

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

- **vs Observation DTW:** Position DTW operates in start-relative coordinates; Observation DTW operates in the agent's egocentric frame. Two levels can have high position DTW (agent takes different shaped paths) but low observation DTW (the local wall patterns look similar). Conversely, the same movement pattern through two different mazes would have low position DTW but potentially high observation DTW.
- **vs Spatial Jaccard:** Position DTW uses start-relative coordinates and preserves temporal ordering. Jaccard uses absolute coordinates and only cares about *which* cells were visited, not *when* or *in what order*. Two agents taking the same path from different starting positions would have low position DTW but potentially low Jaccard (different absolute cells).
- **vs Value DTW:** Position DTW captures the physical path shape; Value DTW captures the agent's internal difficulty assessment along that path.

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
