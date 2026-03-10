# Value Trajectory DTW

**Function:** `value_trajectory_dtw(values_a, dones_a, values_b, dones_b)`

## What it measures

How similar the agent's learned value estimates are over the course of two episodes. The value function reflects the agent's internal assessment of "how well am I doing?" -- this metric captures whether two levels produce similar difficulty profiles over time.

Two levels with low value DTW distance feel similarly difficult to the agent, even if the physical paths differ. This is the only metric that directly measures the agent's *internal state* rather than its observable behavior.

## How it works

1. Truncate each value sequence at the first `done=True` flag.
2. Reshape values to `(T, 1)` and compute DTW, producing a normalized distance, warping path, and local cost profile.
3. DTW captures shape, magnitude, and timing differences with elastic time alignment.

The local cost profile shows *where* along the trajectory the value estimates diverge most -- an LLM can reason over this directly to understand shape, magnitude, and timing without needing separate correlation or L2 metrics.

Complexity: `O(T1 * T2)` for DTW.

## Output

| Field | Type | Description |
|-------|------|-------------|
| `distance` | float | Normalized DTW distance (total cost / path length, lower = more similar) |
| `path` | `(K, 2)` array | Aligned index pairs along the optimal DTW warping path |
| `local_costs` | `(K,)` array | Per-step costs along the warping path (similarity profile) |

## Interpretation

| Value Range | Meaning |
|-------------|---------|
| < 0.05 | Nearly identical value progressions |
| 0.05 - 0.15 | Similar difficulty profiles with minor differences |
| 0.15 - 0.30 | Moderate difference in difficulty progression |
| 0.30+ | Substantially different value trajectories |

## Why this matters for diversity

If the replay buffer is full of levels that all produce the same value trajectory (DTW distance near 0), the agent is not being challenged in diverse ways. A diverse buffer should contain levels with a range of difficulty profiles:
- Levels where the agent quickly becomes confident (rising value curve)
- Levels where the agent struggles throughout (flat or low value curve)
- Levels with non-monotonic difficulty (value rises then drops, indicating dead ends or backtracking)

## Relationship to other metrics

- **vs Position DTW:** Value DTW captures the agent's *internal assessment*; Position DTW captures *physical movement*. A maze where the agent wanders but remains confident scores differently than one where it takes a direct path but with low confidence.
- **vs Observation DTW:** Observation DTW measures what the agent *sees*; Value DTW measures what the agent *thinks* about what it sees.
- **vs Spatial Jaccard:** Value DTW is purely temporal; Spatial Jaccard is purely spatial. They are nearly orthogonal measures.

## Source

Implemented in `metrics/trajectory_metrics.py` as `value_trajectory_dtw()`.
