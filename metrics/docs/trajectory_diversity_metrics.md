# Trajectory Diversity Metrics

Four metrics that measure behavioral diversity from the agent's perspective across maze levels in the ACCEL replay buffer. Each captures a different aspect of how the trained agent's experience varies across levels.

All metrics truncate trajectories at the first `done=True` flag to compare only the active episode.

## Individual Metric Documentation

1. [Observation Sequence DTW](observation_sequence_dtw.md) -- visual experience similarity via flattened egocentric observations
2. [Position Trace DTW](position_trace_dtw.md) -- navigation path similarity in absolute grid coordinates
3. [Value Trajectory DTW](value_trajectory_dtw.md) -- difficulty profile similarity via DTW on raw value sequences
4. [Spatial Footprint Jaccard](spatial_footprint_jaccard.md) -- coarse visited-cell set overlap

## Summary Table

| Metric | Input | Output | Output Range | Measures | Speed |
|--------|-------|--------|-------------|----------|-------|
| Obs DTW | Observations `(T, H, W, C)` | DTW distance | `[0, inf)` | Visual experience similarity | Slow (O(T1*T2*D)) |
| Position DTW | Positions `(T, 2)` | DTW distance | `[0, inf)` | Navigation path similarity | Fast (O(T1*T2)) |
| Value DTW | Values `(T,)` | DTW distance | `[0, inf)` | Difficulty profile similarity | Fast (O(T1*T2)) |
| Spatial Jaccard | Positions `(T, 2)` | Jaccard index | `[0, 1]` | Visited cell overlap | Very fast (O(T)) |

## Pairwise Computation

`compute_pairwise_metrics(trajectories)` computes all metrics for every unique pair `(i, j)` where `i < j` from a list of `N` trajectory dicts. Returns `N*(N-1)/2` values per metric.

Each trajectory dict must contain:
- `observations`: `(T, H, W, C)` -- agent's egocentric observations
- `positions`: `(T, 2)` -- agent's `(x, y)` grid positions
- `values`: `(T,)` -- critic value estimates (or heuristic proxy)
- `dones`: `(T,)` -- boolean episode termination flags

Returns a dict with these arrays (each of length `N*(N-1)/2`):
- `obs_dtw_distances` -- Observation Sequence DTW distances
- `pos_dtw_distances` -- Position Trace DTW distances
- `value_dtw_distances` -- Value Trajectory DTW distances
- `jaccard_indices` -- Spatial Footprint Jaccard indices

## Design Principle

These metrics form a hierarchy from fine-grained (observation DTW) to coarse (spatial Jaccard). A pair of levels can score similarly on one metric but differently on another, and each combination tells a different story about the nature of the diversity in the replay buffer.

All four metrics use DTW or set-based comparison, producing interpretable outputs (distances, cost profiles, Jaccard indices) that an LLM can reason over directly.

## Source

All metrics implemented in `metrics/trajectory_metrics.py`.
