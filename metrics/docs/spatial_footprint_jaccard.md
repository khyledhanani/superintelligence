# Spatial Footprint Jaccard

**Function:** `spatial_footprint_jaccard(pos_a, dones_a, pos_b, dones_b)`

## What it measures

The overlap between the sets of grid cells visited during two episodes. This is the coarsest metric -- it discards all temporal information and only asks "did the agent visit the same cells?"

## How it works

1. Truncate each position trace at the first `done=True` flag.
2. Convert each trace to a set of unique `(x, y)` grid cells visited.
3. Compute Jaccard index: `|A intersection B| / |A union B|`.

The Jaccard index is a standard set similarity measure. It equals 1.0 when the sets are identical, 0.0 when they are completely disjoint, and values in between reflect partial overlap.

Complexity: `O(T)` -- the fastest metric. Set construction is linear in trajectory length, and set intersection/union are linear in set size.

## Output

| Field | Type | Description |
|-------|------|-------------|
| `jaccard` | float | Jaccard index in `[0, 1]`. 1.0 = identical cell sets, 0.0 = no overlap |
| `cells_a` | set | Set of `(x, y)` tuples visited in trajectory A |
| `cells_b` | set | Set of `(x, y)` tuples visited in trajectory B |
| `intersection_size` | int | Number of cells visited by both trajectories |
| `union_size` | int | Number of cells visited by at least one trajectory |

## Interpretation

| Value Range | Meaning |
|-------------|---------|
| 0.0 | Completely disjoint -- the agent visits entirely different areas in the two levels |
| 0.01 - 0.15 | Minimal overlap -- a few shared cells, likely near common start positions |
| 0.15 - 0.35 | Moderate overlap -- some shared corridors but mostly different coverage. Typical for a diverse buffer |
| 0.35 - 0.60 | High overlap -- the levels share many open corridors or the agent has a strong spatial preference |
| 0.60+ | Very high overlap -- the levels produce nearly identical spatial coverage |
| 1.0 | Identical cell sets (very unlikely across different levels) |

## Strengths

- **Fast:** Set operations are `O(T)`, making this the cheapest metric to compute.
- **Intuitive:** "What fraction of the explored area is shared?" is easy to explain and reason about.
- **Order-invariant:** Doesn't depend on *when* or *in what order* cells were visited -- purely about spatial coverage.
- **Visualizable:** The cell sets can be directly overlaid on the maze grid for visual inspection (as in the spatial footprint plots).

## Limitations

- **Loses temporal structure:** Two trajectories that visit the same cells in opposite order score 1.0. An agent that goes A->B->C vs C->B->A are indistinguishable.
- **Loses speed information:** An agent that rushes through cells vs one that slowly explores the same cells score identically.
- **Sensitive to trajectory length:** Longer episodes visit more cells, mechanically inflating intersection sizes. Truncation at first done mitigates this by comparing only the active episode.
- **No notion of density:** A cell visited once and a cell visited 50 times contribute equally to the set.

## Relationship to other metrics

- **vs Position DTW:** Position DTW preserves temporal ordering and traversal speed. Jaccard is a strict information reduction of the position trace -- it keeps only the set of unique cells. Use Position DTW when *how* the agent traverses matters; use Jaccard when only *where* matters.
- **vs Observation DTW:** Observation DTW operates in the agent's egocentric frame; Jaccard operates in absolute grid coordinates. Two levels with identical Jaccard scores can have very different observation DTW if the wall patterns around the shared cells differ.
- **vs Value Correlation:** These are nearly orthogonal. Jaccard is purely spatial; Value Correlation is purely temporal. A pair with high Jaccard but low Value Correlation visits the same cells but with different difficulty progressions.

## Use cases

- **Quick diversity screening:** Jaccard is fast enough to compute for all `N*(N-1)/2` pairs even for large buffers, making it useful as a first-pass diversity filter.
- **Spatial redundancy detection:** High average Jaccard across the buffer indicates the agent is being funneled through the same corridors regardless of level structure.
- **Visualization:** The `cells_a` and `cells_b` sets can be rendered as colored overlays on maze grids to visually assess coverage patterns.

## Typical ranges (trained agent, buffer levels)

- Mean: ~0.23
- Std: ~0.14
- Range: [0.0, 0.61]

## Source

Implemented in `metrics/trajectory_metrics.py` as `spatial_footprint_jaccard()`.
