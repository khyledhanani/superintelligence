# Metric Notes and Quirks

Important properties, limitations, and non-obvious behaviors of the trajectory diversity metrics.

---

## Invariance Properties

**None of the four metrics are rotation invariant.**

- **Position Trace DTW** uses start-relative coordinates, so it is rotation sensitive — the same path shape rotated 90 degrees produces different relative displacement vectors.
- **Spatial Footprint Jaccard** operates on absolute `(x, y)` grid coordinates. Rotating a maze moves all positions, producing completely different cell sets.
- **Observation DTW** depends on the absolute wall layout rendered into the agent's egocentric view. A rotated maze produces different pixel values.
- **Value Trajectory DTW** is the closest to rotation invariant since it only looks at the value curve over time. However, the trained agent's policy (CNN + LSTM) has no rotational equivariance built in, so it will behave differently on rotated versions of the same maze in practice.

**This is by design.** The metrics measure diversity of *agent experience*, not structural diversity of mazes. Since the policy isn't rotation equivariant, a rotated maze genuinely produces different agent behavior.

**Translation invariance varies by metric:**

- **Position Trace DTW** is **translation invariant** — positions are converted to start-relative coordinates before comparison, so two identical path shapes from different starting positions score as identical.
- **Spatial Footprint Jaccard** is **not translation invariant** — it uses absolute coordinates intentionally, to answer "did the agents explore the same region of the grid?"
- **Observation DTW** is **translation invariant in practice** — the agent's egocentric view only depends on local surroundings, not absolute position. The same corridor at different grid locations produces the same observation.
- **Value Trajectory DTW** is **translation invariant in practice** — the value estimate depends on what the agent sees (egocentric), not where it is in absolute terms.

---

## DTW Normalization

The DTW distance is normalized by dividing by the warping path length:

```
normalized_distance = total_accumulated_cost / path_length
```

This means the output is the **average per-step alignment cost**, not the total cost. Without this normalization, longer trajectories would always have higher DTW distances simply because there are more steps to accumulate cost over.

**Consequence:** Two trajectory pairs with the same normalized DTW can have very different total costs if the warping paths are different lengths. The normalization makes the metric comparable across pairs with different episode lengths, but it also means a brief spike in a long trajectory gets diluted.

---

## Observation Flattening

Observation DTW flattens each observation frame from `(H, W, C)` to a 1D vector before computing distances. For the 5x5x3 egocentric view, this means each frame is a 75-dimensional vector.

**Consequence:** All spatial structure within the observation is lost — a wall in the top-left corner is treated the same as a wall in the bottom-right. The Euclidean distance between two flattened frames measures total pixel difference, not perceptual similarity. Two observations that look very different to a human (e.g., wall on left vs wall on right) may have the same Euclidean distance as two observations that look somewhat similar.

**The `agent_dir` field is not included** in the observation DTW — only `obs.image` is used. Two frames with identical views but different agent directions will score as identical.

---

## Truncation Behavior

All metrics truncate at the **first** `done=True` flag. This means:

- Only the first episode is compared, even if the environment auto-resets and the agent plays additional episodes within the same rollout.
- If the agent never reaches done (doesn't solve the maze within `max_steps`), the full trajectory is used.
- The truncation is **inclusive** — the step where `done=True` is included in the truncated trajectory.

**Edge case:** If `done=True` on the very first step, the trajectory has length 1. DTW between two length-1 sequences is just the pointwise distance. Jaccard of single-cell sets returns 1.0 if same cell, 0.0 otherwise.

---

## Jaccard Sensitivity to Coverage

Jaccard is `|intersection| / |union|`. This means:

- Two trajectories that each visit 5 cells with 2 in common: Jaccard = 2/8 = 0.25
- Two trajectories that each visit 50 cells with 20 in common: Jaccard = 20/80 = 0.25

Same Jaccard, very different situations. The second pair covers much more of the maze but the overlap ratio is the same. The raw `intersection_size` and `union_size` in the output provide this context, but the Jaccard scalar alone doesn't.

**Jaccard is biased toward 0 for short trajectories** because small cell sets are unlikely to overlap by chance on a 13x13 grid. A random 5-cell set has only a ~3% chance of overlapping with another random 5-cell set.

---

## DTW Computational Cost

DTW is `O(T1 * T2)` per pair, computed with a double for-loop in pure Python (no C extension or approximation). For 250-step trajectories, this is ~62,500 iterations per pair.

- **Position DTW** (2D inputs): Fast enough for hundreds of pairs.
- **Observation DTW** (75D inputs): The cost matrix computation involves broadcasting over the full observation dimension, making it significantly slower.

For large-scale analysis (e.g., all 4000 buffer levels = ~8M pairs), the current implementation is too slow. Consider FastDTW or restricting to a subset.

---

## Metric Independence

The four metrics are **not independent** — they share the same underlying trajectory data. However, they capture different aspects:

- Position DTW and Spatial Jaccard both use positions but differ in coordinate system (start-relative vs absolute) and what they capture (temporal path shape vs spatial coverage). Two agents taking the same shaped path from different starting positions would have low Position DTW but potentially low Jaccard (different absolute cells).
- Observation DTW and Position DTW are correlated (positions determine what the agent sees) but can diverge (same relative path through different wall layouts = different observations).
- Value DTW is the most independent from the others since it reflects the critic network's internal state, not directly observable quantities.

In practice, Position DTW and Jaccard can diverge in both directions: same cells visited in different order (high Jaccard, high Position DTW), or same path shape from different starts (low Position DTW, low Jaccard).

---

## Auto-Reset Wrapper

The environment uses `AutoReplayWrapper`, which auto-resets when the agent reaches the goal. The `done=True` flag fires on the goal-reaching step, and the next step is already in the new episode. The truncation at first done ensures we only analyze the initial episode, but be aware that the `done` step's observation is from *after* the reset in the wrapper's state, not from the goal state itself.
