# Metric Notes and Quirks

Important properties, limitations, and non-obvious behaviors of the diversity metrics.

---

## Invariance Properties

**Position Trace DTW** uses start-relative coordinates — translation invariant but not rotation invariant. The same path shape rotated 90° produces different relative displacement vectors.

**Value Error Profile** and **Per-Step Regret** are translation invariant in practice — the value estimates depend on egocentric observations, not absolute position.

**Mode Transition Divergence** inherits the invariance of its inputs (value error + entropy), so it is effectively translation invariant.

**This is by design.** The metrics measure diversity of *agent experience*, not structural diversity of mazes. Since the policy isn't rotation equivariant, a rotated maze genuinely produces different agent behavior.

---

## DTW Normalization

DTW distances are normalized by warping path length:

```
normalized_distance = total_accumulated_cost / path_length
```

This makes the metric comparable across different episode lengths, but dilutes brief spikes in long trajectories. A 50-step dead-end loop in a 200-step episode barely registers in the normalized distance.

**This is a fundamental problem with DTW for our use case.** See "Why DTW is problematic" below.

---

## Why DTW is Problematic for Experiential Diversity

DTW assumes two sequences describe the same process at different speeds and finds optimal time-warping alignment. This is wrong for cross-maze trajectory comparison:

1. **Warping hides interesting differences.** A dead-end trap (50 confused steps) gets warped to align with 3 clean steps. But the trap IS the experiential difference.

2. **No natural alignment across mazes.** Step 1 on maze A has no correspondence to step 1 on maze B. The agent might go right-then-down on A and up-then-left on B.

3. **Normalization kills the signal.** A 200-step trajectory and a 20-step trajectory get compared on equal footing, but episode length IS information.

4. **Symmetric when the relationship isn't.** DTW(A,B) = DTW(B,A), but diversity is contextual: "does adding C to {A,B} add new experience?"

**Mode transition divergence avoids all of these** by comparing distributional properties (transition matrices) instead of aligned sequences.

---

## Position DTW ≠ Experiential Diversity

Empirical finding from 16-level buffer analysis (120 pairs):

```
Correlation of Position DTW with Mode Transition Divergence: r = 0.113
```

This is near-zero. Spatial path diversity and experiential diversity are essentially independent. Two mazes where the agent walks completely different paths can produce identical learning signals (same value errors, same confusion patterns). Conversely, similar paths through different wall layouts can produce very different experiences.

**Implication:** Position DTW should not be used as a diversity gate. The `gate.min_pos_dtw` config is disabled (null) by default for this reason.

---

## Mode Classification Thresholds

The experience mode classifier uses two thresholds:

- `error_threshold = 0.3` — |V(s_t) - G_t| above this = "wrong"
- `entropy_threshold = 0.3` — H(π) above this = "uncertain"

These are currently hardcoded. They may need tuning for different agents or training stages. An agent early in training has high error everywhere (all steps classified as "wrong"), while a well-trained agent has low error everywhere (all "correct"). The thresholds should ideally be set relative to the agent's typical error range.

**Classification priority:** Uncertain > error-based modes. If entropy is high, the step is classified as uncertain regardless of error magnitude.

---

## Negative Regret

When a maze is too hard for the agent (0% solve rate across rollouts):
- `max_return = 0` (agent never collected reward)
- `V(s_t) > 0` (agent was trained on other mazes and expects some return)
- `regret = max_return - V(s_t) < 0`

This is not a bug — it correctly indicates the agent overestimates its prospects on an impossible maze. The `min_regret: 1` gate filters these out (negative regret < 1).

With signed value error: `V(s_t) - G_t > 0` on these mazes, correctly flagging the agent as overconfident.

---

## KL Divergence Properties

Mode transition divergence uses symmetric KL: `(KL(P||Q) + KL(Q||P)) / 2` with Laplace smoothing (ε=1e-8).

- **Unbounded:** Unlike DTW distances, KL can be arbitrarily large. Typical range in our buffer: 0-60, mean ~17.
- **Zero only for identical distributions.** Two mazes with identical transition matrices score 0.
- **Sensitive to zero probabilities.** The Laplace smoothing prevents infinities but means unused transitions still affect the score. A mode never visited gets uniform outgoing probabilities (1/5 each).
- **Not a true metric.** Symmetric KL satisfies symmetry and non-negativity but not the triangle inequality. This doesn't matter for pairwise comparison but means you can't build a proper metric space from it.

---

## Truncation Behavior

All metrics truncate at the **first** `done=True` flag:

- Only the first episode is compared, even with auto-reset.
- If the agent never solves the maze, the full trajectory (up to `max_steps`) is used.
- The `done` step is included in the truncated trajectory.

**Edge case:** `done=True` on step 1 → trajectory length 1. DTW between length-1 sequences is pointwise distance. Mode classifier produces a single-step mode with a degenerate 1-transition matrix.

---

## Metric Redundancy

From the correlation analysis:

- **Regret DTW ≈ Mode Divergence** (r=0.890) — mode divergence captures the same signal without DTW's alignment assumptions
- **Value Error ≈ Mode Divergence** (r=0.885) — mode classification is derived from value error, so high correlation is expected
- **Action DTW partially captures Mode Divergence** (r=0.614) — different actions sometimes (but not always) imply different experiences
- **Position DTW is independent** (r=0.113) — spatial and experiential diversity are orthogonal axes

In the current system, mode transition divergence makes regret DTW redundant. If you need a single pairwise diversity metric, mode transition divergence is the best choice.

---

## TD Error EMD Properties

TD error distribution EMD uses Wasserstein-1 distance computed via quantile matching:
- Interpolate both sorted TD error arrays to 200 quantile points
- EMD = mean absolute difference between corresponding quantiles
- This is equivalent to the area between the two ECDFs

**Typical range in our buffer:** 0.009-0.072, mean ~0.023. The small absolute values reflect that TD errors themselves are small (the agent's value function is reasonably well-calibrated). Threshold sensitivity needs care — a gate at 0.05 would reject ~30% of pairs.

**Terminal step handling:** At the last step of an episode, V(s_{t+1}) = 0 (no next state), so `δ_T = r_T - V(s_T)`. If the agent solved the maze, r_T is a large positive reward and the TD error spikes. This terminal spike is informative (it captures whether the agent expected to solve) but can dominate short episodes.

**Task-agnosticism:** TD error EMD requires only `(values, rewards, dones)` — the minimal interface any actor-critic provides. No observation access, no network internals, no entropy, no positions. This is the strongest task-agnostic guarantee of any pairwise metric in the system.

---

## Computational Cost

| Metric | Per-pair cost | 16 levels (120 pairs) | 4000 levels (8M pairs) |
|--------|-------------|----------------------|----------------------|
| Position DTW | O(T1·T2) | ~1s total | Impractical |
| Regret DTW | O(T1·T2) | ~1s total | Impractical |
| Action DTW | O(T1·T2) | ~1s total | Impractical |
| Mode Divergence | O(T) classify + O(1) KL | <1s total | ~minutes |
| TD Error EMD | O(T) compute + O(T log T) sort | <1s total | ~minutes |
| Value Error | O(T) per maze | <1s total | ~seconds |

Mode transition divergence is the cheapest pairwise metric because it reduces each trajectory to a 5×5 matrix before comparison. For large-scale buffer analysis, it's the only viable option without sampling.
