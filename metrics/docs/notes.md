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

- `error_threshold` — |V(s_t) - G_t| above this = "wrong"
- `entropy_threshold` — H(π) above this = "uncertain"

**Adaptive (production):** When `baseline_stats` is provided (computed from reference trajectories), thresholds are set to `mean + 1 std` of |error| and entropy respectively. This means ~16% of steps exceed the threshold, automatically scaling with agent capability and training stage. An early-training agent with high error everywhere gets a higher threshold; a well-trained agent gets a tighter one.

**Fallback:** When `baseline_stats` is not provided, thresholds default to 0.3 for both (the original hardcoded values).

**Classification priority:** Uncertain > error-based modes. If entropy is high, the step is classified as uncertain regardless of error magnitude.

---

## Negative Regret

When a maze is too hard for the agent (0% solve rate across rollouts):
- `max_return = 0` (agent never collected reward)
- `V(s_t) > 0` (agent was trained on other mazes and expects some return)
- `regret = max_return - V(s_t) < 0`

This is not a bug — it correctly indicates the agent overestimates its prospects on an impossible maze. The `gate.difficulty_threshold` filters these out (negative regret < threshold). Alternatively, using `difficulty_metric: sfl` filters by learnability p×(1-p), which naturally rejects both too-hard (p≈0) and too-easy (p≈1) levels.

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
- **Value Error ⊥ Mode Divergence** (r=-0.078 with adaptive thresholds) — the old r=0.885 was an artifact of fixed thresholds that artificially aligned the metrics. With adaptive thresholds, they are nearly uncorrelated
- **Action DTW partially captures Mode Divergence** (r=0.614) — different actions sometimes (but not always) imply different experiences
- **Position DTW is independent** (r=0.113) — spatial and experiential diversity are orthogonal axes

In the current system, mode transition divergence makes regret DTW redundant. If you need a single pairwise diversity metric, mode transition divergence is the best choice.

---

## Normalized TD Error EMD Properties

Normalized TD error distribution EMD uses Wasserstein-1 distance computed via quantile matching on **normalized** TD errors. Each trajectory's TD errors are divided by their total absolute sum before comparison, separating learning signal *shape* from *magnitude*.

**Why normalize?** SFL learnability already captures *how much* learning happens (magnitude). The diversity gate only needs to capture *what kind* of learning (shape/direction). Without normalization, two levels with identical learning signal shapes but different magnitudes would score as diverse — a false positive, since SFL is already filtering by magnitude.

**Computation:**
- Compute raw TD errors per trajectory
- Normalize: `td_norm = td / sum(|td|)` (with epsilon floor to avoid division by zero)
- Sort normalized TD errors → quantile function
- Interpolate both to 200 quantile points
- EMD = mean absolute difference between corresponding quantiles

**Typical range:** Values will differ from the old unnormalized metric (previously 0.009-0.072). The normalized metric is scale-free — thresholds need recalibration.

**Terminal step handling:** At the last step of an episode, V(s_{t+1}) = 0 (no next state), so `δ_T = r_T - V(s_T)`. If the agent solved the maze, r_T is a large positive reward and the TD error spikes. Normalization reduces the dominance of this spike on short episodes since it's divided by the total.

**Task-agnosticism:** Normalized TD error EMD requires only `(values, rewards, dones)` — the minimal interface any actor-critic provides. No observation access, no network internals, no entropy, no positions. This is the strongest task-agnostic guarantee of any pairwise metric in the system.

**Temporal blind spot:** EMD compares distributions, not sequences. Two mazes where the agent is confused early vs confused late produce identical EMD if the overall δ distributions match. This is a real limitation — gradient updates happen in order, so temporal profile affects learning. For temporal sensitivity, use Regret DTW (r=0.290 correlation with EMD, captures temporal shape).

---

## CENIE GMM Properties

CENIE fits a diagonal-covariance GMM on LSTM hidden state + action pairs (257D) from buffer trajectories.

**Silhouette-based K selection:** Tries K from 2 to 10, picks K with highest silhouette score. Automates model complexity. In practice, K=2 consistently wins for the maze buffer — the agent's LSTM states cluster into two modes (exploring vs goal-directed).

**NLL score interpretation:** Novelty = `-mean(log p(x_t | GMM))`. More negative = more familiar (high density). Less negative / positive = more novel (low density). Buffer levels typically score -250 to -110. A good gate threshold is around -200 (rejects the most familiar ~50% of experiences).

**Not pairwise:** CENIE scores a trajectory against a density model, not against another trajectory. It cannot be used for the t-SNE diversity embedding (which requires pairwise distances). The embedding always uses Normalized TD Error EMD regardless of gate metric.

**Architecture coupling:** Uses LSTM hidden states — tied to the agent's architecture. If the agent changes (different hidden dim, no LSTM), the GMM must be refitted and thresholds recalibrated.

**FIFO buffer concept:** The paper uses a sliding window of recent experiences. Our implementation fits on a snapshot of 50 buffer levels (one-shot, not streaming). For integration into the training loop, a ring buffer approach is used (see `vae/cenie_scorer.py` in the training branch).

**Double subsampling problem:** CENIE's "buffer-wide" coverage has two layers of subsampling:

1. **50 of ~4000 levels** from the buffer — `n_cenie_levels = min(size, 50)` in `test_generator.py:1295`. The cap exists because each level requires an agent rollout to extract LSTM hidden states. This gives ~4000 state-action pairs (50 levels × ~80 timesteps), not the full buffer population.

2. **1 rollout per level** — CENIE uses `evaluate_levels()` (single rollout), not `evaluate_level_multi_rollout()` (100 rollouts). Since the policy is stochastic, a single rollout captures one possible behavior — the agent might solve the maze or not, or take an unusual path. The multi-rollout function picks the best-case trajectory, which better represents the agent's capability on that level.

This means CENIE's advantage over pairwise metrics (which subsample ~6 references) is real but narrower than it appears: 50 levels vs 6, not 4000 vs 6. To truly fit on the full buffer, you'd need to either cache LSTM hidden states during training (avoiding re-rollouts) or accept the cost of thousands of rollouts before each generation batch.

---

## Computational Cost

| Metric | Type | Per-eval cost | 16 levels (120 pairs) | 4000 levels |
|--------|------|-------------|----------------------|-------------|
| Position DTW | Pairwise | O(T1·T2) | ~1s total | Impractical (8M pairs) |
| Regret DTW | Pairwise | O(T1·T2) | ~1s total | Impractical |
| Action DTW | Pairwise | O(T1·T2) | ~1s total | Impractical |
| Mode Divergence | Pairwise | O(T) classify + O(1) KL | <1s total | ~minutes |
| Norm TD Error EMD | Pairwise | O(T) compute + O(T log T) sort | <1s total | ~minutes |
| Value Error | Standalone | O(T) per maze | <1s total | ~seconds |
| CENIE Novelty | Standalone | O(T·K) score (K=GMM components) | <1s total | ~seconds + GMM fit |

**Pairwise metrics** scale as O(N²). **Standalone metrics** (Value Error, CENIE) scale as O(N). CENIE has a one-time GMM fitting cost (~3s for 50 levels / 4000 samples) but O(1) per candidate after that — this is its key scaling advantage over pairwise metrics for large buffers.

---

## Reference Selection Strategies

Five strategies are available via `--strategy`:

### `diverse` (greedy max-min)
Uses precomputed pairwise distances (`buffer_td_errors.npz` or `buffer_embeddings.npz`) to greedily select N maximally-spread references from all 4000 buffer levels in ~0.3s (no agent rollouts needed for selection). Starts with the global maximum-distance pair, then iteratively adds the level farthest from the current set. Tends to pick outliers.

### `kmedoids` (PAM cluster medoids)
Uses the same precomputed pairwise distance matrix but applies the PAM (Partitioning Around Medoids) BUILD+SWAP algorithm. Finds K cluster centers that minimize within-cluster distance. Unlike greedy max-min, this picks *representative* levels (cluster centers) rather than maximally-spread outliers. Better when you want references that cover typical buffer behavior.

### `hybrid` (difficulty-filtered + diverse)
Filters buffer to above-mean difficulty levels, then applies greedy max-min on the filtered set. Ensures references are both challenging and diverse.

### `top_regret` / `random`
Simple baselines: highest-regret or uniform random selection.

**Vectorized pairwise computation:** The full 4000×4000 distance matrix is computed via chunked numpy broadcasting on the (4000, 200) quantile matrix — ~15s for exact results (no sampling heuristics). The starting pair is the true global maximum. Memory is managed by processing 500-row chunks, keeping the intermediate tensor under ~800 MB.

**Fallback:** If precomputed data is unavailable, `diverse` and `kmedoids` fall back to rollout-based selection on a stratified candidate pool (slower, less coverage).



---

## LSTM Embedding as Diversity Metric

### Motivation

TD error EMD collapses each timestep to a single scalar (how wrong the value prediction was) and then collapses the time series to a distribution. It cannot distinguish *why* the agent was surprised — a dead end vs an unexpected shortcut can produce identical TD error distributions.

The LSTM hidden state (256-dim) is the agent's full compressed representation at each step. It encodes spatial memory, navigation history, and learned expectations. Two mazes that produce different embedding trajectories are genuinely different *from the agent's perspective*, capturing the "why" that TD error loses.

### Implementation: `metrics/pairwise/embedding_divergence.py`

**Per-dimension quantile EMD:** For each of the 257 dimensions (256 LSTM + 1 action) independently:
1. Extract per-step values for the episode → (ep_len,) vector
2. Compute 200-point quantile representation
3. Compute Wasserstein-1 (EMD) between two trajectories' quantile vectors

Final distance = mean EMD across all 256 dimensions.

This reuses the same quantile-EMD machinery as TD error EMD but applies it to each LSTM dimension separately.

### Status: Production

Fully wired into the pipeline as `diversity_metric: embedding`:

- **Gate:** decision_gate.py computes embedding divergence for accept/reject decisions.
- **Reference selection:** Both `diverse` and `kmedoids` strategies support precomputed embedding distances via `buffer_embeddings.npz` (generated by `llm/precompute_buffer_embeddings.py`).
- **Visualization:** t-SNE and MDS embedding plots use embedding distances when `embedding_metric: embedding`.
- **More dynamic range:** Embedding EMD distances are ~0.14 between diverse mazes (vs ~0.02-0.03 for TD error EMD). The buffer is less of a compressed blob.
- **Richer signal:** 257 dimensions (256 LSTM + 1 action) vs 1 scalar per timestep. Captures what the agent *knows* and *does*, not just how wrong its predictions are. Same state-action pair that CENIE uses.

### Scaling problem

The per-dimension quantile approach creates a (200 × 257) = 51,400-dim representation per level. For the 4000-level buffer:
- Pairwise distance matrix = 8M pairs × 51,400 dims = **257x slower** than TD error EMD
- TD error EMD buffer: ~15s → Embedding EMD buffer: **~1 hour**
- Plus t-SNE/MDS fitting on 4000×4000 is already slow

### Efficiency ideas (not yet implemented)

1. **Mean embedding** — Average the 256-dim LSTM state over the episode. Each level → single 256-dim vector. Pairwise = one matrix multiply. Seconds for 4000 levels. Loses temporal structure but keeps the "what does the agent know" signal.

2. **PCA reduction** — PCA the 256 LSTM dims down to 10-20 before computing quantiles. Representation becomes (200 × 20) = 4,000-dim instead of 51,200. ~13x speedup. Preserves temporal structure.

3. **Final-state embedding** — Use only the last LSTM hidden state before done. Captures the agent's "conclusion" about the maze. 256-dim per level, trivial pairwise. But loses the journey.

4. **Subsample buffer for plots** — Plot 200-500 buffer dots instead of 4000. The embedding plot is for visualization, not gating — fewer dots may be fine. Gate still uses all levels for selection.

5. **Hybrid: mean + final** — Concatenate mean embedding (256) + final embedding (256) = 512-dim. Captures both average experience and final understanding. Still fast (512-dim pairwise).

### Recommendation

Start with **mean embedding** (option 1) for buffer-wide visualization and selection. It's the fastest path to seeing whether the 256-dim space has more structure than TD error. If it does, invest in PCA reduction (option 2) to recover temporal information without the 256x cost.

---

## Embedding Visualization: t-SNE vs MDS

Both t-SNE and MDS embeddings are generated for each batch run:

- **t-SNE** (`diversity_embedding.png`): Preserves local neighborhoods. Good for identifying clusters. Distorts global distances — two clusters that appear far apart may not be proportionally far in metric space.
- **MDS** (`diversity_embedding_mds.png`): Preserves actual pairwise distances. Layout directly reflects metric values. Prefer this when interpreting global distance relationships.

Both plots annotate edges from rejected/accepted levels to their closest reference with actual metric values (not projected distances), so edge labels are reliable in both views.

**Config:** `buffer_embed_samples: -1` (default) shows all buffer levels in the background.

---

# Open items
# /-----/ Need to remove hybrid strategy minimum episode length filter
# /-----/ Need to wire up SFL to diverse/hybrid level selection strategies
# /-----/ Currently we filter by stale regret but fresh TD error