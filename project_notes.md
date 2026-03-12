# LLM Maze Generation — Design Notes

## 1. Goal

Use LLMs (via Ollama cloud) to generate diverse mazes and inject them into the ACCEL replay buffer. Injection is both periodic (every N training updates) and triggered (when buffer diversity drops). The standard ACCEL replay buffer handles prioritization via PLR scoring — no need for a separate MAP-Elites archive.

## 2. Core Principle: Diversity = Different Agent Experience

Two mazes that look different but produce the same agent experience are NOT meaningfully diverse. Diversity must be measured by what the agent actually sees and does:

- **Observation sequences**: the agent has a 5×5 view (`agent_view_size=5`). Walls outside this view cone are irrelevant.
- **Decision points**: junctions where the agent chooses. Same junction topology + different decorative walls = same maze.
- **Solution path topology**: turns, backtracking, dead-ends, deceptive paths.

Structural metrics (VAE latent distance, wall-map Hamming) measure differences the agent may never see. They are not valid diversity metrics for this purpose.

## 3. Four Quantitative Metrics

Compute is not a bottleneck. These are only computed for mazes passed to/from the LLM (~50-100 pairwise comparisons per injection event). All four are cheap enough to run on every pair.

### 3.1 Observation Sequence DTW (primary — diversity gate)

The observation sequence is literally what flows into the neural network. Two mazes producing the same `obs[t]` sequence are mathematically identical from the agent's perspective — same gradients, same learning signal. Everything else is a lossy compression of this.

**Different path lengths** are handled by DTW (Dynamic Time Warping). See section 3.5 for a detailed explanation of how DTW works.

For efficiency, flatten the 5×5 agent view to a 25-dim vector before DTW. Cost: O(T_a × T_b × 25) = ~1.5M ops per pair. Milliseconds on CPU.

**Output:** Not just a scalar — the DTW warping path gives a **similarity profile over time** (see §3.6). This tells you *where* in the trajectory two mazes are similar vs different.

**Used for:** diversity gate — "is this candidate maze a genuinely new experience?"
```python
novelty = min(obs_sequence_distance(candidate, existing) for existing in references)
if novelty > threshold: inject
```

### 3.2 Position Trace DTW (secondary — path structure)

The position trace is a `(T, 2)` matrix — `(col, row)` at each timestep. Captures the navigation route including loops and backtracking (revisited cells produce different traces from non-backtracking — that's a feature, not a bug).

Two complementary views:
- **Spatial footprint** (order-independent): Jaccard similarity on sets of visited cells. "Did the agent explore the same region?"
- **Path trace DTW** (order-dependent): DTW on the `(T, 2)` sequence. "Did the agent navigate the same way?" Handles different speeds naturally.

```python
# Order-independent
jaccard = len(visited_a & visited_b) / len(visited_a | visited_b)

# Order-dependent
dtw_distance(trace_a, trace_b)
```

Cost: O(T_a × T_b × 2) = ~125K ops per pair. Instant.

**Used for:** path structure comparison, clustering buffer levels, and communicating with the LLM (path overlay on grid).

### 3.3 Value Trajectory Correlation (tertiary — difficulty fingerprint)

Decomposes regret over time. Regret is a scalar (overall difficulty). The value curve `V(s_t)` shows the *shape* of the difficulty:

```
[0.1, 0.1, 0.1, 0.8, 0.8, 0.8]  → "hard start, easy finish"
[0.8, 0.8, 0.8, 0.1, 0.1, 0.1]  → "easy start, hard finish" (deceptive)
[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  → "hard throughout"
```

All three can have identical regret but produce completely different learning signals.

**Different path lengths** handled by resampling to a fixed number of points via interpolation:
```python
def normalize_value_trajectory(values, dones, n_points=100):
    ep_len = jnp.argmax(dones)
    v_episode = values[:ep_len]
    x_old = jnp.linspace(0, 1, ep_len)
    x_new = jnp.linspace(0, 1, n_points)
    return jnp.interp(x_new, x_old, v_episode)
```

Compare with Pearson correlation (same difficulty *pattern*?) and L2 distance (same difficulty *magnitude*?).

Cost: O(100) after resampling. Instant.

**Used for:** ensuring LLM targets underrepresented difficulty profiles. "We have enough mazes with this difficulty shape — generate one with a different confidence profile."

### 3.4 Spatial Footprint Jaccard (coarse — region check)

Set of unique cells visited, compared via Jaccard index. Order-independent — doesn't capture how the agent navigated, just where it went.

Cost: O(T) to build sets, O(169) to compare. Instant.

**Used for:** fast coarse filter before more expensive DTW. If two mazes have Jaccard ≈ 1.0, they're probably too similar — skip the DTW.

### 3.5 How DTW Works

DTW (Dynamic Time Warping) finds the optimal alignment between two sequences of different lengths. It answers: "what's the best way to match these two sequences end-to-end, allowing stretching and compressing?"

**Step 1: Build the cost matrix.** For sequences A (length M) and B (length N), compute the distance between every pair of elements. Each cell `C[i,j] = distance(A[i], B[j])`. The distance function is flexible — Euclidean is the default, but Manhattan, cosine, or any other distance works. DTW doesn't care; it just finds the cheapest path through whatever cost matrix you give it. For 2D positions: `C[i,j] = sqrt((x_a - x_b)^2 + (y_a - y_b)^2)`. For observation sequences: L2 distance between flattened 5×5 view vectors (25-dim).

**Concrete example.** Maze 1 agent walks right 4 steps: `A = [(0,0), (1,0), (2,0), (3,0)]`. Maze 2 agent walks right 3 then turns down then right: `B = [(0,0), (1,0), (2,0), (2,1), (3,1)]`.

The cost matrix (Euclidean distance between each pair of positions):

```
              B[0]=(0,0)  B[1]=(1,0)  B[2]=(2,0)  B[3]=(2,1)  B[4]=(3,1)
A[0]=(0,0)   0.0         1.0         2.0         2.2         3.2
A[1]=(1,0)   1.0         0.0         1.0         1.4         2.2
A[2]=(2,0)   2.0         1.0         0.0         1.0         1.4
A[3]=(3,0)   3.0         2.0         1.0         1.4         1.0
```

**Step 2: Find the cheapest path** from top-left `C[0,0]` to bottom-right `C[M-1,N-1]`. At each step, the path can only move:
- **Right** `(i,j)→(i,j+1)`: advance B, hold A (stretches A to match a longer section of B)
- **Down** `(i,j)→(i+1,j)`: advance A, hold B (stretches B)
- **Diagonal** `(i,j)→(i+1,j+1)`: advance both (1:1 match)

This is solved with dynamic programming in O(M×N).

**Step 3: Read the warping path.** The optimal path through the example:

```
(0,0) → (1,1) → (2,2) → (2,3) → (3,4)
```

This means:
- `A[0]=(0,0) ↔ B[0]=(0,0)`: same position. Cost 0. **Identical.**
- `A[1]=(1,0) ↔ B[1]=(1,0)`: same position. Cost 0. **Identical.**
- `A[2]=(2,0) ↔ B[2]=(2,0)`: same position. Cost 0. **Identical.**
- `A[2]=(2,0) ↔ B[3]=(2,1)`: A[2] maps to TWO elements in B. Cost 1.0. **B turned down here, A didn't — DTW stretches A to cover B's extra step.**
- `A[3]=(3,0) ↔ B[4]=(3,1)`: different rows. Cost 1.0. **Both moved right but on different rows now.**

Note: `(0,0)→(1,1)→(2,2)→(3,3)→(3,4)` with total cost 2.4 is NOT optimal — the path above costs 2.0 because it avoids the expensive C[3,3]=1.4 cell. DTW always finds the cheapest valid path.

**Two outputs:**
1. **Scalar**: total cost `0 + 0 + 0 + 1.0 + 1.0 = 2.0` (or normalized: `2.0 / 5 = 0.4`)
2. **Warping path**: the sequence of `(i,j)` pairs with their local costs — this is the similarity profile

**Key property:** DTW preserves global ordering (A[0] always aligns before A[M-1]) and tells you exactly which elements matched. It does NOT preserve where similarity/dissimilarity occurs — that information is in the warping path, not the scalar.

### 3.6 Similarity Profiles, Not Scalars

Collapsing DTW to a single scalar loses critical information. Two mazes that are identical for 90% then diverge at the end get the same scalar as two mazes that are slightly different everywhere. We want to know *where* similarity breaks down.

**The warping path already gives us this.** Each aligned pair `(i,j)` has a local cost `d(A[i], B[j])`. The sequence of local costs IS the similarity profile:

```
From the example above:
Warping path: [(0,0), (1,1), (2,2), (2,3), (3,4)]
Local costs:  [ 0.0,   0.0,   0.0,   1.0,   1.0 ]
               ^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^
               identical first 60%    divergent end
```

This profile can be segmented however coarsely or finely you want:
- **First half vs second half**: "identical opening, divergent ending"
- **Quartiles**: more detail on where the transition happens
- **Event-based**: segment at turns, backtracks, or value drops

No resampling needed — DTW already handled the length alignment. The warping path defines the natural correspondence between the two trajectories.

**This applies to all three DTW-based metrics:**
- Observation sequence DTW → similarity profile of what the agent saw over time
- Position trace DTW → similarity profile of where the agent went over time
- (Value trajectory can also use DTW instead of resampling, giving the same benefit)

**Implications:**
- **Decision gate**: a maze identical for 75% then different for 25% is less diverse than one different throughout. The profile captures this; a scalar cannot.
- **LLM feedback**: "the first half of the agent's experience is too similar to Maze B — the opening section needs to be different" is much more actionable than "too similar overall."
- **Buffer analysis**: if many buffer mazes share the same opening but diverge later, the agent may have a fixed opening strategy. Target that.

### Code change required

One line in `rollout_agent_on_levels` (`regret_fitness.py`):
```python
# Current:
return (...), (reward, value, next_done)
# Updated:
return (...), (reward, value, next_done, action, state.agent_pos, obs)
```

All metrics are downstream computation on the extra outputs. No new models, no training.

## 4. Communicating with the LLM

### The problem
LLMs can't reason about raw numerical arrays. But hand-crafted feature extraction (turn count, dead-end count, etc.) imposes a finite vocabulary that will eventually saturate — the LLM produces mazes "different" by our features but actually similar in ways we didn't capture.

### Solution: path overlay + hints

Overlay the agent's path on the ASCII maze grid (step numbers on visited cells) and let the LLM reason about it directly. The LLM does its own spatial reasoning — no feature bottleneck.

```
Maze with agent path (numbers = visit order):

.. 1  2  3  4  5 .#.......
..........  6 ..#.......
.......... 7 ...........
.......... 8  9 10 ...G
```

Add a few computed stats as **hints** (not primary signal):
- Turn count, backtrack count, episode length
- Value trajectory shape in one sentence ("confidence drops at 40% through episode")

The LLM can notice patterns we didn't think to extract. Feedback works naturally: "your maze produced this path [overlay] — same right-then-down pattern as Maze B. Try forcing leftward navigation."

### Iteration loop

1. LLM generates a candidate maze
2. Validate: parse via `Level.from_str()`, flood-fill solvability, complexity check (min obstacles, min path distance). If invalid → retry (no agent rollout needed)
3. Run agent on valid candidate → get trajectories
4. Compare metrics against reference mazes
5. Decision gate with **specific** feedback:
   - Path not different enough? → "agent took the same right-then-down path as Maze A"
   - Difficulty not similar enough? → "too easy compared to target levels"
   - Both pass → inject into ACCEL replay buffer
   - Either fails → feed specific failure reason back to LLM → retry from step 1

## 5. Selecting Reference Mazes for the LLM Prompt

Don't just pick the top K by regret — that could show K very similar mazes. Cluster first, then sample.

### Clustering space

**Position trace space** (not VAE latent). Consistent with the rest of the pipeline — "similar" means "similar agent experience."

### Algorithm

**k-medoids** (not k-means). k-means needs a vector space for centroids; k-medoids works with the precomputed DTW distance matrix directly.

### Workflow

1. Take buffer levels above a regret threshold
2. Compute pairwise position trace DTW → distance matrix
3. k-medoids → K clusters of behaviorally similar levels
4. From each cluster, pick the highest-regret member
5. These K representatives become the reference mazes in the LLM prompt

The LLM sees one example from each behaviorally distinct group of challenging mazes and can aim for the gaps.

### Clustering cost

Pairwise DTW is O(N^2). Mitigations for large buffers:
- Recluster periodically, not every injection event
- Subsample: top 100 by regret, then cluster those
- Cache distance matrix, update incrementally

## 6. Architecture

```
ACCEL Training Loop (GPU, JAX)
  |
  +-- normal PLR replay/mutation cycle
  |
  +-- every N updates OR diversity drops:
  |     |
  |     +-- Cluster buffer (position trace DTW + k-medoids)
  |     +-- Select K representative high-regret mazes
  |     +-- Build prompt (grids + path overlays + hints)
  |     |
  |     +-- LLM Generator (Ollama, async on CPU)
  |           +-- generate candidate maze
  |           +-- validate (format, solvability, complexity)
  |           +-- run agent → compute 4 metrics
  |           +-- decision gate: inject or retry with feedback
  |           +-- loop up to max_retries
  |
  +-- valid candidates injected into replay buffer
      (PLR scoring handles prioritization naturally)
```

### Ollama integration
- Free tier with hourly/weekly rate limits
- OpenAI-compatible API: `base_url="https://api.ollama.com/v1"`
- Available models: deepseek-v3.1:671b-cloud, qwen3-coder:480b-cloud, gpt-oss
- Local fallback: `base_url="http://localhost:11434/v1"` — zero rate limits
- 13×13 ASCII maze generation is well within capability of smaller models

### Maze constraints
- 13×13 grid, `Level.from_str()` compatible (`#` wall, `.` empty, `>v<^` agent, `G` goal)
- Solvable (flood-fill via `flood_fill_solvable()`)
- Minimum obstacle count and agent-goal Manhattan distance

## 7. Possible Extension: Experience-Aligned VAE Latent Space

**Status: not needed for LLM injection, but could revive CMA-ES.**

### Why CMA-ES collapsed

The VAE latent space is organized by structure, not agent experience. CMA-ES optimizes regret in this space — it finds a high-regret region and converges. But structurally different, experientially identical mazes can be spread across the space, while experientially diverse mazes sit right next to each other. CMA-ES sees regret going up and keeps converging to one type.

MAP-Elites worked better because behavior descriptors provided crude diversity pressure. CMA-ES had none, and the latent space didn't help.

### Fix: contrastive loss on trajectories

Add a loss that aligns latent distance with trajectory distance during VAE training:

```python
latent_dist = ||z_i - z_j||^2
traj_dist = DTW(traj_i, traj_j)  # normalized
contrastive_loss = (latent_dist - traj_dist)^2

# Or triplet loss with anchor/positive/negative
# Total: recon + kl_weight * kl + contrastive_weight * contrastive
```

### Benefits for CMA-ES
- Latent directions correspond to different agent experiences → smooth diversity landscape
- Convergence to a single point = convergence to one challenge type → detectable, counteractable
- CMA-ME (CMA-ES + MAP-Elites) works naturally — latent cells = genuinely different challenges
- Latent L2 becomes a valid fast proxy for trajectory distance

### Practical approach
- **Precompute trajectories**: run frozen agent on all 20K training mazes once, cache results. No agent inference during VAE training.
- **Cheaper alternative**: BFS shortest path instead of agent rollout. Aligns latent space with optimal navigation structure (zero compute cost). May be enough to fix collapse.

### Hybrid potential
LLM generates structurally creative seeds → CMA-ES refines them in the aligned latent space for optimal regret while maintaining experiential diversity.

## 8. Open Questions

- What is the ACCEL replay buffer size? Determines whether full pairwise DTW clustering is feasible or needs subsampling.
- Exact threshold for the diversity monitor — what observation sequence DTW distance means "diversity is low"?
- How many LLM retries before giving up on a candidate?
- How many reference mazes to include in the LLM prompt? (context bloat vs. sufficient coverage)
- BFS optimal path vs. agent's actual (suboptimal) path for the overlay? The gap between them is itself informative.
- Can LLMs reliably reason over path overlays on 13×13 grids? Likely yes, but needs validation.
- Can we fine-tune a small local model on validated maze outputs to improve generation hit rate over time?

## 9. Note: Chicken-and-Egg Problem with Experience-Aligned VAE

The contrastive alignment loss (§7) requires agent trajectories, but the agent needs environments to train on, and the environments benefit from the VAE. Circular dependency.

**Options:**
1. **Two-stage**: Train VAE (recon+KL only) → train agent → retrain VAE with alignment loss using frozen agent trajectories. Works but doubles VAE training, and the latent space changes under the agent (agent was trained with mazes from the old space).
2. **Agent-free proxy (BFS)**: Use BFS shortest path as the "experience distance" — deterministic, instant, no agent needed. Captures navigation structure (turns, bottlenecks, path topology) without a trained agent. Breaks the circular dependency completely. Weaker signal (won't capture agent-specific behavior like getting tricked by dead-ends) but likely enough to fix CMA-ES collapse, since collapse was driven by structural similarity.
3. **Alternating/co-training**: Train VAE a bit → train agent a bit → retrain VAE with alignment → repeat. Co-evolution. Complex and potentially unstable.

**Recommendation:** Start with option 2 (BFS proxy) — zero dependencies, cheap, directly addresses the CMA-ES problem. Layer on agent-trajectory alignment (option 1) later as a refinement once both VAE and agent exist.
