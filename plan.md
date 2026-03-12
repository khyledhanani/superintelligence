# Implementation Plan: LLM-Driven Diversity Injection for ACCEL

Base branch: `dev/es_in_accel_implementation`
Each phase is independently testable and mergeable.

---

## Phase 1: Metrics Infrastructure & Baseline Measurement

**Goal:** Compute the 4 diversity metrics on the ACCEL replay buffer during CMA-ES+VAE training. Establish baseline measurements of how metrics evolve as the agent trains. Build the trajectory caching infrastructure needed by later phases.

**Branch:** `feat/phase1-metrics`

### Step 1.1: Extend `rollout_agent_on_levels` to return full trajectory data

- **File:** `examples/maze_plr.py` (the rollout logic in `on_new_levels`, `on_replay_levels`, `on_mutate_levels`)
- **Change:** The agent rollout scan currently returns `(reward, value, done)`. Add `action`, `state.agent_pos`, and `obs` to the scan output.
- **Output:** Each rollout now produces:
  - `positions`: `(T, pop_size, 2)` — agent position trace
  - `actions`: `(T, pop_size)` — action sequence
  - `observations`: `(T, pop_size, *obs_shape)` — full observation sequence
  - (rewards, values, dones already exist)
- **Test:** Run a short training (100 updates), verify trajectory data shapes and values are sensible. Check that training speed is not significantly impacted by the extra scan outputs.

### Step 1.2: Implement the 4 metrics

- **New file:** `metrics/trajectory_metrics.py`
- **Implement:**
  1. **Observation Sequence DTW** — DTW on flattened 5×5 obs vectors (25-dim). Returns scalar distance + warping path (local cost sequence = similarity profile over time). Normalize by warping path length.
  2. **Position Trace DTW** — DTW on `(T, 2)` position sequences. Returns scalar + warping path.
  3. **Value Trajectory Correlation** — Resample value curves to 100 points via `jnp.interp`. Compute Pearson correlation and L2 distance.
  4. **Spatial Footprint Jaccard** — Set of unique visited cells, Jaccard index.
- **Each metric returns:** scalar summary + similarity profile (local costs along warping path) where applicable.
- **Dependencies:** Use `dtw-python` or implement DTW from scratch (simple DP, ~30 lines). Prefer a pure NumPy implementation to avoid extra dependencies.
- **Test:** Unit tests with hand-crafted trajectory pairs:
  - Two identical trajectories → distance ≈ 0, Jaccard = 1.0, correlation ≈ 1.0
  - Two completely different trajectories → high distance, low Jaccard, low correlation
  - Same path but different speed (one has a long straight) → DTW should align correctly, low normalized distance

### Step 1.3: Trajectory cache for replay buffer levels

- **New file:** `metrics/trajectory_cache.py`
- **Purpose:** Store trajectory data (positions, values, observations, dones) per buffer level, keyed by level index.
- **Design:**
  - Dictionary mapping buffer index → trajectory data
  - Updated whenever levels are evaluated during training (on_new_levels, on_replay_levels, on_mutate_levels)
  - Evicted when levels are evicted from the replay buffer
  - Stores only first-episode data (truncate at first `done=True`) to save memory
- **Integration:** Hook into `maze_plr.py` to populate the cache during training.
- **Test:** Verify cache grows with buffer, entries are valid, eviction works.

### Step 1.4: Logging metrics during training

- **File:** `examples/maze_plr.py`
- **Change:** Every N updates (e.g., every 100), compute pairwise metrics on a subsample of the replay buffer (e.g., 20 random levels) and log summary statistics to wandb:
  - Mean/std/min/max of pairwise observation DTW distances
  - Mean/std of pairwise position trace DTW distances
  - Mean/std of pairwise value trajectory correlations
  - Mean/std of pairwise Jaccard indices
  - Distribution plots (histograms) of each metric
- **Test:** Run full CMA-ES+VAE training for the standard number of updates. Verify metrics are logged and visualize how they change over training. This is the baseline.

### Step 1.5: Baseline experiment

- **Run:** Standard CMA-ES+VAE ACCEL training (no LLM injection) with metrics logging enabled.
- **Deliverable:** Wandb dashboard showing:
  - How buffer diversity (all 4 metrics) evolves over training
  - Whether diversity collapses (metrics converging toward 0 distance / 1.0 Jaccard)
  - Correlation between diversity metrics and agent performance (solve rate, return)
- **This establishes the baseline that Phase 2 aims to improve.**

---

## Phase 2: LLM Diversity Injection

**Goal:** Use Kimi K2.5 (via Ollama) to generate new maze levels and inject them into the ACCEL replay buffer. First periodic injection, then add triggered injection based on diversity metrics.

**Branch:** `feat/phase2-llm-injection`

### Step 2.1: LLM maze generator

- **New file:** `llm/maze_generator.py`
- **Implement:**
  - Ollama client setup (OpenAI-compatible API, model: `kimi-k2.5:cloud`)
  - System prompt: maze format specification (13×13, `#.>v<^G` characters), constraints
  - Generation function: takes reference mazes (ASCII grids with path overlays + hint stats) → returns candidate maze string
  - Parsing: extract ASCII grid from LLM response → `Level.from_str()`
  - Validation pipeline: format check → `flood_fill_solvable()` → complexity check (min obstacles, min path distance)
  - Retry logic: on invalid maze, feed specific error back to LLM ("row 5 has 14 characters, need exactly 13"), retry up to max_retries
- **Config:** Ollama base URL, model name, max retries, temperature — all configurable
- **Test:** Standalone script that generates 20 mazes, reports success rate, visualizes valid ones using existing `visualize_envs.py`.

### Step 2.2: Path overlay and prompt builder

- **New file:** `llm/prompt_builder.py`
- **Implement:**
  - `overlay_path_on_grid(level, positions)`: takes a Level and position trace, returns ASCII grid with step numbers on visited cells
  - `build_generation_prompt(reference_mazes, trajectories, metrics)`: assembles the full prompt with:
    - Reference maze grids with path overlays
    - Hint stats per maze (turn count, backtrack count, episode length, value curve shape)
    - Instruction to generate a maze with similar difficulty but different navigation path
  - `build_feedback_prompt(candidate_grid, candidate_overlay, metrics_comparison)`: when a candidate fails the decision gate, build specific feedback ("agent took the same path as Maze A — try forcing leftward navigation")
- **Test:** Generate prompts from real buffer data, verify they're well-formed and fit within context limits.

### Step 2.3: Decision gate

- **New file:** `llm/decision_gate.py`
- **Implement:**
  - Takes candidate maze trajectory metrics + reference maze metrics
  - Checks:
    - Path different enough? (obs DTW and position trace DTW above thresholds)
    - Difficulty similar enough? (value trajectory correlation above threshold)
  - Returns: accept/reject + specific failure reason string (for LLM feedback)
  - Uses similarity profiles (warping path local costs) to identify WHERE similarity is too high
- **Thresholds:** Start with conservative values, tune based on Phase 1 baseline distributions.
- **Test:** Unit tests with known-similar and known-different trajectory pairs.

### Step 2.4: Periodic injection integration

- **File:** `examples/maze_plr.py`
- **Change:** Every N training updates (configurable, e.g., every 500):
  1. Sample K reference mazes from the buffer (for now, top K by regret — clustering comes in Phase 3)
  2. Get their cached trajectories from Step 1.3
  3. Build prompt (Step 2.2) and call LLM (Step 2.1)
  4. Validate candidate maze
  5. Run agent on candidate → get trajectory → compute metrics
  6. Decision gate (Step 2.3): accept or retry with feedback
  7. If accepted, convert to Level and insert into replay buffer via `level_sampler.insert_batch()`
  8. Log: injection success rate, candidate metrics, number of retries
- **LLM calls run synchronously** (simplest first — async in a later optimization)
- **Test:** Run training with periodic injection enabled. Verify:
  - Mazes are actually injected into the buffer
  - Injected mazes have different trajectory metrics from existing buffer levels
  - Training doesn't crash or slow down catastrophically
  - Compare diversity metrics (from Phase 1 logging) with vs without injection

### Step 2.5: Triggered injection (diversity-based)

- **New file:** `llm/diversity_monitor.py`
- **Implement:**
  - Monitors the pairwise metric distributions logged in Step 1.4
  - Trigger condition: when mean pairwise observation DTW distance drops below a threshold (i.e., buffer levels are becoming too similar)
  - When triggered: run a larger injection batch (e.g., 20-30 mazes instead of 5-10)
  - Cooldown period after triggered injection to avoid spamming
- **File:** `examples/maze_plr.py`
- **Change:** Add diversity monitor check alongside the periodic check. Either can trigger injection; triggered injection requests a larger batch and uses more targeted prompts.
- **Test:** Artificially reduce buffer diversity (e.g., fill buffer with copies of similar mazes) and verify the monitor triggers. Run full training and verify triggered injection fires when diversity drops.

---

## Phase 3: Clustering-Based Reference Selection

**Goal:** Replace "top K by regret" reference selection with behaviorally meaningful clusters, so the LLM sees one example from each distinct type of challenge.

**Branch:** `feat/phase3-clustering`

### Step 3.1: Pairwise distance matrix computation

- **New file:** `metrics/buffer_clustering.py`
- **Implement:**
  - Compute pairwise position trace DTW distances for buffer levels (using cached trajectories from Phase 1)
  - Subsample if buffer is large: take top N by regret (e.g., N=100), compute pairwise on those
  - Cache the distance matrix, update incrementally when new levels enter/leave the buffer
- **Test:** Compute distance matrix on a trained buffer, verify it's symmetric, diagonal is zero, values are in expected range.

### Step 3.2: k-medoids clustering

- **File:** `metrics/buffer_clustering.py`
- **Implement:**
  - k-medoids on the precomputed distance matrix (use `sklearn_extra.cluster.KMedoids` or implement PAM algorithm)
  - Configurable K (number of clusters)
  - Returns: cluster assignments, medoid indices
- **Test:** Cluster a trained buffer, visualize clusters (e.g., plot representative mazes from each cluster, verify they look structurally different).

### Step 3.3: Integrate clustering into reference selection

- **File:** `examples/maze_plr.py` and `llm/prompt_builder.py`
- **Change:** Replace "top K by regret" reference selection with:
  1. Recluster periodically (not every injection — e.g., every 5 injection events or every 2500 updates)
  2. From each cluster, pick the highest-regret member
  3. These K representatives become the reference mazes in the LLM prompt
- **Test:** Compare LLM-generated mazes when using clustered vs unclustered reference selection. Verify that clustered references produce more diverse candidates (measure via the 4 metrics).

---

## Phase 4: Experience-Aligned VAE Latent Space

**Goal:** Retrain the VAE with a contrastive alignment loss so that latent distance reflects agent experience distance, not just structural similarity. Then retrain the agent with the aligned VAE. Multi-stage: VAE → Agent → VAE (aligned) → Agent.

**Branch:** `feat/phase4-aligned-vae`

### Step 4.1: BFS shortest path computation

- **New file:** `metrics/bfs_paths.py`
- **Implement:**
  - BFS from agent position to goal on the 13×13 grid (existing `flood_fill_solvable` is close but doesn't return the path)
  - Returns: ordered list of `(row, col)` positions along the shortest path
  - Batch-friendly: works on the full VAE training dataset (20K-200K mazes)
- **Test:** Run on all prefab mazes, verify paths are correct. Run on training dataset, verify all solvable mazes have valid paths.

### Step 4.2: Precompute pairwise BFS path distances

- **Compute:** For the VAE training dataset (20K mazes), compute BFS shortest paths, then pairwise DTW distances between all paths.
- **Storage:** Save as a precomputed distance matrix (`.npy` file). For 20K mazes, the full matrix is 20K × 20K = 400M entries — may need to subsample or use approximate methods.
- **Alternative:** Don't precompute full matrix. Instead, during VAE training, compute BFS DTW distances within each batch (batch_size=512, so 512×512 = 262K pairs per step). This is feasible given the costs from the notes.
- **Test:** Verify distance matrix properties (symmetric, zero diagonal, triangle inequality approximately holds).

### Step 4.3: Add contrastive alignment loss to VAE training

- **File:** `vae/train_vae.py` (or new `vae/train_vae_aligned.py` to keep original intact)
- **Implement:**
  - New loss term alongside reconstruction + KL:
    ```python
    latent_dist = ||z_i - z_j||^2
    bfs_dist = precomputed_or_batch_computed_dtw(maze_i, maze_j)
    alignment_loss = mean((latent_dist - bfs_dist)^2)
    total = recon_loss + kl_weight * kl_loss + align_weight * alignment_loss
    ```
  - `align_weight` as a new hyperparameter (anneal separately from KL, or fixed)
  - Sample pairs within each batch for the contrastive loss (don't need all pairs — random subsample of e.g., 64 pairs per batch is sufficient)
- **Config:** Add `align_weight`, `num_contrastive_pairs` to `vae_train_config.yml`
- **Test:** Train aligned VAE on the existing dataset. Verify:
  - Reconstruction quality is not significantly degraded
  - KL divergence is not significantly different
  - Latent distances correlate with BFS path DTW distances (scatter plot, Pearson correlation)

### Step 4.4: Stage 1 — Train VAE (standard, no alignment)

- Use existing VAE training pipeline (already done — checkpoints exist)
- Or retrain from scratch with current config for reproducibility
- **Deliverable:** Trained VAE checkpoint (standard)

### Step 4.5: Stage 2 — Train agent with standard VAE

- Run CMA-ES+VAE ACCEL training with Phase 1 metrics logging
- **Deliverable:** Trained agent checkpoint + baseline metrics

### Step 4.6: Stage 3 — Retrain VAE with alignment loss

- Use the agent from Step 4.5 to compute agent trajectories on the VAE training dataset (optional upgrade from BFS-only: use both BFS + agent trajectories as target distances)
- Train aligned VAE from scratch or fine-tune from Step 4.4 checkpoint
- **Deliverable:** Aligned VAE checkpoint

### Step 4.7: Stage 4 — Retrain agent with aligned VAE

- Run CMA-ES+VAE ACCEL training using the aligned VAE
- Log Phase 1 metrics for comparison
- **Deliverable:** Trained agent checkpoint + metrics with aligned VAE

### Step 4.8: Comparison

- Compare standard VAE agent (Step 4.5) vs aligned VAE agent (Step 4.7):
  - Agent performance: solve rate, return
  - Buffer diversity: all 4 metrics over training
  - CMA-ES behavior: does it still collapse? Does it explore more of the latent space?
  - Latent space structure: PCA/t-SNE of buffer levels colored by cluster assignment (from Phase 3)
- **Deliverable:** Comparison plots and analysis

---

## Summary

| Phase | Branch | Depends on | Key deliverable |
|-------|--------|-----------|-----------------|
| 1 | `feat/phase1-metrics` | None | 4 metrics + trajectory cache + baseline measurements |
| 2 | `feat/phase2-llm-injection` | Phase 1 | LLM maze generation + periodic/triggered injection |
| 3 | `feat/phase3-clustering` | Phase 1, 2 | k-medoids clustering for smarter reference selection |
| 4 | `feat/phase4-aligned-vae` | Phase 1 | Experience-aligned VAE + multi-stage training comparison |

Phases 2, 3, and 4 all depend on Phase 1 but are independent of each other — they can be developed in parallel after Phase 1 is complete.
