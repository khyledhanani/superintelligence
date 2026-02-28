# Phase 1: Foundation - Research

**Researched:** 2026-02-28
**Domain:** JAX/Flax RL agent verification + JAX-native behavior signature extraction from maze rollouts
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Agent Verification (FOUND-01)**
- Approach: Code comparison first, then smoke test — do NOT run DCD repo to compare training curves in Phase 1. Build our side first; full comparison happens after ES is working.
- Scope: Verify PPO update logic and regret computation (MaxMC). Level sampling will intentionally differ (no random env generation in our version) — document this as intentional.
- Smoke test: Run `maze_plr.py` with ACCEL for ~10k steps. Passing bar: no crash, regret > 0 and changing, solve rate between 0-1, WandB logs successfully.
- Output: Write all differences (code + smoke test results) to `.planning/phases/01-foundation/AGENT_VERIFICATION.md`. Classify differences as a flat list — no tiers needed. Document everything found.

**Behavior Signature (FOUND-02)**
- What it captures: DEFER TO RESEARCH. The specific implementation (visit-count histogram, action sequences, stats) should be whatever the researcher recommends for this JAX/maze setup. Flag this as a critical open question.
- Grid resolution: DEFER TO RESEARCH.
- Normalization: DEFER TO RESEARCH.
- Key constraint: Must be JAX-JIT-compatible. All ops via `jax.lax.*`, fixed-size arrays.
- CRITICAL FLAG: The behavior signature is the most important design decision in the entire project. The wrong choice leads to mode collapse or novelty reward hacking. This is explicitly NOT finalized in Phase 1 — Phase 1 builds a v1 implementation that works, and the design is revisited after NS-ES validates the approach.

**Extractor Location and Interface**
- Location: Extend `es/regret_fitness.py` (not a new file). Behavior extraction sits alongside regret computation since they share the rollout.
- Execution: Run as a separate pass from regret computation for simplicity during development. Efficiency optimization deferred.
- Interface: Claude's discretion — make it modular/swappable. Research will determine exact signature.
- Marking: Add `# TODO: EXPERIMENTAL v1 — behavior signature design is NOT final. See .planning/DECISIONS.md` comment in code AND document in `.planning/DECISIONS.md`.
- Documentation: Docstring with usage example (no standalone demo script needed).

**DECISIONS.md**
- Create `.planning/DECISIONS.md` — a living document tracking key architectural decisions for collaborators.
- First entry: behavior signature design rationale, what v1 implements, and why this needs revisiting.

### Claude's Discretion

- Interface design for `extract_behavior_signature()` — modular/swappable, exact signature up to researcher
- Normalization of the histogram (normalize by total steps vs. leave as counts)
- Grid resolution for the histogram (full 13x13 = 169 cells vs. coarser)
- Whether to accumulate counts across episodes (reset on done) or per whole rollout

### Deferred Ideas (OUT OF SCOPE)

- Full training curve comparison vs DCD (deferred to after ES is working)
- Behavior signature redesign based on NS-ES results (Phase 3/5)
- Normalization choices for behavior signatures (left to researcher — now addressed below)
- Grid resolution for histogram (left to researcher — now addressed below)
- Standalone demo script for extractor (out of scope Phase 1)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| FOUND-01 | Agent PPO/ACCEL training verified to match DCD repo implementation | PPO hyperparameter table documents exact parameter mapping; MaxMC confirmed as correct score function; identified key differences (gae_lambda, entropy_coef, score strategy) to document in AGENT_VERIFICATION.md |
| FOUND-02 | Behavior signature vector extracted from agent rollout on any level (visit-count histogram over grid cells, JAX-compatible) | JIT-compatible histogram construction via `jnp.zeros` + scatter approach documented; rollout_agent_on_levels already returns next_state; full implementation pattern with fixed-shape arrays provided |
</phase_requirements>

---

## Summary

Phase 1 has two fully independent tasks. The first (FOUND-01) is a code audit and smoke test: read the DCD repo `arguments.py` and our `maze_plr.py` in parallel and document all differences in a Markdown file. No training is run against DCD; only our codebase is smoke-tested. The second (FOUND-02) is a new function in `es/regret_fitness.py` that takes the existing rollout trajectory and computes a fixed-length JAX array representing cell visit counts.

Both tasks are straightforward extensions of existing code. There are no new library dependencies. The key technical risk is JIT compatibility for the behavior signature extractor — the implementation must use `jax.lax.scatter` patterns with fixed-size arrays rather than Python control flow or dynamic shapes. The rollout infrastructure in `regret_fitness.py` already returns `next_state` in the scan carry, so position data is available without modifying `rollout_agent_on_levels`.

The most important judgment call in this phase is the behavior signature design. Research recommends a full-resolution 13x13 = 169-cell visit-count histogram, L1-normalized (divide by total steps), accumulated across all rollout steps regardless of episode boundary. This is deliberately simple: it encodes "where did the agent spend time?" without committing to trajectory ordering or action sequence structure. The experimental flag and DECISIONS.md entry make the revisit path explicit.

**Primary recommendation:** Add `extract_behavior_signature(rollout_states, levels, num_steps)` to `es/regret_fitness.py` as a separate function that takes the batched `EnvState` array from `jax.lax.scan` output and returns a `(pop_size, 169)` float32 array of L1-normalized visit counts over the 13x13 grid.

---

## Standard Stack

### Core (already installed — no new dependencies)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| JAX | installed | JIT compilation, vmap, lax ops | All existing code is JAX; non-negotiable |
| Flax | installed | Neural network (ActorCritic), structs | Checkpoint format; network.apply interface |
| jaxued | installed | Maze env, LevelSampler, max_mc, compute_max_returns | Core library this project extends |
| distrax | installed | Categorical distribution for agent actions | Already used in ActorCritic |
| WandB | installed | Logging smoke test metrics | Required for smoke test validation |
| orbax | installed | Checkpoint loading (agent_params.pkl fallback) | Existing checkpoint format |

### No New Installations Required

The full stack for Phase 1 is already present. This phase adds zero new pip packages.

---

## Architecture Patterns

### Existing Rollout Infrastructure

`rollout_agent_on_levels` in `es/regret_fitness.py` already runs a full trajectory via `jax.lax.scan`. The scan currently collects `(reward, value, next_done)` per step. To support behavior signature extraction, the scan must also collect `next_state` (which contains `agent_pos`).

**Current scan output:**
```python
_, (rewards, values, dones) = jax.lax.scan(
    step_fn, init_carry, None, length=num_steps
)
# rewards, values, dones: (num_steps, pop_size)
```

**Required change — extend step_fn output to include states:**
```python
# In step_fn, change return from:
return carry, (reward, value, next_done)
# To:
return carry, (reward, value, next_done, next_state)
# next_state is EnvState pytree with agent_pos shape (pop_size, 2)
```

This is a minimal change to `rollout_agent_on_levels`. The output becomes:
```python
rewards, values, dones, states = rollout
# states.agent_pos: (num_steps, pop_size, 2)  [x=col, y=row]
```

### Pattern 1: JIT-Compatible Visit-Count Histogram

**What:** Convert agent position trajectory into a fixed-size cell visit count array without Python loops or dynamic indexing.

**When to use:** Any time trajectory data must be aggregated into a fixed-length descriptor in a JIT context.

**Key constraint:** All array shapes must be static at compile time. No `if/for` loops. Use `jnp.zeros` + `jnp.add.at` scatter pattern (or `jax.lax.scatter`).

```python
# Source: JAX documentation — pure functional scatter
def extract_behavior_signature(states, levels, num_steps, grid_h=13, grid_w=13):
    """
    Extract visit-count behavior signature from rollout trajectory.

    # TODO: EXPERIMENTAL v1 — behavior signature design is NOT final.
    # See .planning/DECISIONS.md for rationale and planned revisit criteria.

    Args:
        states: EnvState pytree from jax.lax.scan, agent_pos shape (num_steps, pop_size, 2).
                Coordinate convention: agent_pos = [x=col, y=row].
        levels: Batched Level pytree (unused in v1, kept for API consistency).
        num_steps: Rollout length (used for L1 normalization).
        grid_h: Grid height (13 for CLUTTR mazes).
        grid_w: Grid width (13 for CLUTTR mazes).

    Returns:
        behavior_sig: (pop_size, grid_h * grid_w) float32 array.
                      L1-normalized visit counts over grid cells.

    Usage example:
        rewards, values, dones, states = rollout_agent_on_levels(...)
        sigs = extract_behavior_signature(states, levels, num_steps=256)
        # sigs[i] is the behavior signature for the i-th level in the batch
    """
    pop_size = states.agent_pos.shape[1]
    num_cells = grid_h * grid_w  # 169 for 13x13

    # agent_pos: (num_steps, pop_size, 2), [x=col, y=row]
    # Convert to linear cell index: row * width + col
    col = states.agent_pos[:, :, 0].astype(jnp.int32)  # (num_steps, pop_size)
    row = states.agent_pos[:, :, 1].astype(jnp.int32)  # (num_steps, pop_size)
    cell_idx = row * grid_w + col                        # (num_steps, pop_size)

    # Scatter-add: count visits per cell per level
    # Initialize histogram: (pop_size, num_cells)
    hist = jnp.zeros((pop_size, num_cells), dtype=jnp.float32)

    # Vectorized scatter using one-hot encoding (JIT-safe, vmap-safe)
    one_hot_visits = jax.nn.one_hot(cell_idx, num_classes=num_cells, dtype=jnp.float32)
    # one_hot_visits: (num_steps, pop_size, num_cells)
    hist = one_hot_visits.sum(axis=0)  # (pop_size, num_cells)

    # L1 normalize: divide by total steps so signatures are comparable
    # across different rollout lengths
    total = hist.sum(axis=-1, keepdims=True)  # (pop_size, 1)
    hist = hist / jnp.maximum(total, 1.0)     # avoid div-by-zero

    return hist
```

**Why one-hot sum instead of `at[].add`:** `jnp.zeros(...).at[cell_idx].add(1)` is not vectorizable with `vmap` over the step dimension and requires Python-level looping. `jax.nn.one_hot(...).sum(axis=0)` is a single fused operation that is vmap-safe, JIT-safe, and compiles to an efficient matmul-like reduction.

### Pattern 2: JIT-Compatibility Test

Before running any training, verify the extractor compiles cleanly:

```python
# Source: JAX JIT compilation documentation
def test_jit_compatibility(network, env, env_params, agent_params, pop_size=4):
    """Verify extract_behavior_signature passes jit.lower().compile()."""
    import jax
    rng = jax.random.PRNGKey(0)
    # Create dummy levels (minimal valid levels)
    dummy_levels = ...  # use make_level_generator or hardcoded test levels

    def full_pipeline(rng, levels):
        rewards, values, dones, states = rollout_agent_on_levels(
            rng, env, env_params, agent_params, network, levels
        )
        sig = extract_behavior_signature(states, levels, num_steps=256)
        return sig

    # This will raise if any dynamic shape or Python control flow is present
    lowered = jax.jit(full_pipeline).lower(rng, dummy_levels)
    lowered.compile()
    print("JIT compilation: PASSED")
```

### Pattern 3: Agent Verification — PPO Hyperparameter Comparison

**DCD `arguments.py` defaults vs. our `maze_plr.py` defaults:**

| Parameter | DCD default | Our default | Match? | Note |
|-----------|------------|-------------|--------|------|
| `lr` | 1e-4 | 1e-4 | YES | |
| `gamma` | 0.995 | 0.995 | YES | |
| `gae_lambda` | 0.95 | 0.98 | **NO** | Document as difference |
| `clip_eps` (clip_param) | 0.2 | 0.2 | YES | |
| `entropy_coeff` | 0.0 | 1e-3 | **NO** | Document as difference |
| `critic_coeff` (value_loss_coef) | 0.5 | 0.5 | YES | |
| `max_grad_norm` | 0.5 | 0.5 | YES | |
| `epoch_ppo` (ppo_epoch) | 5 | 5 | YES | |
| `num_minibatches` (num_mini_batch) | 1 | 1 | YES | |
| `num_steps` | 256 | 256 | YES | |
| `num_train_envs` (num_processes) | 32 | 32 | YES | |
| Score function | `value_l1`/`positive_value_loss` | `MaxMC` | **NO** | Our codebase uses MaxMC by default; DCD uses value_l1 (PVL) as default. **ACCEL config uses MaxMC** — verify against ACCEL JSON config |

**Important nuance on score function:** DCD's default for the generic training script is `value_l1`, but the ACCEL-specific configuration files use MaxMC regret. Our `maze_plr.py` defaults to `MaxMC`, which matches the ACCEL configuration. This should be documented as "score function matches ACCEL config, differs from DCD generic default."

**Key structural difference — Level sampling:**
- DCD ACCEL: Uses PLR buffer + random mutation (minimax mutator) of existing levels
- Our implementation: Uses PLR buffer + MAP-Elites / ES latent-space mutation
- Verdict: Intentional architectural difference, NOT a bug. Document as "level sampling differs by design — we use VAE latent-space ES mutation instead of random minimax mutation."

**Key structural similarity — Regret computation:**
Both use `max_mc`: `max_return - V(s_t)` averaged over steps, where `max_return` is the maximum cumulative reward achievable across episodes on that level. Both use `compute_max_returns(dones, rewards)` then `max_mc(dones, values, max_returns)`.

### Recommended Project Structure Extension

```
es/
├── regret_fitness.py         # Extended: add extract_behavior_signature()
├── agent_loader.py           # Unchanged
├── env_bridge.py             # Unchanged
├── metrics.py                # Unchanged
└── ...

.planning/
├── DECISIONS.md              # NEW: key architectural decisions log
└── phases/
    └── 01-foundation/
        └── AGENT_VERIFICATION.md   # NEW: code diff + smoke test results
```

### Anti-Patterns to Avoid

- **Python-level loop over steps:** `for t in range(num_steps): hist = hist.at[cell_idx[t]].add(1)` — this unrolls the loop at trace time, making compilation take forever and the lowered program enormous.
- **Dynamic shape in behavior extractor:** `jnp.unique(cell_idx)` or `jnp.nonzero(hist)` — both create dynamic shapes that break JIT.
- **Collecting full `states` pytree when only `agent_pos` needed:** Collecting the full `EnvState` in the scan output is wasteful but correct. If memory is a concern, extract only `next_state.agent_pos` in the step function output instead of the full state.
- **Running DCD codebase for comparison in Phase 1:** Explicitly out of scope. Code comparison only.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Visit count accumulation | Custom Python loop with dict | `jax.nn.one_hot(...).sum(axis=0)` | JIT-incompatible; JAX one-hot is fused O(1) op |
| JIT-safe scatter | `jnp.zeros.at[idx].add(counts)` with dynamic idx | One-hot sum | `.at[]` scatter with dynamic multi-dim indices requires careful shape handling; one-hot is simpler and vmappable |
| Solvability check | Custom BFS in Python | `flood_fill_solvable` in `env_bridge.py` | Already JIT-compatible; fori_loop bounded to H*W steps |
| Agent loading | Direct orbax/pickle calls | `load_agent()` in `agent_loader.py` | Handles pickle-first fallback; glibc compatibility across machines |
| Architecture contract test | Manual shape checks | `verify_agent_contract()` in `agent_loader.py` | Already does dummy forward pass; catches checkpoint/arch mismatch |

**Key insight:** The one-hot-sum pattern is the correct JAX idiom for converting dynamic integer indices to fixed-size histograms in JIT contexts. It avoids all scatter-related tracing complications.

---

## Common Pitfalls

### Pitfall 1: JIT Trace Failure — Dynamic Shapes in Histogram

**What goes wrong:** Using `jnp.bincount`, `jnp.unique`, or Python loops in `extract_behavior_signature` causes `ConcretizationTypeError` or silent fallback to eager mode during `jax.jit` tracing.

**Why it happens:** JAX traces shapes at compile time. Any operation that produces a shape depending on array values (like `bincount` with `minlength=None` or `nonzero`) breaks the static shape invariant.

**How to avoid:** Use `jax.nn.one_hot(cell_idx, num_classes=169).sum(axis=0)`. The shape `(pop_size, 169)` is always static regardless of which cells were visited.

**Warning signs:** `ConcretizationTypeError`, `abstract_value` errors in JAX stack traces, or very long compilation times (loop unrolling).

### Pitfall 2: Wrong Coordinate Convention in Position Indexing

**What goes wrong:** `agent_pos` in `EnvState` is `[x=col, y=row]` (Level convention). Using `agent_pos[0]` as row and `agent_pos[1]` as col produces transposed histograms.

**Why it happens:** The maze Level dataclass uses `[x, y] = [col, row]` convention, but `wall_map` is indexed as `wall_map[row, col]`. This is documented in `env_bridge.py` but easy to miss.

**How to avoid:** Always use `col = agent_pos[:, :, 0]`, `row = agent_pos[:, :, 1]`, then `linear_idx = row * grid_width + col`.

**Warning signs:** Behavior signatures look similar for topologically different levels (transposed histograms are still valid histograms, just wrong).

### Pitfall 3: AutoReplayWrapper Resets Between Episodes

**What goes wrong:** The agent replays the same level multiple times within a single rollout (AutoReplayWrapper). Position data across episode boundaries still belongs to the same level, so accumulating all positions across all steps gives a multi-episode visit histogram — which is what we want. However, if the intent is single-episode behavior, dones must be used to mask the accumulation.

**Why it happens:** `rollout_agent_on_levels` uses `AutoReplayWrapper`, so within the 256-step rollout, multiple episodes may complete and restart.

**How to avoid:** For v1, accumulate ALL steps unconditionally (multi-episode histogram is correct — it reflects how the agent navigates the level overall, not just the first episode). Document this choice in the function docstring.

**Warning signs:** Expecting single-episode behavior but seeing broader coverage — this is expected behavior, not a bug.

### Pitfall 4: Smoke Test With Wrong score_function

**What goes wrong:** Running the smoke test with `--score_function pvl` instead of `MaxMC` shows regret = 0 because the PVL score function has different semantics and may not produce visible regret curves in short runs.

**How to avoid:** Always run smoke test with `--score_function MaxMC --use_accel`.

**Warning signs:** Regret stays exactly 0.0 throughout the smoke test run.

### Pitfall 5: Smoke Test Duration — 10k Steps vs 10k Updates

**What goes wrong:** `--num_updates` controls gradient update steps, not environment steps. At `num_train_envs=32` and `num_steps=256`, each update processes 8192 environment steps. For ~10k env steps, we only need ~2 updates.

**How to avoid:** For a true 10k-environment-step smoke test use `--num_updates 2` (not `--num_updates 10000`). Alternatively, use `--num_env_steps 10000` which computes updates = 10000 // (32 * 256) = 1 update.

**Warning signs:** Smoke test takes hours instead of minutes.

### Pitfall 6: Collecting Full EnvState vs. Only agent_pos

**What goes wrong:** `jax.lax.scan` collects a stacked pytree of every output. If the full `EnvState` is collected (including `maze_map` which is shape `(pop_size, padded_H, padded_W, 3)`), memory usage for a 32-level rollout of 256 steps is: `32 * 256 * (padding_size * padding_size * 3) * 4 bytes`. This may exceed GPU VRAM.

**How to avoid:** In the extended `step_fn`, emit only `next_state.agent_pos` (shape `(pop_size, 2)`) rather than the full `EnvState`. Then the scan output is `(num_steps, pop_size, 2)` — tiny.

**Warning signs:** OOM errors on GPU during JIT compilation or first run.

---

## Code Examples

### Behavior Signature Function (Complete Implementation)

```python
# Location: es/regret_fitness.py
# Source: derived from existing jax.lax.scan pattern in rollout_agent_on_levels

def extract_behavior_signature(agent_positions, num_steps, grid_h=13, grid_w=13):
    """Extract visit-count behavior signature from agent position trajectory.

    # TODO: EXPERIMENTAL v1 — behavior signature design is NOT final.
    # See .planning/DECISIONS.md for design rationale and planned revisit criteria.
    # v1: full-resolution grid histogram (13x13=169 cells), L1-normalized.

    Args:
        agent_positions: (num_steps, pop_size, 2) int array.
                         Coordinate convention: [..., 0]=col (x), [..., 1]=row (y).
                         This is the agent_pos field from EnvState stacked by scan.
        num_steps: Rollout length. Used only for documentation; normalization uses
                   actual sum, not num_steps, to handle masked steps correctly.
        grid_h: Grid height. Default 13 (CLUTTR mazes).
        grid_w: Grid width. Default 13 (CLUTTR mazes).

    Returns:
        behavior_sig: (pop_size, grid_h * grid_w) float32 array.
                      L1-normalized cell visit counts.
                      Sum along axis=-1 equals 1.0 for each level.

    Usage example:
        # After extending rollout_agent_on_levels to return agent_positions:
        agent_positions = rollout_agent_on_levels_v2(...)
        sigs = extract_behavior_signature(agent_positions, num_steps=256)
        # Verify JIT compatibility:
        jax.jit(extract_behavior_signature).lower(
            agent_positions, num_steps
        ).compile()
    """
    num_cells = grid_h * grid_w  # 169

    # Convert (col, row) to linear index: row * width + col
    col = agent_positions[..., 0].astype(jnp.int32)  # (num_steps, pop_size)
    row = agent_positions[..., 1].astype(jnp.int32)  # (num_steps, pop_size)
    cell_idx = row * grid_w + col                     # (num_steps, pop_size)

    # One-hot encode each step's cell visit, then sum over steps
    # Shape: (num_steps, pop_size, num_cells) -> sum -> (pop_size, num_cells)
    one_hot = jax.nn.one_hot(cell_idx, num_classes=num_cells, dtype=jnp.float32)
    hist = one_hot.sum(axis=0)  # (pop_size, num_cells)

    # L1 normalize (each row sums to 1.0)
    total = hist.sum(axis=-1, keepdims=True)      # (pop_size, 1)
    hist = hist / jnp.maximum(total, 1.0)         # safe div-by-zero guard

    return hist
```

### Extended rollout_agent_on_levels to Emit agent_pos

```python
# Minimal change to existing rollout_agent_on_levels in es/regret_fitness.py:
# Change step_fn return from:
#   return (rng, hstate, next_obs, next_state, next_done), (reward, value, next_done)
# To:
#   return (rng, hstate, next_obs, next_state, next_done), (reward, value, next_done, next_state.agent_pos)
#
# Then the scan returns:
#   _, (rewards, values, dones, agent_positions) = jax.lax.scan(...)
# Where agent_positions has shape (num_steps, pop_size, 2).
```

### Agent Verification Smoke Test Command

```bash
# From project root. Run with ACCEL enabled, MaxMC score function.
# ~2 gradient updates = enough to confirm no crash, regret > 0.
cd /path/to/project

WANDB_MODE=offline python examples/maze_plr.py \
  --use_accel \
  --score_function MaxMC \
  --num_updates 5 \
  --num_train_envs 32 \
  --num_steps 256 \
  --seed 42 \
  --project FOUNDATION_SMOKE_TEST \
  --run_name smoke_test_accel_maxmc \
  --checkpoint_save_interval 999
```

**Passing criteria:**
1. No Python exception or JAX error
2. WandB (offline) logs a non-zero, changing regret value across updates
3. Solve rate (eval) is between 0.0 and 1.0
4. Script exits cleanly after 5 updates

### DECISIONS.md First Entry Template

```markdown
# Key Architectural Decisions

## DECISION-01: Behavior Signature v1 Design

**Date:** 2026-02-28
**Status:** EXPERIMENTAL — subject to revision after NS-ES validation (Phase 3)

**Decision:** v1 behavior signature is a 169-element L1-normalized visit-count
histogram over the 13x13 maze grid.

**Rationale:**
- Visit counts encode "where did the agent spend time?" — captures maze traversal
  pattern without committing to trajectory ordering or action semantics
- 13x13=169 cells matches the CLUTTR maze resolution — no lossy binning in v1
- L1 normalization makes signatures comparable across rollout lengths
- One-hot sum pattern is JIT-safe without scatter complications

**Known limitations of v1:**
- Ignores action sequence (turn left 3x vs. turn left 1x look the same)
- Does not capture temporal ordering of visits
- May conflate different maze structures that happen to produce similar paths
- If the agent rarely reaches the far side of large mazes, many histogram
  bins will always be near-zero and carry no discriminatory signal

**Planned revisit criteria:**
- After Phase 3 NS-ES validation: check if regret curves improve vs. vanilla CMA-ES
- If novelty reward hacks (high novelty, flat regret): redesign signature
- If mode collapse persists: consider action-sequence or temporal features
- See INFRA-04, ALGO-01 for the downstream consumers of this design

**Code location:** `es/regret_fitness.py::extract_behavior_signature()`
**Marked with:** `# TODO: EXPERIMENTAL v1`
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| DCD (PyTorch, non-JIT) | jaxued (JAX, fully JIT) | 2023 (minimax paper) | 100x faster training; requires JAX-native ops for all new code |
| Global histogram (bincount) | One-hot sum pattern | JAX idiom | `bincount` not JIT-compatible; one-hot sum is the JAX equivalent |
| CMA-ES for diversity | MAP-Elites (existing) / NS-ES (Phase 3) | Documented in REGRET_PIPELINE_README.md | CMA-ES collapses; MAP-Elites covers space; NS-ES is the MVP thesis contribution |

**Key historical finding from codebase (REGRET_PIPELINE_README.md):**
> "CMA-ES diversity collapse: After ~200 generations, CMA-ES converges to one environment type. All environments share the same structure. MAP-Elites solves this: 100% archive coverage across all 48 behavior cells."

This is the documented motivation for Phase 1's behavior signature — Phase 3 NS-ES needs a behavior descriptor to avoid this collapse.

---

## Open Questions

1. **Should `rollout_agent_on_levels` be modified in-place or should a new `rollout_agent_on_levels_with_positions` function be created?**
   - What we know: The existing function signature is used by `regret_fitness()` which is called from `map_elites.py`, `evolve_envs.py`, and potentially `maze_plr.py`. Changing the return signature breaks callers.
   - What's unclear: How many callers destructure the return value? Would adding a fourth return element break them silently?
   - Recommendation: Create a thin wrapper `rollout_agent_on_levels_with_positions()` that calls the existing function with a modified internal step_fn, returning both the existing tuple AND agent_positions. This preserves backward compatibility.

2. **Should behavior extraction share the same rollout as regret computation, or use a separate pass?**
   - What we know: CONTEXT.md locks "separate pass from regret computation for simplicity during development."
   - What's unclear: Whether the separate pass doubles GPU time during ES evaluation.
   - Recommendation: Implement as a separate function `extract_behavior_signature_from_levels()` that internally calls the rollout and returns only agent_positions. Phase 2/3 can merge if profiling shows it's a bottleneck.

3. **What is the smoke test checkpoint directory convention?**
   - What we know: `maze_plr.py` creates checkpoints based on `--checkpoint_directory` and `--run_name`. Default save interval is 2 updates.
   - Recommendation: Pass `--checkpoint_save_interval 999` to suppress checkpointing during the smoke test (avoids filling disk). Use `WANDB_MODE=offline` to avoid network dependency.

---

## Behavior Signature Design Recommendation

This section addresses the CONTEXT.md deferral of grid resolution and normalization to research.

**Recommendation: 13x13 full-resolution histogram, L1-normalized, multi-episode**

| Design choice | Recommendation | Alternative | When to switch |
|---------------|----------------|-------------|----------------|
| Grid resolution | 13x13 = 169 cells (full res) | 7x7 = 49 cells (coarser) | If 169-dim vectors cause k-NN slowness in Phase 2 |
| Normalization | L1 (divide by total steps) | No normalization (raw counts) | Raw counts make shorter/longer rollouts incomparable; L1 is always correct |
| Episode boundary | Ignore (accumulate all steps) | Reset per episode | Ignoring boundaries gives richer coverage signal |
| Temporal order | Ignored (histogram = bag of visits) | Sequence encoding | Too complex for v1; upgrade path exists via recurrent descriptors |

**Rationale for 13x13 full-resolution:**
- The maze is 13x13 by construction (CLUTTR constraint, `inner_dim=13` in `regret_fitness.py`)
- Coarser binning loses information about which specific corridors were traversed
- 169-dimensional vectors are tiny by any standard — no scalability concern
- k-NN over 4000 buffer entries × 169 dimensions = trivially fast with `jnp.linalg.norm` broadcasting (Phase 2 concern)

---

## Agent Verification Scope

### What to Compare (FOUND-01)

The verification produces `AGENT_VERIFICATION.md`. Structure it as a flat list of all found differences, tagged as intentional or potential bug:

**PPO Implementation — items to verify:**
1. `compute_gae()` — GAE formula: `delta = r + γ * V(s') * (1-d) - V(s)`, `gae = delta + γ * λ * (1-d) * gae`. Confirm matching formula in both codebases.
2. `update_epoch()` — clipped surrogate objective, value loss clipping, entropy bonus. Confirm same loss formula.
3. `max_mc()` — confirm it matches DCD's regret computation. Our `max_mc` comes from jaxued which is the JAX reimplementation of DCD.
4. `compute_max_returns()` — confirm this matches DCD's max-return computation.

**Known differences to document as intentional:**
- `gae_lambda`: DCD default 0.95 vs ours 0.98 — document as "tuned differently, not a bug"
- `entropy_coeff`: DCD default 0.0 vs ours 1e-3 — document as "minor difference; may affect exploration"
- Level sampling: DCD uses minimax random mutation; we use ES latent-space mutation — document as "intentional architectural difference (the thesis contribution)"
- Score function name: DCD calls it `level_replay_strategy=value_l1` (generic default); ACCEL configs use MaxMC. Our code uses MaxMC as the default for ACCEL runs — document as matching ACCEL configuration.

---

## Sources

### Primary (HIGH confidence — direct codebase inspection)

- `es/regret_fitness.py` — rollout_agent_on_levels signature, max_mc usage, step_fn pattern
- `es/env_bridge.py` — coordinate conventions (col=x, row=y), grid dimensions (13x13)
- `es/agent_loader.py` — ActorCritic architecture, load_agent(), verify_agent_contract()
- `es/REGRET_PIPELINE_README.md` — CMA-ES collapse documentation, MAP-Elites results
- `es/metrics.py` — existing diversity metrics (latent L2, Hamming) — NOT to be reused for behavior sig
- `examples/maze_plr.py` — PPO hyperparameter defaults (argparse), score_function choices, ACCEL branch structure, compute_gae() formula
- `src/jaxued/level_sampler.py` — levels_extra dict pattern, insert/update API
- `src/jaxued/environments/maze/env.py` — EnvState.agent_pos field, coordinate convention

### Secondary (MEDIUM confidence — official repository, web-fetched)

- `github.com/facebookresearch/dcd arguments.py` — DCD default PPO hyperparameters: lr=1e-4, gamma=0.995, gae_lambda=0.95, clip_param=0.2, entropy_coef=0.0, value_loss_coef=0.5, ppo_epoch=5, num_mini_batch=1, num_steps=256, num_processes=32
- JAX one-hot documentation — `jax.nn.one_hot(x, num_classes=N).sum(axis=0)` as JIT-safe histogram pattern

### Tertiary (LOW confidence — requires empirical validation)

- Behavior signature dimensionality choice (13x13=169 vs coarser) — recommended based on grid structure; effectiveness needs NS-ES ablation in Phase 5
- Optimal smoke test duration (5 updates) — should show regret > 0; may need adjustment if ACCEL warm-up takes more steps

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all components exist; verified by direct file inspection
- Architecture patterns: HIGH — code patterns derived directly from `regret_fitness.py` step_fn; one-hot histogram is standard JAX idiom
- Agent verification scope: HIGH — PPO hyperparameter table derived from direct codebase + DCD repo inspection
- Behavior signature design: MEDIUM — recommended design is well-motivated but effectiveness is empirically unvalidated; explicit revisit path documented
- Pitfalls: HIGH — all pitfalls derived from direct code structure analysis, not speculation

**Research date:** 2026-02-28
**Valid until:** 2026-03-30 (stable JAX codebase; 30-day window appropriate)
