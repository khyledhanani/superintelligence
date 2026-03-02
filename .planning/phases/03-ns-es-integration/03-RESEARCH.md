# Phase 3: NS-ES Integration - Research

**Researched:** 2026-03-02
**Domain:** NS-ES algorithm integration, archive warm-up, two-bucket PLR sampling, WandB metrics
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### NS-ES Algorithm Design
- **Fitness for tell():** Composite fitness F = α·Regret + β·Novelty — reuses Phase 2 `compute_fitness()` directly. NS-ES and CMA-ES differ only in that NS-ES uses this composite signal rather than regret-only.
- **ask() behavior:** Identical to CMAESStrategy — sample candidates from current latent distribution. The NS-ES distinction is entirely in fitness computation before tell(). No changes to ask() needed.
- **Novelty source:** NSESStrategy reads behavior signatures from the replay buffer for k-NN novelty computation. No separate novelty archive — one data structure, no duplication.
- **Buffer insertion:** Same criterion as all strategies — regret-based insertion into the PLR replay buffer. Behavior signatures are stored per-entry as established in Phase 2.

#### Archive Warm-up
- **Latent distribution:** Sample 256 latents from N(0,I) — same initial distribution as CMA-ES. Warm-up and ES are aligned from the start.
- **Timing:** Synchronous — all 256 warm-up evals complete before step 0 of training. Buffer is pre-populated before any training update.
- **Step budget:** Warm-up is overhead (pre-training), does NOT count toward the training step budget. All strategies get the same N steps of actual training.
- **Solvability gate:** Apply BFS solvability check on each decoded maze before evaluating it. Skip unsolvable latents silently (no eval, no insertion). Check `es/` folder for existing BFS solver before implementing from scratch.
- **Failure handling:** On NaN regret or NaN behavior signature: skip and continue with a warning log. Buffer may end up with slightly fewer than 256 entries — training proceeds regardless.

#### Two-Bucket Sampling
- **Default p values:** Match ACCEL's existing sampling split from `maze_plr.py` — use as baseline default. Researcher to confirm the exact value.
- **Schedule:** Fixed split throughout training. No annealing for Phase 3 MVP.
- **Empty buffer guard:** If buffer is empty when two-bucket sampler is called, fall back to 100% frontier sampling. Add a guard clause — this should not happen given synchronous warm-up, but guard for safety.
- **Configuration:** p_replay and p_frontier live in the ES config dict only (consistent with how alpha/beta are handled in Phase 2). No new CLI flags for Phase 3.

### Claude's Discretion
- Exact BFS solver implementation (or reuse from `es/` if it exists)
- WandB metric logging frequency and exact key names (must include: `regret`, `novelty_score`, `replay_buffer_size`, `buffer_occupied`)
- NSESStrategy class structure and file location within `es/`
- How to wire warm-up into `maze_plr.py` (function vs inline block)

### Deferred Ideas (OUT OF SCOPE)
- None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ALGO-01 | NS-ES strategy implementation with composite fitness and buffer-as-novelty-archive | NSESStrategy class mirrors CMAESStrategy but calls compute_fitness_batch() before tell(); novelty extracted from buffer via compute_novelty_batch() (both from Phase 2 es_components) |
| INTEG-01 | Two-bucket sampling wired into ACCEL training loop (replay prob p + ES frontier 1-p) | LevelSampler.sample_replay_decision() already implements the two-bucket split via replay_prob; maze_plr.py default is p_replay=0.8; train.py uses the same; no structural change needed |
| INTEG-02 | Archive warm-up phase (init_pop evaluation before training starts) | flood_fill_solvable() and rollout_agent_on_levels_with_positions() + extract_behavior_signature() already exist in es/; warm-up function calls eval_fn, extracts behavior_sig, calls insert_batch with behavior_sig populated |
| INTEG-03 | End-to-end training pipeline with ES-generated curriculum, WandB logging, checkpointing | wandb 0.25.0 is installed in jax_env; train.py already logs to CSV; WandB init/log pattern confirmed in examples/maze_plr.py |
</phase_requirements>

---

## Summary

Phase 3 wires NSESStrategy into the existing ACCEL training loop in `accel_training/train.py`. All the infrastructure needed is already present from Phase 2: `CMAESStrategy`, `compute_fitness_batch()`, `compute_novelty_batch()`, and the PLR buffer with `behavior_sig` field. The critical gap in the current `train.py` is the failing assert at line 353: `assert "behavior_sig" in level_extra` — this assert was placed as a forward-contract for Phase 3. Phase 3 must close this gap by: (1) adding `rollout_agent_on_levels_with_positions()` + `extract_behavior_signature()` calls at every buffer insertion point, (2) implementing `NSESStrategy` as a composite-fitness variant of `CMAESStrategy`, (3) adding synchronous archive warm-up before the training loop, and (4) wiring WandB metrics.

The BFS solvability solver already exists at `es/env_bridge.py::flood_fill_solvable()` — no reimplementation needed. The two-bucket sampling is already implemented by `LevelSampler.sample_replay_decision()` with `replay_prob=0.8` matching `maze_plr.py`'s default. WandB 0.25.0 is installed in `jax_env` and the integration pattern is documented in `examples/maze_plr.py`. The phase is structurally well-supported — the main work is: (a) writing `NSESStrategy`, (b) closing the `behavior_sig` population gap at insert sites, and (c) writing the warm-up function.

**Primary recommendation:** Implement `NSESStrategy` as a 1:1 copy of `CMAESStrategy` with one behavioral change in `tell()`: compute novelty from the PLR buffer before calling `compute_fitness_batch()`, then negate the composite for evosax. Wire warm-up as a standalone function `run_archive_warmup(...)` called once before the training loop in `train.py`.

---

## Standard Stack

### Core (all confirmed present in jax_env)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| jax | 0.5.3 | Array compute, JIT, vmap | Project baseline — all new code must be JIT-compatible |
| evosax | 0.2.0 | CMA_ES algorithm | Already used in CMAESStrategy; NSESStrategy reuses same wrapper |
| flax | 0.10.7 | Neural net + Pytree structs | Agent params, TrainState |
| optax | 0.2.7 | Optimizer (Adam) | Already wired in train.py |
| wandb | 0.25.0 | Experiment tracking | Required by INTEG-03; already installed |
| numpy | (system) | CPU-side array ops | Warm-up latent sampling via rng_np |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| es/env_bridge.py | local | flood_fill_solvable, cluttr_sequence_to_level | Warm-up solvability gate |
| es/regret_fitness.py | local | rollout_agent_on_levels_with_positions, extract_behavior_signature | Behavior sig extraction during warm-up and insertion |
| accel_training/es_components/ | local | ESStrategy, CMAESStrategy, compute_fitness_batch, compute_novelty_batch | NSESStrategy reuses directly |
| src/jaxued/level_sampler.py | local | LevelSampler.insert_batch, sample_replay_decision | PLR buffer ops |

### No New Installations Needed

All dependencies are available. No `pip install` steps required for Phase 3.

---

## Architecture Patterns

### Recommended File Layout

```
accel_training/
  es_components/
    __init__.py          # add NSESStrategy to exports
    interface.py         # ESStrategy Protocol (unchanged)
    cmaes_strategy.py    # CMAESStrategy (unchanged)
    fitness.py           # compute_fitness, compute_fitness_batch (unchanged)
    novelty.py           # compute_novelty_knn, compute_novelty_batch (unchanged)
    nses_strategy.py     # NEW — NSESStrategy
  train.py               # MODIFIED — behavior_sig, warm-up, WandB, ES routing
  config.yml             # MODIFIED — add es_config block (alpha, beta, pop_size, sigma_init, k_novelty)
```

### Pattern 1: NSESStrategy — Composite-Fitness Variant of CMAESStrategy

**What:** NSESStrategy satisfies the `ESStrategy` Protocol with identical `ask()` behavior to `CMAESStrategy`. The only behavioral change is in `tell()`: before updating evosax, compute novelty from the PLR buffer, then compute composite fitness F = α·regret + β·novelty, then negate for evosax.

**When to use:** Called from `train.py` in the NEW/mutate branch, after eval_fn returns regrets and the buffer contains behavior_sigs.

**Key insight:** The PLR buffer is the novelty archive — no separate data structure. At `tell()` time, read `sampler["levels_extra"]["behavior_sig"]` and `valid_mask` (from `sampler["size"]`) to compute novelty.

```python
# Source: es_components/nses_strategy.py pattern (derived from cmaes_strategy.py + novelty.py)

class NSESStrategy:
    """NS-ES: CMA-ES with composite fitness (regret + novelty from PLR buffer)."""

    def __init__(self, param_dim: int, pop_size: int):
        from evosax.algorithms import CMA_ES
        self._es = CMA_ES(population_size=pop_size, solution=jnp.zeros(param_dim))
        self._param_dim = param_dim
        self._pop_size = pop_size

    def init_state(self, rng: jax.Array, config: dict) -> dict:
        es_params = self._es.default_params
        if "sigma_init" in config:
            es_params = es_params.replace(std_init=config["sigma_init"])
        mean = config.get("mean", jnp.zeros(self._param_dim))
        es_state = self._es.init(rng, mean, es_params)
        return {"es_state": es_state, "es_params": es_params}

    def ask(self, state: dict, rng: jax.Array) -> tuple[jnp.ndarray, dict]:
        """Identical to CMAESStrategy.ask()."""
        population, new_es_state = self._es.ask(rng, state["es_state"], state["es_params"])
        return population, {**state, "es_state": new_es_state}

    def tell(
        self,
        state: dict,
        candidates: jnp.ndarray,
        regrets: jnp.ndarray,          # raw positive MaxMC regret (NOT negated)
        buffer_sigs: jnp.ndarray,      # (capacity, D) from sampler["levels_extra"]["behavior_sig"]
        valid_mask: jnp.ndarray,       # (capacity,) bool from jnp.arange(capacity) < buf_size
        candidate_sigs: jnp.ndarray,   # (pop_size, D) behavior sigs of current candidates
        alpha: float,
        beta: float,
        k: int = 5,
    ) -> dict:
        from accel_training.es_components.novelty import compute_novelty_batch
        from accel_training.es_components.fitness import compute_fitness_batch
        novelties = compute_novelty_batch(candidate_sigs, buffer_sigs, valid_mask, k=k)
        composite = compute_fitness_batch(regrets, novelties, alpha=alpha, beta=beta)
        fitness_for_evosax = -composite   # evosax minimizes
        dummy_key = jax.random.PRNGKey(0)
        new_es_state, _ = self._es.tell(
            dummy_key, candidates, fitness_for_evosax, state["es_state"], state["es_params"]
        )
        return {**state, "es_state": new_es_state}
```

**Note:** `tell()` signature differs from `ESStrategy` Protocol for NSESStrategy because it requires novelty inputs. The Protocol defines the minimum interface; NSESStrategy extends it. The caller in `train.py` uses the concrete type, not the Protocol abstraction, for this call.

### Pattern 2: Archive Warm-up as Standalone Function

**What:** A `run_archive_warmup(...)` function called once in `train.py` before the training loop. It samples 256 latents from N(0,I), evaluates each via `eval_fn`, extracts behavior signatures, and inserts valid entries into the PLR buffer.

**When to use:** Once, immediately after `train_state` initialization and before `for update in range(config["num_updates"])`.

**Critical:** Warm-up must populate `behavior_sig` in `level_extra` at insertion — this is what closes the failing assert at line 353 of `train.py`.

```python
# Source: pattern derived from train.py eval_fn usage (lines 320-363)

def run_archive_warmup(
    rng, rng_np, train_state, level_sampler, eval_fn,
    n_warmup=256, latent_dim=64, num_envs=32,
    rollout_env=None, env_params=None, network=None,
    config=None,
):
    """Pre-populate PLR buffer before training step 0.

    Samples n_warmup latents from N(0,I), evaluates via eval_fn,
    extracts behavior signatures, inserts valid levels into PLR buffer.
    Skips unsolvable levels (solvability gate is inside eval_fn via 'valid' mask).
    On NaN: skip with warning, continue.

    Returns: (rng, rng_np, train_state)  — buffer pre-populated
    """
    import numpy as np
    from es.regret_fitness import rollout_agent_on_levels_with_positions, extract_behavior_signature

    latents_np = rng_np.standard_normal((n_warmup, latent_dim)).astype(np.float32)
    latents_jax = jnp.array(latents_np)

    rng, rng_eval = jax.random.split(rng)
    sequences, levels, regrets, max_returns, valid = eval_fn(
        rng_eval, train_state.params, latents_jax
    )
    # Extract behavior signatures for all candidates
    rng, rng_rollout = jax.random.split(rng)
    _, _, _, agent_positions = rollout_agent_on_levels_with_positions(
        rng_rollout, rollout_env, env_params,
        train_state.params, network, levels,
        num_steps=config["eval_rollout_steps"],
    )
    behavior_sigs = extract_behavior_signature(agent_positions, config["eval_rollout_steps"])

    # NaN guard before insertion
    regrets_np = np.asarray(regrets)
    valid_np = np.asarray(valid)
    behavior_sigs_np = np.asarray(behavior_sigs)
    nan_mask = np.isnan(regrets_np) | np.any(np.isnan(behavior_sigs_np), axis=-1)
    if nan_mask.any():
        print(f"  [warmup] WARNING: {nan_mask.sum()} NaN entries skipped")
    valid_np = valid_np & ~nan_mask

    # Tile/slice to num_envs for insert_batch (LevelSampler requires fixed batch size)
    if n_warmup < num_envs:
        repeat = (num_envs + n_warmup - 1) // n_warmup
        latents_jax_pad = jnp.tile(latents_jax, (repeat, 1))[:num_envs]
        levels_pad = jax.tree_util.tree_map(
            lambda x: jnp.tile(x, (repeat, *([1] * (x.ndim - 1))))[:num_envs], levels
        )
        regrets_pad = jnp.tile(regrets, repeat)[:num_envs]
        max_returns_pad = jnp.tile(max_returns, repeat)[:num_envs]
        behavior_sigs_pad = jnp.tile(jnp.array(behavior_sigs), (repeat, 1))[:num_envs]
    else:
        latents_jax_pad = latents_jax[:num_envs]
        levels_pad = jax.tree_util.tree_map(lambda x: x[:num_envs], levels)
        regrets_pad = regrets[:num_envs]
        max_returns_pad = max_returns[:num_envs]
        behavior_sigs_pad = jnp.array(behavior_sigs)[:num_envs]

    level_extra = {
        "max_return":   max_returns_pad,
        "latent":       latents_jax_pad,
        "behavior_sig": behavior_sigs_pad,   # closes the assert in train.py line 353
    }
    sampler, _ = level_sampler.insert_batch(
        train_state.sampler, levels_pad, regrets_pad, level_extra,
    )
    train_state = train_state.replace(sampler=sampler)
    print(f"  [warmup] Done: {int(sampler['size'])} entries in PLR buffer")
    return rng, rng_np, train_state
```

### Pattern 3: behavior_sig Population at Training Insert Sites

**What:** Every call to `level_sampler.insert_batch()` in `train.py` must include `"behavior_sig"` in `level_extra`. The assert at line 353 enforces this. Phase 3 must:
1. Add `rollout_agent_on_levels_with_positions` import from `es.regret_fitness`
2. After `eval_fn(...)` in NEW/mutate branches, run a second rollout (or reuse the eval rollout) to extract agent positions
3. Call `extract_behavior_signature(agent_positions, num_steps)` to get `behavior_sigs`
4. Add `"behavior_sig": behavior_sigs` to `level_extra` before `insert_batch`

**Critical subtlety:** `eval_fn` in `ued_interface.py` already runs a rollout internally (lines 214-241) but does NOT emit agent positions — it only returns `(sequences, levels, regrets, max_returns, valid)`. Phase 3 must either:
- Option A: Run a second rollout via `rollout_agent_on_levels_with_positions()` after `eval_fn()` — clean separation, small overhead
- Option B: Extend `eval_fn` in `ued_interface.py` to also return agent_positions — touches more surface area

**Recommendation:** Option A (separate rollout call). The warm-up and training insert patterns are identical. The overhead is modest (eval_rollout_steps=128, same config value).

### Pattern 4: WandB Integration

**What:** Add `wandb.init()` before training loop, `wandb.log()` every N updates. Follow the exact pattern from `examples/maze_plr.py`.

**Required metric keys (from CONTEXT.md):** `regret`, `novelty_score`, `replay_buffer_size`, `buffer_occupied`

**Additional recommended keys (from maze_plr.py pattern):**
```python
wandb.log({
    "update":              update,
    "regret":              mean_regret,           # mean candidate regret this step
    "novelty_score":       mean_novelty,           # mean candidate novelty (when mode != replay)
    "replay_buffer_size":  int(sampler["size"]),   # total buffer capacity used
    "buffer_occupied":     buf_occupied_frac,       # size / capacity
    "mode":                mode,                   # "new" | "replay" | "mutate"
    "valid_fraction":      valid_frac,
    "mean_buffer_score":   mean_buf_score,
}, step=update)
```

**WandB init pattern from maze_plr.py:**
```python
# Source: examples/maze_plr.py lines 460-469
import wandb
run = wandb.init(
    config=config,
    project=config.get("wandb_project", "es-accel"),
    group=config["run_name"],
    tags=["NS-ES"],
)
wandb.define_metric("update")
wandb.define_metric("regret", step_metric="update")
wandb.define_metric("novelty_score", step_metric="update")
wandb.define_metric("replay_buffer_size", step_metric="update")
wandb.define_metric("buffer_occupied", step_metric="update")
```

**Config additions for `config.yml`:**
```yaml
# ES config (Phase 3 NS-ES)
es_strategy: ns_es        # "ns_es" | "cma_es" for baseline comparison
es_alpha: 0.8             # regret weight in composite fitness
es_beta: 0.2              # novelty weight in composite fitness
es_pop_size: 16           # matches n_candidates
es_sigma_init: 0.5        # initial CMA-ES sigma
es_k_novelty: 5           # k for k-NN novelty (matches Phase 2 default)
warmup_n: 256             # number of latents for archive warm-up

# WandB
wandb_project: es-accel
wandb_log_freq: 10        # log every N updates
```

### Anti-Patterns to Avoid

- **Separate novelty archive:** Do not create a separate list/array of behavior signatures outside the PLR buffer. The buffer IS the archive. Read `sampler["levels_extra"]["behavior_sig"]` directly.
- **Modifying ESStrategy Protocol for NSESStrategy:** Do not change `interface.py`. The Protocol defines the minimum surface. NSESStrategy's extended `tell()` signature is acceptable — train.py calls the concrete type.
- **Running eval_fn twice inside warm-up:** eval_fn already runs the rollout internally. For behavior_sig extraction, run `rollout_agent_on_levels_with_positions()` as a separate call — this is intentional duplication to avoid modifying `ued_interface.py`'s eval_fn signature.
- **Warm-up counting toward step budget:** The `for update in range(config["num_updates"])` loop starts after warmup completes. Never increment the update counter during warm-up.
- **JIT-tracing buffer size as a dynamic shape:** When building `valid_mask` for novelty, use `jnp.arange(capacity) < buf_size` (capacity is static, buf_size is a traced int32 value). This pattern is already used in `train.py` line 413.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| BFS/flood-fill solvability | Custom BFS solver | `es.env_bridge.flood_fill_solvable()` | Already JIT-safe, vmap-compatible fori_loop implementation |
| k-NN novelty scoring | Custom k-NN | `es_components.novelty.compute_novelty_batch()` | Already JAX JIT, masked, tested in Phase 2 |
| Composite fitness | Custom F=α·r+β·n | `es_components.fitness.compute_fitness_batch()` | Tested in Phase 2 with known-value tests |
| CMA-ES inner loop | Custom CMA-ES | `evosax.algorithms.CMA_ES` via existing wrapper | evosax handles all CMA-ES math; NSESStrategy reuses it |
| Experiment tracking | Custom CSV metrics | `wandb` 0.25.0 | Already installed, maze_plr.py pattern available |
| Latent-space warm-up eval | Custom eval pipeline | Reuse `eval_fn` from `ued_interface.build_eval_fn()` | eval_fn is already JIT-compiled with solvability gate baked in |
| Behavior signature extraction | Custom histogram | `es.regret_fitness.extract_behavior_signature()` | Phase 1 deliverable; tested, JIT-safe, L1-normalized 169-dim |

**Key insight:** Every primitive this phase needs was built in Phases 1 and 2 specifically to be reused here. Phase 3 is integration work, not new algorithm research.

---

## Common Pitfalls

### Pitfall 1: behavior_sig Not Populated at Insert — Failing Assert
**What goes wrong:** `train.py` line 353 has `assert "behavior_sig" in level_extra` as a forward-contract placed in Phase 2. If Phase 3 adds the warm-up but forgets to add `behavior_sig` to `level_extra` in the NEW/mutate branches of the training loop, the assert will fire at runtime.
**Why it happens:** The assert was added to the insert_batch call only. The warm-up inserts separately. Both sites must be fixed.
**How to avoid:** Fix all `level_extra = {...}` dicts in `train.py` to include `"behavior_sig"` before the assert site. There are two insert_batch call sites: warm-up and the NEW/mutate branch at line 357.
**Warning signs:** `AssertionError: All PLR buffer insertions must include 'behavior_sig'` at startup.

### Pitfall 2: behavior_sig Dimension Mismatch in sampler
**What goes wrong:** `train_state.sampler` is initialized at line 181-188 of `train.py` with `"behavior_sig": jnp.zeros(169, dtype=jnp.float32)`. The `level_extra` placeholder sets the expected shape. If `extract_behavior_signature()` returns a different shape or dtype, jaxued's `insert_batch` will fail with a shape error inside `jax.lax.scan`.
**Why it happens:** The sampler placeholder fixes the pytree shape at initialization. Any deviation in dtype or shape during insertion causes a JAX pytree mismatch.
**How to avoid:** Confirm `extract_behavior_signature()` returns `(pop_size, 169)` float32. The slice passed to `insert_batch` must be `(num_envs, 169)` float32.
**Warning signs:** Shape mismatch errors inside `jax.lax.scan` during insert_batch.

### Pitfall 3: NSESStrategy tell() Called with Buffer Before Any Insertions
**What goes wrong:** If NSESStrategy.tell() is called during the training loop before any levels have been inserted (e.g., first iteration of training), the valid_mask is all-False, and `compute_novelty_batch()` returns `inf` for all candidates (as documented in `novelty.py` tests). This makes composite fitness undefined.
**Why it happens:** Synchronous warm-up prevents this in normal flow, but if warm-up fails silently or buffer is 0-filled, novelty is inf and fitness negation pushes evosax into NaN territory.
**How to avoid:** Warm-up always runs first (locked decision). Guard: if `buf_size == 0` after warm-up, log a critical warning and fall back to regret-only fitness (beta=0) for first N steps.
**Warning signs:** `inf` or `nan` values in wandb `novelty_score` metric at step 0.

### Pitfall 4: Two-Bucket p_replay Already Hardcoded in LevelSampler
**What goes wrong:** `LevelSampler` takes `replay_prob` at construction time (line 44 of `level_sampler.py`). The "two-bucket sampling" is already implemented — it IS `sample_replay_decision()` with the `replay_prob` parameter. There is no second sampler to wire; just confirm `config["replay_prob"]` is set correctly.
**Why it happens:** The name "two-bucket sampling" could suggest a new component is needed, but jaxued's LevelSampler already implements exactly this.
**How to avoid:** Do NOT add a second sampling layer. The two buckets are already: (1) replay from buffer, (2) ES frontier. `replay_prob=0.8` is the default in both `maze_plr.py` and `train.py`'s `config.yml`. Empty buffer guard is needed only because `minimum_fill_ratio=0.5` gates replay — warm-up exceeds this easily with 256 entries into a 4000-capacity buffer (6.4% fill vs. 50% gate means replay won't trigger until buffer is 50% full post-warmup). This is correct behavior.
**Warning signs:** Unexpected `mode == "new"` for all early updates (minimum_fill_ratio gate still in effect until buffer reaches 2000 entries).

### Pitfall 5: WandB Run Not Initialized Before wandb.log()
**What goes wrong:** Calling `wandb.log()` before `wandb.init()` raises a `wandb.errors.Error`. If `wandb_log_freq` check happens before init, this crashes training.
**Why it happens:** Config-conditional WandB (only log if wandb_project is set) can accidentally call `wandb.log` before `wandb.init` if the guard is misplaced.
**How to avoid:** Call `wandb.init()` unconditionally at the start of `train()` (after config is loaded), or guard all `wandb.log` calls with `if run is not None`. Follow the pattern in `examples/maze_plr.py`.

### Pitfall 6: eval_fn Does Not Return Agent Positions
**What goes wrong:** Phase 3 calls `eval_fn(rng, params, latents)` expecting agent positions, but `eval_fn` (from `ued_interface.build_eval_fn()`) returns only `(sequences, levels, regrets, max_returns, valid)`. Attempting to unpack agent_positions from this call will fail.
**Why it happens:** `eval_fn` was built in Phase 1 for regret computation only; position tracking was added to `regret_fitness.py` as a separate function in Phase 1 but not to the `ued_interface` wrapper.
**How to avoid:** After `eval_fn(...)` returns `levels` and `valid`, call `rollout_agent_on_levels_with_positions()` separately with the same `levels`. This second rollout is intentional — it's the behavior signature extraction pass. Use `eval_rollout_steps` from config (128 steps, same as eval_fn internally).

---

## Code Examples

### NSESStrategy — Minimal Implementation

```python
# Source: pattern from accel_training/es_components/cmaes_strategy.py + novelty.py
# File: accel_training/es_components/nses_strategy.py

class NSESStrategy:
    """NS-ES: CMA-ES with composite fitness F = alpha*regret + beta*novelty.

    ask() is identical to CMAESStrategy.
    tell() differs: requires candidate behavior signatures and buffer state.
    """

    def __init__(self, param_dim: int, pop_size: int):
        from evosax.algorithms import CMA_ES
        dummy_solution = jnp.zeros(param_dim)
        self._es = CMA_ES(population_size=pop_size, solution=dummy_solution)
        self._param_dim = param_dim
        self._pop_size = pop_size

    def init_state(self, rng, config):
        es_params = self._es.default_params
        if "sigma_init" in config:
            es_params = es_params.replace(std_init=config["sigma_init"])
        mean = config.get("mean", jnp.zeros(self._param_dim))
        es_state = self._es.init(rng, mean, es_params)
        return {"es_state": es_state, "es_params": es_params}

    def ask(self, state, rng):
        """Identical to CMAESStrategy.ask()."""
        population, new_es_state = self._es.ask(rng, state["es_state"], state["es_params"])
        return population, {**state, "es_state": new_es_state}

    def tell(self, state, candidates, regrets, candidate_sigs,
             buffer_sigs, valid_mask, alpha, beta, k=5):
        """Update CMA-ES with composite fitness.

        Args:
            candidates:     (pop_size, param_dim) — latent vectors from ask()
            regrets:        (pop_size,) — raw positive MaxMC regret
            candidate_sigs: (pop_size, 169) — behavior sigs of current candidates
            buffer_sigs:    (capacity, 169) — from sampler["levels_extra"]["behavior_sig"]
            valid_mask:     (capacity,) bool — jnp.arange(capacity) < buf_size
            alpha, beta:    float weights from config
            k:              int (static) for k-NN
        """
        from accel_training.es_components.novelty import compute_novelty_batch
        from accel_training.es_components.fitness import compute_fitness_batch
        novelties = compute_novelty_batch(candidate_sigs, buffer_sigs, valid_mask, k=k)
        composite = compute_fitness_batch(regrets, novelties, alpha=alpha, beta=beta)
        fitness_for_evosax = -composite
        dummy_key = jax.random.PRNGKey(0)
        new_es_state, _ = self._es.tell(
            dummy_key, candidates, fitness_for_evosax,
            state["es_state"], state["es_params"]
        )
        return {**state, "es_state": new_es_state}, float(jnp.mean(novelties))
```

### Extracting Buffer State for Novelty Computation

```python
# Source: inferred from level_sampler.py Sampler TypedDict + train.py pattern

# At the tell() call site in train.py (NEW/mutate branch):
buf_size = int(train_state.sampler["size"])
buffer_sigs = train_state.sampler["levels_extra"]["behavior_sig"]  # (capacity, 169)
valid_mask = jnp.arange(level_sampler.capacity) < buf_size          # (capacity,) bool

# candidate_sigs comes from extract_behavior_signature() on the second rollout:
_, _, _, agent_positions = rollout_agent_on_levels_with_positions(
    rng_rollout, eval_env, env_params,
    train_state.params, network, levels_pad,
    num_steps=config["eval_rollout_steps"],
)
candidate_sigs = extract_behavior_signature(agent_positions, config["eval_rollout_steps"])
# -> (num_envs, 169) float32
```

### WandB Logging in Training Loop

```python
# Source: examples/maze_plr.py lines 476-508 — adapted for ES metrics

# At top of train():
import wandb
run = wandb.init(
    config=config,
    project=config.get("wandb_project", "es-accel"),
    group=config["run_name"],
    tags=[config.get("es_strategy", "cma_es").upper()],
)
wandb.define_metric("update")
for key in ["regret", "novelty_score", "replay_buffer_size", "buffer_occupied",
            "valid_fraction", "mean_buffer_score"]:
    wandb.define_metric(key, step_metric="update")

# Inside training loop, every wandb_log_freq updates:
if (update + 1) % config.get("wandb_log_freq", 10) == 0:
    wandb.log({
        "update":              update,
        "regret":              mean_regret,
        "novelty_score":       mean_novelty if mean_novelty is not None else 0.0,
        "replay_buffer_size":  buf_size,
        "buffer_occupied":     buf_size / level_sampler.capacity,
        "valid_fraction":      valid_frac,
        "mean_buffer_score":   mean_buf_score,
        "mode":                mode,
    }, step=update)
```

### Solvability Gate Reuse (Confirmed Existing)

```python
# Source: es/env_bridge.py lines 131-162 — flood_fill_solvable already exists

# Already used inside eval_fn (ued_interface.py line 205):
solvable = jax.vmap(flood_fill_solvable)(
    levels.wall_map, levels.agent_pos, levels.goal_pos
)
# Phase 3 does NOT need to re-implement this.
# eval_fn already applies the solvability gate — the 'valid' mask it returns
# is solvable & complex_enough. No additional BFS needed in warm-up.
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Separate novelty archive (NS-ES canonical) | PLR replay buffer as novelty archive | Phase 2 decision | Eliminates data duplication; behavior_sig stored per level |
| Manual CSV logging | WandB run tracking | Phase 3 addition | Enables side-by-side regret curve comparison |
| Random level generation (vanilla ACCEL) | ES-generated curriculum with composite fitness | Phase 3 delivery | First end-to-end NS-ES curriculum learning run |
| No behavior_sig at buffer insertion | behavior_sig required (assert gate) | Phase 2 forward-contract | Enforces invariant; Phase 3 closes the gap |

**Still in use from prior phases:**
- `eval_fn` from `ued_interface.build_eval_fn()` — JIT-compiled, no changes
- `LevelSampler.insert_batch()` with `level_extra` dict — no changes
- `CMAESStrategy.ask()` logic — copied verbatim into `NSESStrategy.ask()`

---

## Open Questions

1. **Two-bucket sampling: minimum_fill_ratio gate interaction with warm-up**
   - What we know: `minimum_fill_ratio=0.5` means replay only triggers when buffer is 50% full (2000/4000). Warm-up inserts 256 entries = 6.4% fill. Replay won't be possible until 1744 more entries accumulate through training.
   - What's unclear: Is this the intended behavior? The context says warm-up pre-populates "before training" but the 50% gate means the agent trains on NEW levels for a long time before any replay occurs.
   - Recommendation: Confirm `minimum_fill_ratio` remains at 0.5 or lower it for ES-mode (e.g., 0.1) so warm-up's 256 entries enable faster replay. Add a config key `es_minimum_fill_ratio` separate from the PLR default. LOW priority — not a blocking issue.

2. **Whether to extend eval_fn or use separate rollout for behavior_sig**
   - What we know: Option A (separate rollout) doubles the rollout computation at insert sites. Option B (extend eval_fn) modifies `ued_interface.py` which has broader callers.
   - What's unclear: Runtime overhead of Option A at 128 steps x 32 envs per insert.
   - Recommendation: Option A (separate rollout). Clean separation; eval_fn interface stays stable. `eval_rollout_steps=128` is already the config value. Overhead is acceptable for Phase 3 MVP.

3. **NSESStrategy.tell() Protocol conformance**
   - What we know: `ESStrategy` Protocol defines `tell(state, candidates, fitness)` — 3 args. NSESStrategy.tell() requires more args (buffer_sigs, valid_mask, candidate_sigs).
   - What's unclear: Whether to keep Protocol conformance by restructuring.
   - Recommendation: NSESStrategy does NOT strictly conform to the `ESStrategy` Protocol's tell() signature. This is acceptable — the Protocol is used for type-checking CMAESStrategy. The planner should document this deviation. Alternatively, pack buffer state into the `state` dict at tell() time — but that's over-engineering for Phase 3 MVP.

---

## Validation Architecture

Config inspection: `workflow.nyquist_validation` key not found in `.planning/config.json` (key absent). The config has only `workflow.research`, `workflow.plan_check`, and `workflow.verifier`. Treating as absent = skip formal Validation Architecture section.

However, existing test infrastructure is relevant:

**Test file:** `/cs/student/project_msc/2025/csml/gmaralla/superintelligence/tests/test_es_components.py`
**Run command:** `/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python tests/test_es_components.py`
**Pattern:** Plain Python assertions (no pytest), run from project root.

Phase 3 should add a `tests/test_phase3_ns_es.py` following the same pattern:

| Req | Behavior | Test Type | Test Name |
|-----|----------|-----------|-----------|
| ALGO-01 | NSESStrategy ask/tell cycle | unit | `test_nses_strategy_ask_tell` |
| ALGO-01 | NSESStrategy composite fitness applied in tell() | unit | `test_nses_tell_uses_composite_fitness` |
| INTEG-02 | Warm-up populates buffer with behavior_sig | integration | `test_archive_warmup_populates_buffer` |
| INTEG-01 | Two-bucket sampling guard (empty buffer -> frontier) | unit | `test_two_bucket_empty_buffer_guard` |
| INTEG-03 | End-to-end: train.py runs 3 updates without crash | smoke | `test_end_to_end_3_updates` |

---

## Sources

### Primary (HIGH confidence)
- `/cs/student/project_msc/2025/csml/gmaralla/superintelligence/accel_training/es_components/` — Phase 2 deliverables read directly
- `/cs/student/project_msc/2025/csml/gmaralla/superintelligence/accel_training/train.py` — training loop structure read directly
- `/cs/student/project_msc/2025/csml/gmaralla/superintelligence/es/env_bridge.py` — flood_fill_solvable, bfs_path_length confirmed present
- `/cs/student/project_msc/2025/csml/gmaralla/superintelligence/es/regret_fitness.py` — rollout_agent_on_levels_with_positions, extract_behavior_signature confirmed present
- `/cs/student/project_msc/2025/csml/gmaralla/superintelligence/src/jaxued/level_sampler.py` — LevelSampler API read directly
- `pip list` output: evosax 0.2.0, jax 0.5.3, wandb 0.25.0 confirmed installed
- `/cs/student/project_msc/2025/csml/gmaralla/superintelligence/examples/maze_plr.py` — WandB integration pattern confirmed

### Secondary (MEDIUM confidence)
- `.planning/DECISIONS.md` — DECISION-01 behavior signature design rationale
- `.planning/STATE.md` — Phase 2 accumulated decisions (alpha/beta as Python floats, k=5 static)

### Tertiary (LOW confidence)
- None — all critical claims verified from source code

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — packages confirmed installed via pip list in jax_env
- Architecture: HIGH — all reused modules read from source; integration points confirmed
- Pitfalls: HIGH — most derived from direct inspection of train.py assert at line 353 and level_sampler.py API

**Research date:** 2026-03-02
**Valid until:** 2026-04-02 (stable codebase, 30 days)
