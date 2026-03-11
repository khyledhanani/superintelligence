# Architecture Research

**Domain:** CNN-VAE decoder adapter integration into CMA-ES maze generation pipeline
**Researched:** 2026-03-11
**Confidence:** HIGH (derived directly from codebase inspection)

## Standard Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    maze_plr.py  (Training Orchestrator)             │
│                                                                     │
│  on_new_levels()                                                    │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────┐    │
│  │ CMAESManager │───>│  decode_fn (VAE) │───>│ Level objects  │   │
│  │   .ask()     │    │  z→grid logits   │    │ (batched)      │   │
│  └──────────────┘    └──────────────────┘    └───────┬────────┘   │
│         │                                            │             │
│  .tell(-scores)                              env.reset_to_level()  │
│         │                                            │             │
│  ┌──────┴──────┐                           ┌────────┴──────────┐  │
│  │  es_state   │                           │  Trajectory       │  │
│  │ (in pytree) │                           │  Rollout + PPO    │  │
│  └─────────────┘                           └───────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    vae/  (Decoder Layer)                            │
│                                                                     │
│  ┌─────────────────┐        ┌──────────────────────────────────┐   │
│  │   cnn_vae_model │        │   NEW: cnn_vae_level_utils.py    │   │
│  │   .py           │        │                                  │   │
│  │  CnnLstmDecoder │───────>│  decode_latent_to_levels_grid()  │   │
│  │  z (64,)        │        │  grid_to_level() (per-sample)    │   │
│  │  → wall_logits  │        │  wall masking (argmax w/ mask)   │   │
│  │  → goal_logits  │        │                                  │   │
│  │  → agent_logits │        └──────────────────────────────────┘   │
│  │  each (B,13,13) │                                               │
│  └─────────────────┘                                               │
│                                                                     │
│  ┌─────────────────┐   (kept as fallback, --use_clutr_vae flag)   │
│  │   vae_model.py  │                                               │
│  │   CluttrVAE     │───>  vae_level_utils.decode_latent_to_levels  │
│  │   z → (52,170)  │      tokens_to_level()                       │
│  └─────────────────┘                                               │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    Checkpoint Layer                                 │
│                                                                     │
│  GCS: gs://cnn-vae-maze-checkpoints/run10/ (step 200000)           │
│    ↓  gsutil cp                                                     │
│  vae/checkpoints/cnn_vae/   (local Orbax format)                   │
│    ↓  orbax.checkpoint.PyTreeCheckpointer                           │
│  params["decoder"]   → CnnLstmDecoder params pytree                │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| `CMAESManager` (`vae/cmaes_manager.py`) | Maintains CMA-ES state; provides `ask()` (sample latent batch) and `tell()` (update from fitness scores) | `maze_plr.py` training loop; receives `z_population (popsize, 64)`, returns updated `es_state` |
| `CnnLstmDecoder` (`vae/cnn_vae_model.py`) | Flax module that decodes `z (B, 64)` → `(wall_logits, goal_logits, agent_logits)` each `(B, 13, 13)` | Called via frozen `decode_fn` closure in training loop; no mutable state |
| `decode_latent_to_levels_grid()` (NEW — `vae/cnn_vae_level_utils.py`) | Convert CNN-VAE output grids to batched `Level` objects; wall masking for goal/agent placement | Receives `(wall_logits, goal_logits, agent_logits)` each `(B, 13, 13)`; returns batched `Level` |
| `grid_to_level()` (NEW — single-sample inner fn) | Convert one set of `(13,13)` logit grids → one `Level`; vmapped by caller | Used inside `decode_latent_to_levels_grid` via `jax.vmap` |
| Checkpoint loader (NEW — startup code in `maze_plr.py`) | Download Orbax checkpoint from GCS to local path; extract `params["decoder"]` | Runs once at startup before training; produces frozen `decoder_params` dict |
| `CluttrVAE` + `decode_latent_to_levels` (`vae/vae_model.py`, `vae/vae_level_utils.py`) | Existing token-based decoder path; kept as `--use_clutr_vae` fallback | No changes required |
| `TrainState` / training loop (`maze_plr.py`) | Holds `es_state` in JAX pytree; calls decode path during `on_new_levels()`; passes `Level` objects to `env.reset_to_level()` | All components above; outputs levels to Maze environment |
| `Level` (`src/jaxued/environments/maze/level.py`) | Flax struct dataclass representing one maze: `wall_map (13,13)`, `goal_pos (2,)`, `agent_pos (2,)`, `agent_dir`, `width`, `height` | Consumed by Maze environment; vmapped in batches across `num_train_envs` |

## Data Flow: z to Level (CNN-VAE Path)

### Full Flow (per training step, `on_new_levels`)

```
CMAESManager.ask(rng, es_state)
    → z_population: (popsize=num_train_envs, latent_dim=64)
    → es_state (updated)

          ↓

cnn_decode_fn(z_population)        [closure capturing decoder_params]
  = CnnLstmDecoder.apply(
        {"params": decoder_params},
        z_population
    )
    → wall_logits:  (B, 13, 13)   [raw sigmoid logits, unbounded]
    → goal_logits:  (B, 13, 13)   [raw logits, position dist]
    → agent_logits: (B, 13, 13)   [raw logits, position dist]

          ↓

decode_latent_to_levels_grid(wall_logits, goal_logits, agent_logits, rng)
  [vmapped over batch dimension B]
  Per-sample: grid_to_level(wall_logits_i, goal_logits_i, agent_logits_i, rng_i)

          ↓  [inside grid_to_level — single sample]

  wall_map = (wall_logits > 0)                  # sigmoid threshold → bool (13,13)

  # Wall-masked position selection:
  free_mask = ~wall_map.reshape(-1)             # (169,) — cells without walls
  masked_goal = goal_logits.reshape(-1) + jnp.where(wall_map.reshape(-1), -1e9, 0)
  goal_flat = jnp.argmax(masked_goal)           # (,) 0-based flat index
  goal_pos = (goal_flat % 13, goal_flat // 13)  # (x=col, y=row) → uint32

  # Mask out goal cell for agent placement:
  goal_cell_mask = (jnp.arange(169) == goal_flat)
  masked_agent = agent_logits.reshape(-1) + jnp.where(wall_map.reshape(-1) | goal_cell_mask, -1e9, 0)
  agent_flat = jnp.argmax(masked_agent)
  agent_pos = (agent_flat % 13, agent_flat // 13)  # uint32

  # Clear wall at goal/agent positions (defensive):
  wall_map = wall_map.at[goal_pos[1], goal_pos[0]].set(False)
  wall_map = wall_map.at[agent_pos[1], agent_pos[0]].set(False)

  # Randomize agent direction:
  agent_dir = jax.random.randint(rng, (), 0, 4).astype(jnp.uint8)

  → Level(wall_map, goal_pos, agent_pos, agent_dir, width=13, height=13)

          ↓

new_levels: batched Level (each field shape (B, ...))

          ↓

env.reset_to_level(rngs, new_levels, env_params)   [vmapped]
    → init_obs, init_env_state

          ↓

sample_trajectories_rnn(...)   → trajectory data

          ↓

compute_level_scores(...)   → scores (B,)

          ↓

CMAESManager.tell(rng, z_population, -scores, es_state)
    → es_state (updated for next step)
```

### Coordinate Convention (Critical)

The `Level` struct uses `(x, y) = (col, row)` for positions but `wall_map[row, col]` for indexing. This is the same convention used by `tokens_to_level()` in the existing CluttrVAE path. The conversion from flat index:

```
flat_idx → col = flat_idx % 13, row = flat_idx // 13
pos = (col, row)  # Level expects (x=col, y=row)
wall_map[row, col] = wall_map[pos[1], pos[0]]
```

This matches exactly how `tokens_to_level()` converts 1-based indices:
```python
agent_pos = jnp.array([agent_0 % GRID_SIZE, agent_0 // GRID_SIZE], dtype=jnp.uint32)
# agent_pos[0] = col = x, agent_pos[1] = row = y
```

## Architectural Patterns

### Pattern 1: Frozen Decode Function Closure

**What:** The VAE decoder is wrapped in a Python closure at startup, capturing frozen params. The closure presents a pure function `z (latent_dim,) -> logits` to the training loop. The training loop never touches decoder params directly.

**When to use:** Any time you need to inject a learned model into a JIT-compiled functional training loop without carrying model params in `TrainState`.

**Why it works:** JAX traces closures that capture static Python objects (frozen dicts of params). The closure `cnn_decode_fn = lambda z: CnnLstmDecoder(...).apply({"params": decoder_params}, z)` becomes a compile-time constant inside `jax.jit`.

**Example (existing CluttrVAE pattern, replicate for CNN-VAE):**
```python
# Startup: load params once
decoder_params = load_cnn_vae_checkpoint(config["cnn_vae_checkpoint_path"])

# Define pure closure — captured params are compile-time constants in JIT
def cnn_decode_fn(z_batch):
    # z_batch: (B, 64)
    decoder = CnnLstmDecoder(latent_dim=64)
    return decoder.apply({"params": decoder_params["decoder"]}, z_batch)

# Inside on_new_levels (JIT-traced):
z_population, es_state = cmaes_mgr.ask(rng_ask, es_state)
new_levels = decode_latent_to_levels_grid(cnn_decode_fn, z_population, rng_decode)
```

### Pattern 2: vmap-Over-Batch for Level Conversion

**What:** The per-sample grid-to-Level conversion function is written for a single sample `(13,13) → Level`, then lifted to batch via `jax.vmap`. This keeps the scalar logic simple and JIT-compatible.

**Why required:** The training loop batches `num_train_envs` environments. All level generation must produce batched Level structs with matching leading dimension. `jax.vmap` handles this without Python loops (which break JIT).

**Example:**
```python
def grid_to_level(wall_logits, goal_logits, agent_logits, rng):
    # wall_logits: (13, 13)  — single sample
    wall_map = (wall_logits > 0)
    # ... argmax with wall masking (see data flow above) ...
    return Level(wall_map=wall_map, goal_pos=goal_pos, ...)

def decode_latent_to_levels_grid(decode_fn, z_batch, rng):
    # z_batch: (B, 64)
    wall_logits, goal_logits, agent_logits = decode_fn(z_batch)
    rngs = jax.random.split(rng, z_batch.shape[0])
    return jax.vmap(grid_to_level)(wall_logits, goal_logits, agent_logits, rngs)
```

### Pattern 3: Flag-Based Decoder Path Dispatch

**What:** The training loop selects the decode path at Python-level (config flag), not inside JIT. The `if config["use_cmaes"]` / `if config.get("use_clutr_vae")` branches are compile-time specializations of the training function.

**When to use:** When two paths have incompatible JAX pytree structures (CluttrVAE produces token logits; CNN-VAE produces grid logits) that cannot be unified under `jax.lax.cond`.

**Implementation:** The `on_new_levels` function captures `decode_fn` from the outer scope. Switching VAEs means replacing the closure; the training loop structure stays identical.

```python
# Startup dispatch — Python level, before JIT
if config["use_cmaes"] and not config.get("use_clutr_vae"):
    decode_fn = cnn_decode_fn        # z_batch → (wall, goal, agent) logits
    level_converter = decode_latent_to_levels_grid
else:
    decode_fn = vae_decode_fn        # z_batch → token logits (seq, vocab)
    level_converter = decode_latent_to_levels

# Inside on_new_levels (JIT-traced):
new_levels = level_converter(decode_fn, z_population, rng_decode)
```

### Pattern 4: Orbax Checkpoint Loading for Decoder-Only

**What:** The CNN-VAE checkpoint contains the full model params tree (`encoder`, `mean_layer`, `logvar_layer`, `decoder`). Only `params["decoder"]` is needed at inference time. Extract it once at startup.

**Why:** Avoids carrying unused encoder params in memory during training. Also future-proofs against model structure changes.

**Example:**
```python
import orbax.checkpoint as ocp

def load_cnn_vae_decoder_params(checkpoint_dir):
    """Load CNN-VAE checkpoint, return decoder params only."""
    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(checkpoint_dir)
    # Checkpoint tree: {"params": {"encoder": ..., "mean_layer": ...,
    #                               "logvar_layer": ..., "decoder": ...}}
    return restored["params"]["decoder"]
```

## Anti-Patterns

### Anti-Pattern 1: Python Loop in Batch Decoding

**What people do:** Write `levels = [grid_to_level(w[i], g[i], a[i]) for i in range(B)]` then stack.

**Why it's wrong:** Python loops don't compile under `jax.jit`. The loop runs at trace time, unrolling every iteration into the computation graph. For `B=32` this inflates compile time and graph size. The function exits JIT cleanly but breaks `jax.lax.scan` compatibility.

**Do this instead:** Write the per-sample function, lift with `jax.vmap`. The `jax.vmap(grid_to_level)(wall_logits, goal_logits, agent_logits, rngs)` call compiles to a single vectorized kernel.

### Anti-Pattern 2: Placing Agent/Goal Without Wall Masking

**What people do:** `goal_pos = jnp.argmax(goal_logits)` without checking whether that cell is a wall.

**Why it's wrong:** `Level.is_well_formatted()` checks `~wall_map[agent_pos[1], agent_pos[0]]` and `~wall_map[goal_pos[1], goal_pos[0]]`. Invalid levels pass through `decode_latent_to_levels_grid` and enter the replay buffer, corrupting training with unreachable goals or unplaceable agents. WandB `cmaes/valid_structure_pct` will drop.

**Do this instead:** Apply a `-1e9` additive mask over wall cells to the goal/agent logits before `jnp.argmax`. This is JIT-compatible and guaranteed valid (the decoder has 96% prior solvability, so free cells almost always exist).

### Anti-Pattern 3: Loading Full VAE Params into TrainState

**What people do:** Add `vae_params: chex.ArrayTree` field to `TrainState`, include in Orbax checkpoint.

**Why it's wrong:** (a) Doubles checkpoint size with static params that never change. (b) Orbax then serializes/deserializes the full VAE tree on every checkpoint. (c) `TrainState` becomes non-comparable with checkpoints from the ACCEL baseline run (different pytree shape).

**Do this instead:** Load VAE params once at startup, close over them in a Python lambda. They live in Python heap memory, not in the JAX pytree hierarchy.

### Anti-Pattern 4: Attempting to JIT the Orbax Checkpoint Load

**What people do:** Put `orbax.checkpoint.restore()` inside `@jax.jit` or `jax.lax.scan`.

**Why it's wrong:** Orbax restoration is a Python/host operation. It cannot be traced by JAX. The JAX tracer will raise at compile time.

**Do this instead:** Load checkpoint once in Python before defining any jitted functions. Pass params to jitted code via Python closure capture.

## Integration Points

### New File: `vae/cnn_vae_level_utils.py`

This is the primary new component. It mirrors `vae/vae_level_utils.py` but operates on grid logits instead of token sequences.

```
vae/vae_level_utils.py          (existing — token path)
vae/cnn_vae_level_utils.py      (new — grid path)
```

Public API surface:

```python
def decode_latent_to_levels_grid(
    decode_fn,     # z_batch (B, 64) -> (wall_logits, goal_logits, agent_logits) each (B, 13, 13)
    z_batch,       # (B, 64)
    rng,           # PRNGKey for agent_dir randomization
) -> Level:        # batched Level, each field shape (B, ...)
```

```python
def grid_to_level(
    wall_logits,   # (13, 13) — single sample
    goal_logits,   # (13, 13)
    agent_logits,  # (13, 13)
    rng,           # PRNGKey
) -> Level:        # single Level
```

### Existing File Modifications: `examples/maze_plr.py`

Changes are additive and localized to the VAE setup block (lines ~503–546):

1. Add import: `from cnn_vae_model import CnnLstmDecoder`
2. Add import: `from cnn_vae_level_utils import decode_latent_to_levels_grid`
3. Add `--use_clutr_vae` argparse flag (inverse of CNN-VAE default)
4. In `_needs_vae` block: branch on flag to instantiate CluttrVAE vs CNN-VAE
5. Replace `decode_latent_to_levels(vae_decode_fn, ...)` calls with `decode_latent_to_levels_grid(cnn_decode_fn, ...)` in the default path
6. Keep CluttrVAE path fully intact under `use_clutr_vae` flag

Changes to `on_new_levels()` are a one-line replacement:
```python
# Before (CluttrVAE path):
new_levels = decode_latent_to_levels(vae_decode_fn, z_population, rng_decode)

# After (CNN-VAE default path):
new_levels = level_converter(decode_fn, z_population, rng_decode)
# where level_converter is set at startup based on --use_clutr_vae flag
```

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `CMAESManager` → decode fn | `z_population (B, 64)` numpy/jax array | `CMAESManager.ask()` returns JAX array directly from evosax; no conversion needed |
| `decode_fn` → `decode_latent_to_levels_grid` | `(wall_logits, goal_logits, agent_logits)` each `(B, 13, 13)` | Decoder is called inside JIT; output arrays stay on device |
| `grid_to_level` → `Level` | Flax struct construction with correct dtypes | `wall_map: bool`, `goal_pos/agent_pos: uint32`, `agent_dir: uint8` — must match exactly |
| `Level` → `env.reset_to_level` | Batched Level pytree, `(B, ...)` fields | Maze env validates via `is_well_formatted()` — wall masking prevents failures here |
| Orbax loader → `decoder_params` | Python dict pytree at startup | `restored["params"]["decoder"]` — key path determined by `CnnLstmVAE` module naming contract |

## Build Order (Phase Dependencies)

The dependency graph drives build ordering:

```
Phase 1: Checkpoint acquisition
    gsutil cp gs://cnn-vae-maze-checkpoints/run10/ vae/checkpoints/cnn_vae/
    Verify: orbax restore works, params["decoder"] present
    No code changes required.

Phase 2: grid_to_level() — single-sample converter
    File: vae/cnn_vae_level_utils.py
    Depends on: Level dataclass (exists), coordinate convention (documented above)
    Test: decode z=0 -> Level, check is_well_formatted()

Phase 3: decode_latent_to_levels_grid() — batch wrapper
    File: vae/cnn_vae_level_utils.py (same file, adds vmapped batch fn)
    Depends on: grid_to_level() (Phase 2), CnnLstmDecoder (exists)
    Test: decode z_batch (32, 64) -> batched Level, check valid_structure_pct

Phase 4: maze_plr.py integration
    File: examples/maze_plr.py (additive changes only)
    Depends on: Phases 1-3 complete, cnn_vae_level_utils imported
    Test: smoke test (1000-step CMA-ES run), cmaes/valid_structure_pct > 90%

Phase 5: 20k comparison run
    File: launch scripts (adapt existing)
    Depends on: Phase 4 validated
    Output: WandB logs for CNN-VAE CMA-ES vs ACCEL baseline
```

**Critical path:** Phase 2 (grid_to_level) blocks everything. It contains the coordinate convention and wall masking logic that determines whether generated levels are valid. Get this right before wiring it into the training loop.

## Scaling Considerations

This integration runs on a fixed-size GPU node (sideswipe/prowl). Scaling is not a concern. The relevant resource dimensions are:

| Parameter | Typical Value | Constraint |
|-----------|--------------|------------|
| `num_train_envs` (= popsize) | 32 | Must equal CMA-ES popsize — these are linked |
| `latent_dim` | 64 | Fixed by CNN-VAE checkpoint; CMAESManager is latent-dim agnostic |
| Grid size | 13x13 | Fixed by Maze environment and CNN-VAE architecture |
| Checkpoint size | ~50MB (Orbax) | Loaded once at startup; not in training hot loop |

## Sources

- Direct codebase inspection: `vae/cnn_vae_model.py`, `vae/vae_level_utils.py`, `examples/maze_plr.py` (lines 503-1030), `vae/cmaes_manager.py`, `src/jaxued/environments/maze/level.py`
- Existing coordinate convention: `vae/vae_level_utils.py` `tokens_to_level()` function (lines 46-86)
- Checkpoint param tree contract: `vae/cnn_vae_model.py` module docstring (lines 1-13)
- CNN-VAE decoder output spec: `PROJECT.md` "CNN-VAE architecture" section
- Wall masking requirement: `PROJECT.md` "Active" requirements, `Level.is_well_formatted()` source

---
*Architecture research for: CNN-VAE decoder adapter integration with CMA-ES maze pipeline*
*Researched: 2026-03-11*
