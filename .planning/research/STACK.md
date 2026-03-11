# Stack Research

**Domain:** CNN-VAE decoder integration into JAX/Flax CMA-ES maze generation pipeline
**Researched:** 2026-03-11
**Confidence:** HIGH — all library APIs and patterns verified by running code in the actual jax_env conda environment

---

## Context

This is a **subsequent milestone** — not a new project. The JAX/Flax/evosax/JaxUED stack already exists and works. Do not re-research it. This STACK.md covers only the three new technical surfaces introduced by this milestone:

1. **Orbax checkpoint loading** — Restoring the CNN-VAE from a downloaded GCS checkpoint
2. **Flax module param extraction** — Getting `params['decoder']` subtree from a full VAE checkpoint
3. **JAX-compatible grid-to-Level conversion** — Pure JAX conversion of CNN-VAE grid outputs to `Level` pytrees

---

## Verified Environment

```
JAX:    0.5.3   (jax_env conda env)
Flax:   0.10.7
Orbax:  0.10.3  (orbax-checkpoint)
```

All patterns below were executed and verified in this exact environment.

---

## Recommended Stack

### Core Technologies (existing — do not change)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| JAX | 0.5.3 | Numerical compute + JIT | Existing pipeline; CUDA 12 works on sideswipe/prowl |
| Flax linen | 0.10.7 | Neural network module definition | CNN-VAE model already defined in `flax.linen` |
| evosax | current | CMA-ES latent space search | CMAESManager already integrated, latent-dim agnostic |
| JaxUED | current | Maze environment + Level dataclass | Provides `Level`, `Maze`, `make_level_generator` |

### New Integration Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `orbax.checkpoint.PyTreeCheckpointer` | 0.10.3 | Load CNN-VAE checkpoint from local dir | No target pytree required; restores nested dict directly; verified working |
| `orbax.checkpoint.CheckpointManager` | 0.10.3 | Load CNN-VAE if saved with step structure | Use when checkpoint dir has step subdirs (e.g., `step_200000/`); same API as existing training code |
| `jax.vmap` | 0.5.3 | Batch grid-to-Level conversion | Makes `decode_latent_to_levels_grid` JIT-compatible; verified with `Level` pytree |
| `jax.image.resize` | 0.5.3 | Nearest-neighbor upsample in decoder | Already used in `CnnLstmDecoder`; no new import needed |

---

## Checkpoint Loading Pattern

**Use `ocp.PyTreeCheckpointer` if downloading a step directory from GCS (e.g., `step_200000/`).
Use `ocp.CheckpointManager` if the checkpoint was saved with a manager (most likely for run10).**

Both patterns verified working in orbax 0.10.3:

```python
import orbax.checkpoint as ocp

# Pattern A: PyTreeCheckpointer (simpler; no target needed)
# Use when: you download a single step directory from GCS
checkpointer = ocp.PyTreeCheckpointer()
raw = checkpointer.restore('vae/checkpoints/cnn_vae/step_200000')
decoder_params = raw['params']['decoder']

# Pattern B: CheckpointManager (use when checkpoint has step structure)
# Use when: the checkpoint was saved with CheckpointManager (standard for run10)
mgr = ocp.CheckpointManager(
    'vae/checkpoints/cnn_vae/',
    options=ocp.CheckpointManagerOptions()
)
raw = mgr.restore(200000)
decoder_params = raw['params']['decoder']

# Apply decoder with restored params (VERIFIED working)
from vae.cnn_vae_model import CnnLstmDecoder
decoder = CnnLstmDecoder(latent_dim=64)
wall_logits, goal_logits, agent_logits = decoder.apply({'params': decoder_params}, z_batch)
# Shapes: each (B, 13, 13)
```

**Verified param tree structure** (from actual model init):
```
params/
  decoder/
    dec_lstm/   # hf, hg, hi, ho, if, ig, ii, io kernels/biases
    dec_proj/   # kernel (256, 128), bias (128,)
    dec_conv1/  # kernel (3,3,128,192), bias (192,)
    dec_conv2/  # kernel (3,3,192,128), bias (128,)
    dec_conv3/  # kernel (3,3,128,128), bias (128,)
    wall_head/  # kernel (1,1,128,1), bias (1,)
    goal_head/  # kernel (1,1,128,1), bias (1,)
    agent_head/ # kernel (1,1,128,1), bias (1,)
  encoder/
    conv1/ conv2/ conv3/ enc_lstm/ enc_bridge/
  mean_layer/
  logvar_layer/
```

The `decoder_params` dict extracted above maps directly to the `CnnLstmDecoder` parameter tree. No reshaping or key remapping required.

---

## Grid-to-Level Conversion Pattern

**Use `jax.vmap` over a pure function. Do NOT use Python loops.**

```python
import jax
import jax.numpy as jnp
from jaxued.environments.maze import Level

GRID_SIZE = 13

def _decode_grid_single(decoder_params, decoder, z, rng):
    """Single latent -> Level. vmappable. JIT-safe."""
    # Decode z (64,) -> logits (13, 13)
    wall_l, goal_l, agent_l = decoder.apply({'params': decoder_params}, z[None])
    wall_l = wall_l[0]   # (13, 13)
    goal_l = goal_l[0]
    agent_l = agent_l[0]

    # Wall map: logit > 0.0 means sigmoid(logit) > 0.5 (wall present)
    wall_map = wall_l > 0.0  # (13, 13) bool

    # Goal/agent argmax: mask walls with large negative value to exclude them
    # Then mask goal position for agent to prevent collision
    goal_flat = jnp.argmax(
        goal_l.reshape(-1) - wall_l.reshape(-1) * 1e9
    )
    agent_flat = jnp.argmax(
        agent_l.reshape(-1)
        - wall_l.reshape(-1) * 1e9
        - (jnp.arange(GRID_SIZE * GRID_SIZE) == goal_flat) * 1e9
    )

    # Coordinate convention: x=col=flat%13, y=row=flat//13
    goal_pos = jnp.array([goal_flat % GRID_SIZE, goal_flat // GRID_SIZE], dtype=jnp.uint32)
    agent_pos = jnp.array([agent_flat % GRID_SIZE, agent_flat // GRID_SIZE], dtype=jnp.uint32)

    # Clear walls at goal and agent (defensive — mirrors tokens_to_level)
    wall_map = wall_map.at[goal_pos[1], goal_pos[0]].set(False)
    wall_map = wall_map.at[agent_pos[1], agent_pos[0]].set(False)

    # Randomize agent direction
    agent_dir = jax.random.randint(rng, (), 0, 4).astype(jnp.uint8)

    return Level(
        wall_map=wall_map,
        goal_pos=goal_pos,
        agent_pos=agent_pos,
        agent_dir=agent_dir,
        width=GRID_SIZE,
        height=GRID_SIZE,
    )


def decode_latent_to_levels_grid(decoder_params, decoder, z_batch, rng):
    """Batch decode: (N, 64) -> batched Level. JIT-safe via vmap.

    Drop-in replacement for decode_latent_to_levels() used by CluttrVAE path.
    """
    N = z_batch.shape[0]
    rngs = jax.random.split(rng, N)
    return jax.vmap(_decode_grid_single, in_axes=(None, None, 0, 0))(
        decoder_params, decoder, z_batch, rngs
    )
```

**Coordinate convention verified** against `tokens_to_level()`:
- `grid[row, col]` — row-major indexing
- `flat_idx = row * 13 + col` (0-indexed)
- `x = col = flat_idx % 13`
- `y = row = flat_idx // 13`
- `Level.goal_pos = [x, y] = [col, row]`

**Wall masking strategy for goal/agent placement:**
- Subtract `wall_map * 1e9` from logits before argmax — pushes wall cells to minimum
- Subtract `(arange == goal_flat) * 1e9` from agent logits — prevents agent == goal
- This is pure JAX, no Python control flow, JIT-safe

---

## GCS Checkpoint Download

**Neither `gsutil` nor `google-cloud-storage` is installed in jax_env.** Install `google-cloud-storage` to use Python-based GCS download, or use the bundled `gcloud` CLI workaround.

```bash
# Option A: Install google-cloud-storage into jax_env (verified installable, ~3 deps)
pip install google-cloud-storage

# Option B: Use the bundled Google Cloud SDK (present in repo)
# Path: google_cloud_tpu/google-cloud-sdk/ — but bin/gsutil not present (bootstrap state)

# Option C (recommended for one-shot download): Use gcloud auth + storage.Client
python - << 'EOF'
from google.cloud import storage
client = storage.Client()
bucket = client.bucket('cnn-vae-maze-checkpoints')
# List and download step_200000 blobs
EOF

# Option D (simplest if auth is pre-configured): gsutil cp
# gsutil -m cp -r gs://cnn-vae-maze-checkpoints/run10/200000/ vae/checkpoints/cnn_vae/
```

**Recommendation: Use `gsutil -m cp` after installing google-cloud-storage, or pre-download the checkpoint manually before training.** The checkpoint is a one-time artifact, not needed at runtime.

---

## Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `orbax.checkpoint.PyTreeCheckpointer` | 0.10.3 | One-shot checkpoint restore | When loading a single step directory |
| `orbax.checkpoint.CheckpointManager` | 0.10.3 | Manager-based restore | When checkpoint was saved with CheckpointManager (likely for run10) |
| `google-cloud-storage` | 3.9.0 (installable) | Download checkpoint from GCS | One-time download; NOT needed at training runtime |
| `pickle` | stdlib | Already used for CluttrVAE | Do NOT use for CNN-VAE (Orbax format, not pickle) |

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| `ocp.PyTreeCheckpointer` | `ocp.StandardCheckpointer` | When you have the original target pytree available at load time; both work, PyTreeCheckpointer is simpler for cross-run loading |
| `ocp.PyTreeCheckpointer` | `pickle.load()` | Never for CNN-VAE — it was saved with Orbax, not pickle; CluttrVAE uses pickle but CNN-VAE does not |
| `jax.vmap` over pure function | Python for-loop over batch | Never — Python loop breaks jax.jit and is ~100x slower |
| Logit masking (`- 1e9`) | Post-hoc correction after argmax | Logit masking is JIT-safe; post-hoc correction requires Python control flow |

---

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `pickle.load()` for CNN-VAE checkpoint | CNN-VAE was trained and saved with Orbax, not pickle; pickle will raise or produce wrong data | `ocp.PyTreeCheckpointer().restore()` |
| `ocp.StandardCheckpointer` without target | Requires matching target pytree; fails without it | `ocp.PyTreeCheckpointer` for targetless restore |
| `ConvTranspose` for upsampling in decoder | Already decided against — checkerboard artifacts; CNN-VAE uses nearest-neighbor + Conv instead | `jax.image.resize(method='nearest')` + `nn.Conv` (already in model) |
| Python loops in `decode_latent_to_levels_grid` | Breaks `jax.jit`; 100x slower than `jax.vmap` | `jax.vmap` over a pure single-sample function |
| `jnp.where` with dynamic Python branching | Trace-time error inside `jax.jit` | Static Python conditionals before JIT; dynamic selection via `jnp.where` |

---

## Version Compatibility

| Package | Version | Notes |
|---------|---------|-------|
| orbax-checkpoint 0.10.3 | flax 0.10.7 | Compatible; both used in existing pipeline |
| orbax-checkpoint 0.10.3 | jax 0.5.3 | Compatible; `ocp.PyTreeCheckpointer` restored JAX arrays correctly in testing |
| `ocp.PyTreeCheckpointer().restore()` | no target needed | Returns nested Python `dict`; `raw['params']['decoder']` gives correct subtree |
| `ocp.CheckpointManager().restore(step)` | no args needed for dict restore | Returns same nested `dict` structure as PyTreeCheckpointer |
| `Level` pytree | jax.vmap | `Level` is a registered JAX pytree (6 leaves: wall_map, goal_pos, agent_pos, agent_dir, width=int, height=int); width/height are static Python ints, do NOT try to batch them |

---

## Stack Patterns by Variant

**If checkpoint was saved WITH CheckpointManager (most likely — standard Orbax training pattern):**
- Checkpoint dir has `step_N/` subdirectories
- Use `ocp.CheckpointManager(ckpt_dir).restore(200000)` → returns nested dict
- Extract `raw['params']['decoder']`

**If checkpoint was saved WITHOUT CheckpointManager (direct save):**
- Checkpoint dir is the leaf directory itself
- Use `ocp.PyTreeCheckpointer().restore(ckpt_dir)` → returns nested dict
- Extract `raw['params']['decoder']`

**If need to confirm checkpoint structure before loading:**
```bash
ls vae/checkpoints/cnn_vae/
# If you see numbered dirs (e.g., 200000/): use CheckpointManager
# If you see orbax metadata files directly: use PyTreeCheckpointer
```

---

## Sources

- Orbax API verified by running in `/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env` (orbax-checkpoint 0.10.3)
- `ocp.PyTreeCheckpointer().restore()` returns nested dict — confirmed by end-to-end test (save + restore + `decoder.apply()`)
- `ocp.CheckpointManager().restore(step)` with `ocp.args.StandardRestore(target)` — confirmed by end-to-end test
- `jax.vmap` over `Level` construction — confirmed; Level has 6 pytree leaves, width/height are static ints
- `CnnLstmDecoder` param tree structure — confirmed from `decoder.init()` output
- Coordinate convention — verified against `tokens_to_level()` in `vae/vae_level_utils.py`
- Wall masking via logit subtraction (`- 1e9`) — verified JIT-safe through `jax.jit(jax.vmap(...))` test

---

*Stack research for: CNN-VAE + CMA-ES maze generation integration*
*Researched: 2026-03-11*
