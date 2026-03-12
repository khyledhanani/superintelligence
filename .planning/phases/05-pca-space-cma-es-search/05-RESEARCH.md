# Phase 5: PCA-Space CMA-ES Search - Research

**Researched:** 2026-03-12
**Domain:** Dimensionality reduction (SVD/PCA) applied to VAE latent space for CMA-ES search
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Two-Stage Architecture (LOCKED)**
- Stage 1 (steps 0–10k): Variance-pruned latent search — encode training mazes to mu vectors, rank the 64 latent dims by per-dimension variance, keep top ~30, CMA-ES searches in this reduced subspace from step 0
- Stage 2 (steps 10k+): Buffer PCA — encode replay buffer levels to mu vectors, compute PCA, switch CMA-ES to top K PCA components
- Transition point is configurable (default 10k updates). User wants this two-stage design, NOT immediate PCA from step 0.

**Dimension Selection for Stage 1 (LOCKED)**
- Use training data to identify high-variance latent dims vs near-zero-variance (noise)
- Claude picks the cutoff (elbow, cumulative variance threshold like 85-90%, or fixed K)
- Expected: ~30 dims retained from 64

**PCA Source for Stage 2 (LOCKED)**
- PCA computed from replay buffer levels (NOT training data)
- Buffer levels encoded via CNN-VAE encoder to get mu vectors
- Ensures PCA reflects current training distribution

**Core Architecture from PRD (LOCKED)**
- New file: `vae/cnn_vae_pca_utils.py` with encode, PCA, and projection functions
- `make_pc_decode_fn` wraps base decode_fn transparently — CMAESManager unaware of PCA
- Whitened PCA projection: `z_latent = mu_mean + (z_pc * pc_stds) @ pc_axes`
- CMA-ES tell() uses z_pc (PC space), never z_latent (full 64-dim)

**Integration Points (LOCKED)**
- Flags: `--use_pca_search`, `--pca_components`, `--pca_dataset_size`, plus stage-transition flag
- CNN-VAE checkpoint already loaded in maze_plr.py (Phase 3 done)
- CMAESManager is latent_dim-agnostic — just pass K instead of 64
- `decode_latent_to_levels_grid` is unchanged

**Encoder Architecture (LOCKED from PRD)**
- `h = CnnEncoder(name='encoder')(x)` → `(B, 512)`
- `mu = jnp.tanh(nn.Dense(latent_dim, name='mean_layer')(h)) * 4.0` → `(B, 64)`, bounded [-4, 4]
- `params['mean_layer']` is at TOP LEVEL (NOT inside `params['encoder']`)

**Dataset for Initial Variance Analysis (LOCKED)**
- `gs://ucl-ued-project-bucket/vae/datasets/train_1M_envs.npy`
- Format: `(N, 52)` int32 CLUTR sequences — use `clutr_to_grid()` to convert
- Only need 10k-20k mazes for variance analysis / initial PCA

### Claude's Discretion
- Exact variance cutoff method for Stage 1 dimension selection (elbow, threshold, or fixed K)
- How to handle the Stage 1→2 transition in CMA-ES state (reinitialize or project?)
- Whether to log PCA diagnostics (explained variance, etc.) to WandB
- Buffer encoding batch size and frequency
- Whether Stage 2 PCA should be recomputed periodically or just once at transition

### Deferred Ideas (OUT OF SCOPE)
- Periodic PCA recomputation during training (recompute every N steps)
- Adaptive K selection (change number of components during training)
- Comparison of Stage 1 only vs Stage 1+2 vs full PCA from start
</user_constraints>

---

## Summary

Phase 5 adds two-stage dimensionality reduction to the CMA-ES level generation loop in `examples/maze_plr.py`. The core insight is that valid maze latent vectors don't fill all 64 dims equally — some have high variance (informative) and others near-zero (noise from KL regularization). Stage 1 identifies the high-variance dimensions from training data offline and restricts CMA-ES to those ~30 dims from the start, giving it a better-conditioned search space immediately. Stage 2 switches to a proper PCA basis derived from the replay buffer after 10k updates, reflecting the current task distribution rather than the static training set.

The implementation is architecturally clean: both stages are expressed as Python closures (`pruned_decode_fn` for Stage 1, `pc_decode_fn` for Stage 2) that wrap the existing `cnn_base_decode_fn`. CMAESManager and `decode_latent_to_levels_grid` are completely unaware of the transformation — they see K-dimensional vectors in both stages. The transition between stages happens at the Python-level outer loop (`for eval_step in range(...)`) where `cmaes_mgr` and `vae_decode_fn` can be reassigned without JAX tracing constraints.

All critical components are verified: the CNN-VAE checkpoint exposes `params['encoder']` and `params['mean_layer']` at the correct paths; `numpy.linalg.svd` is available and fast; the GCS dataset blob (`208MB, 1M mazes`) is accessible via legacy credentials; and `make_pc_decode_fn` closures are JIT and vmap compatible.

**Primary recommendation:** Implement `vae/cnn_vae_pca_utils.py` with `encode_mazes_to_mu`, `compute_variance_pruned_dims`, `compute_pca_axes`, and `make_pc_decode_fn`; then modify `maze_plr.py` to add Stage 1 at startup and Stage 2 at the transition hook in the outer Python loop.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | 2.2.6 (in jax_env) | SVD via `np.linalg.svd`, dataset loading with `np.load` | Already installed; SVD verified working |
| jax / jax.numpy | (jax_env default) | JIT-compiled encoder forward pass, vmap over batch | Required for CNN-VAE encode_batch |
| flax.linen | (jax_env default) | `nn.Dense` for mean_layer forward pass | Already used in cnn_vae_model.py |
| orbax.checkpoint | (jax_env default) | CNN-VAE checkpoint loading (already done in Phase 3) | Already wired in maze_plr.py |
| google-cloud-storage | (jax_env) | GCS dataset download | Already installed and auth verified |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy.linalg.svd | numpy 2.2.6 | PCA via `_, S, Vt = np.linalg.svd(mu_centered, full_matrices=False)` | Only during setup (not in training loop) |
| vae/cnn_vae_data.clutr_to_grid | local | Convert (N, 52) int32 CLUTR seqs to (N, 13, 13, 3) grids | Dataset → encoder input conversion |
| vae/vae_level_utils.level_to_tokens | local | Convert buffer Level objects to CLUTR tokens for Stage 2 encoding | Buffer Level → clutr_to_grid input |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `np.linalg.svd` | `sklearn.decomposition.PCA` | sklearn not installed in jax_env; numpy SVD verified working and sufficient |
| numpy SVD | jax.numpy.linalg.svd | numpy is faster for one-shot offline PCA; jax SVD unnecessary overhead |
| Full re-encode each run | Cache to disk | Dataset is 208MB but only 10k-20k mazes needed — memory fits, no caching needed |

**Installation:** No new packages required. All dependencies verified present in jax_env.

---

## Architecture Patterns

### Recommended Project Structure

```
vae/
├── cnn_vae_pca_utils.py    # NEW: encode_mazes_to_mu, compute_variance_pruned_dims,
│                           #      compute_pca_axes, make_variance_pruned_decode_fn,
│                           #      make_pc_decode_fn
├── cnn_vae_model.py        # UNCHANGED: CnnEncoder, CnnLstmDecoder, CnnLstmVAE
├── cnn_vae_level_utils.py  # UNCHANGED: decode_latent_to_levels_grid
├── cmaes_manager.py        # UNCHANGED: latent_dim-agnostic, pass K
└── cnn_vae_data.py         # UNCHANGED: clutr_to_grid (used by encode_mazes_to_mu)

examples/
└── maze_plr.py             # MODIFIED: new flags, Stage 1 setup block, Stage 2 hook in outer loop
```

### Pattern 1: Two-Stage Decode Function Wrapping

**What:** Both stages produce a Python closure that accepts K-dimensional input and outputs the same tuple as `cnn_base_decode_fn`. CMAESManager and `decode_latent_to_levels_grid` never see more than K dims.

**Stage 1 (variance-pruned):**
```python
# Source: verified working with jax.vmap in research shell session 2026-03-12
def make_variance_pruned_decode_fn(base_decode_fn, mu_mean, kept_dims):
    """Stage 1: search top-variance dims; pad rest with mu_mean.

    Args:
        base_decode_fn: z (64,) -> (wall_logits, goal_logits, agent_logits) each (13,13)
        mu_mean: (64,) float32 — mean of training mu vectors
        kept_dims: (K,) int32 — indices of top-variance dims (from np.argsort(-per_dim_var)[:K])

    Returns:
        pruned_decode_fn: z_reduced (K,) -> same output as base_decode_fn
    """
    def pruned_decode_fn(z_reduced):
        # Start at mu_mean, add the reduced-space offsets at the kept dimensions
        z_latent = mu_mean.at[kept_dims].set(mu_mean[kept_dims] + z_reduced)
        return base_decode_fn(z_latent)
    return pruned_decode_fn
```

**Stage 2 (PCA):**
```python
# Source: verified working with jax.vmap in research shell session 2026-03-12
def make_pc_decode_fn(base_decode_fn, mu_mean, pc_axes, pc_stds):
    """Stage 2: search in whitened PCA space.

    Args:
        base_decode_fn: z (64,) -> (wall_logits, goal_logits, agent_logits)
        mu_mean: (64,) mean of mu vectors from buffer levels
        pc_axes: (K, 64) top K principal component directions (unit vectors)
        pc_stds: (K,) std of each PC coordinate in the data (for whitening)

    Returns:
        pc_decode_fn: z_pc (K,) -> same output as base_decode_fn
    """
    def pc_decode_fn(z_pc):
        z_latent = mu_mean + (z_pc * pc_stds) @ pc_axes  # (64,)
        return base_decode_fn(z_latent)
    return pc_decode_fn
```

Both closures are `jax.jit` and `jax.vmap` compatible because `mu_mean`, `kept_dims`, `pc_axes`, `pc_stds` are JAX arrays closed over as static constants.

### Pattern 2: Stage 1 Variance Analysis (Offline, Pre-Training)

```python
# Source: vae/pca_search_prompt.md (PRD) — algorithm verified 2026-03-12
def encode_mazes_to_mu(params, clutr_seqs, latent_dim=64, batch_size=256):
    """Encode CLUTR sequences to mu vectors using CNN-VAE encoder.

    CRITICAL: mean_layer params are at params['mean_layer'] (TOP LEVEL),
    NOT inside params['encoder']. Confirmed via checkpoint inspection 2026-03-12.
    """
    grids = np.array([clutr_to_grid(seq) for seq in clutr_seqs])  # (N, 13, 13, 3)
    mean_layer = nn.Dense(latent_dim, name='mean_layer')

    @jax.jit
    def encode_batch(batch):
        h = CnnEncoder(name='encoder').apply({'params': params['encoder']}, batch)
        mu = jnp.tanh(mean_layer.apply({'params': params['mean_layer']}, h)) * 4.0
        return mu

    all_mu = []
    for i in range(0, len(grids), batch_size):
        batch = jnp.array(grids[i:i+batch_size])
        all_mu.append(np.array(encode_batch(batch)))
    return np.concatenate(all_mu, axis=0)  # (N, 64)


def compute_variance_pruned_dims(mu, K=None, cumvar_threshold=0.85):
    """Identify top-variance dimensions for Stage 1.

    Returns kept_dims (sorted indices) and per_dim_var.
    If K is None, use cumvar_threshold to determine K automatically (elbow method).
    """
    per_dim_var = mu.var(axis=0)                              # (64,)
    sorted_idx = np.argsort(-per_dim_var)                     # descending by variance
    if K is None:
        cumvar = np.cumsum(per_dim_var[sorted_idx]) / per_dim_var.sum()
        K = int(np.searchsorted(cumvar, cumvar_threshold)) + 1
        K = min(K, len(per_dim_var))
    kept_dims = np.sort(sorted_idx[:K])                       # keep sorted for stable indexing
    return kept_dims, per_dim_var, K
```

### Pattern 3: Stage Transition (Python-Level, Outside jit)

**What:** The Stage 1→2 transition happens at the Python-level outer loop in `maze_plr.py`. The transition requires:
1. Encoding current buffer levels to mu vectors
2. Computing PCA on those mu vectors
3. Creating a new `pc_decode_fn` wrapping `cnn_base_decode_fn`
4. Creating a new `CMAESManager` with `latent_dim=K_stage2`
5. Reinitializing `es_state` for the new manager

**Why reinitialize (not project):** CMA-ES state (covariance matrix `C`, evolution paths `p_c`, `p_std`) is shaped `(K, K)` for Stage 1 and `(K2, K2)` for Stage 2. These cannot be projected — they must be reinitialized. The new CMA-ES starts fresh in PCA space, which is fine because Stage 1 has already warmed up the replay buffer.

```python
# Hook in the outer for loop (line ~1374 in maze_plr.py):
for eval_step in range(config["num_updates"] // config["eval_freq"]):
    ...
    updates_so_far = (eval_step + 1) * config["eval_freq"]

    # Stage 1 -> Stage 2 transition
    if (config.get("use_pca_search") and
        config.get("pca_stage2_transition") and
        updates_so_far == config["pca_stage2_step"]):

        ts = runner_state[1]  # extract train_state
        sampler = ts.sampler
        size = int(sampler["size"])
        print(f"[PCA Stage 2] Buffer has {size} levels — computing PCA...")

        # Step 1: Get buffer levels as CLUTR tokens (jax -> numpy)
        buffer_levels = jax.tree_util.tree_map(lambda x: x[:size], sampler["levels"])
        tokens_jax = jax.vmap(level_to_tokens)(buffer_levels)   # (size, 52) jax array
        tokens_np = np.array(tokens_jax)                         # to numpy

        # Step 2: Encode to mu vectors
        mu_buf = encode_mazes_to_mu(cnn_vae_params, tokens_np, latent_dim=64)

        # Step 3: Compute PCA
        K2 = config["pca_components"]
        mu_mean2, pc_axes2, pc_stds2, evr2 = compute_pca_axes(mu_buf, K2)
        print(f"[PCA Stage 2] Top {K2} PCs explain {evr2.sum()*100:.1f}% variance")

        # Step 4: Wrap decode fn
        vae_decode_fn = make_pc_decode_fn(
            cnn_base_decode_fn,
            jnp.array(mu_mean2), jnp.array(pc_axes2), jnp.array(pc_stds2)
        )

        # Step 5: New CMAESManager + reinitialized es_state
        cmaes_mgr = CMAESManager(
            popsize=config["num_train_envs"],
            latent_dim=K2,
            sigma_init=config["cmaes_sigma_init"],
        )
        rng_cur, ts_cur = runner_state
        rng_cur, rng_es_init = jax.random.split(rng_cur)
        new_es_state = cmaes_mgr.initialize(rng_es_init)
        ts_cur = ts_cur.replace(es_state=new_es_state)
        runner_state = (rng_cur, ts_cur)
```

**CRITICAL:** `train_and_eval_step` is `@jax.jit`-compiled with `cmaes_mgr` and `vae_decode_fn` captured via closure. After reassigning these Python variables, `train_and_eval_step` must be **recompiled**. This requires either:
- Redefining `train_and_eval_step` as a non-decorated function and calling `jax.jit(train_and_eval_step)` again, OR
- Using `functools.partial` with the mutable objects passed explicitly

The simplest fix: wrap `train_and_eval_step` in a factory function so it can be re-jit-compiled at transition. See Anti-Patterns below.

### Pattern 4: Stage 2 Buffer Encoding (Level -> Grid)

Buffer levels are `Level` objects (JAX pytrees). To encode them:

```python
# 1. Level -> CLUTR tokens (jax, fast)
from vae_level_utils import level_to_tokens
tokens_jax = jax.vmap(level_to_tokens)(buffer_levels)  # (N, 52)

# 2. Tokens to numpy (device transfer)
tokens_np = np.array(tokens_jax)  # (N, 52) int32

# 3. Encode to mu (batched JAX encode)
mu_buf = encode_mazes_to_mu(cnn_vae_params, tokens_np, latent_dim=64)
# Note: encode_mazes_to_mu internally calls clutr_to_grid(seq) for each seq
```

This is the same pipeline used in `buffer_latent_analysis.py` (which uses `level_to_tokens` + `encode_tokens` for CluttrVAE). For CNN-VAE, `clutr_to_grid` replaces the CluttrVAE encoder.

### Anti-Patterns to Avoid

- **JIT closure capture with mutable Python references:** `train_and_eval_step` is `@jax.jit` — it captures `cmaes_mgr` and `vae_decode_fn` at compile time. If you reassign these variables at the Stage 2 transition, JAX will NOT recompile automatically and will use stale closures. Fix: refactor `train_and_eval_step` to accept `cmaes_mgr` and `vae_decode_fn` as explicit arguments and call `jax.jit(train_and_eval_step)(...)` fresh after transition.

- **Passing z_latent to cmaes_mgr.tell():** After wrapping with `pc_decode_fn`, `z_population` from `cmaes_mgr.ask()` is already in K-dim PC space. The `vae_decode_fn` closure handles the K→64 projection internally. Never pass the 64-dim z_latent to `tell()` — CMA-ES must update in the space it sampled from.

- **Skipping whitening in Stage 2:** Without whitening (`pc_stds`), PC1 may have std 3-4× larger than PC20. CMA-ES adapts its covariance eventually but wastes generations. With whitening, `sigma_init=0.5` is appropriate for all K dims.

- **Using stage2 transition inside jax.lax.scan:** The Stage 1→2 transition cannot happen inside `train_step` or `train_and_eval_step` because it requires Python-level reinitialization of `CMAESManager` and recompilation of `jax.jit`. It MUST happen in the outer Python `for eval_step in range(...)` loop.

- **Not converting tokens to numpy before clutr_to_grid:** `clutr_to_grid()` expects `np.ndarray`, not JAX arrays. Must do `np.array(tokens_jax)` before passing to `encode_mazes_to_mu`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| PCA computation | Custom eigendecomp | `np.linalg.svd(mu_centered, full_matrices=False)` | SVD is numerically stable, handles rank-deficient cases, already in numpy |
| Whitening | Manual scale normalization | `pc_coords.std(axis=0)` per-PC std from training data | Automatically handles heterogeneous PC scales |
| Level encoding | Custom grid-to-jax converter | `level_to_tokens` + `clutr_to_grid` (both already written and tested) | Avoids reimplementing coordinate conventions |
| CMA-ES reinit | Manually zero covariance matrix | `cmaes_mgr.initialize(rng)` | Evosax handles all internal CMA-ES state fields |

**Key insight:** The projection math (Stage 1 and Stage 2) is 2-3 lines of numpy/jax. The complexity is in the integration (JIT recompilation at transition, correct closure capture), not the math.

---

## Common Pitfalls

### Pitfall 1: JIT Closure Staleness at Stage Transition
**What goes wrong:** `train_and_eval_step` is `@jax.jit`-decorated and captures `cmaes_mgr` and `vae_decode_fn` from the enclosing scope. Reassigning these Python variables does NOT trigger JAX recompilation — the next call uses the old compiled function with old K.
**Why it happens:** Python closures capture variable bindings, but JAX traces at first call. After transition, the Python variable points to a new object, but JAX still has the old trace.
**How to avoid:** Refactor so `train_and_eval_step` is not a bare `@jax.jit`-decorated function that captures mutable state. Options:
  - Option A: `jax.jit(make_train_step(cmaes_mgr, vae_decode_fn))` — factory that returns a new jitted function
  - Option B: Pass `cmaes_mgr.es_params` and `effective_latent_dim` as static arguments
  - Option C: Add a Python flag that conditionally selects which decode/manager path to use (simpler but less clean)
**Warning signs:** After transition, `es_state.mean.shape` is still `(K1,)` even though the new manager expects `(K2,)`.

### Pitfall 2: mean_layer at Wrong Param Path
**What goes wrong:** `params['encoder']['mean_layer']` raises `KeyError`.
**Why it happens:** `mean_layer` is defined at the top level of `CnnLstmVAE.__call__`, NOT inside `CnnEncoder`. This is intentional for checkpoint compatibility.
**How to avoid:** Always use `params['mean_layer']` (top level). Confirmed in checkpoint: `params.keys() = ['decoder', 'encoder', 'logvar_layer', 'mean_layer']`.
**Warning signs:** `KeyError: 'mean_layer'` inside a `params['encoder']` subtree lookup.

### Pitfall 3: Buffer Too Small at Stage 2 Transition
**What goes wrong:** At 10k updates, the replay buffer may not be full. `sampler["size"]` could be 0 to `capacity` depending on how many DR steps have run.
**Why it happens:** `minimum_fill_ratio=1.0` means replay only starts when buffer is full. DR steps populate the buffer but CMA-ES popsize (e.g., 32) × 10k = 320k level insertions into a buffer of size ~4000 — buffer will be full well before 10k.
**How to avoid:** Check `int(sampler["size"]) > K_stage2 * 10` before triggering Stage 2 — need at least ~200 levels for PCA to be meaningful. Log a warning if too few.
**Warning signs:** `np.linalg.svd` on a matrix with fewer rows than columns produces degenerate PCA (S has zeros, some PCs undefined).

### Pitfall 4: Dataset Download Hanging at Training Start
**What goes wrong:** `np.load(gs://...)` is not supported — numpy cannot load from GCS URIs.
**Why it happens:** GCS URLs require the google-cloud-storage SDK; numpy only reads local paths.
**How to avoid:** The dataset must be pre-downloaded to a local path before training. Add a download step at the beginning of the `--use_pca_search` setup block. Use the legacy credentials pattern: `storage.Client(project='open-endedness-personal', credentials=legacy_creds)`.
**Warning signs:** `FileNotFoundError: gs://...` or similar when numpy tries to open the path.

### Pitfall 5: Stage 1 K Selection Too Aggressive
**What goes wrong:** Keeping only ~5 dims (very aggressive pruning) means CMA-ES searches a space too small to represent diverse valid mazes, causing premature convergence.
**Why it happens:** If cumulative variance threshold is set too high (e.g., 99%), K could be 50+; too low (e.g., 50%) gives K ≈ 10.
**How to avoid:** Use 85% cumulative variance threshold as default. For this CNN-VAE (free_bits=4.0 active on all 64 dims), expect K in range 25-35. Verify: decode `z_pc=zeros(K)` → `wall_map.sum()` should be 20-40 (reasonable maze density).
**Warning signs:** `z_pc=zeros(K)` decodes to an empty maze (all open) or all-wall maze.

### Pitfall 6: CMA-ES sigma_init in Stage 1 (Non-Whitened Dims)
**What goes wrong:** In Stage 1, the CMA-ES searches directly in the latent space (not whitened). Latent dims are bounded [-4, 4] by `tanh * 4`. `sigma_init=1.0` (current default) is fine. If Stage 1 used a whitened representation, sigma_init would need adjustment — but Stage 1 does NOT whiten, it searches in the original latent space for the kept dims.
**How to avoid:** Keep `sigma_init=1.0` for Stage 1 (same as current CMA-ES default). Use `sigma_init=0.5` for Stage 2 (whitened PCA space, unit std per PC).
**Warning signs:** Stage 1 CMA-ES collapses to std < 0.01 within 500 steps (the IPOP restart threshold).

---

## Code Examples

Verified patterns from codebase inspection and interactive testing (2026-03-12):

### Checkpoint Params Inspection (Verified)
```python
# Source: verified in jax_env 2026-03-12
import orbax.checkpoint as ocp
restored = ocp.PyTreeCheckpointer().restore('/abs/path/to/vae/checkpoints/cnn_vae/default')
# restored['params'].keys() = ['decoder', 'encoder', 'logvar_layer', 'mean_layer']
# mean_layer is at TOP LEVEL — NOT inside encoder
params = restored['params']
```

### Encoder Forward Pass (Verified)
```python
# Source: verified working in jax_env 2026-03-12 — mu.shape=(4,64), range=[-1.36, 1.34] for zeros input
import jax, jax.numpy as jnp, flax.linen as nn
from cnn_vae_model import CnnEncoder

@jax.jit
def encode_batch(batch):
    h = CnnEncoder(name='encoder').apply({'params': params['encoder']}, batch)
    mu = jnp.tanh(nn.Dense(64, name='mean_layer').apply({'params': params['mean_layer']}, h)) * 4.0
    return mu  # (B, 64)
```

### numpy SVD for PCA (Verified)
```python
# Source: verified in jax_env 2026-03-12 — np.version 2.2.6, full_matrices=False essential for efficiency
import numpy as np
mu_mean = mu.mean(axis=0)              # (64,)
mu_centered = mu - mu_mean             # (N, 64)
_, S, Vt = np.linalg.svd(mu_centered, full_matrices=False)  # Vt: (64, 64) or (N, 64) if N<64
pc_axes = Vt[:K]                       # (K, 64) top K PCs
pc_coords = mu_centered @ pc_axes.T    # (N, K)
pc_stds = pc_coords.std(axis=0)        # (K,) — for whitening
evr = (S[:K]**2) / (S**2).sum()       # (K,) explained variance ratio
```

### pc_decode_fn JIT+vmap Compatibility (Verified)
```python
# Source: verified with jax.vmap(lambda z: pc_decode_fn(z))(z_batch) where z_batch=(32,K) — 2026-03-12
# The closure captures jnp arrays (mu_mean, pc_axes, pc_stds) — these are static constants in JAX trace
def make_pc_decode_fn(base_decode_fn, mu_mean, pc_axes, pc_stds):
    def pc_decode_fn(z_pc):           # z_pc: (K,)
        z_latent = mu_mean + (z_pc * pc_stds) @ pc_axes   # (64,)
        return base_decode_fn(z_latent)
    return pc_decode_fn
# Usage: vae_decode_fn = make_pc_decode_fn(cnn_base_decode_fn, jnp.array(mu_mean), ...)
```

### Stage 1 variance-pruned decode (Verified)
```python
# Source: verified with jax.vmap — 2026-03-12
def make_variance_pruned_decode_fn(base_decode_fn, mu_mean, kept_dims):
    # mu_mean: (64,) jnp array; kept_dims: (K,) jnp int32 array
    def pruned_decode_fn(z_reduced):  # z_reduced: (K,)
        z_latent = mu_mean.at[kept_dims].set(mu_mean[kept_dims] + z_reduced)
        return base_decode_fn(z_latent)
    return pruned_decode_fn
```

### GCS Dataset Download (Verified Pattern)
```python
# Source: legacy_credentials pattern from Phase 1 STATE.md — verified credentials exist 2026-03-12
import json, os
from google.oauth2.credentials import Credentials
from google.cloud import storage

cred_path = os.path.expanduser('~/.config/gcloud/legacy_credentials/giacomo.maralla@gmail.com/adc.json')
with open(cred_path) as f:
    cred_data = json.load(f)
creds = Credentials(
    token=cred_data.get('access_token'), refresh_token=cred_data.get('refresh_token'),
    token_uri='https://oauth2.googleapis.com/token',
    client_id=cred_data.get('client_id'), client_secret=cred_data.get('client_secret'),
)
client = storage.Client(project='open-endedness-personal', credentials=creds)
blob = client.bucket('ucl-ued-project-bucket').blob('vae/datasets/train_1M_envs.npy')
blob.download_to_filename('/tmp/train_1M_envs.npy')  # 208MB
data = np.load('/tmp/train_1M_envs.npy')             # (1_000_000, 52) int32
clutr_seqs = data[:config['pca_dataset_size']]       # e.g., first 10_000
```

### Buffer Level Encoding for Stage 2
```python
# Source: combination of maze_plr.py buffer access pattern + encode_mazes_to_mu
from vae_level_utils import level_to_tokens
from cnn_vae_data import clutr_to_grid

# Get current buffer levels (outside jit, in Python outer loop)
ts = runner_state[1]  # train_state
size = int(ts.sampler["size"])
buffer_levels = jax.tree_util.tree_map(lambda x: x[:size], ts.sampler["levels"])
tokens_jax = jax.vmap(level_to_tokens)(buffer_levels)  # (size, 52)
tokens_np = np.array(tokens_jax)                        # to numpy

# Encode to mu
mu_buf = encode_mazes_to_mu(cnn_vae_params, tokens_np, latent_dim=64)  # (size, 64)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Full 64-dim latent search from random init | Two-stage: variance-pruned → PCA subspace | Phase 5 (now) | O(K²) CMA-ES sample complexity vs O(64²); ~10× cheaper with K=20 |
| No warm-start covariance for CMA-ES | PC axes give free covariance structure | Phase 5 (now) | No burn-in; CMA-ES starts in valid maze manifold |
| sklearn.decomposition.PCA | numpy.linalg.svd (direct) | Phase 5 (now) | sklearn not installed; numpy SVD equivalent for offline analysis |

**Deprecated/outdated:**
- sklearn PCA: not in jax_env — use `np.linalg.svd` directly

---

## Open Questions

1. **JIT recompilation at Stage 1→2 transition**
   - What we know: `train_and_eval_step` is `@jax.jit` with closure capture of `cmaes_mgr` and `vae_decode_fn`
   - What's unclear: Exact refactoring needed to allow recompilation without restructuring the whole function
   - Recommendation: During planning, decide whether to (A) use `jax.jit` with `donate_argnums` and factory pattern, or (B) restructure `on_new_levels` to accept decode_fn as an argument. Option B is cleanest and matches existing `on_replay_levels` pattern where `level_sampler` is passed in. The planner should spec this concretely.

2. **Stage 1 K selection recommendation**
   - What we know: CNN-VAE has free_bits=4.0 so all 64 dims are active; expect per-dim variance spread across dims
   - What's unclear: Actual variance distribution of THIS CNN-VAE (only testable with real mazes)
   - Recommendation: Use cumulative 85% variance threshold as default; add `--pca_stage1_k` override flag. During smoke test, print the full variance curve and pick elbow visually.

3. **sigma_init for Stage 2**
   - What we know: With whitened PCs (unit std), sigma_init=0.5 means ±0.5 std exploration per PC
   - What's unclear: Whether sigma_init=0.5 or 1.0 works better in practice for Stage 2
   - Recommendation: Default sigma_init for Stage 2 to 0.5 (separate from Stage 1 which keeps 1.0). Add `--pca_sigma_init` flag.

4. **Flag for `--dataset_path`**
   - What we know: `--dataset_path` does NOT currently exist in maze_plr.py
   - What's unclear: Whether to hardcode `/tmp/train_1M_envs.npy` as default or require explicit flag
   - Recommendation: Add `--pca_dataset_path` flag with default `/tmp/train_1M_envs.npy`. Download step should be a separate `scripts/download_pca_dataset.sh` or inline in setup block.

5. **Stage 2 minimum buffer size**
   - What we know: `LevelSampler.capacity` is set by `--num_train_levels` (default likely 4000); buffer fills fast with popsize=32 CMA-ES
   - What's unclear: Exact `num_train_levels` default and whether 10k update step guarantees buffer is full
   - Recommendation: Add guard: `if size < K_stage2 * 10: skip transition and log warning`. Standard PLR minimum_fill_ratio=1.0 means at 10k updates the buffer will be full for typical configs.

---

## Sources

### Primary (HIGH confidence)
- Local codebase: `vae/cnn_vae_model.py`, `vae/cmaes_manager.py`, `vae/cnn_vae_level_utils.py`, `vae/cnn_vae_data.py` — direct inspection
- Local codebase: `examples/maze_plr.py` — grep analysis of integration points, outer loop structure (lines 1266-1376), TrainState fields
- Local codebase: `src/jaxued/level_sampler.py` — Sampler dict structure (`levels`, `scores`, `size`), `level_to_tokens`
- jax_env verification: `numpy.linalg.svd` working (2026-03-12); encoder forward pass produces `(B, 64)` mu with correct param paths; GCS dataset blob confirmed 208MB = 1M mazes × 52 × 4 bytes; `pc_decode_fn` and `pruned_decode_fn` JIT+vmap verified
- `vae/pca_search_prompt.md` (PRD) — implementation spec with verified code patterns

### Secondary (MEDIUM confidence)
- `vae/buffer_latent_analysis.py` — PCA pipeline pattern for buffer analysis; uses sklearn PCA (not available) but confirms `level_to_tokens` → encode pipeline
- `.planning/phases/01-checkpoint/01-01-SUMMARY.md` — GCS auth pattern (legacy_credentials), project='open-endedness-personal'
- `.planning/STATE.md` — Accumulated decisions, confirmed mean_layer at top level

### Tertiary (LOW confidence)
- Expected variance distribution: "85% in top 20 PCs for CNN-VAE with free_bits=4.0" from PRD — not yet empirically verified against THIS checkpoint; must verify in smoke test

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all dependencies verified present in jax_env; SVD, encoder, GCS all tested
- Architecture: HIGH — code patterns verified working (vmap, JIT, closure capture); transition design derived from outer loop structure inspection
- Pitfalls: HIGH for params/JIT/whitening (derived from code analysis); MEDIUM for buffer-size pitfall (depends on runtime config)

**Research date:** 2026-03-12
**Valid until:** 2026-04-12 (stable — no fast-moving dependencies; numpy/jax/flax APIs unchanged)
