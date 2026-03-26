# GSD Feature Prompt: PCA-Space CMA-ES Search

Use this prompt AFTER the CNN-VAE integration is working (`--use_cnn_vae` flag live in `examples/maze_plr.py`).

---

## Prompt

I want to add PCA-space CMA-ES search as an option in the training pipeline. This builds on the existing `--use_cnn_vae` flag.

### The core idea (why this works)

CMA-ES starts with identity covariance and has to learn the structure of good solutions from scratch. But valid mazes don't fill all 64 latent dims equally — when you encode real mazes, the mu vectors form an ellipsoid (some dims vary a lot, others are near-zero). By pre-computing PCA of encoded real mazes and searching in the top K PC axes instead, you:

1. **Give CMA-ES the covariance for free** — no burn-in needed
2. **Reduce dimensionality 64→K** — CMA-ES sample complexity is O(d²), so K=20 is ~10× cheaper than 64
3. **Stay on the valid maze manifold** — PC subspace IS where real mazes live; raw latent search can drift to garbage regions
4. **Perfect z=0 init** — origin of PC space maps to `mu_mean` (the average training maze), not decoder's default

### Existing files to build on

- `vae/cnn_vae_model.py` — `CnnLstmVAE`, `CnnEncoder`, `CnnLstmDecoder`
- `vae/cnn_vae_data.py` — `clutr_to_grid()`, `grid_to_clutr()`
- `vae/cnn_vae_level_utils.py` — `decode_latent_to_levels_grid(decode_fn, z_batch, rng)`
- `vae/cmaes_manager.py` — `CMAESManager(popsize, latent_dim)` — latent_dim-agnostic, just change to K
- `examples/maze_plr.py` — already has `--use_cnn_vae` flag and CNN-VAE checkpoint loading

### Checkpoint + dataset

- CNN-VAE checkpoint: `gs://cnn-vae-maze-checkpoints/run10/` (Orbax, step 200000)
- Checkpoint param keys: `params/encoder/...`, `params/mean_layer/...`, `params/decoder/...`
- Maze dataset for PCA: `gs://ucl-ued-project-bucket/vae/datasets/train_1M_envs.npy`
  - Format: `(N, 52)` int32 CLUTR sequences — use `clutr_to_grid()` to convert before encoding
  - Only need 10k-20k mazes for PCA (not all 1M)

### Architecture: how the encoder produces mu

```python
# From vae/cnn_vae_model.py CnnLstmVAE.__call__:
h = CnnEncoder(name='encoder')(x)                              # (B, 512)
mu = jnp.tanh(nn.Dense(latent_dim, name='mean_layer')(h)) * 4.0  # (B, 64), bounded [-4, 4]
```

To get mu vectors for PCA, call encoder + mean_layer directly (no reparameterization, no decoder needed).

### What to build

#### File 1: `vae/cnn_vae_pca_utils.py` (new file)

```python
"""PCA utilities for CMA-ES latent space search."""

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

from vae.cnn_vae_model import CnnEncoder
from vae.cnn_vae_data import clutr_to_grid


def encode_mazes_to_mu(params: dict, clutr_seqs: np.ndarray,
                        latent_dim: int = 64, batch_size: int = 256) -> np.ndarray:
    """Encode CLUTR maze sequences to mu vectors using the CNN-VAE encoder.

    Args:
        params: Full VAE params dict with keys 'encoder', 'mean_layer', etc.
        clutr_seqs: (N, 52) int32 CLUTR sequences
        latent_dim: latent dimension (64)
        batch_size: encoding batch size

    Returns:
        mu: (N, latent_dim) float32 mu vectors
    """
    # Convert CLUTR → grids
    grids = np.array([clutr_to_grid(seq) for seq in clutr_seqs])  # (N, 13, 13, 3)

    # Define batched encoder forward pass
    # CRITICAL: mean_layer params are at TOP LEVEL of params dict (params['mean_layer']),
    # NOT inside params['encoder']. This matches CnnLstmVAE's __call__ design.
    mean_layer = nn.Dense(latent_dim, name='mean_layer')

    @jax.jit
    def encode_batch(batch):
        h = CnnEncoder(name='encoder').apply({'params': params['encoder']}, batch)
        mu = jnp.tanh(mean_layer.apply({'params': params['mean_layer']}, h)) * 4.0
        return mu

    # Encode in batches
    all_mu = []
    for i in range(0, len(grids), batch_size):
        batch = jnp.array(grids[i:i+batch_size])
        all_mu.append(np.array(encode_batch(batch)))
    return np.concatenate(all_mu, axis=0)  # (N, 64)


def compute_pca_axes(mu: np.ndarray, K: int):
    """Compute whitened PCA axes from mu vectors.

    Whitening: divide PC coordinates by their std so all K dims have unit variance.
    This is critical for CMA-ES — without whitening, PC1 might have std=2.0 and
    PC20 might have std=0.1, and CMA-ES wastes generations learning the scale.

    Args:
        mu: (N, latent_dim) mu vectors from encode_mazes_to_mu
        K: number of principal components to keep

    Returns:
        mu_mean: (latent_dim,) mean of mu vectors — z=0 in PC space maps here
        pc_axes: (K, latent_dim) top K principal component directions (unit vectors)
        pc_stds: (K,) standard deviation of each PC coordinate in the data
        explained_variance_ratio: (K,) fraction of variance explained by each PC
    """
    mu_mean = mu.mean(axis=0)          # (64,)
    mu_centered = mu - mu_mean          # (N, 64)

    # SVD: mu_centered = U @ diag(S) @ Vt
    # Vt rows = principal component directions (unit vectors)
    _, S, Vt = np.linalg.svd(mu_centered, full_matrices=False)  # Vt: (64, 64)

    pc_axes = Vt[:K]                    # (K, 64) top K PCs
    pc_coords = mu_centered @ pc_axes.T # (N, K) each maze's coordinates in PC space
    pc_stds = pc_coords.std(axis=0)     # (K,) std of each PC — use for whitening

    total_var = (S**2).sum()
    explained_variance_ratio = (S[:K]**2) / total_var  # (K,)

    return mu_mean, pc_axes, pc_stds, explained_variance_ratio


def make_pc_decode_fn(base_decode_fn, mu_mean: jnp.ndarray,
                      pc_axes: jnp.ndarray, pc_stds: jnp.ndarray):
    """Wrap a latent decode_fn to accept PC-space coordinates instead.

    CMA-ES samples z_pc ~ N(0, sigma*I) in K-dim PC space.
    This wrapper converts: z_pc (K,) → z_latent (64,) → decoder output.

    The whitened projection is:
        z_latent = mu_mean + (z_pc * pc_stds) @ pc_axes

    Why multiply by pc_stds? Because CMA-ES works in normalized space (unit std),
    but we want to move in real latent space units. pc_stds undoes the whitening.

    Args:
        base_decode_fn: z (64,) → (wall_logits, goal_logits, agent_logits) each (13,13)
        mu_mean: (64,) from compute_pca_axes
        pc_axes: (K, 64) from compute_pca_axes
        pc_stds: (K,) from compute_pca_axes

    Returns:
        pc_decode_fn: z_pc (K,) → same output as base_decode_fn
    """
    def pc_decode_fn(z_pc):
        # z_pc: (K,) — whitened PC coordinates from CMA-ES
        z_latent = mu_mean + (z_pc * pc_stds) @ pc_axes  # (64,)
        return base_decode_fn(z_latent)
    return pc_decode_fn
```

#### Changes to `examples/maze_plr.py`

Add these flags (near the other CNN-VAE flags):
```python
parser.add_argument('--use_pca_search', action='store_true',
                    help='Search in PCA subspace of latent codes instead of full latent space')
parser.add_argument('--pca_components', type=int, default=20,
                    help='Number of PCA components to keep for CMA-ES search')
parser.add_argument('--pca_dataset_size', type=int, default=10000,
                    help='Number of mazes to use for PCA computation')
```

In the VAE setup block (after CNN-VAE checkpoint is loaded, inside `if config["use_cnn_vae"]:`):
```python
if config["use_pca_search"]:
    from vae.cnn_vae_pca_utils import encode_mazes_to_mu, compute_pca_axes, make_pc_decode_fn
    import numpy as np

    # Load a subset of the training dataset for PCA
    print(f"Computing PCA from {config['pca_dataset_size']} mazes...")
    clutr_seqs = np.load(config["dataset_path"])[:config["pca_dataset_size"]]

    # Encode to mu vectors
    mu_vectors = encode_mazes_to_mu(cnn_vae_params, clutr_seqs, latent_dim=64)

    # Compute PCA
    K = config["pca_components"]
    mu_mean, pc_axes, pc_stds, evr = compute_pca_axes(mu_vectors, K)
    print(f"PCA: top {K} components explain {evr.sum()*100:.1f}% of variance")
    print(f"Variance per component: {evr[:5]*100}")  # show first 5

    # Convert to jnp (these are closed over in pc_decode_fn — fully JIT safe)
    mu_mean_j = jnp.array(mu_mean)
    pc_axes_j = jnp.array(pc_axes)
    pc_stds_j = jnp.array(pc_stds)

    # Wrap the decode function to accept K-dim PC coords
    vae_decode_fn = make_pc_decode_fn(cnn_base_decode_fn, mu_mean_j, pc_axes_j, pc_stds_j)

    # CRITICAL: CMA-ES must use K dimensions, not 64
    effective_latent_dim = K
else:
    vae_decode_fn = cnn_base_decode_fn
    effective_latent_dim = 64

# Use effective_latent_dim when constructing CMAESManager:
cmaes_mgr = CMAESManager(
    popsize=config["num_train_envs"],
    latent_dim=effective_latent_dim,   # K if PCA, else 64
    sigma_init=config["cmaes_sigma_init"],
)
```

In `on_new_levels` — this part does NOT change at all:
```python
z_population, es_state = cmaes_mgr.ask(rng_ask, es_state)       # (popsize, K or 64)
new_levels = decode_latent_to_levels_grid(vae_decode_fn, z_population, rng_decode)
# ...
es_state = cmaes_mgr.tell(rng_tell, z_population, -scores, es_state)
```

The magic: `vae_decode_fn` is now `pc_decode_fn` which accepts (K,) and outputs Level-compatible logits. CMAESManager and `decode_latent_to_levels_grid` are completely unaware of PCA — it's transparent.

#### Launch script: `examples/launch_vae_comparison.sh`

Add new condition:
```bash
python examples/maze_plr.py \
  --use_cmaes --use_accel \
  --use_cnn_vae \
  --vae_checkpoint_path gs://cnn-vae-maze-checkpoints/run10/ \
  --dataset_path /tmp/train_1M_envs.npy \
  --use_pca_search \
  --pca_components 20 \
  --run_name cmaes_cnn_vae_pca20 \
  --project $PROJECT \
  --seed $SEED
```

### Implementation order

1. Create `vae/cnn_vae_pca_utils.py` with the three functions above
2. Add flags + PCA setup block to `examples/maze_plr.py`
3. Smoke test locally (CPU): load checkpoint, encode 100 mazes, compute PCA, decode z_pc=zeros(20) → check it produces a valid Level
4. Update launch script with `cmaes_cnn_vae_pca20` condition

### Key gotchas to watch for

**1. params key structure**
`params['mean_layer']` is at TOP LEVEL of the params dict — NOT inside `params['encoder']`. This is a deliberate design decision in CnnLstmVAE to match the ES pipeline checkpoint contract. If you get `KeyError: 'mean_layer'` inside encoder params, you're looking in the wrong place.

**2. Tell must use z_pc, not z_latent**
CMA-ES updates its distribution in the space it sampled from. `z_population` from `cmaes_mgr.ask()` is in K-dim PC space — pass THAT to `cmaes_mgr.tell()`. Never pass the projected `z_latent` (64-dim) to tell. The code above is correct as written since `vae_decode_fn` closes over the projection.

**3. Whitening vs no whitening**
We whiten (multiply by pc_stds) so CMA-ES sees unit-variance dimensions. If you skip whitening, PC1 will have variance ~(S[0]/sqrt(N))² which could be 10-100× larger than PC20. CMA-ES will adapt eventually but wastes generations. The whitened version is strongly preferred.

**4. sigma_init in PC space**
With whitened PCs (unit std), `sigma_init=0.5` means CMA-ES initially explores ±0.5 std in each PC direction — reasonable. If you don't whiten, sigma_init needs to be tuned per-PC. The whitened approach lets you reuse the same sigma_init as before.

**5. Dataset path for PCA**
`maze_plr.py` might not have a `--dataset_path` flag yet. Add it if missing. The dataset at `gs://ucl-ued-project-bucket/vae/datasets/train_1M_envs.npy` needs to be downloaded to `/tmp/` on the TPU VM first (same as in the CNN-VAE training — standard startup step).

**6. How many PCs?**
Plot `evr.cumsum()` to decide K. For our CNN-VAE with free_bits=4.0 (all 64 dims active), expect ~85% variance in top 20 PCs. Run with `--pca_components 64` first to see the full curve, then pick the elbow. Typical sweet spot: K where cumulative variance crosses 85-90%.

### What success looks like

Smoke test passes:
```
PCA: top 20 components explain 87.3% of variance
Variance per component: [18.2  12.1   8.7   6.3   4.9]
Decoded z_pc=zeros(20) → Level(wall_map.sum()=24, is_well_formatted()=True)
```

Short CMA-ES run (500 steps): fitness should improve faster than raw 64-dim search, and sigma should NOT collapse to near-zero within 200 generations (collapse = the latent space is too rough).

### GCS paths reference
- CNN-VAE checkpoint: `gs://cnn-vae-maze-checkpoints/run10/` (Orbax, step 200000)
- Training dataset: `gs://ucl-ued-project-bucket/vae/datasets/train_1M_envs.npy`
- CluttrVAE checkpoint (baseline): `vae/runs/runs/20260227_185835_.../checkpoints/checkpoint_260000.pkl`
