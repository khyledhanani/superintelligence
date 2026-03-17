"""PCA utilities for two-stage CMA-ES latent space search.

Stage 1: variance-pruned latent search (top-variance dims from training data)
Stage 2: whitened PCA search (buffer-derived PCA axes)

Two-stage architecture:
  Stage 1 (steps 0-10k): Weight-norm pruned search.
    - Read mean_layer kernel (512, 64) from checkpoint.
    - Compute L2 norm of each column to identify active vs dead dims.
    - Keep only top ~30 dims by cumulative norm threshold.
    - CMA-ES searches in this reduced K-dim subspace from step 0.
    - No dataset encoding needed — checkpoint weights are sufficient.

  Stage 2 (steps 10k+): Buffer PCA search.
    - Encode replay buffer levels to mu vectors via encode_mazes_to_mu.
    - Compute PCA on buffer-derived mu vectors.
    - Switch CMA-ES to top K PCA components with whitened projection.

Key contracts:
  - params['mean_layer'] is at TOP LEVEL (NOT inside params['encoder'])
  - clutr_to_grid expects numpy arrays (NOT JAX arrays)
  - All compute functions use numpy (offline); closures use jnp (JIT/vmap)
"""
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

from cnn_vae_model import CnnEncoder
from cnn_vae_data import clutr_to_grid


def encode_mazes_to_mu(params, clutr_seqs, latent_dim=64, batch_size=256):
    """Encode CLUTR maze sequences to mu vectors using the CNN-VAE encoder.

    Uses the encoder and mean_layer from the VAE checkpoint to compute
    the posterior mean (mu) for each maze. The mean_layer is at the TOP
    LEVEL of params — NOT inside params['encoder'].

    Args:
        params: Full VAE params dict from orbax checkpoint.
                Must have params['encoder'] and params['mean_layer'] keys.
        clutr_seqs: (N, 52) int32 numpy array of CLUTR maze sequences.
        latent_dim: Latent dimension of the VAE (default 64).
        batch_size: Number of mazes to process in each JIT batch.

    Returns:
        mu: (N, latent_dim) float32 numpy array. Values in [-4, 4]
            due to tanh * 4.0 in the VAE bottleneck.
    """
    # Convert all CLUTR sequences to grids (loop — clutr_to_grid expects single np.ndarray)
    N = len(clutr_seqs)
    grids = np.stack([clutr_to_grid(np.array(clutr_seqs[i])) for i in range(N)])
    # grids: (N, 13, 13, 3) float32

    # JIT-compiled batch encoder function
    @jax.jit
    def encode_batch(batch):
        # batch: (B, 13, 13, 3) float32
        # Step 1: CNN encoder -> (B, 512)
        h = CnnEncoder(name='encoder').apply({'params': params['encoder']}, batch)
        # Step 2: mean_layer at TOP LEVEL (NOT inside encoder) -> (B, latent_dim)
        # CRITICAL: params['mean_layer'] is a top-level key, NOT params['encoder']['mean_layer']
        mu = jnp.tanh(
            nn.Dense(latent_dim, name='mean_layer').apply({'params': params['mean_layer']}, h)
        ) * 4.0
        return mu

    # Process in batches
    all_mu = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = grids[start:end]
        mu_batch = encode_batch(batch)
        all_mu.append(np.array(mu_batch))

    return np.concatenate(all_mu, axis=0)  # (N, latent_dim) float32


def compute_active_dims(mean_layer_kernel, K=None, cum_threshold=0.85):
    """Select the most active latent dimensions via weight norm analysis.

    Dims with large mean_layer kernel norms produce varying mu values across
    inputs (active); dims with small norms produce near-constant mu regardless
    of input (dead). This allows Stage 1 to work without any dataset encoding.

    Args:
        mean_layer_kernel: (512, 64) numpy weight matrix from
                           params['mean_layer']['kernel'].
        K: Number of dims to keep. If None, auto-select via cum_threshold.
        cum_threshold: Cumulative norm fraction threshold for auto K selection
                       (default 0.85 = keep dims covering 85% of total norm).

    Returns:
        kept_dims: (K,) int32 numpy array of kept dimension indices,
                   sorted ascending for stable indexing.
        per_dim_norms: (64,) float32 numpy array of per-dimension L2 norms.
        K: int, number of kept dimensions.
    """
    # Per-dim L2 norms: axis=0 collapses the 512-dim input axis -> (64,) norms
    per_dim_norms = np.linalg.norm(mean_layer_kernel, axis=0)  # (64,)

    # Sort dimensions by descending norm
    sorted_idx = np.argsort(-per_dim_norms)  # descending

    if K is None:
        # Auto-select K via cumulative norm threshold
        cum_norms = np.cumsum(per_dim_norms[sorted_idx]) / per_dim_norms.sum()
        K = int(np.searchsorted(cum_norms, cum_threshold)) + 1
        K = min(K, len(per_dim_norms))

    # Sort kept dims ascending for stable indexing in pruned_decode_fn
    kept_dims = np.sort(sorted_idx[:K]).astype(np.int32)  # (K,) ascending

    return kept_dims, per_dim_norms, K


def compute_pca_axes(mu, K):
    """Compute PCA axes from a collection of mu vectors.

    Performs truncated SVD on centered mu vectors to extract the top K
    principal components. Uses full_matrices=False for efficiency when N > 64.

    Args:
        mu: (N, 64) numpy array of encoded mu vectors.
        K: Number of principal components to retain.

    Returns:
        mu_mean: (64,) float32 numpy array — mean of the mu vectors.
        pc_axes: (K, 64) float32 numpy array — top K principal component
                 directions (rows are unit vectors).
        pc_stds: (K,) float32 numpy array — standard deviation of mu projections
                 onto each PC (used for whitening in make_pc_decode_fn).
        evr: (K,) float32 numpy array — explained variance ratio per component.
             evr.sum() gives the total fraction of variance explained by K PCs.
    """
    mu_mean = mu.mean(axis=0)       # (64,)
    mu_centered = mu - mu_mean      # (N, 64)

    # Truncated SVD: full_matrices=False is ESSENTIAL for efficiency when N > 64
    # Vt shape: (min(N,64), 64) with full_matrices=False (vs (N, 64) with True)
    _, S, Vt = np.linalg.svd(mu_centered, full_matrices=False)

    # Top K principal component axes (rows are PC directions)
    pc_axes = Vt[:K]                # (K, 64)

    # Project centered mu onto PCs to compute per-component std
    pc_coords = mu_centered @ pc_axes.T  # (N, K)
    pc_stds = pc_coords.std(axis=0)     # (K,)

    # Explained variance ratio: S are singular values of mu_centered
    # Variance of each PC is proportional to S^2
    evr = (S[:K] ** 2) / (S ** 2).sum()  # (K,)

    return mu_mean.astype(np.float32), pc_axes.astype(np.float32), pc_stds.astype(np.float32), evr.astype(np.float32)


def make_variance_pruned_decode_fn(base_decode_fn, mu_mean, kept_dims):
    """Create a decode function for Stage 1 (weight-norm pruned) CMA-ES search.

    Returns a closure that maps a K-dim search vector to the full 64-dim
    latent space by injecting it into the active dimensions of mu_mean,
    then calling the base decoder. Dropped dimensions are held at mu_mean.

    The closure captures jnp arrays (mu_mean, kept_dims) and is JIT and
    vmap compatible via jax.vmap(pruned_decode_fn).

    Args:
        base_decode_fn: Callable z (64,) -> (wall_logits, goal_logits,
                        agent_logits) each (13, 13). Must be UNBATCHED.
        mu_mean: (64,) jnp array — mean of training mu vectors. Dropped dims
                 are held at this mean. Use jnp.zeros(64) as the VAE prior mean
                 if training data is not available.
        kept_dims: (K,) jnp int32 array — indices of top-K active dimensions.

    Returns:
        pruned_decode_fn: Callable z_reduced (K,) -> (wall_logits, goal_logits,
                          agent_logits). Maps K-dim CMA-ES search vector to
                          full latent space and decodes.
    """
    def pruned_decode_fn(z_reduced):
        """Decode a K-dim variance-pruned latent vector.

        Args:
            z_reduced: (K,) jnp array — CMA-ES search vector in reduced space.

        Returns:
            Tuple of (wall_logits, goal_logits, agent_logits) each (13, 13).
        """
        # Inject z_reduced into active dims, hold dropped dims at mu_mean
        z_latent = mu_mean.at[kept_dims].set(mu_mean[kept_dims] + z_reduced)
        return base_decode_fn(z_latent)

    return pruned_decode_fn


def make_pc_decode_fn(base_decode_fn, mu_mean, pc_axes, pc_stds):
    """Create a decode function for Stage 2 (whitened PCA) CMA-ES search.

    Returns a closure that maps a K-dim PCA search vector to the full 64-dim
    latent space using the whitened inverse PCA projection:
        z_latent = mu_mean + (z_pc * pc_stds) @ pc_axes

    This ensures that unit changes in z_pc correspond to one standard deviation
    in the direction of each principal component, giving CMA-ES well-scaled
    dimensions to search over (sigma_init=0.5 is a reasonable starting value).

    The closure captures jnp arrays (mu_mean, pc_axes, pc_stds) and is JIT and
    vmap compatible via jax.vmap(pc_decode_fn).

    Args:
        base_decode_fn: Callable z (64,) -> (wall_logits, goal_logits,
                        agent_logits) each (13, 13). Must be UNBATCHED.
        mu_mean: (64,) jnp array — mean of mu vectors (origin of PCA space).
        pc_axes: (K, 64) jnp array — top K principal component directions
                 (rows are unit vectors from compute_pca_axes).
        pc_stds: (K,) jnp array — standard deviation along each PC axis
                 (from compute_pca_axes). Used for whitening: z_pc=1 -> 1 std.

    Returns:
        pc_decode_fn: Callable z_pc (K,) -> (wall_logits, goal_logits,
                      agent_logits). Maps K-dim PCA search vector to full
                      latent space and decodes.
    """
    def pc_decode_fn(z_pc):
        """Decode a K-dim whitened PCA latent vector.

        Args:
            z_pc: (K,) jnp array — CMA-ES search vector in PCA space.
                  z_pc = zeros(K) maps to the average training maze (mu_mean).

        Returns:
            Tuple of (wall_logits, goal_logits, agent_logits) each (13, 13).
        """
        # Whitened inverse PCA: scale by per-PC std, then project from PC axes
        # (z_pc * pc_stds): (K,) — scale each PC coord by its std
        # @ pc_axes: (K,) @ (K, 64) = (64,) — project back to full latent space
        z_latent = mu_mean + (z_pc * pc_stds) @ pc_axes
        return base_decode_fn(z_latent)

    return pc_decode_fn
