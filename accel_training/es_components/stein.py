"""Stein Variational Gradient Descent repulsion for SV-CMA-ES.

Implements SVGD (Liu and Wang 2016) with behavior-space RBF kernel and latent-space
gradient direction. Kernel K is NxN from behavior signature pairwise distances; repulsion
gradient applied to CMA means in latent space (D=64). Standard median heuristic:
h = median(sq_dists) / log(N).
"""

from __future__ import annotations

import jax.numpy as jnp


def compute_stein_repulsion(
    means: jnp.ndarray,
    bsigs: jnp.ndarray,
    epsilon: float = 0.01,
) -> jnp.ndarray:
    """Compute Stein repulsion delta for N CMA-ES particle means.

    The kernel K is computed in behavior space (D_bsig) to measure similarity
    between particles. The repulsion gradient pushes means apart in latent space
    (param_dim). These are two distinct spaces — K is NxN, means is (N, param_dim).

    Standard SVGD repulsion (Liu & Wang 2016, equation 15):
        phi(x_i) = (1/N) * sum_j [ K(x_j, x_i) * grad_{x_j} log p(x_j)
                                   + grad_{x_j} K(x_j, x_i) ]

    Here we use behavior sigs for the kernel and means for the gradient direction,
    applying the repulsion term only (no target distribution gradient).

    Bandwidth: h = median(sq_dists_b) / log(N).  Standard median heuristic.
    Floor at 1e-8 prevents NaN when all bsigs are identical (median = 0) or N = 1.

    Args:
        means:   (N, param_dim) float32 — current CMA-ES particle means in latent space.
        bsigs:   (N, D_bsig) float32 — per-particle behavior signatures.
        epsilon: float — step size (default 0.01).

    Returns:
        (N, param_dim) float32 — delta to add to each particle's mean.
        For N=1 the repulsion term is zero (single particle has no peers).
    """
    N = means.shape[0]
    N_f = jnp.float32(N)

    # N=1: no peer particles, repulsion is exactly zero.
    if N == 1:
        return jnp.zeros_like(means)

    # Pairwise squared distances in behavior space: (N, N)
    diff_b = bsigs[:, None, :] - bsigs[None, :, :]        # (N, N, D_bsig)
    sq_dists_b = jnp.sum(diff_b ** 2, axis=-1)             # (N, N)

    # Median heuristic bandwidth.
    # Use log(N+1) so the denominator is always >= log(2) > 0 for any N >= 1.
    # jnp.maximum floor prevents h=0 when all bsigs identical (median_sq = 0).
    h = jnp.maximum(
        jnp.median(sq_dists_b) / jnp.log(N_f + 1.0),
        1e-8,
    )

    # RBF kernel matrix: K[i,j] = exp(-||bsig_i - bsig_j||^2 / h)
    K = jnp.exp(-sq_dists_b / h)                           # (N, N)

    # Row sums: K_rowsum[i] = sum_j K[i,j]
    K_rowsum = K.sum(axis=1)                               # (N,)

    # Repulsion gradient (push-apart term only).
    # K @ means: (N,N)@(N,param_dim) = (N, param_dim) — cross-particle weighted means.
    # K_rowsum[:, None] * means: (N, param_dim) — self-weighted means.
    # repulsion[i] pushes mean_i away from the kernel-weighted center of other means.
    repulsion = (1.0 / (N_f * h)) * (
        K_rowsum[:, None] * means - K @ means
    )  # (N, param_dim)

    return epsilon * repulsion


def mean_pairwise_behavior_dist(bsigs: jnp.ndarray) -> float:
    """Compute mean Euclidean distance over all N*(N-1) off-diagonal particle pairs.

    Used as a diversity metric: logged to WandB before and after Stein repulsion
    to track whether particles maintain behavioral separation.

    Args:
        bsigs: (N, D_bsig) float32 — per-particle behavior signatures.

    Returns:
        Python float: mean pairwise Euclidean distance.
        Returns 0.0 immediately if N <= 1 (no pairs to compare).
    """
    N = bsigs.shape[0]
    if N <= 1:
        return 0.0

    # Pairwise Euclidean distances: (N, N)
    diff = bsigs[:, None, :] - bsigs[None, :, :]          # (N, N, D_bsig)
    sq = jnp.sum(diff ** 2, axis=-1)                       # (N, N)
    d = jnp.sqrt(jnp.maximum(sq, 0.0))                    # (N, N), no negative sqrt

    # Mask out diagonal (self-distances = 0) and average over off-diagonal pairs.
    mask = 1.0 - jnp.eye(N)                               # (N, N), 0 on diagonal
    return float(jnp.sum(d * mask) / (N * (N - 1)))
