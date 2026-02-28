"""JIT-compatible k-NN novelty scoring for ES-ACCEL behavioral diversity.

Implements brute-force masked k-NN distance over the replay buffer.
Avoids ConcretizationTypeError by using jnp.where masking (never dynamic slicing).

Memory: 4000 entries x 169 dims x 4 bytes ~= 2.7 MB distance matrix -- within GPU SRAM.
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import functools


@functools.partial(jax.jit, static_argnames=("k",))
def compute_novelty_knn(
    candidate_sig: jax.Array,
    buffer_sigs: jax.Array,
    valid_mask: jax.Array,
    k: int = 5,
) -> jax.Array:
    """Compute novelty of one candidate against the buffer via k-NN mean distance.

    Uses masked brute-force L2 distance. Empty buffer slots (valid_mask=False) are
    excluded by setting their distance to inf before top-k selection.
    JIT-compatible: no Python branches on traced values.

    Args:
        candidate_sig: (D,) float32 -- the query behavior signature
        buffer_sigs:   (capacity, D) float32 -- all stored signatures incl. empty slots
        valid_mask:    (capacity,) bool -- True for filled slots with real signatures
        k:             number of nearest neighbors (static, must be a compile-time constant)

    Returns:
        novelty: scalar float32 -- mean Euclidean distance to k nearest neighbors
                 (0.0 if fewer than k valid entries exist, handled via inf distances)
    """
    # Squared L2 distances: (capacity,)
    diffs = buffer_sigs - candidate_sig[None, :]        # (capacity, D)
    sq_dists = jnp.sum(diffs ** 2, axis=-1)             # (capacity,)

    # Mask out empty/invalid slots -- they become inf and are never selected as neighbors
    masked = jnp.where(valid_mask, sq_dists, jnp.inf)   # (capacity,)

    # jax.lax.top_k returns the k LARGEST values; negate to get k smallest
    neg_masked = -masked
    neg_top_k, _ = jax.lax.top_k(neg_masked, k)         # (k,)
    top_sq_dists = -neg_top_k                            # k smallest squared distances

    # Mean Euclidean distance (sqrt of squared distances)
    # jnp.maximum guards against tiny negatives from floating point
    novelty = jnp.mean(jnp.sqrt(jnp.maximum(top_sq_dists, 0.0)))
    return novelty


def compute_novelty_batch(
    candidate_sigs: jax.Array,
    buffer_sigs: jax.Array,
    valid_mask: jax.Array,
    k: int = 5,
) -> jax.Array:
    """Compute novelty for a batch of candidates via vmap over compute_novelty_knn.

    Args:
        candidate_sigs: (pop_size, D) float32 -- batch of query signatures
        buffer_sigs:    (capacity, D) float32 -- full buffer (including empty slots)
        valid_mask:     (capacity,) bool -- True for filled slots
        k:              number of nearest neighbors (static)

    Returns:
        novelty_scores: (pop_size,) float32 -- one novelty score per candidate
    """
    _fn = functools.partial(compute_novelty_knn, k=k)
    return jax.vmap(_fn, in_axes=(0, None, None))(candidate_sigs, buffer_sigs, valid_mask)
