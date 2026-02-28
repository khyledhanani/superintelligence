"""Composite fitness function F = alpha * regret + beta * novelty.

Sign convention:
    - regret:  raw positive MaxMC regret (NOT negated). Higher = harder level.
    - novelty: k-NN novelty score. Higher = more diverse.
    - F:       higher is better for curriculum quality.
    - Callers MUST negate F before passing to evosax (which minimizes).

Weights alpha and beta live in the ES config dict (Python-side), not in JAX state.
No normalization: F = alpha * regret + beta * novelty (raw combination).
"""

from __future__ import annotations
import jax.numpy as jnp


def compute_fitness(
    regret: jnp.ndarray,
    novelty: jnp.ndarray,
    alpha: float,
    beta: float,
) -> jnp.ndarray:
    """Compute composite fitness for a single candidate (or batch of scalars).

    F = alpha * regret + beta * novelty

    Args:
        regret:  scalar float32 -- raw MaxMC regret (positive; higher = harder)
        novelty: scalar float32 -- k-NN novelty score (positive; higher = more diverse)
        alpha:   float -- weight for regret component (from ES config, static per run)
        beta:    float -- weight for novelty component (from ES config, static per run)

    Returns:
        F: scalar float32 composite fitness (higher = better curriculum quality).
           Caller must negate before passing to evosax (which minimizes).
    """
    return alpha * regret + beta * novelty


def compute_fitness_batch(
    regrets: jnp.ndarray,
    novelties: jnp.ndarray,
    alpha: float,
    beta: float,
) -> jnp.ndarray:
    """Compute composite fitness for a batch of candidates.

    Args:
        regrets:   (pop_size,) float32 -- raw MaxMC regrets (positive)
        novelties: (pop_size,) float32 -- k-NN novelty scores (positive)
        alpha:     float -- regret weight
        beta:      float -- novelty weight

    Returns:
        fitness: (pop_size,) float32 composite fitness scores.
                 Caller must negate before passing to evosax.

    Example:
        composite = compute_fitness_batch(regrets, novelties, alpha=0.8, beta=0.2)
        fitness_for_evosax = -composite  # evosax minimizes
    """
    return alpha * regrets + beta * novelties
