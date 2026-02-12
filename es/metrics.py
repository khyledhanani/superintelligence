"""
Per-generation metrics tracking for ES environment evolution.

Tracks: regret, solvability rate, latent/sequence diversity, CMA-ES sigma,
agent return, and timing. Saves as .npy arrays + JSON summary.
"""

import jax.numpy as jnp
import numpy as np
import json
import os
import time
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Diversity metrics (JIT-friendly)
# ---------------------------------------------------------------------------

def compute_latent_diversity(population):
    """Mean pairwise L2 distance in latent space.

    Args:
        population: (pop_size, latent_dim) array.

    Returns:
        Scalar: mean pairwise L2 distance.
    """
    diff = population[:, None, :] - population[None, :, :]
    dists = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))
    n = population.shape[0]
    mask = jnp.triu(jnp.ones((n, n), dtype=jnp.bool_), k=1)
    return jnp.sum(dists * mask) / jnp.sum(mask)


def compute_sequence_diversity(sequences):
    """Mean pairwise Hamming distance (normalized by seq_len).

    Args:
        sequences: (pop_size, seq_len) integer array.

    Returns:
        Scalar: mean pairwise Hamming distance in [0, 1].
    """
    n = sequences.shape[0]
    seq_len = sequences.shape[1]
    hamming = (sequences[:, None, :] != sequences[None, :, :]).sum(axis=-1).astype(jnp.float32)
    mask = jnp.triu(jnp.ones((n, n), dtype=jnp.bool_), k=1)
    return jnp.sum(hamming * mask) / (jnp.sum(mask) * seq_len)


# ---------------------------------------------------------------------------
# Metrics tracker
# ---------------------------------------------------------------------------

@dataclass
class EvolutionMetrics:
    """Accumulate and save per-generation metrics."""

    best_regret: List[float] = field(default_factory=list)
    mean_regret: List[float] = field(default_factory=list)
    solvability_rate: List[float] = field(default_factory=list)
    unsolvable_rate: List[float] = field(default_factory=list)
    latent_diversity: List[float] = field(default_factory=list)
    sequence_diversity: List[float] = field(default_factory=list)
    cma_sigma: List[float] = field(default_factory=list)
    eval_return_mean: List[float] = field(default_factory=list)
    best_fitness: List[float] = field(default_factory=list)
    mean_fitness: List[float] = field(default_factory=list)
    gen_time_seconds: List[float] = field(default_factory=list)

    def record(self, gen_data: dict):
        """Record metrics for one generation."""
        for key in [
            'best_regret', 'mean_regret', 'solvability_rate', 'unsolvable_rate',
            'latent_diversity', 'sequence_diversity', 'cma_sigma',
            'eval_return_mean', 'best_fitness', 'mean_fitness', 'gen_time_seconds',
        ]:
            getattr(self, key).append(float(gen_data.get(key, 0.0)))

    def save(self, output_dir: str):
        """Save all metrics to output directory."""
        os.makedirs(output_dir, exist_ok=True)
        for name in [
            'best_regret', 'mean_regret', 'solvability_rate', 'unsolvable_rate',
            'latent_diversity', 'sequence_diversity', 'cma_sigma',
            'eval_return_mean', 'best_fitness', 'mean_fitness', 'gen_time_seconds',
        ]:
            arr = np.array(getattr(self, name))
            np.save(os.path.join(output_dir, f'{name}.npy'), arr)

        # JSON summary
        summary = {
            'total_generations': len(self.best_regret),
            'final_best_regret': self.best_regret[-1] if self.best_regret else 0,
            'final_mean_regret': self.mean_regret[-1] if self.mean_regret else 0,
            'final_solvability_rate': self.solvability_rate[-1] if self.solvability_rate else 0,
            'avg_gen_time_seconds': float(np.mean(self.gen_time_seconds)) if self.gen_time_seconds else 0,
        }
        with open(os.path.join(output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Metrics saved to {output_dir}/")


def compute_generation_metrics(info, population, sequences, es_state, gen_time):
    """Compute all metrics for a single generation.

    Args:
        info: Dict from regret_fitness with 'regret', 'solvable', 'max_returns', 'any_solvable'.
        population: (pop_size, latent_dim) latent vectors.
        sequences: (pop_size, 52) decoded sequences.
        es_state: CMA-ES state (has .std attribute).
        gen_time: Wall-clock seconds for this generation.

    Returns:
        Dict of metric name -> value.
    """
    regret = info['regret']
    solvable = info['solvable']
    max_returns = info['max_returns']
    any_solvable = info['any_solvable']

    solv_rate = float(solvable.mean())
    num_solvable = float(solvable.sum())

    # Regret stats (only over solvable envs)
    if any_solvable:
        solvable_regret = jnp.where(solvable, regret, -jnp.inf)
        best_reg = float(jnp.max(solvable_regret))
        mean_reg = float(jnp.where(solvable, regret, 0.0).sum() / jnp.maximum(num_solvable, 1))
        solvable_returns = jnp.where(solvable, max_returns, 0.0)
        return_mean = float(solvable_returns.sum() / jnp.maximum(num_solvable, 1))
    else:
        best_reg = 0.0
        mean_reg = 0.0
        return_mean = 0.0

    # CMA-ES sigma
    sigma = float(es_state.sigma) if hasattr(es_state, 'sigma') else (
        float(es_state.std) if hasattr(es_state, 'std') and es_state.std.ndim == 0
        else float(es_state.std.mean()) if hasattr(es_state, 'std') else 0.0
    )

    return {
        'best_regret': best_reg,
        'mean_regret': mean_reg,
        'solvability_rate': solv_rate,
        'unsolvable_rate': 1.0 - solv_rate,
        'latent_diversity': float(compute_latent_diversity(population)),
        'sequence_diversity': float(compute_sequence_diversity(sequences)),
        'cma_sigma': sigma,
        'eval_return_mean': return_mean,
        'gen_time_seconds': gen_time,
    }
