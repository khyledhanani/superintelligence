"""Buffer statistics extractor for live sampler state to ReferenceMaze conversion.

Provides two entry points:
  1. BufferStatsExtractor — extracts reference mazes from a live JAX sampler dict
     or a sampler dict built from a .npz buffer dump via npz_to_sampler().
  2. npz_to_sampler() — converts a buffer dump .npz file into a sampler dict
     compatible with BufferStatsExtractor.
"""

import logging
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from llm.prompt_builder import MetricEntry, ReferenceMaze

logger = logging.getLogger(__name__)


def npz_to_sampler(npz_path: str) -> dict:
    """Load a buffer dump .npz and return a sampler dict for BufferStatsExtractor.

    Args:
        npz_path: Path to .npz file with keys: tokens (N, 52), scores (N,), size (int).

    Returns:
        dict with keys: levels (batched Level pytree), scores (jnp array), size (int),
        and levels_extra if ancestor_id data is present.
    """
    from vae.vae_level_utils import tokens_to_level

    data = np.load(npz_path, allow_pickle=True)
    tokens = data["tokens"]
    scores = data["scores"]
    size = int(data["size"])

    if size == 0:
        raise ValueError(f"Buffer at {npz_path} is empty (size=0)")

    levels_list = []
    for i in range(size):
        levels_list.append(tokens_to_level(jnp.array(tokens[i])))

    levels_pytree = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs), *levels_list
    )

    sampler = {
        "levels": levels_pytree,
        "scores": jnp.array(scores[:size], dtype=jnp.float32),
        "size": size,
    }

    # Carry over levels_extra if present
    if "ancestor_ids" in data:
        ancestor_ids = data["ancestor_ids"][:size]
        sampler["levels_extra"] = {
            "ancestor_id": jnp.array(ancestor_ids, dtype=jnp.int32),
        }

    logger.info(f"[Buffer] Loaded {size} levels from {npz_path}")
    logger.info(f"[Buffer] Score range: [{scores[:size].min():.4f}, "
                f"{scores[:size].max():.4f}], mean={scores[:size].mean():.4f}")
    return sampler


def _kmedoids_select(dist_matrix: np.ndarray, n: int, weights: Optional[np.ndarray] = None) -> list:
    """Run PAM k-medoids on a precomputed distance matrix.

    Args:
        dist_matrix: (N, N) pairwise distance matrix
        n: number of medoids to select
        weights: optional (N,) per-point weights. If provided, objective is
            weighted sum of distances (density-weighted k-medoids).

    Returns:
        List of medoid indices into the distance matrix.
    """
    n_valid = len(dist_matrix)
    if n_valid <= n:
        return list(range(n_valid))

    w = weights if weights is not None else np.ones(n_valid)

    # PAM BUILD: greedy initialization
    weighted_total = (dist_matrix * w[np.newaxis, :]).sum(axis=1)
    medoids = [int(np.argmin(weighted_total))]

    nearest_medoid_dist = dist_matrix[medoids[0]].copy()
    for _ in range(1, n):
        gains = np.full(n_valid, -1.0)
        for c in range(n_valid):
            if c in medoids:
                continue
            improvement = np.maximum(0, nearest_medoid_dist - dist_matrix[c])
            gains[c] = (improvement * w).sum()
        best = int(np.argmax(gains))
        medoids.append(best)
        nearest_medoid_dist = np.minimum(nearest_medoid_dist, dist_matrix[best])

    # PAM SWAP: iterative improvement (up to 100 iterations)
    for iteration in range(100):
        medoid_dists = dist_matrix[medoids]
        assignments = np.argmin(medoid_dists, axis=0)

        improved = False
        for mi, m in enumerate(medoids):
            cluster_members = np.where(assignments == mi)[0]
            best_swap, best_delta = None, 0.0

            for candidate in cluster_members:
                if candidate == m:
                    continue
                delta = 0.0
                for j in range(n_valid):
                    old_d = dist_matrix[medoids[assignments[j]], j]
                    if assignments[j] == mi:
                        new_d = dist_matrix[candidate, j]
                        for mk in range(len(medoids)):
                            if mk != mi:
                                new_d = min(new_d, dist_matrix[medoids[mk], j])
                    else:
                        new_d = min(old_d, dist_matrix[candidate, j])
                    delta += (new_d - old_d) * w[j]

                if delta < best_delta:
                    best_delta = delta
                    best_swap = candidate

            if best_swap is not None:
                medoids[mi] = best_swap
                improved = True
                break

        if not improved:
            break

    return medoids


class BufferStatsExtractor:
    """Extracts reference mazes and summary statistics from a live PLR sampler.

    Converts the JAX array-backed sampler dict into Python-native ReferenceMaze
    objects by converting JAX arrays to numpy and using Level.to_str() for ASCII
    grid generation.

    Args:
        n_references: Number of reference mazes to select (default 5)
        strategy: Reference selection strategy. Currently supports:
            - "hardest" (alias "top_regret"): highest score mazes shown first
            - "random": random selection from active levels
            - "diverse": greedy max-min distance on 257D behavior embeddings
            - "kmedoid": k-medoids clustering on 257D behavior embeddings
            - "hybrid-kmedoid": density-weighted k-medoids (biased toward dense regions)
        buffer_embeddings: Optional (capacity, 257) numpy array of behavior embeddings.
            Required for diverse, kmedoid, and hybrid-kmedoid strategies.
    """

    def __init__(self, n_references: int = 5, strategy: str = "hardest",
                 density_radius_frac: float = 0.5,
                 hybrid_difficulty_percentile: float = 50.0) -> None:
        self.n_references = n_references
        self.strategy = strategy
        self.density_radius_frac = density_radius_frac
        self.hybrid_difficulty_percentile = hybrid_difficulty_percentile
        self._buffer_embeddings = None  # set by injector from training loop

    def extract_references_with_levels(self, sampler: dict) -> Tuple[List[ReferenceMaze], list]:
        """Return both ReferenceMaze objects (for prompt) and Level objects (for rollouts).

        Combines ReferenceMaze construction and raw Level extraction in a single pass
        to avoid iterating the sampler pytree twice. Used by Plan 02-02 injector to
        compute trajectories for the diversity gate.

        Args:
            sampler: PLR sampler dict from train_state.sampler, with keys:
                - "levels": batched Level pytree (capacity-first dimension)
                - "scores": float[capacity] regret scores
                - "size": int number of active levels

        Returns:
            (references, levels) tuple where:
                - references: List[ReferenceMaze] for prompt context
                - levels: list of raw Level pytree objects (one per selected index)
                  suitable for passing to AgentEvaluator.evaluate_levels()
        """
        size = int(np.asarray(sampler["size"]))
        if size == 0:
            return [], []

        scores = np.asarray(sampler["scores"])[:size]
        levels_pytree = sampler["levels"]

        # Select indices by strategy
        n = min(self.n_references, size)
        if self.strategy in ("hardest", "top_regret"):
            selected_indices = np.argsort(scores)[::-1][:n]
        elif self.strategy == "random":
            selected_indices = np.random.choice(size, n, replace=False)
        elif self.strategy == "diverse":
            selected_indices = self._select_diverse_indices(scores, size, n)
        elif self.strategy in ("kmedoid", "hybrid-kmedoid"):
            difficulty_mask = None
            if self.strategy == "hybrid-kmedoid":
                pct = self.hybrid_difficulty_percentile
                threshold = float(np.percentile(scores, pct))
                difficulty_mask = scores >= threshold
                n_pass = int(difficulty_mask.sum())
                logger.info(f"Hybrid difficulty filter (p{pct:.0f}): "
                            f"{n_pass}/{size} levels above {threshold:.4f}")
            selected_indices = self._select_kmedoid_indices(
                scores, size, n, difficulty_mask=difficulty_mask)
        else:
            raise ValueError(
                f"Unknown reference selection strategy: {self.strategy!r}. "
                "Expected 'hardest', 'top_regret', 'random', 'diverse', "
                "'kmedoid', or 'hybrid-kmedoid'."
            )

        references = []
        level_objects = []
        for i, idx in enumerate(selected_indices):
            # Extract single Level from batched pytree
            level = jax.tree_util.tree_map(lambda x: x[idx], levels_pytree)
            ascii_grid = level.to_str()

            metric = MetricEntry(
                name="Regret Score",
                value=float(scores[idx]),
                description="Agent's learning potential",
                higher_is="more to learn",
                metric_key="scalar_regret",
            )
            ref = ReferenceMaze(
                grid=ascii_grid,
                label=f"Maze {chr(65 + i)}",
                metrics=[metric],
            )
            references.append(ref)
            level_objects.append(level)

        return references, level_objects

    def _select_diverse_indices(self, scores: np.ndarray, size: int, n: int) -> np.ndarray:
        """Select reference indices via greedy max-min distance on embeddings."""
        emb = self._buffer_embeddings
        if emb is None:
            logger.warning("diverse strategy requires buffer embeddings; falling back to hardest")
            return np.argsort(scores)[::-1][:n]

        emb = emb[:size]
        norms = np.sqrt(np.sum(emb ** 2, axis=1))
        valid_mask = norms > 1e-6
        valid_indices = np.where(valid_mask)[0]
        n_valid = len(valid_indices)
        logger.info(f"Diverse selection: {n_valid} candidate levels from {size} active")

        if n_valid <= n:
            return valid_indices

        # L2 pairwise distance matrix
        valid_emb = emb[valid_indices]
        dist_matrix = np.zeros((n_valid, n_valid), dtype=np.float32)
        chunk_size = max(1, min(500, n_valid))
        for start in range(0, n_valid, chunk_size):
            end = min(start + chunk_size, n_valid)
            diff = valid_emb[start:end, np.newaxis, :] - valid_emb[np.newaxis, :, :]
            dist_matrix[start:end] = np.sqrt(np.sum(diff ** 2, axis=2))

        # Greedy max-min: start with the pair having largest distance
        flat_max = int(np.argmax(dist_matrix))
        i0, i1 = divmod(flat_max, n_valid)
        selected = [i0, i1]

        # Track minimum distance from each point to any selected point
        min_dists = np.minimum(dist_matrix[i0], dist_matrix[i1])

        while len(selected) < n:
            # Zero out already-selected points
            min_dists[selected[-1]] = -1.0
            next_idx = int(np.argmax(min_dists))
            selected.append(next_idx)
            min_dists = np.minimum(min_dists, dist_matrix[next_idx])

        result = valid_indices[selected]
        for i, s in enumerate(selected):
            idx = int(result[i])
            min_d = float(min(dist_matrix[s, o] for o in selected if o != s))
            logger.info(f"  Diverse {i+1}: idx={idx}, score={scores[idx]:.4f}, min_dist={min_d:.4f}")

        return result

    def _select_kmedoid_indices(self, scores: np.ndarray, size: int, n: int,
                                difficulty_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Select reference indices via k-medoids on buffer embeddings."""
        emb = self._buffer_embeddings
        if emb is None:
            logger.warning("kmedoid strategy requires buffer embeddings; falling back to hardest")
            return np.argsort(scores)[::-1][:n]

        emb = emb[:size]
        norms = np.sqrt(np.sum(emb ** 2, axis=1))
        valid_mask = norms > 1e-6
        if difficulty_mask is not None:
            valid_mask &= difficulty_mask
        valid_indices = np.where(valid_mask)[0]
        n_valid = len(valid_indices)
        logger.info(f"K-medoids selection: {n_valid} candidate levels from {size} active")

        if n_valid <= n:
            return valid_indices

        # L2 pairwise distance matrix
        valid_emb = emb[valid_indices]
        dist_matrix = np.zeros((n_valid, n_valid), dtype=np.float32)
        chunk_size = max(1, min(500, n_valid))
        for start in range(0, n_valid, chunk_size):
            end = min(start + chunk_size, n_valid)
            diff = valid_emb[start:end, np.newaxis, :] - valid_emb[np.newaxis, :, :]
            dist_matrix[start:end] = np.sqrt(np.sum(diff ** 2, axis=2))

        # Density weights for hybrid-kmedoid
        weights = None
        if self.strategy == "hybrid-kmedoid":
            median_dist = float(np.median(dist_matrix[np.triu_indices(n_valid, k=1)]))
            radius = median_dist * self.density_radius_frac
            weights = np.sum(dist_matrix < radius, axis=1).astype(np.float64)
            weights = weights / weights.mean()
            logger.info(f"  Density weights: radius={radius:.4f}, "
                        f"min={weights.min():.2f}, max={weights.max():.2f}")

        medoid_local = _kmedoids_select(dist_matrix, n, weights=weights)
        selected = valid_indices[medoid_local]

        for i, m in enumerate(medoid_local):
            idx = int(selected[i])
            min_d = float(min(dist_matrix[m, o] for o in medoid_local if o != m))
            logger.info(f"  Medoid {i+1}: idx={idx}, score={scores[idx]:.4f}, min_dist={min_d:.4f}")

        return selected

    def extract_references(self, sampler: dict) -> List[ReferenceMaze]:
        """Convert live sampler state to a list of ReferenceMaze objects.

        Reads the live PLR buffer (JAX pytree) and returns the top-N reference
        mazes according to the configured selection strategy. All JAX arrays are
        converted to numpy before Python operations.

        Delegates to extract_references_with_levels() for DRY selection logic.

        Args:
            sampler: PLR sampler dict from train_state.sampler, with keys:
                - "levels": batched Level pytree (capacity-first dimension)
                - "scores": float[capacity] regret scores
                - "size": int number of active levels

        Returns:
            List of ReferenceMaze objects, one per selected reference level.
            Each has label "Maze A", "Maze B", etc. and a RegretScore metric.
        """
        references, _ = self.extract_references_with_levels(sampler)
        return references

    def extract_buffer_summary(self, sampler: dict) -> dict:
        """Return summary statistics for the active PLR buffer.

        Useful for WandB logging and global_metrics in prompt building.

        Args:
            sampler: PLR sampler dict from train_state.sampler

        Returns:
            dict with keys:
                - buffer_size: int, number of active levels
                - mean_score: float
                - max_score: float
                - min_score: float
                - score_std: float
        """
        size = int(np.asarray(sampler["size"]))
        if size == 0:
            return {
                "buffer_size": 0,
                "mean_score": 0.0,
                "max_score": 0.0,
                "min_score": 0.0,
                "score_std": 0.0,
            }

        scores = np.asarray(sampler["scores"])[:size]
        return {
            "buffer_size": size,
            "mean_score": float(np.mean(scores)),
            "max_score": float(np.max(scores)),
            "min_score": float(np.min(scores)),
            "score_std": float(np.std(scores)),
        }

    @staticmethod
    def extract_global_metrics(buffer_summary: dict) -> List[MetricEntry]:
        """Convert buffer summary stats to MetricEntry list for prompt context.

        Converts the dict returned by extract_buffer_summary() into a structured
        list of MetricEntry objects for use in prompt_builder global_metrics.

        Args:
            buffer_summary: dict with keys buffer_size, mean_score, max_score,
                min_score, score_std (as returned by extract_buffer_summary())

        Returns:
            List of MetricEntry objects representing global training state.
        """
        return [
            MetricEntry(
                name="Buffer Mean Regret",
                value=buffer_summary["mean_score"],
                description="Mean regret score across all active buffer levels",
                higher_is="more challenging curriculum",
                metric_key="scalar_regret",
            ),
            MetricEntry(
                name="Buffer Max Regret",
                value=buffer_summary["max_score"],
                description="Highest regret score in buffer (hardest level for agent)",
                higher_is="harder top level",
                metric_key="scalar_regret",
            ),
            MetricEntry(
                name="Buffer Size",
                value=buffer_summary["buffer_size"],
                description="Number of active levels in the PLR replay buffer",
                metric_key="",
            ),
        ]
