"""Buffer statistics extractor for live sampler state to ReferenceMaze conversion.

Converts a live train_state.sampler JAX dict into ReferenceMaze objects suitable
for LLM prompt context. This replaces the .npz-based flow in test_generator.py
with a live-buffer flow that reads directly from the training state.
"""

from typing import List

import jax
import numpy as np

from llm.prompt_builder import MetricEntry, ReferenceMaze


class BufferStatsExtractor:
    """Extracts reference mazes and summary statistics from a live PLR sampler.

    Converts the JAX array-backed sampler dict into Python-native ReferenceMaze
    objects by converting JAX arrays to numpy and using Level.to_str() for ASCII
    grid generation.

    Args:
        n_references: Number of reference mazes to select (default 5)
        strategy: Reference selection strategy. Currently supports:
            - "hardest": top-regret, highest score mazes shown first
            - "random": random selection from active levels
            - "diverse": spread selection across score range (quartile-based)
    """

    def __init__(self, n_references: int = 5, strategy: str = "hardest") -> None:
        self.n_references = n_references
        self.strategy = strategy

    def extract_references(self, sampler: dict) -> List[ReferenceMaze]:
        """Convert live sampler state to a list of ReferenceMaze objects.

        Reads the live PLR buffer (JAX pytree) and returns the top-N reference
        mazes according to the configured selection strategy. All JAX arrays are
        converted to numpy before Python operations.

        Args:
            sampler: PLR sampler dict from train_state.sampler, with keys:
                - "levels": batched Level pytree (capacity-first dimension)
                - "scores": float[capacity] regret scores
                - "size": int number of active levels

        Returns:
            List of ReferenceMaze objects, one per selected reference level.
            Each has label "Maze A", "Maze B", etc. and a RegretScore metric.
        """
        size = int(np.asarray(sampler["size"]))
        if size == 0:
            return []

        scores = np.asarray(sampler["scores"])[:size]
        levels = sampler["levels"]

        # Select indices by strategy
        n = min(self.n_references, size)
        if self.strategy == "hardest":
            selected_indices = np.argsort(scores)[::-1][:n]
        elif self.strategy == "random":
            selected_indices = np.random.choice(size, n, replace=False)
        elif self.strategy == "diverse":
            sorted_idx = np.argsort(scores)
            step = max(1, len(sorted_idx) // n)
            selected_indices = sorted_idx[::step][:n]
        else:
            raise ValueError(
                f"Unknown reference selection strategy: {self.strategy!r}. "
                "Expected 'hardest', 'random', or 'diverse'."
            )

        references = []
        for i, idx in enumerate(selected_indices):
            # Extract single Level from batched pytree
            level = jax.tree_util.tree_map(lambda x: x[idx], levels)
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
