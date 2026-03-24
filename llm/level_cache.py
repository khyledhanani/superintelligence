"""Disk cache for accepted LLM-generated maze levels.

Provides an audit trail so reviewers can reconstruct which levels were injected
into which run. Each accepted level is saved as:
  - A .npy file: the wall_map as a bool array
  - A JSON sidecar: metadata including SHA-256 hash, injection step, gate scores, timestamp

File naming convention: step_{step:05d}_idx_{idx:03d}.{ext}
  e.g.  step_05000_idx_003.npy  /  step_05000_idx_003.json

Only LLM-generated levels that passed the gate are cached. Rejected levels
and mutation-amplified variants are NOT cached.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class LevelCache:
    """Saves accepted LLM-generated maze levels to disk for reproducibility.

    Args:
        run_dir: Directory for this run's cached levels
                 (e.g. "results/<run_name>/llm_levels/<seed>").
                 Created on init via Path.mkdir(parents=True, exist_ok=True).
    """

    def __init__(self, run_dir: str) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def save_accepted(
        self,
        level,
        step: int,
        idx: int,
        gate_metrics: dict,
    ) -> str:
        """Save an accepted LLM level to disk and return its wall_map hash.

        Writes two files:
          - <run_dir>/step_{step:05d}_idx_{idx:03d}.npy  — wall_map as bool array
          - <run_dir>/step_{step:05d}_idx_{idx:03d}.json — JSON sidecar with metadata

        Args:
            level:        Level object with a wall_map attribute (JAX/numpy array)
            step:         Training step at which injection occurred
            idx:          Batch index of this level within the injection event
            gate_metrics: Dict with gate scores (td_error_emd, solve_rate etc.)
                          Pass empty dict for ungated code path.

        Returns:
            SHA-256 hash of the wall_map as a 16-character hex string.
        """
        wall_map_np = np.asarray(level.wall_map, dtype=bool)

        # Compute hash from deterministic bool representation
        wall_map_hash = hashlib.sha256(wall_map_np.tobytes()).hexdigest()[:16]

        # File base name
        stem = f"step_{step:05d}_idx_{idx:03d}"

        # Save .npy
        npy_path = self.run_dir / f"{stem}.npy"
        np.save(str(npy_path), wall_map_np)

        # Build sidecar metadata
        sidecar = {
            "wall_map_hash": wall_map_hash,
            "injection_step": step,
            "batch_index": idx,
            "gate_scores": {
                "td_error_emd": gate_metrics.get("mean_diversity", None),
                "solve_rate": gate_metrics.get("regret", None),
            },
            "accept": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Save .json sidecar
        json_path = self.run_dir / f"{stem}.json"
        json_path.write_text(json.dumps(sidecar, indent=2))

        return wall_map_hash

    @staticmethod
    def compute_hash(level) -> str:
        """Compute the SHA-256 hash of a level's wall_map without saving to disk.

        Useful for WandB logging when the cache is disabled.

        Args:
            level: Level object with a wall_map attribute

        Returns:
            SHA-256 hash as a 16-character hex string (same algorithm as save_accepted).
        """
        wall_map_np = np.asarray(level.wall_map, dtype=bool)
        return hashlib.sha256(wall_map_np.tobytes()).hexdigest()[:16]
