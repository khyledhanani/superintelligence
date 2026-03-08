"""
Trajectory cache for ACCEL replay buffer levels.

Stores per-level trajectory data (positions, values, observations, dones)
keyed by buffer index. Updated during training as levels are evaluated.
Entries are evicted when levels leave the buffer.
"""

import numpy as np
from typing import Dict, Optional, Set


class TrajectoryCache:
    """Cache storing first-episode trajectory data for replay buffer levels.

    Stores trajectory data truncated at first done=True to save memory.
    Keyed by buffer level index (int).
    """

    def __init__(self):
        self._cache: Dict[int, Dict[str, np.ndarray]] = {}

    def update(self, level_idx: int, observations: np.ndarray, positions: np.ndarray,
               values: np.ndarray, dones: np.ndarray, actions: np.ndarray):
        """Store or update trajectory data for a buffer level.

        Automatically truncates at the first done=True to store only first-episode data.

        Args:
            level_idx: Buffer index for this level
            observations: (T, *obs_shape) observations
            positions: (T, 2) agent positions
            values: (T,) value estimates
            dones: (T,) done flags
            actions: (T,) actions taken
        """
        # Truncate at first done
        done_indices = np.where(dones)[0]
        if len(done_indices) > 0:
            end = done_indices[0] + 1
            observations = observations[:end]
            positions = positions[:end]
            values = values[:end]
            dones = dones[:end]
            actions = actions[:end]

        self._cache[int(level_idx)] = {
            "observations": observations,
            "positions": positions,
            "values": values,
            "dones": dones,
            "actions": actions,
        }

    def get(self, level_idx: int) -> Optional[Dict[str, np.ndarray]]:
        """Get cached trajectory data for a buffer level.

        Returns:
            Dict with keys 'observations', 'positions', 'values', 'dones', 'actions',
            or None if not cached.
        """
        return self._cache.get(int(level_idx))

    def has(self, level_idx: int) -> bool:
        return int(level_idx) in self._cache

    def evict(self, level_idx: int):
        """Remove a level's trajectory data from the cache."""
        self._cache.pop(int(level_idx), None)

    def sync_with_buffer(self, active_indices: Set[int]):
        """Remove cached entries for levels no longer in the buffer.

        Args:
            active_indices: Set of buffer indices that are currently active.
        """
        stale = set(self._cache.keys()) - active_indices
        for idx in stale:
            del self._cache[idx]

    def update_batch(self, level_indices: np.ndarray, observations: np.ndarray,
                     positions: np.ndarray, values: np.ndarray, dones: np.ndarray,
                     actions: np.ndarray):
        """Update cache for a batch of levels.

        Args:
            level_indices: (N,) buffer indices
            observations: (T, N, *obs_shape) observations
            positions: (T, N, 2) agent positions
            values: (T, N) value estimates
            dones: (T, N) done flags
            actions: (T, N) actions
        """
        n = len(level_indices)
        for i in range(n):
            self.update(
                level_idx=int(level_indices[i]),
                observations=observations[:, i],
                positions=positions[:, i],
                values=values[:, i],
                dones=dones[:, i],
                actions=actions[:, i],
            )

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> list:
        """Sample n random cached trajectories for pairwise metric computation.

        Args:
            n: Number of trajectories to sample (capped at cache size)
            rng: Optional numpy random generator

        Returns:
            List of trajectory dicts, each with keys:
                'observations', 'positions', 'values', 'dones', 'actions', 'level_idx'
        """
        if rng is None:
            rng = np.random.default_rng()

        indices = list(self._cache.keys())
        n = min(n, len(indices))
        if n == 0:
            return []

        selected = rng.choice(indices, size=n, replace=False)
        result = []
        for idx in selected:
            entry = self._cache[idx].copy()
            entry["level_idx"] = idx
            result.append(entry)
        return result

    @property
    def size(self) -> int:
        return len(self._cache)

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, level_idx: int) -> bool:
        return int(level_idx) in self._cache
