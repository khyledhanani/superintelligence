"""
MAP-Elites archive for diverse environment storage.

Maintains a 2D grid of elite environments binned by behavior descriptors
(num_obstacles, manhattan_distance). Extracted from the legacy es/map_elites.py
standalone runner — this module contains only the live archive infrastructure
used by train.py and ued_interface.py.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Behavior descriptor bins
# ---------------------------------------------------------------------------

OBS_BINS = np.array([5, 10, 15, 20, 25, 30, 35, 40, 50])   # 8 bins
DIST_BINS = np.array([3, 6, 9, 12, 15, 18, 24])             # 6 bins
N_OBS_BINS = len(OBS_BINS) - 1   # 8
N_DIST_BINS = len(DIST_BINS) - 1  # 6


def compute_behavior_descriptors(sequences):
    """Compute (num_obstacles, manhattan_dist) for a batch of sequences.

    Args:
        sequences: (batch, 52) integer arrays.

    Returns:
        obs_counts: (batch,) int array
        dists: (batch,) int array
    """
    seqs_np = np.asarray(sequences)
    obs_counts = (seqs_np[:, :50] > 0).sum(axis=1)
    goal_idx = seqs_np[:, 50]
    agent_idx = seqs_np[:, 51]
    goal_row = (goal_idx - 1) // 13
    goal_col = (goal_idx - 1) % 13
    agent_row = (agent_idx - 1) // 13
    agent_col = (agent_idx - 1) % 13
    dists = np.abs(goal_row - agent_row) + np.abs(goal_col - agent_col)
    return obs_counts, dists


def descriptor_to_cell(obs_count, dist):
    """Map a single (obs_count, dist) to archive grid indices.

    Returns (row, col) or None if out of bounds.
    """
    row = np.searchsorted(OBS_BINS, obs_count, side='right') - 1
    col = np.searchsorted(DIST_BINS, dist, side='right') - 1
    if 0 <= row < N_OBS_BINS and 0 <= col < N_DIST_BINS:
        return (row, col)
    return None


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------

class Archive:
    """2D MAP-Elites archive storing elite environments per behavior cell."""

    def __init__(self, latent_dim=64):
        self.latent_dim = latent_dim
        self.grid = {}  # (row, col) -> {latent, sequence, regret}
        self.total_attempts = 0
        self.total_insertions = 0

    def try_insert(self, latent, sequence, regret, obs_count, dist):
        """Attempt to insert a solution. Returns True if inserted."""
        cell = descriptor_to_cell(obs_count, dist)
        if cell is None:
            return False

        self.total_attempts += 1
        existing = self.grid.get(cell)
        if existing is None or regret > existing['regret']:
            self.grid[cell] = {
                'latent': np.array(latent),
                'sequence': np.array(sequence),
                'regret': float(regret),
                'obs_count': int(obs_count),
                'dist': int(dist),
            }
            self.total_insertions += 1
            return True
        return False

    @property
    def num_filled(self):
        return len(self.grid)

    @property
    def total_cells(self):
        return N_OBS_BINS * N_DIST_BINS

    @property
    def coverage(self):
        return self.num_filled / self.total_cells

    def sample_parents(self, rng, n):
        """Sample n parent latent vectors from occupied cells (uniform)."""
        cells = list(self.grid.values())
        if not cells:
            return None
        indices = rng.integers(0, len(cells), size=n)
        return np.stack([cells[i]['latent'] for i in indices])

    def get_arrays(self):
        """Extract archive contents as numpy arrays."""
        if not self.grid:
            return None
        items = list(self.grid.values())
        return {
            'sequences': np.stack([it['sequence'] for it in items]),
            'latents': np.stack([it['latent'] for it in items]),
            'regrets': np.array([it['regret'] for it in items]),
            'descriptors': np.array([[it['obs_count'], it['dist']] for it in items]),
        }

    def regret_heatmap(self):
        """Build a 2D array of regret values for visualization."""
        hmap = np.full((N_OBS_BINS, N_DIST_BINS), np.nan)
        for (r, c), entry in self.grid.items():
            hmap[r, c] = entry['regret']
        return hmap
