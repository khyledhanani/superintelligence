"""Shared utilities for metrics modules."""

import numpy as np


def truncate_at_done(arr: np.ndarray, dones: np.ndarray) -> np.ndarray:
    """Return array up to (and including) the first done=True step."""
    done_idx = np.where(dones.astype(bool))[0]
    end = done_idx[0] + 1 if len(done_idx) > 0 else len(arr)
    return arr[:end]


def downsample(arr: np.ndarray, max_points: int = 30) -> np.ndarray:
    """Downsample array to at most max_points via uniform index selection."""
    if len(arr) <= max_points:
        return arr
    indices = np.linspace(0, len(arr) - 1, max_points, dtype=int)
    return arr[indices]


def format_vector(arr: np.ndarray, decimals: int = 3) -> str:
    """Format a 1D array as a compact string."""
    return "[" + ", ".join(f"{v:.{decimals}f}" for v in arr) + "]"
