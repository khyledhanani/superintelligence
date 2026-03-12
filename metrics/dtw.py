"""Core DTW algorithm — shared by all pairwise DTW metrics."""

from typing import Tuple
import numpy as np


def dtw_with_path(seq_a: np.ndarray, seq_b: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute DTW distance between two sequences with full warping path.

    Args:
        seq_a: (T1, D) or (T1,) first sequence
        seq_b: (T2, D) or (T2,) second sequence

    Returns:
        distance: Normalized DTW distance (total cost / path length)
        path: (K, 2) array of (i, j) index pairs along the optimal warping path
        local_costs: (K,) array of pointwise costs along the warping path
    """
    if seq_a.ndim == 1:
        seq_a = seq_a[:, None]
    if seq_b.ndim == 1:
        seq_b = seq_b[:, None]

    n, m = len(seq_a), len(seq_b)
    cost = np.sum((seq_a[:, None, :] - seq_b[None, :, :]) ** 2, axis=-1)  # (n, m)
    cost = np.sqrt(cost)  # Euclidean distance

    # Accumulated cost matrix
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dtw[i, j] = cost[i - 1, j - 1] + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    # Backtrack to find optimal path
    path = []
    i, j = n, m
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            argmin = np.argmin([dtw[i - 1, j - 1], dtw[i - 1, j], dtw[i, j - 1]])
            if argmin == 0:
                i, j = i - 1, j - 1
            elif argmin == 1:
                i = i - 1
            else:
                j = j - 1
    path.reverse()
    path = np.array(path)  # (K, 2)

    local_costs = np.array([cost[p[0], p[1]] for p in path])
    total_cost = dtw[n, m]
    normalized_distance = total_cost / len(path)

    return normalized_distance, path, local_costs
