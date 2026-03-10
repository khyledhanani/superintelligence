"""
Trajectory-based diversity metrics for ACCEL replay buffer analysis.

Four metrics that measure diversity from the agent's perspective:
1. Observation Sequence DTW - ground truth of agent experience
2. Position Trace DTW - navigation path structure
3. Value Trajectory DTW - difficulty fingerprint
4. Spatial Footprint Jaccard - coarse set-based measure
"""

import numpy as np
from typing import Tuple, Dict


def dtw_with_path(seq_a: np.ndarray, seq_b: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute DTW distance between two sequences with full warping path.

    Args:
        seq_a: (T1, D) or (T1,) first sequence
        seq_b: (T2, D) or (T2,) second sequence

    Returns:
        distance: Normalized DTW distance (total cost / path length)
        path: (K, 2) array of (i, j) index pairs along the optimal warping path
        local_costs: (K,) array of pointwise costs along the warping path (similarity profile)
    """
    if seq_a.ndim == 1:
        seq_a = seq_a[:, None]
    if seq_b.ndim == 1:
        seq_b = seq_b[:, None]

    n, m = len(seq_a), len(seq_b)
    # Cost matrix: squared Euclidean distance between all pairs
    # For large D, this is efficient via broadcasting
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

    # Local costs along the warping path (similarity profile)
    local_costs = np.array([cost[p[0], p[1]] for p in path])

    total_cost = dtw[n, m]
    normalized_distance = total_cost / len(path)

    return normalized_distance, path, local_costs


def truncate_at_first_done(data: np.ndarray, dones: np.ndarray) -> np.ndarray:
    """Truncate trajectory data at the first done=True (inclusive).

    Args:
        data: (T, ...) trajectory data
        dones: (T,) boolean done flags

    Returns:
        Truncated data up to and including the first done step.
        If no done found, returns the full trajectory.
    """
    done_indices = np.where(dones)[0]
    if len(done_indices) == 0:
        return data
    return data[:done_indices[0] + 1]


# ============================================================
# Metric 1: Observation Sequence DTW
# ============================================================

def observation_dtw(obs_a: np.ndarray, dones_a: np.ndarray,
                    obs_b: np.ndarray, dones_b: np.ndarray) -> Dict:
    """Compute DTW distance between two observation sequences.

    Observations are flattened to 1D vectors before comparison.
    Sequences are truncated at first episode completion.

    Args:
        obs_a: (T1, *obs_shape) observations for trajectory A
        dones_a: (T1,) done flags for trajectory A
        obs_b: (T2, *obs_shape) observations for trajectory B
        dones_b: (T2,) done flags for trajectory B

    Returns:
        Dict with 'distance' (normalized scalar), 'path', 'local_costs'
    """
    obs_a = truncate_at_first_done(obs_a, dones_a)
    obs_b = truncate_at_first_done(obs_b, dones_b)

    # Flatten obs to (T, D) where D = product of obs_shape
    flat_a = obs_a.reshape(len(obs_a), -1).astype(np.float32)
    flat_b = obs_b.reshape(len(obs_b), -1).astype(np.float32)

    distance, path, local_costs = dtw_with_path(flat_a, flat_b)
    return {
        "distance": distance,
        "path": path,
        "local_costs": local_costs,
    }


# ============================================================
# Metric 2: Position Trace DTW
# ============================================================

def position_trace_dtw(pos_a: np.ndarray, dones_a: np.ndarray,
                       pos_b: np.ndarray, dones_b: np.ndarray) -> Dict:
    """Compute DTW distance between two position traces.

    Args:
        pos_a: (T1, 2) agent positions for trajectory A
        dones_a: (T1,) done flags for trajectory A
        pos_b: (T2, 2) agent positions for trajectory B
        dones_b: (T2,) done flags for trajectory B

    Returns:
        Dict with 'distance' (normalized scalar), 'path', 'local_costs'
    """
    pos_a = truncate_at_first_done(pos_a, dones_a).astype(np.float32)
    pos_b = truncate_at_first_done(pos_b, dones_b).astype(np.float32)

    # Relative to start position (translation invariant)
    pos_a = pos_a - pos_a[0]
    pos_b = pos_b - pos_b[0]

    distance, path, local_costs = dtw_with_path(pos_a, pos_b)
    return {
        "distance": distance,
        "path": path,
        "local_costs": local_costs,
    }


# ============================================================
# Metric 3: Value Trajectory DTW
# ============================================================

def value_trajectory_dtw(values_a: np.ndarray, dones_a: np.ndarray,
                         values_b: np.ndarray, dones_b: np.ndarray) -> Dict:
    """Compare two value trajectories using DTW.

    DTW captures shape, magnitude, and timing differences with a local cost
    profile that an LLM can reason over directly.

    Args:
        values_a: (T1,) value estimates for trajectory A
        dones_a: (T1,) done flags for trajectory A
        values_b: (T2,) value estimates for trajectory B
        dones_b: (T2,) done flags for trajectory B

    Returns:
        Dict with 'distance' (normalized DTW), 'path', 'local_costs'
    """
    values_a = truncate_at_first_done(values_a, dones_a).astype(np.float64)
    values_b = truncate_at_first_done(values_b, dones_b).astype(np.float64)

    distance, path, local_costs = dtw_with_path(
        values_a.reshape(-1, 1), values_b.reshape(-1, 1)
    )

    return {
        "distance": distance,
        "path": path,
        "local_costs": local_costs,
    }


# ============================================================
# Metric 4: Spatial Footprint Jaccard
# ============================================================

def spatial_footprint_jaccard(pos_a: np.ndarray, dones_a: np.ndarray,
                              pos_b: np.ndarray, dones_b: np.ndarray) -> Dict:
    """Compute Jaccard index of visited cell sets.

    Args:
        pos_a: (T1, 2) agent positions for trajectory A
        dones_a: (T1,) done flags for trajectory A
        pos_b: (T2, 2) agent positions for trajectory B
        dones_b: (T2,) done flags for trajectory B

    Returns:
        Dict with 'jaccard' (float in [0, 1]), 'cells_a' (set), 'cells_b' (set),
        'intersection_size', 'union_size'
    """
    pos_a = truncate_at_first_done(pos_a, dones_a)
    pos_b = truncate_at_first_done(pos_b, dones_b)

    cells_a = set(map(tuple, pos_a.tolist()))
    cells_b = set(map(tuple, pos_b.tolist()))

    intersection = cells_a & cells_b
    union = cells_a | cells_b

    jaccard = len(intersection) / len(union) if len(union) > 0 else 1.0

    return {
        "jaccard": jaccard,
        "cells_a": cells_a,
        "cells_b": cells_b,
        "intersection_size": len(intersection),
        "union_size": len(union),
    }


# ============================================================
# Pairwise computation helpers
# ============================================================

def compute_pairwise_metrics(trajectories: list) -> Dict:
    """Compute all pairwise metrics for a list of trajectory data.

    Args:
        trajectories: List of dicts, each with keys:
            'observations': (T, *obs_shape)
            'positions': (T, 2)
            'values': (T,)
            'dones': (T,)

    Returns:
        Dict with arrays of pairwise metric values:
            'obs_dtw_distances': (N*(N-1)/2,)
            'pos_dtw_distances': (N*(N-1)/2,)
            'value_dtw_distances': (N*(N-1)/2,)
            'jaccard_indices': (N*(N-1)/2,)
    """
    n = len(trajectories)
    obs_dtw_dists = []
    pos_dtw_dists = []
    value_dtw_dists = []
    jaccards = []

    for i in range(n):
        for j in range(i + 1, n):
            ti, tj = trajectories[i], trajectories[j]

            obs_result = observation_dtw(
                ti["observations"], ti["dones"],
                tj["observations"], tj["dones"],
            )
            obs_dtw_dists.append(obs_result["distance"])

            pos_result = position_trace_dtw(
                ti["positions"], ti["dones"],
                tj["positions"], tj["dones"],
            )
            pos_dtw_dists.append(pos_result["distance"])

            val_result = value_trajectory_dtw(
                ti["values"], ti["dones"],
                tj["values"], tj["dones"],
            )
            value_dtw_dists.append(val_result["distance"])

            jac_result = spatial_footprint_jaccard(
                ti["positions"], ti["dones"],
                tj["positions"], tj["dones"],
            )
            jaccards.append(jac_result["jaccard"])

    return {
        "obs_dtw_distances": np.array(obs_dtw_dists),
        "pos_dtw_distances": np.array(pos_dtw_dists),
        "value_dtw_distances": np.array(value_dtw_dists),
        "jaccard_indices": np.array(jaccards),
    }
