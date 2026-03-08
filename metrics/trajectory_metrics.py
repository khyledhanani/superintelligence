"""
Trajectory-based diversity metrics for ACCEL replay buffer analysis.

Four metrics that measure diversity from the agent's perspective:
1. Observation Sequence DTW - ground truth of agent experience
2. Position Trace DTW - navigation path structure
3. Value Trajectory Correlation - difficulty fingerprint
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

    distance, path, local_costs = dtw_with_path(pos_a, pos_b)
    return {
        "distance": distance,
        "path": path,
        "local_costs": local_costs,
    }


# ============================================================
# Metric 3: Value Trajectory Correlation
# ============================================================

def value_trajectory_correlation(values_a: np.ndarray, dones_a: np.ndarray,
                                 values_b: np.ndarray, dones_b: np.ndarray,
                                 num_points: int = 100) -> Dict:
    """Compute correlation between two value trajectories.

    Resamples both value curves to a fixed number of points via linear
    interpolation, then computes Pearson correlation and L2 distance.

    Args:
        values_a: (T1,) value estimates for trajectory A
        dones_a: (T1,) done flags for trajectory A
        values_b: (T2,) value estimates for trajectory B
        dones_b: (T2,) done flags for trajectory B
        num_points: Number of interpolation points (default 100)

    Returns:
        Dict with 'correlation' (Pearson r), 'l2_distance', 'resampled_a', 'resampled_b'
    """
    values_a = truncate_at_first_done(values_a, dones_a).astype(np.float64)
    values_b = truncate_at_first_done(values_b, dones_b).astype(np.float64)

    # Resample to fixed length
    t_uniform = np.linspace(0, 1, num_points)
    t_a = np.linspace(0, 1, len(values_a))
    t_b = np.linspace(0, 1, len(values_b))
    resampled_a = np.interp(t_uniform, t_a, values_a)
    resampled_b = np.interp(t_uniform, t_b, values_b)

    # Pearson correlation
    std_a = np.std(resampled_a)
    std_b = np.std(resampled_b)
    if std_a < 1e-8 or std_b < 1e-8:
        # Constant value trajectory — correlation undefined
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(resampled_a, resampled_b)[0, 1])

    # L2 distance
    l2_distance = float(np.sqrt(np.mean((resampled_a - resampled_b) ** 2)))

    return {
        "correlation": correlation,
        "l2_distance": l2_distance,
        "resampled_a": resampled_a,
        "resampled_b": resampled_b,
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
            'value_correlations': (N*(N-1)/2,)
            'value_l2_distances': (N*(N-1)/2,)
            'jaccard_indices': (N*(N-1)/2,)
    """
    n = len(trajectories)
    obs_dtw_dists = []
    pos_dtw_dists = []
    value_corrs = []
    value_l2s = []
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

            val_result = value_trajectory_correlation(
                ti["values"], ti["dones"],
                tj["values"], tj["dones"],
            )
            value_corrs.append(val_result["correlation"])
            value_l2s.append(val_result["l2_distance"])

            jac_result = spatial_footprint_jaccard(
                ti["positions"], ti["dones"],
                tj["positions"], tj["dones"],
            )
            jaccards.append(jac_result["jaccard"])

    return {
        "obs_dtw_distances": np.array(obs_dtw_dists),
        "pos_dtw_distances": np.array(pos_dtw_dists),
        "value_correlations": np.array(value_corrs),
        "value_l2_distances": np.array(value_l2s),
        "jaccard_indices": np.array(jaccards),
    }
