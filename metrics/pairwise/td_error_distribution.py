"""TD error distribution divergence — task-agnostic experiential diversity.

Compares the distribution of TD errors (δ_t = r_t + γV(s_{t+1}) - V(s_t))
between two trajectories using Earth Mover's Distance (Wasserstein-1).

This is the most task-agnostic pairwise diversity metric:
- Uses only values, rewards, and dones (any actor-critic has these)
- No thresholds, no mode taxonomy, no architecture knowledge
- TD error is the raw learning signal — different distributions mean
  different gradient directions (in aggregate)

EMD is bounded, interpretable ("cost to reshape one histogram into another"),
and well-defined even for very different episode lengths.
"""

import numpy as np
from typing import Dict, Optional

from metrics.utils import truncate_at_done


def compute_td_errors(
    values: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    gamma: float = 1.0,
) -> np.ndarray:
    """Compute per-step TD errors for the first episode.

    δ_t = r_t + γ * V(s_{t+1}) - V(s_t)

    For the terminal step, V(s_{t+1}) = 0.

    Args:
        values: (T,) value estimates V(s_t)
        rewards: (T,) rewards r_t
        dones: (T,) done flags
        gamma: discount factor

    Returns:
        (ep_len,) array of TD errors
    """
    ep_values = truncate_at_done(values, dones).astype(np.float64)
    ep_rewards = truncate_at_done(rewards, dones).astype(np.float64)
    ep_len = len(ep_values)

    if ep_len == 0:
        return np.array([])

    td_errors = np.zeros(ep_len, dtype=np.float64)
    for t in range(ep_len - 1):
        td_errors[t] = ep_rewards[t] + gamma * ep_values[t + 1] - ep_values[t]
    # Terminal step: next value is 0
    td_errors[ep_len - 1] = ep_rewards[ep_len - 1] - ep_values[ep_len - 1]

    return td_errors


def td_error_divergence(
    traj_a: dict,
    dones_a: np.ndarray,
    traj_b: dict,
    dones_b: np.ndarray,
    gamma: float = 1.0,
    n_bins: int = 50,
) -> Dict:
    """Compute distributional divergence of TD errors between two trajectories.

    Uses Wasserstein-1 (Earth Mover's Distance) on the TD error distributions.

    Args:
        traj_a: Trajectory dict with 'values', 'rewards' keys
        dones_a: Done flags for trajectory A
        traj_b: Trajectory dict with 'values', 'rewards' keys
        dones_b: Done flags for trajectory B
        gamma: Discount factor
        n_bins: Number of histogram bins (for visualization only; EMD uses raw samples)

    Returns:
        Dict with:
            emd: float — Earth Mover's Distance between TD error distributions
            td_errors_a: (ep_len_a,) TD errors for trajectory A
            td_errors_b: (ep_len_b,) TD errors for trajectory B
            mean_a, std_a: summary stats for A
            mean_b, std_b: summary stats for B
            histogram_a, histogram_b: (n_bins,) normalized histograms (for plotting)
            bin_edges: (n_bins+1,) shared bin edges
    """
    td_a = compute_td_errors(traj_a["values"], traj_a["rewards"], dones_a, gamma)
    td_b = compute_td_errors(traj_b["values"], traj_b["rewards"], dones_b, gamma)

    # Earth Mover's Distance on sorted samples (Wasserstein-1 for 1D)
    # W_1 = integral |F_a(x) - F_b(x)| dx, computed via sorted sample means
    if len(td_a) == 0 or len(td_b) == 0:
        emd = 0.0
    else:
        sorted_a = np.sort(td_a)
        sorted_b = np.sort(td_b)
        # Interpolate both to same number of quantile points
        n_points = max(len(sorted_a), len(sorted_b), 200)
        quantiles = np.linspace(0, 1, n_points)
        qa = np.quantile(sorted_a, quantiles)
        qb = np.quantile(sorted_b, quantiles)
        emd = float(np.mean(np.abs(qa - qb)))

    # Shared histogram for visualization
    all_td = np.concatenate([td_a, td_b]) if (len(td_a) > 0 and len(td_b) > 0) else np.array([0.0])
    bin_edges = np.linspace(all_td.min() - 0.01, all_td.max() + 0.01, n_bins + 1)
    hist_a = np.histogram(td_a, bins=bin_edges, density=True)[0] if len(td_a) > 0 else np.zeros(n_bins)
    hist_b = np.histogram(td_b, bins=bin_edges, density=True)[0] if len(td_b) > 0 else np.zeros(n_bins)

    return {
        "emd": emd,
        "td_errors_a": td_a,
        "td_errors_b": td_b,
        "mean_a": float(np.mean(td_a)) if len(td_a) > 0 else 0.0,
        "std_a": float(np.std(td_a)) if len(td_a) > 0 else 0.0,
        "mean_b": float(np.mean(td_b)) if len(td_b) > 0 else 0.0,
        "std_b": float(np.std(td_b)) if len(td_b) > 0 else 0.0,
        "histogram_a": hist_a,
        "histogram_b": hist_b,
        "bin_edges": bin_edges,
    }
