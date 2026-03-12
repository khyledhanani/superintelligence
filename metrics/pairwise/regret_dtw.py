"""Regret Curve DTW — difficulty profile similarity between two levels.

Measures how similar the temporal shape of difficulty is across two levels.
Low distance = agent struggles at same relative points in both levels.
"""

from typing import Optional, Dict
import numpy as np

from metrics.dtw import dtw_with_path
from metrics.standalone.per_step_regret import compute_per_step_regret


def regret_curve_dtw(
    values_a: np.ndarray,
    rewards_a: np.ndarray,
    dones_a: np.ndarray,
    values_b: np.ndarray,
    rewards_b: np.ndarray,
    dones_b: np.ndarray,
    stored_max_return_a: Optional[float] = None,
    stored_max_return_b: Optional[float] = None,
) -> Dict:
    """Compute DTW distance between two per-step regret curves.

    Args:
        values_a, rewards_a, dones_a: Trajectory data for level A
        values_b, rewards_b, dones_b: Trajectory data for level B
        stored_max_return_a: Previously stored max_return for level A
        stored_max_return_b: Previously stored max_return for level B

    Returns:
        Dict with 'distance', 'path', 'local_costs'
    """
    info_a = compute_per_step_regret(values_a, rewards_a, dones_a,
                                     stored_max_return=stored_max_return_a)
    info_b = compute_per_step_regret(values_b, rewards_b, dones_b,
                                     stored_max_return=stored_max_return_b)

    curve_a = info_a["regret_curve"]
    curve_b = info_b["regret_curve"]

    if len(curve_a) == 0 or len(curve_b) == 0:
        return {
            "distance": 0.0,
            "path": np.array([]),
            "local_costs": np.array([]),
        }

    distance, path, local_costs = dtw_with_path(
        curve_a.reshape(-1, 1), curve_b.reshape(-1, 1)
    )

    return {
        "distance": distance,
        "path": path,
        "local_costs": local_costs,
    }
