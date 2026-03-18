"""Signed value error profile — where and how is the agent wrong?

Value error = V(s_t) - G_t (signed), where G_t is the actual return from step t.
Positive = agent overestimates (overconfident).
Negative = agent underestimates (underconfident, same direction as regret).

This is strictly more informative than unsigned regret: the sign reveals whether
the agent is confidently wrong (trap/dead-end) vs cautiously wrong (hidden shortcut).
"""

import numpy as np
from typing import Optional

from metrics.utils import truncate_at_done


def compute_value_error(
    values: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    gamma: float = 1.0,
) -> dict:
    """Compute signed per-step value error for the first episode.

    Args:
        values: (T,) value estimates V(s_t)
        rewards: (T,) rewards r_t
        dones: (T,) done flags
        gamma: discount factor (1.0 = undiscounted, matching ACCEL)

    Returns:
        Dict with:
            error_curve: (ep_len,) signed V(s_t) - G_t
            values: (ep_len,) truncated value estimates
            returns: (ep_len,) actual returns G_t from each step
            episode_length: int
            mean_error: float (positive = overconfident on average)
            mean_abs_error: float
            overconfident_frac: float (fraction of steps where V > G)
            max_overconfident: float (peak overestimation)
            max_overconfident_step: int
            max_underconfident: float (peak underestimation, as positive number)
            max_underconfident_step: int
    """
    ep_values = truncate_at_done(values, dones).astype(np.float64)
    ep_rewards = truncate_at_done(rewards, dones).astype(np.float64)
    ep_len = len(ep_values)

    if ep_len == 0:
        return {
            "error_curve": np.array([]),
            "values": np.array([]),
            "returns": np.array([]),
            "episode_length": 0,
            "mean_error": 0.0,
            "mean_abs_error": 0.0,
            "overconfident_frac": 0.0,
            "max_overconfident": 0.0,
            "max_overconfident_step": 0,
            "max_underconfident": 0.0,
            "max_underconfident_step": 0,
        }

    # Compute actual return G_t from each step
    returns = np.zeros(ep_len, dtype=np.float64)
    g = 0.0
    for t in range(ep_len - 1, -1, -1):
        g = ep_rewards[t] + gamma * g
        returns[t] = g

    # Signed error: positive = overconfident, negative = underconfident
    error_curve = ep_values - returns

    overconfident_mask = error_curve > 0
    overconfident_frac = float(np.mean(overconfident_mask)) if ep_len > 0 else 0.0

    # Peak overconfidence (max positive error)
    if np.any(overconfident_mask):
        max_over = float(np.max(error_curve))
        max_over_step = int(np.argmax(error_curve))
    else:
        max_over = 0.0
        max_over_step = 0

    # Peak underconfidence (max negative error, reported as positive magnitude)
    underconfident_mask = error_curve < 0
    if np.any(underconfident_mask):
        max_under = float(-np.min(error_curve))
        max_under_step = int(np.argmin(error_curve))
    else:
        max_under = 0.0
        max_under_step = 0

    return {
        "error_curve": error_curve,
        "values": ep_values,
        "returns": returns,
        "episode_length": ep_len,
        "mean_error": float(np.mean(error_curve)),
        "mean_abs_error": float(np.mean(np.abs(error_curve))),
        "overconfident_frac": overconfident_frac,
        "max_overconfident": max_over,
        "max_overconfident_step": max_over_step,
        "max_underconfident": max_under,
        "max_underconfident_step": max_under_step,
    }
