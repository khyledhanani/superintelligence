"""Action Sequence Alignment — behavioral similarity between two levels.

Uses DTW with binary mismatch cost (0 = same action, 1 = different action)
instead of Euclidean distance, because actions are categorical labels, not
ordinal values. The normalized distance is the fraction of aligned steps
where the agent chose a different action.

Low distance = agent behaves the same way = functionally identical experience.
High distance = different behavioral fingerprints = diverse experience.
"""

from typing import Dict
import numpy as np

from metrics.utils import truncate_at_done


def action_sequence_distance(
    actions_a: np.ndarray,
    dones_a: np.ndarray,
    actions_b: np.ndarray,
    dones_b: np.ndarray,
) -> Dict:
    """Compute DTW alignment distance between two action sequences using mismatch cost.

    Cost function: 0 if actions match, 1 if they differ. This treats actions
    as categorical labels rather than ordinal values.

    The normalized distance is in [0, 1]: fraction of aligned steps with
    different actions. 0 = identical behavior, 1 = completely different.

    Args:
        actions_a: (T1,) actions for trajectory A
        dones_a: (T1,) done flags for trajectory A
        actions_b: (T2,) actions for trajectory B
        dones_b: (T2,) done flags for trajectory B

    Returns:
        Dict with:
            'distance': normalized mismatch fraction in [0, 1]
            'path': (K, 2) warping path indices
            'local_costs': (K,) binary costs along warping path (0 or 1)
    """
    act_a = truncate_at_done(actions_a, dones_a).astype(np.int32)
    act_b = truncate_at_done(actions_b, dones_b).astype(np.int32)

    if len(act_a) == 0 or len(act_b) == 0:
        return {
            "distance": 0.0,
            "path": np.array([]),
            "local_costs": np.array([]),
        }

    n, m = len(act_a), len(act_b)

    # Binary mismatch cost matrix
    cost = (act_a[:, None] != act_b[None, :]).astype(np.float64)  # (n, m)

    # DTW accumulation
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dtw[i, j] = cost[i - 1, j - 1] + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    # Backtrack
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
    normalized_distance = dtw[n, m] / len(path)  # mismatch fraction

    return {
        "distance": normalized_distance,
        "path": path,
        "local_costs": local_costs,
    }
