"""LSTM embedding divergence — behavioral diversity via agent representation.

Compares the mean LSTM hidden state + action (257-dim) between two
trajectories using L2 distance.

The LSTM embedding is the agent's full compressed understanding at each step —
it encodes spatial memory, navigation history, and learned expectations.
The action dimension captures what the agent *did* in response. Averaging
over the episode gives a compact summary of the agent's overall experience.

Pairwise distance is L2 on the 257-dim mean vectors.
"""

import numpy as np
from typing import Dict

from metrics.utils import truncate_at_done


def _mean_state_action(traj: dict, dones: np.ndarray) -> np.ndarray:
    """Compute mean LSTM hidden state + action over the episode.

    Returns (257,) vector: [mean_hstate(256), mean_action(1)].
    """
    hstates = truncate_at_done(traj["hstates"], dones).astype(np.float64)
    actions = truncate_at_done(traj["actions"], dones).astype(np.float64)
    if len(hstates) == 0:
        return np.zeros(257)
    mean_h = np.mean(hstates, axis=0)  # (256,)
    mean_a = np.mean(actions)           # scalar
    return np.concatenate([mean_h, [mean_a]])  # (257,)


def embedding_divergence(
    traj_a: dict,
    dones_a: np.ndarray,
    traj_b: dict,
    dones_b: np.ndarray,
) -> Dict:
    """Compute L2 distance between mean LSTM state-action embeddings.

    Args:
        traj_a: Trajectory dict with 'hstates' (T, 256) and 'actions' (T,)
        dones_a: (T,) done flags for trajectory A
        traj_b: Trajectory dict with 'hstates' (T, 256) and 'actions' (T,)
        dones_b: (T,) done flags for trajectory B

    Returns:
        Dict with:
            distance: float — L2 distance between mean embeddings
            mean_a: (257,) mean state-action for A
            mean_b: (257,) mean state-action for B
    """
    mean_a = _mean_state_action(traj_a, dones_a)
    mean_b = _mean_state_action(traj_b, dones_b)

    distance = float(np.linalg.norm(mean_a - mean_b))

    return {
        "distance": distance,
        "mean_a": mean_a,
        "mean_b": mean_b,
    }


def compute_mean_embedding(
    hstates: np.ndarray,
    dones: np.ndarray,
    actions: np.ndarray = None,
) -> np.ndarray:
    """Compute mean embedding for a single trajectory.

    Used for precomputation / fast pairwise distance on the full buffer.

    Args:
        hstates: (T, 256) LSTM hidden states
        dones: (T,) done flags
        actions: (T,) action indices. If provided, appended as 257th dimension.

    Returns:
        (D,) mean vector where D=257 if actions provided, else 256.
    """
    ep_h = truncate_at_done(hstates, dones).astype(np.float64)
    if len(ep_h) == 0:
        n_dims = 257 if actions is not None else (hstates.shape[1] if hstates.ndim == 2 else 256)
        return np.zeros(n_dims)

    mean_h = np.mean(ep_h, axis=0)  # (256,)
    if actions is not None:
        ep_a = truncate_at_done(actions, dones).astype(np.float64)
        mean_a = np.mean(ep_a)
        return np.concatenate([mean_h, [mean_a]])  # (257,)
    return mean_h
