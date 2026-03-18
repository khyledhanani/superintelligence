"""Mode transition divergence — experiential diversity between two mazes.

Classifies each timestep into an experiential mode based on value error
and policy entropy, then compares the mode transition distributions.

Modes capture multi-step agent experience patterns (traps, exploration,
confident navigation) without requiring temporal alignment.

Thresholds are adaptive: derived from the agent's own error/entropy
distributions across reference trajectories (mean + 1 std), so they
automatically scale with agent capability and training stage.

KL divergence between transition matrices measures whether two mazes
put the agent through fundamentally different learning processes.
"""

import numpy as np
from typing import Dict, List, Optional

from metrics.utils import truncate_at_done


# Experience modes
MODE_CONFIDENT_CORRECT = 0   # Low |error|, low entropy — agent knows what it's doing
MODE_CONFIDENT_WRONG = 1     # High |error|, low entropy — confidently mistaken (traps)
MODE_UNCERTAIN = 2           # High entropy — decision point, unsure
MODE_RECOVERING = 3          # |error| decreasing — agent figuring it out
MODE_DEGRADING = 4           # |error| increasing — agent getting more confused
NUM_MODES = 5

MODE_NAMES = [
    "confident_correct",
    "confident_wrong",
    "uncertain",
    "recovering",
    "degrading",
]


def compute_baseline_stats(trajectories: List[Dict]) -> Dict:
    """Compute error and entropy statistics from reference trajectories.

    These stats define what's "normal" for this agent at this training stage.
    Mode thresholds are set at mean + 1 std, so ~16% of steps are classified
    as "wrong" or "uncertain" — the tail of the distribution.

    Args:
        trajectories: List of trajectory dicts, each with 'values', 'rewards',
            'dones', and optionally 'entropy'.

    Returns:
        Dict with:
            error_mean, error_std, error_threshold: mean + 1 std of |error|
            entropy_mean, entropy_std, entropy_threshold: mean + 1 std of entropy
            n_steps: total steps used for computation
    """
    all_abs_errors = []
    all_entropies = []

    for traj in trajectories:
        ep_values = truncate_at_done(traj["values"], traj["dones"]).astype(np.float64)
        ep_rewards = truncate_at_done(traj["rewards"], traj["dones"]).astype(np.float64)
        ep_len = len(ep_values)

        if ep_len == 0:
            continue

        # Compute actual returns G_t
        returns = np.zeros(ep_len, dtype=np.float64)
        g = 0.0
        for t in range(ep_len - 1, -1, -1):
            g = ep_rewards[t] + g  # gamma=1.0
            returns[t] = g

        abs_error = np.abs(ep_values - returns)
        all_abs_errors.append(abs_error)

        if "entropy" in traj:
            ep_entropy = truncate_at_done(traj["entropy"], traj["dones"]).astype(np.float64)
            all_entropies.append(ep_entropy)

    if not all_abs_errors:
        return {
            "error_mean": 0.3, "error_std": 0.0, "error_threshold": 0.3,
            "entropy_mean": 0.3, "entropy_std": 0.0, "entropy_threshold": 0.3,
            "n_steps": 0,
        }

    all_abs_errors = np.concatenate(all_abs_errors)
    error_mean = float(np.mean(all_abs_errors))
    error_std = float(np.std(all_abs_errors))

    if all_entropies:
        all_entropies = np.concatenate(all_entropies)
        entropy_mean = float(np.mean(all_entropies))
        entropy_std = float(np.std(all_entropies))
    else:
        entropy_mean = 0.0
        entropy_std = 0.0

    return {
        "error_mean": error_mean,
        "error_std": error_std,
        "error_threshold": error_mean + error_std,
        "entropy_mean": entropy_mean,
        "entropy_std": entropy_std,
        "entropy_threshold": entropy_mean + entropy_std,
        "n_steps": len(all_abs_errors),
    }


def classify_modes(
    values: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    entropy: Optional[np.ndarray] = None,
    gamma: float = 1.0,
    baseline_stats: Optional[Dict] = None,
) -> dict:
    """Classify each timestep into an experiential mode.

    Thresholds are adaptive when baseline_stats is provided (mean + 1 std
    from reference trajectories). Falls back to fixed thresholds if not.

    Args:
        values: (T,) value estimates V(s_t)
        rewards: (T,) rewards r_t
        dones: (T,) done flags
        entropy: (T,) policy entropy (optional; if None, uncertain mode is disabled)
        gamma: discount factor
        baseline_stats: Output of compute_baseline_stats(). If None, uses
            fixed fallback thresholds (error=0.3, entropy=0.3).

    Returns:
        Dict with:
            modes: (ep_len,) int array of mode indices
            mode_names: list of mode name strings
            mode_fractions: (NUM_MODES,) fraction of time in each mode
            transition_matrix: (NUM_MODES, NUM_MODES) transition counts (row=from, col=to)
            transition_probs: (NUM_MODES, NUM_MODES) transition probabilities (row-normalized)
            episode_length: int
            error_curve: (ep_len,) signed value error used for classification
            thresholds: dict with error_threshold and entropy_threshold used
    """
    # Resolve thresholds
    if baseline_stats is not None:
        error_threshold = baseline_stats["error_threshold"]
        entropy_threshold = baseline_stats["entropy_threshold"]
    else:
        error_threshold = 0.3
        entropy_threshold = 0.3

    ep_values = truncate_at_done(values, dones).astype(np.float64)
    ep_rewards = truncate_at_done(rewards, dones).astype(np.float64)
    ep_len = len(ep_values)

    if entropy is not None:
        ep_entropy = truncate_at_done(entropy, dones).astype(np.float64)
    else:
        ep_entropy = np.zeros(ep_len)

    empty = {
        "modes": np.array([], dtype=np.int32),
        "mode_names": MODE_NAMES,
        "mode_fractions": np.zeros(NUM_MODES),
        "transition_matrix": np.zeros((NUM_MODES, NUM_MODES)),
        "transition_probs": np.zeros((NUM_MODES, NUM_MODES)),
        "episode_length": 0,
        "error_curve": np.array([]),
        "thresholds": {"error": error_threshold, "entropy": entropy_threshold},
    }

    if ep_len == 0:
        return empty

    # Compute actual returns G_t
    returns = np.zeros(ep_len, dtype=np.float64)
    g = 0.0
    for t in range(ep_len - 1, -1, -1):
        g = ep_rewards[t] + gamma * g
        returns[t] = g

    error_curve = ep_values - returns
    abs_error = np.abs(error_curve)

    # Classify each step
    modes = np.zeros(ep_len, dtype=np.int32)
    for t in range(ep_len):
        is_uncertain = ep_entropy[t] > entropy_threshold
        is_wrong = abs_error[t] > error_threshold

        if is_uncertain:
            modes[t] = MODE_UNCERTAIN
        elif t > 0:
            error_delta = abs_error[t] - abs_error[t - 1]
            if is_wrong:
                if error_delta > 0:
                    modes[t] = MODE_DEGRADING
                else:
                    modes[t] = MODE_RECOVERING if error_delta < -0.01 else MODE_CONFIDENT_WRONG
            else:
                modes[t] = MODE_CONFIDENT_CORRECT
        else:
            modes[t] = MODE_CONFIDENT_WRONG if is_wrong else MODE_CONFIDENT_CORRECT

    # Mode fractions
    mode_fractions = np.zeros(NUM_MODES)
    for m in range(NUM_MODES):
        mode_fractions[m] = np.mean(modes == m)

    # Transition matrix
    transition_matrix = np.zeros((NUM_MODES, NUM_MODES))
    for t in range(ep_len - 1):
        transition_matrix[modes[t], modes[t + 1]] += 1

    # Row-normalize to probabilities
    transition_probs = np.zeros((NUM_MODES, NUM_MODES))
    for i in range(NUM_MODES):
        row_sum = transition_matrix[i].sum()
        if row_sum > 0:
            transition_probs[i] = transition_matrix[i] / row_sum
        else:
            transition_probs[i] = 1.0 / NUM_MODES

    return {
        "modes": modes,
        "mode_names": MODE_NAMES,
        "mode_fractions": mode_fractions,
        "transition_matrix": transition_matrix,
        "transition_probs": transition_probs,
        "episode_length": ep_len,
        "error_curve": error_curve,
        "thresholds": {"error": error_threshold, "entropy": entropy_threshold},
    }


def mode_transition_divergence(
    traj_a: dict,
    dones_a: np.ndarray,
    traj_b: dict,
    dones_b: np.ndarray,
    entropy_a: Optional[np.ndarray] = None,
    entropy_b: Optional[np.ndarray] = None,
    baseline_stats: Optional[Dict] = None,
) -> Dict:
    """Compute experiential divergence between two trajectories.

    Args:
        traj_a: Trajectory dict with 'values', 'rewards' keys
        dones_a: Done flags for trajectory A
        traj_b: Trajectory dict with 'values', 'rewards' keys
        dones_b: Done flags for trajectory B
        entropy_a: Policy entropy for trajectory A (optional)
        entropy_b: Policy entropy for trajectory B (optional)
        baseline_stats: Output of compute_baseline_stats(). If None, uses
            fixed fallback thresholds.

    Returns:
        Dict with:
            kl_divergence: float — symmetric KL divergence between transition matrices
            mode_fractions_a: (NUM_MODES,) mode distribution for A
            mode_fractions_b: (NUM_MODES,) mode distribution for B
            transition_probs_a: (NUM_MODES, NUM_MODES) transition probs for A
            transition_probs_b: (NUM_MODES, NUM_MODES) transition probs for B
            fraction_distance: float — L1 distance between mode fraction vectors
            thresholds: dict with error_threshold and entropy_threshold used
    """
    modes_a = classify_modes(
        traj_a["values"], traj_a["rewards"], dones_a,
        entropy=entropy_a, baseline_stats=baseline_stats,
    )
    modes_b = classify_modes(
        traj_b["values"], traj_b["rewards"], dones_b,
        entropy=entropy_b, baseline_stats=baseline_stats,
    )

    # Symmetric KL divergence with Laplace smoothing
    eps = 1e-8
    P = modes_a["transition_probs"] + eps
    Q = modes_b["transition_probs"] + eps
    P = P / P.sum(axis=1, keepdims=True)
    Q = Q / Q.sum(axis=1, keepdims=True)

    kl_pq = np.sum(P * np.log(P / Q))
    kl_qp = np.sum(Q * np.log(Q / P))
    sym_kl = (kl_pq + kl_qp) / 2.0

    frac_dist = float(np.sum(np.abs(
        modes_a["mode_fractions"] - modes_b["mode_fractions"]
    )))

    return {
        "kl_divergence": float(sym_kl),
        "mode_fractions_a": modes_a["mode_fractions"],
        "mode_fractions_b": modes_b["mode_fractions"],
        "transition_probs_a": modes_a["transition_probs"],
        "transition_probs_b": modes_b["transition_probs"],
        "fraction_distance": frac_dist,
        "thresholds": modes_a["thresholds"],
    }
