"""CENIE: Coverage-based Evaluation of Novelty In Environments.

Implements the novelty scoring component from:
"Improving Environment Novelty Quantification for Effective Unsupervised
Environment Design" (Teoh, Li, Varakantham, NeurIPS 2024)
https://arxiv.org/abs/2502.05726

Fits a GMM on LSTM hidden state + action pairs from buffer trajectories.
Scores candidate levels by their average negative log-likelihood under
the GMM — higher NLL = more novel experience.

Per the paper, MiniGrid uses the LSTM hidden state (agent's belief
representation) concatenated with the action as the state-action pair.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class CENIEModel:
    """Fitted CENIE GMM model for novelty scoring."""
    gmm: object  # sklearn GaussianMixture
    n_components: int
    n_samples: int  # number of samples used to fit


@dataclass
class CENIENoveltyInfo:
    """Novelty score for a candidate level."""
    novelty: float = 0.0       # mean NLL under GMM (higher = more novel)
    n_pairs: int = 0           # number of state-action pairs scored


def extract_state_action_pairs(trajectory: dict) -> Optional[np.ndarray]:
    """Extract LSTM hidden state + action pairs from a trajectory.

    Per the CENIE paper (Appendix D.3), MiniGrid uses LSTM hidden states
    as the state representation, concatenated with the action.

    Args:
        trajectory: Dict with 'hstates' (T, 256) and 'actions' (T,) and 'dones' (T,)

    Returns:
        (T_ep, 257) array of [hstate, action] pairs truncated at first done,
        or None if hstates not available.
    """
    if "hstates" not in trajectory:
        return None

    hstates = np.asarray(trajectory["hstates"])   # (T, 256)
    actions = np.asarray(trajectory["actions"])     # (T,)
    dones = np.asarray(trajectory["dones"])         # (T,)

    # Truncate at first done
    done_idx = np.where(dones.astype(bool))[0]
    ep_len = int(done_idx[0] + 1) if len(done_idx) > 0 else len(dones)

    hstates = hstates[:ep_len]
    actions = actions[:ep_len].astype(np.float32)

    # Concatenate: [hstate (256), action (1)] = 257D
    return np.concatenate([hstates, actions[:, None]], axis=-1)


def fit_cenie_model(
    trajectories: List[dict],
    max_components: int = 10,
    min_components: int = 2,
) -> Optional[CENIEModel]:
    """Fit a CENIE GMM model on trajectories from the buffer.

    Uses silhouette score to select optimal number of GMM components,
    following the paper's approach.

    Args:
        trajectories: List of trajectory dicts with 'hstates', 'actions', 'dones'
        max_components: Maximum GMM components to try
        min_components: Minimum GMM components to try

    Returns:
        Fitted CENIEModel, or None if insufficient data
    """
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score

    # Collect all state-action pairs from buffer trajectories
    all_pairs = []
    for traj in trajectories:
        pairs = extract_state_action_pairs(traj)
        if pairs is not None:
            all_pairs.append(pairs)

    if not all_pairs:
        logger.warning("[CENIE] No trajectories with hstates — cannot fit GMM")
        return None

    data = np.concatenate(all_pairs, axis=0).astype(np.float64)
    n_samples = len(data)

    # Need enough samples for the GMM
    min_samples = max_components * 20
    if n_samples < min_samples:
        logger.warning(f"[CENIE] Only {n_samples} samples, need {min_samples} — skipping")
        return None

    # Select optimal K via silhouette score (paper Section 3.2)
    k_range = range(min_components, min(max_components + 1, n_samples // 20 + 1))
    best_k = min_components
    best_score = -1.0
    best_gmm = None

    for k in k_range:
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type='diag',
                max_iter=100,
                reg_covar=1e-4,
                random_state=42,
            )
            gmm.fit(data)
            labels = gmm.predict(data)

            # Silhouette score needs at least 2 unique labels
            if len(np.unique(labels)) >= 2:
                # Subsample for speed if data is large
                if n_samples > 5000:
                    idx = np.random.RandomState(42).choice(n_samples, 5000, replace=False)
                    score = silhouette_score(data[idx], labels[idx])
                else:
                    score = silhouette_score(data, labels)

                if score > best_score:
                    best_score = score
                    best_k = k
                    best_gmm = gmm
        except Exception as e:
            logger.debug(f"[CENIE] GMM fit failed for k={k}: {e}")
            continue

    if best_gmm is None:
        # Fallback: fit with min_components
        best_gmm = GaussianMixture(
            n_components=min_components,
            covariance_type='diag',
            max_iter=100,
            reg_covar=1e-4,
            random_state=42,
        )
        best_gmm.fit(data)
        best_k = min_components

    logger.info(
        f"[CENIE] GMM fitted: k={best_k} (silhouette={best_score:.3f}), "
        f"{n_samples} samples from {len(all_pairs)} trajectories"
    )

    return CENIEModel(gmm=best_gmm, n_components=best_k, n_samples=n_samples)


def compute_cenie_novelty(
    candidate_trajectory: dict,
    cenie_model: CENIEModel,
) -> CENIENoveltyInfo:
    """Compute CENIE novelty score for a candidate level.

    Novelty = mean negative log-likelihood of the candidate's state-action
    pairs under the fitted GMM. Higher = more novel.

    Args:
        candidate_trajectory: Dict with 'hstates', 'actions', 'dones'
        cenie_model: Fitted CENIEModel from fit_cenie_model()

    Returns:
        CENIENoveltyInfo with novelty score
    """
    pairs = extract_state_action_pairs(candidate_trajectory)
    if pairs is None or len(pairs) == 0:
        return CENIENoveltyInfo(novelty=0.0, n_pairs=0)

    # Score under GMM: negative mean log-likelihood (higher = more novel)
    log_probs = cenie_model.gmm.score_samples(pairs.astype(np.float64))  # (T,)
    novelty = -float(np.mean(log_probs))

    return CENIENoveltyInfo(novelty=novelty, n_pairs=len(pairs))
