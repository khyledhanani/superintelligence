"""Decision gate: evaluate whether a candidate maze is diverse enough.

Modular design: each metric (diversity, regret, or any future
metric) is an independent check. Metrics are only computed when their threshold
is set (not None). Each metric produces its own issue string for LLM feedback.

To add a new metric:
1. Add a threshold field to DiversityThresholds
2. Add a compute function in metrics/<name>.py
3. Add a _check_* function or use the metric module's check function
4. Wire it into evaluate_candidate
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np

from metrics.standalone.regret import (
    RegretInfo,
    compute_regret,
    check_regret,
)
from metrics.standalone.learnability import (
    LearnabilityInfo,
    compute_learnability,
    check_learnability,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DiversityThresholds:
    """Thresholds for the decision gate.

    Set any threshold to None to disable that metric entirely
    (skips computation and feedback).

    difficulty_metric: "regret" (MaxMC regret, default) or "sfl" (SFL learnability p*(1-p))
    diversity_metric: "normalized_td_error_emd", "experience_divergence", "position_dtw", or "cenie"
    """
    difficulty_threshold: Optional[float] = None  # Min difficulty score to accept
    difficulty_metric: str = "regret"             # "regret" or "sfl"
    min_diversity: Optional[float] = 0.04         # pairwise diversity vs references
    diversity_metric: str = "normalized_td_error_emd"  # "normalized_td_error_emd", "experience_divergence", "position_dtw"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PairGateMetrics:
    """Full metrics for a candidate vs one reference maze.

    Only populated fields are those for enabled metrics.
    """
    ref_label: str = ""
    diversity_distance: float = 0.0
    diversity_metric: str = ""
    # Legacy pos_dtw fields kept for visualization compatibility
    pos_dtw_distance: float = 0.0
    pos_dtw_local_costs: Optional[np.ndarray] = None
    pos_dtw_path: Optional[np.ndarray] = None


@dataclass
class GateResult:
    """Result of a decision gate evaluation.

    Attributes:
        accepted: Whether the candidate passed all enabled checks
        issues: List of specific issues (for LLM feedback), one per failed check
        pair_metrics: Per-reference detailed metrics (only for enabled pairwise metrics)
        summary: Scalar summary stats for logging/saving
        most_similar_ref: Label of the most similar reference maze
        regret_info: Regret metrics (if regret is enabled)
    """
    accepted: bool = False
    issues: List[str] = field(default_factory=list)
    pair_metrics: List[PairGateMetrics] = field(default_factory=list)
    summary: Dict[str, float] = field(default_factory=dict)
    most_similar_ref: str = ""
    regret_info: Optional[RegretInfo] = None
    learnability_info: Optional[LearnabilityInfo] = None


# ---------------------------------------------------------------------------
# Per-metric diversity computation
# ---------------------------------------------------------------------------

def _compute_pairwise_diversity(
    candidate_trajectory: dict,
    reference_trajectory: dict,
    metric: str,
    baseline_stats: Optional[dict] = None,
) -> float:
    """Compute a single pairwise diversity distance.

    Args:
        candidate_trajectory: Candidate trajectory dict
        reference_trajectory: Reference trajectory dict
        metric: One of "normalized_td_error_emd", "experience_divergence", "position_dtw"
        baseline_stats: For experience_divergence, mode classification thresholds

    Returns:
        Scalar distance (higher = more diverse)
    """
    cand = candidate_trajectory
    ref = reference_trajectory

    if metric in ("normalized_td_error_emd", "td_error_emd"):
        from metrics.pairwise.td_error_distribution import td_error_divergence
        result = td_error_divergence(
            cand, cand["dones"], ref, ref["dones"],
            normalize=(metric == "normalized_td_error_emd"),
        )
        return result["emd"]

    elif metric == "experience_divergence":
        from metrics.pairwise.mode_transition import mode_transition_divergence
        result = mode_transition_divergence(
            cand, cand["dones"], ref, ref["dones"],
            entropy_a=cand.get("entropy"), entropy_b=ref.get("entropy"),
            baseline_stats=baseline_stats,
        )
        return result["kl_divergence"]

    elif metric == "position_dtw":
        from metrics.pairwise.pos_dtw import position_trace_dtw
        result = position_trace_dtw(
            cand["positions"], cand["dones"],
            ref["positions"], ref["dones"],
        )
        return result["distance"]

    elif metric == "embedding":
        from metrics.pairwise.embedding_divergence import embedding_divergence
        result = embedding_divergence(cand, cand["dones"], ref, ref["dones"])
        return result["distance"]

    else:
        raise ValueError(f"Unknown diversity metric: {metric}")


# ---------------------------------------------------------------------------
# Main gate function
# ---------------------------------------------------------------------------

def evaluate_candidate(
    candidate_trajectory: dict,
    reference_trajectories: List[dict],
    reference_labels: List[str],
    thresholds: Optional[DiversityThresholds] = None,
    max_steps: int = 250,
    stored_max_return: Optional[float] = None,
    baseline_stats: Optional[dict] = None,
    cenie_model: Optional[object] = None,
) -> GateResult:
    """Evaluate a candidate maze's trajectory against reference mazes.

    Only computes metrics whose thresholds are enabled (not None).

    Args:
        candidate_trajectory: Dict with keys:
            'positions': (T, 2) agent positions
            'values': (T,) value estimates
            'dones': (T,) done flags
            'rewards': (T,) rewards
        reference_trajectories: List of dicts with same keys
        reference_labels: List of labels like ["Maze A", "Maze B", ...]
        thresholds: Diversity thresholds (None fields = skip that metric)
        max_steps: Max steps per episode for regret calculation
        stored_max_return: Previously stored max_return for accumulated regret
        baseline_stats: For experience_divergence mode classification
        cenie_model: Fitted CENIEModel for CENIE diversity metric (required
            when diversity_metric="cenie")

    Returns:
        GateResult with accept/reject decision, per-metric feedback, and metrics
    """
    thresholds = thresholds or DiversityThresholds()
    result = GateResult()

    # --- Difficulty gate (standalone, no references needed) ---
    if thresholds.difficulty_threshold is not None:
        if thresholds.difficulty_metric == "sfl":
            result.learnability_info = compute_learnability(candidate_trajectory)
        else:
            # Default: regret
            result.regret_info = compute_regret(
                candidate_trajectory, max_steps,
                stored_max_return=stored_max_return,
            )

    if not reference_trajectories:
        issues = []
        if thresholds.difficulty_threshold is not None:
            if thresholds.difficulty_metric == "sfl" and result.learnability_info is not None:
                check_learnability(result.learnability_info, thresholds.difficulty_threshold, issues)
            elif result.regret_info is not None:
                check_regret(result.regret_info, thresholds.difficulty_threshold, issues)
        result.issues = issues
        result.accepted = len(issues) == 0
        return result

    compute_diversity = thresholds.min_diversity is not None
    is_cenie = thresholds.diversity_metric == "cenie"

    # --- CENIE novelty (standalone, uses pre-fitted GMM) ---
    if compute_diversity and is_cenie:
        if cenie_model is not None:
            from metrics.standalone.cenie import compute_cenie_novelty
            novelty_info = compute_cenie_novelty(candidate_trajectory, cenie_model)
            result.summary["cenie_novelty"] = novelty_info.novelty
            result.summary["cenie_n_pairs"] = novelty_info.n_pairs
            result.summary["min_diversity"] = novelty_info.novelty
            result.summary["mean_diversity"] = novelty_info.novelty
            result.summary["diversity_metric"] = "cenie"

    # --- Pairwise metrics (only compute what's needed, skip for CENIE) ---
    if not is_cenie:
        for ref, label in zip(reference_trajectories, reference_labels):
            pair = PairGateMetrics(ref_label=label)

            if compute_diversity:
                pair.diversity_distance = _compute_pairwise_diversity(
                    candidate_trajectory, ref,
                    thresholds.diversity_metric,
                    baseline_stats=baseline_stats,
                )
                pair.diversity_metric = thresholds.diversity_metric

            result.pair_metrics.append(pair)

        # --- Summary scalars (only for computed metrics) ---
        if compute_diversity and result.pair_metrics:
            min_pair = min(result.pair_metrics, key=lambda p: p.diversity_distance)
            result.most_similar_ref = min_pair.ref_label
            distances = [p.diversity_distance for p in result.pair_metrics]
            result.summary["min_diversity"] = min(distances)
            result.summary["mean_diversity"] = float(np.mean(distances))
            result.summary["diversity_metric"] = thresholds.diversity_metric

    if result.regret_info is not None:
        result.summary["regret"] = result.regret_info.regret
        result.summary["max_return"] = result.regret_info.max_return
        result.summary["episode_length"] = result.regret_info.episode_length

    if result.learnability_info is not None:
        result.summary["learnability"] = result.learnability_info.learnability
        result.summary["solve_rate"] = result.learnability_info.solve_rate
        result.summary["difficulty_metric"] = "sfl"

    if result.regret_info is not None and result.learnability_info is None:
        result.summary["difficulty_metric"] = "regret"

    # --- Check thresholds (each metric independently) ---
    issues = []

    if thresholds.difficulty_threshold is not None:
        if thresholds.difficulty_metric == "sfl" and result.learnability_info is not None:
            check_learnability(result.learnability_info, thresholds.difficulty_threshold, issues)
        elif result.regret_info is not None:
            check_regret(result.regret_info, thresholds.difficulty_threshold, issues)

    if compute_diversity and is_cenie:
        novelty = result.summary.get("cenie_novelty", 0.0)
        if novelty < thresholds.min_diversity:
            issues.append(
                f"Maze produces experiences too similar to the training buffer "
                f"(CENIE novelty = {novelty:.4f}, "
                f"need > {thresholds.min_diversity:.4f}).\n"
                f"  The agent's state-action trajectory on this maze overlaps heavily "
                f"with past curriculum experiences.\n"
                f"  Design a maze that forces the agent into unfamiliar states "
                f"and actions it hasn't encountered before."
            )
    elif compute_diversity and result.pair_metrics:
        distances = [p.diversity_distance for p in result.pair_metrics]
        min_dist = min(distances)
        if min_dist < thresholds.min_diversity:
            metric_name = {
                "normalized_td_error_emd": "Normalized TD Error EMD",
                "td_error_emd": "TD Error EMD",
                "experience_divergence": "Experience Divergence",
                "position_dtw": "Position DTW",
                "embedding": "LSTM Embedding L2",
            }.get(thresholds.diversity_metric, thresholds.diversity_metric)
            closest = min(result.pair_metrics, key=lambda p: p.diversity_distance)
            issues.append(
                f"Maze is too similar to {closest.ref_label} "
                f"(min {metric_name} = {min_dist:.4f}, "
                f"need > {thresholds.min_diversity:.4f}).\n"
                f"  Per-reference distances: "
                + ", ".join(f"{p.ref_label}={p.diversity_distance:.4f}" for p in result.pair_metrics)
                + f"\n  Most similar to: {closest.ref_label} ({closest.diversity_distance:.4f})"
                + "\n  Design a maze that produces fundamentally different agent behavior."
            )

    result.issues = issues
    result.accepted = len(issues) == 0

    return result
