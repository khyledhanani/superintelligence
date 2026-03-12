"""Decision gate: evaluate whether a candidate maze is diverse enough.

Modular design: each metric (position DTW, regret, or any future
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

from metrics.pairwise.pos_dtw import position_trace_dtw
from metrics.standalone.regret import (
    RegretInfo,
    compute_regret,
    check_regret,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DiversityThresholds:
    """Thresholds for the decision gate.

    Set any threshold to None to disable that metric entirely
    (skips computation and feedback).
    """
    min_pos_dtw: Optional[float] = 0.5   # position trace DTW (spatial diversity)
    min_regret: Optional[float] = None    # MaxMC regret (rejects trivial mazes)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PairGateMetrics:
    """Full metrics for a candidate vs one reference maze.

    Only populated fields are those for enabled metrics.
    """
    ref_label: str = ""
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


# ---------------------------------------------------------------------------
# Per-metric issue generation
# ---------------------------------------------------------------------------

def _check_pos_dtw(min_pair: PairGateMetrics, thresholds: DiversityThresholds, issues: List[str]):
    """Check position DTW threshold."""
    if thresholds.min_pos_dtw is None:
        return

    if min_pair.pos_dtw_distance < thresholds.min_pos_dtw:
        profile_str = _format_profile(min_pair.pos_dtw_local_costs)
        issues.append(
            f"Navigation path too similar to {min_pair.ref_label} "
            f"(position DTW = {min_pair.pos_dtw_distance:.3f}, "
            f"need > {thresholds.min_pos_dtw:.3f}).\n"
            f"  Similarity profile (per-step cost): {profile_str}\n"
            f"  Change wall layout to force a completely different route."
        )


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

    Returns:
        GateResult with accept/reject decision, per-metric feedback, and metrics
    """
    thresholds = thresholds or DiversityThresholds()
    result = GateResult()

    # --- Regret (standalone, no references needed) ---
    if thresholds.min_regret is not None:
        result.regret_info = compute_regret(
            candidate_trajectory, max_steps,
            stored_max_return=stored_max_return,
        )

    if not reference_trajectories:
        issues = []
        if result.regret_info is not None:
            check_regret(result.regret_info, thresholds.min_regret, issues)
        result.issues = issues
        result.accepted = len(issues) == 0
        return result

    cand = candidate_trajectory
    compute_pos = thresholds.min_pos_dtw is not None

    # --- Pairwise metrics (only compute what's needed) ---
    for ref, label in zip(reference_trajectories, reference_labels):
        pair = PairGateMetrics(ref_label=label)

        if compute_pos:
            pos_result = position_trace_dtw(
                cand["positions"], cand["dones"],
                ref["positions"], ref["dones"],
            )
            pair.pos_dtw_distance = pos_result["distance"]
            pair.pos_dtw_local_costs = pos_result["local_costs"]
            pair.pos_dtw_path = pos_result["path"]

        result.pair_metrics.append(pair)

    # --- Summary scalars (only for computed metrics) ---
    if compute_pos:
        min_pair = min(result.pair_metrics, key=lambda p: p.pos_dtw_distance)
        result.most_similar_ref = min_pair.ref_label
        result.summary["min_pos_dtw"] = min(p.pos_dtw_distance for p in result.pair_metrics)
        result.summary["mean_pos_dtw"] = float(np.mean([p.pos_dtw_distance for p in result.pair_metrics]))

    if result.regret_info is not None:
        result.summary["regret"] = result.regret_info.regret
        result.summary["max_return"] = result.regret_info.max_return
        result.summary["episode_length"] = result.regret_info.episode_length

    # --- Check thresholds (each metric independently) ---
    issues = []

    if result.regret_info is not None and thresholds.min_regret is not None:
        check_regret(result.regret_info, thresholds.min_regret, issues)

    if compute_pos:
        min_pair = min(result.pair_metrics, key=lambda p: p.pos_dtw_distance)
        _check_pos_dtw(min_pair, thresholds, issues)

    result.issues = issues
    result.accepted = len(issues) == 0

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_profile(local_costs: np.ndarray, max_points: int = 30) -> str:
    """Format a local_costs vector as a compact string for the LLM."""
    if local_costs is None or len(local_costs) == 0:
        return "[]"

    if len(local_costs) > max_points:
        indices = np.linspace(0, len(local_costs) - 1, max_points, dtype=int)
        sampled = local_costs[indices]
    else:
        sampled = local_costs

    return "[" + ", ".join(f"{v:.2f}" for v in sampled) + "]"
