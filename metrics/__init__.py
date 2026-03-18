"""Metrics package — standalone per-step signals and pairwise comparisons.

Modules:
    base        — DiversityAnalyzer ABC, AnalysisSection
    dtw         — Core DTW algorithm (shared by pairwise metrics)
    utils       — Shared helpers (truncate, downsample, format)

    standalone/ — Per-level metrics (no reference needed)
        per_step_entropy  — Per-step policy entropy
        per_step_regret   — Per-step regret curve
        per_step_action   — Per-step action sequence
        regret            — Scalar MaxMC regret
        value_error       — Signed value error profile (V(s_t) - G_t)

    pairwise/   — Level-vs-level comparison metrics
        pos_dtw           — Position trace DTW (spatial diversity)
        regret_dtw        — Regret curve DTW (difficulty diversity)
        action_dtw_binary — Action sequence DTW with binary mismatch cost (behavioral diversity)
        mode_transition   — Experience mode transition divergence (experiential diversity)
"""

from metrics.base import DiversityAnalyzer, AnalysisSection

# Standalone
from metrics.standalone.per_step_entropy import PolicyEntropyAnalyzer, compute_per_step_entropy
from metrics.standalone.per_step_regret import PerStepRegretAnalyzer, compute_per_step_regret
from metrics.standalone.per_step_action import PerStepActionAnalyzer, compute_per_step_action
from metrics.standalone.regret import RegretInfo, compute_regret, check_regret
from metrics.standalone.value_error import compute_value_error

# Pairwise
from metrics.pairwise.pos_dtw import PositionDTWAnalyzer, position_trace_dtw
from metrics.pairwise.regret_dtw import regret_curve_dtw
from metrics.pairwise.action_dtw_binary import action_sequence_distance
from metrics.pairwise.mode_transition import (
    classify_modes, mode_transition_divergence, compute_baseline_stats,
    MODE_NAMES, NUM_MODES,
)
from metrics.pairwise.td_error_distribution import (
    compute_td_errors, td_error_divergence,
)

__all__ = [
    "DiversityAnalyzer", "AnalysisSection",
    # Standalone
    "PolicyEntropyAnalyzer", "compute_per_step_entropy",
    "PerStepRegretAnalyzer", "compute_per_step_regret",
    "PerStepActionAnalyzer", "compute_per_step_action",
    "RegretInfo", "compute_regret", "check_regret",
    "compute_value_error",
    # Pairwise
    "PositionDTWAnalyzer", "position_trace_dtw",
    "regret_curve_dtw",
    "action_sequence_distance",
    "classify_modes", "mode_transition_divergence", "compute_baseline_stats",
    "MODE_NAMES", "NUM_MODES",
    "compute_td_errors", "td_error_divergence",
]
