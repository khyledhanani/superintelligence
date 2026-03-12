"""Position Trace DTW — spatial path similarity between two levels.

Translation-invariant: positions are made relative to start before comparison.
"""

from typing import Dict
import numpy as np

from metrics.base import DiversityAnalyzer
from metrics.dtw import dtw_with_path
from metrics.utils import truncate_at_done, downsample, format_vector


def position_trace_dtw(pos_a: np.ndarray, dones_a: np.ndarray,
                       pos_b: np.ndarray, dones_b: np.ndarray) -> Dict:
    """Compute DTW distance between two position traces.

    Args:
        pos_a: (T1, 2) agent positions for trajectory A
        dones_a: (T1,) done flags for trajectory A
        pos_b: (T2, 2) agent positions for trajectory B
        dones_b: (T2,) done flags for trajectory B

    Returns:
        Dict with 'distance' (normalized scalar), 'path', 'local_costs'
    """
    pos_a = truncate_at_done(pos_a, dones_a).astype(np.float32)
    pos_b = truncate_at_done(pos_b, dones_b).astype(np.float32)

    # Relative to start position (translation invariant)
    pos_a = pos_a - pos_a[0]
    pos_b = pos_b - pos_b[0]

    distance, path, local_costs = dtw_with_path(pos_a, pos_b)
    return {
        "distance": distance,
        "path": path,
        "local_costs": local_costs,
    }


class PositionDTWAnalyzer(DiversityAnalyzer):
    """Formats Position DTW results for LLM prompt injection."""

    def __init__(
        self,
        candidate_positions: np.ndarray,
        candidate_dones: np.ndarray,
        reference_positions: np.ndarray,
        reference_dones: np.ndarray,
        reference_label: str = "Reference",
        max_points: int = 30,
    ):
        self.ref_label = reference_label
        self.max_points = max_points

        result = position_trace_dtw(
            candidate_positions, candidate_dones,
            reference_positions, reference_dones,
        )
        self.distance = result["distance"]
        self.local_costs = result["local_costs"]

        cand_pos = truncate_at_done(candidate_positions, candidate_dones)
        ref_pos = truncate_at_done(reference_positions, reference_dones)
        self.cand_steps = len(cand_pos)
        self.ref_steps = len(ref_pos)

    @property
    def section_title(self) -> str:
        return "POSITION DTW PROFILE"

    def analyze(self) -> str:
        lines = []
        ds = downsample(self.local_costs, self.max_points)

        lines.append(
            f"Position DTW vs {self.ref_label}: distance={self.distance:.3f} "
            f"(candidate {self.cand_steps} steps, {self.ref_label} {self.ref_steps} steps)"
        )
        lines.append(f"Local cost profile (per warping step):")
        lines.append(f"  {format_vector(ds, decimals=2)}")
        lines.append(
            "\nHigh cost = paths diverge at that point. "
            "Low cost = paths are similar."
        )

        return "\n".join(lines)
