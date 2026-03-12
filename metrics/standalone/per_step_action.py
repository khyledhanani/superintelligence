"""Per-step action sequence — what the agent actually does.

The action sequence is the agent's behavioral fingerprint for a level.
Task-agnostic: works with any discrete-action environment.
"""

from typing import Optional
import numpy as np

from metrics.base import DiversityAnalyzer
from metrics.utils import truncate_at_done, downsample, format_vector


def compute_per_step_action(actions: np.ndarray, dones: np.ndarray) -> dict:
    """Extract first-episode action sequence.

    Args:
        actions: (T,) discrete actions at each timestep
        dones: (T,) done flags

    Returns:
        Dict with:
            'actions': (ep_len,) action sequence for first episode
            'episode_length': int
            'action_counts': dict mapping action -> count
            'num_unique_actions': int
            'dominant_action': int (most frequent action)
            'dominant_fraction': float (fraction of steps using dominant action)
    """
    ep_actions = truncate_at_done(actions, dones).astype(np.int32)
    ep_len = len(ep_actions)

    if ep_len == 0:
        return {
            "actions": np.array([], dtype=np.int32),
            "episode_length": 0,
            "action_counts": {},
            "num_unique_actions": 0,
            "dominant_action": -1,
            "dominant_fraction": 0.0,
        }

    unique, counts = np.unique(ep_actions, return_counts=True)
    action_counts = dict(zip(unique.tolist(), counts.tolist()))
    dominant_idx = np.argmax(counts)

    return {
        "actions": ep_actions,
        "episode_length": ep_len,
        "action_counts": action_counts,
        "num_unique_actions": len(unique),
        "dominant_action": int(unique[dominant_idx]),
        "dominant_fraction": float(counts[dominant_idx] / ep_len),
    }


class PerStepActionAnalyzer(DiversityAnalyzer):
    """Formats per-step action sequence for LLM prompt injection."""

    def __init__(
        self,
        candidate_actions: np.ndarray,
        candidate_dones: np.ndarray,
        label: str = "Candidate",
        reference_actions: Optional[np.ndarray] = None,
        reference_dones: Optional[np.ndarray] = None,
        reference_label: str = "Reference",
        max_points: int = 30,
    ):
        self.cand_info = compute_per_step_action(candidate_actions, candidate_dones)
        self.label = label
        self.max_points = max_points

        self.ref_info = None
        self.ref_label = reference_label
        if reference_actions is not None and reference_dones is not None:
            self.ref_info = compute_per_step_action(reference_actions, reference_dones)

    @property
    def section_title(self) -> str:
        return "PER-STEP ACTION SEQUENCE"

    def analyze(self) -> str:
        lines = []
        ci = self.cand_info
        ds_cand = downsample(ci["actions"].astype(np.float64), self.max_points)

        lines.append(
            f"{self.label} action sequence "
            f"(episode length={ci['episode_length']}, "
            f"{ci['num_unique_actions']} unique actions, "
            f"dominant=action {ci['dominant_action']} "
            f"({ci['dominant_fraction']:.0%} of steps)):"
        )
        lines.append(f"  {format_vector(ds_cand, decimals=0)}")

        if self.ref_info is not None:
            ri = self.ref_info
            ds_ref = downsample(ri["actions"].astype(np.float64), self.max_points)
            lines.append(
                f"\n{self.ref_label} action sequence "
                f"(episode length={ri['episode_length']}, "
                f"{ri['num_unique_actions']} unique actions, "
                f"dominant=action {ri['dominant_action']} "
                f"({ri['dominant_fraction']:.0%} of steps)):"
            )
            lines.append(f"  {format_vector(ds_ref, decimals=0)}")

        lines.append(
            "\nAction sequence = the agent's behavioral response to the level. "
            "Different action patterns = different agent experience."
        )

        return "\n".join(lines)
