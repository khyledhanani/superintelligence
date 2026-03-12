"""Per-step policy entropy — where is the agent uncertain?

High entropy = agent unsure what action to take = decision point.
Task-agnostic: works with any environment that produces a policy distribution.
"""

from typing import Optional
import numpy as np

from metrics.base import DiversityAnalyzer
from metrics.utils import truncate_at_done, downsample, format_vector


def compute_per_step_entropy(entropy: np.ndarray, dones: np.ndarray) -> dict:
    """Extract first-episode per-step entropy.

    Args:
        entropy: (T,) policy entropy at each timestep
        dones: (T,) done flags

    Returns:
        Dict with 'entropy', 'episode_length', 'mean', 'max', 'max_step', 'std'
    """
    ep_entropy = truncate_at_done(entropy, dones).astype(np.float64)
    ep_len = len(ep_entropy)

    if ep_len == 0:
        return {
            "entropy": np.array([]),
            "episode_length": 0,
            "mean": 0.0, "max": 0.0, "max_step": 0, "std": 0.0,
        }

    return {
        "entropy": ep_entropy,
        "episode_length": ep_len,
        "mean": float(np.mean(ep_entropy)),
        "max": float(np.max(ep_entropy)),
        "max_step": int(np.argmax(ep_entropy)),
        "std": float(np.std(ep_entropy)),
    }


class PolicyEntropyAnalyzer(DiversityAnalyzer):
    """Formats per-step policy entropy for LLM prompt injection."""

    def __init__(
        self,
        candidate_entropy: np.ndarray,
        candidate_dones: np.ndarray,
        label: str = "Candidate",
        reference_entropy: Optional[np.ndarray] = None,
        reference_dones: Optional[np.ndarray] = None,
        reference_label: str = "Reference",
        max_points: int = 30,
    ):
        self.cand_info = compute_per_step_entropy(candidate_entropy, candidate_dones)
        self.label = label
        self.max_points = max_points

        self.ref_info = None
        self.ref_label = reference_label
        if reference_entropy is not None and reference_dones is not None:
            self.ref_info = compute_per_step_entropy(reference_entropy, reference_dones)

    @property
    def section_title(self) -> str:
        return "PER-STEP POLICY ENTROPY"

    def analyze(self) -> str:
        lines = []
        ci = self.cand_info
        ds_cand = downsample(ci["entropy"], self.max_points)

        lines.append(
            f"{self.label} policy entropy "
            f"(episode length={ci['episode_length']}, "
            f"mean={ci['mean']:.3f}, max={ci['max']:.3f} at step {ci['max_step']}):"
        )
        lines.append(f"  {format_vector(ds_cand)}")

        if self.ref_info is not None:
            ri = self.ref_info
            ds_ref = downsample(ri["entropy"], self.max_points)
            lines.append(
                f"\n{self.ref_label} policy entropy "
                f"(episode length={ri['episode_length']}, "
                f"mean={ri['mean']:.3f}, max={ri['max']:.3f} at step {ri['max_step']}):"
            )
            lines.append(f"  {format_vector(ds_ref)}")

            lines.append(
                f"\nEntropy comparison: "
                f"{self.label} mean={ci['mean']:.3f} vs "
                f"{self.ref_label} mean={ri['mean']:.3f}. "
            )
            if ci["mean"] > ri["mean"] + 0.05:
                lines.append(
                    f"  {self.label} has higher overall uncertainty — "
                    f"the agent finds this maze harder to navigate."
                )
            elif ri["mean"] > ci["mean"] + 0.05:
                lines.append(
                    f"  {self.ref_label} has higher overall uncertainty — "
                    f"the agent finds that maze harder."
                )
            else:
                lines.append(
                    f"  Both mazes produce similar overall uncertainty levels."
                )

        lines.append(
            "\nHigh entropy (>0.5) = agent unsure what to do. "
            "Low entropy (<0.2) = agent confident."
        )

        return "\n".join(lines)
