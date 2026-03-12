"""Per-step regret curve — where is the agent confused about value?

Per-step regret = max_return - V(s_t), matching ACCEL's MaxMC formula
but kept as a vector instead of averaged to a scalar.

High regret = agent's value estimate is far below actual return = confused.
Task-agnostic: only needs value estimates, rewards, and done flags.
"""

from typing import Optional
import numpy as np

from metrics.base import DiversityAnalyzer
from metrics.utils import truncate_at_done, downsample, format_vector


def compute_per_step_regret(
    values: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    stored_max_return: Optional[float] = None,
) -> dict:
    """Compute per-step regret curve for the first episode.

    Args:
        values: (T,) value estimates
        rewards: (T,) rewards
        dones: (T,) done flags
        stored_max_return: Previously stored max_return (None for new levels)

    Returns:
        Dict with 'regret_curve', 'values', 'max_return', 'episode_length',
        'mean_regret', 'max_regret', 'max_regret_step', 'min_regret'
    """
    ep_values = truncate_at_done(values, dones).astype(np.float64)
    ep_rewards = truncate_at_done(rewards, dones).astype(np.float64)
    ep_len = len(ep_values)

    if ep_len == 0:
        return {
            "regret_curve": np.array([]),
            "values": np.array([]),
            "max_return": 0.0,
            "episode_length": 0,
            "mean_regret": 0.0, "max_regret": 0.0,
            "max_regret_step": 0, "min_regret": 0.0,
        }

    rollout_return = float(np.sum(ep_rewards))
    if stored_max_return is not None:
        max_return = max(stored_max_return, rollout_return)
    else:
        max_return = rollout_return

    regret_curve = max_return - ep_values

    return {
        "regret_curve": regret_curve,
        "values": ep_values,
        "max_return": max_return,
        "episode_length": ep_len,
        "mean_regret": float(np.mean(regret_curve)),
        "max_regret": float(np.max(regret_curve)),
        "max_regret_step": int(np.argmax(regret_curve)),
        "min_regret": float(np.min(regret_curve)),
    }


class PerStepRegretAnalyzer(DiversityAnalyzer):
    """Formats per-step regret curve for LLM prompt injection."""

    def __init__(
        self,
        candidate_values: np.ndarray,
        candidate_rewards: np.ndarray,
        candidate_dones: np.ndarray,
        label: str = "Candidate",
        stored_max_return: Optional[float] = None,
        reference_values: Optional[np.ndarray] = None,
        reference_rewards: Optional[np.ndarray] = None,
        reference_dones: Optional[np.ndarray] = None,
        reference_label: str = "Reference",
        reference_stored_max_return: Optional[float] = None,
        max_points: int = 30,
    ):
        self.cand_info = compute_per_step_regret(
            candidate_values, candidate_rewards, candidate_dones,
            stored_max_return=stored_max_return,
        )
        self.label = label
        self.max_points = max_points

        self.ref_info = None
        self.ref_label = reference_label
        if (reference_values is not None
                and reference_rewards is not None
                and reference_dones is not None):
            self.ref_info = compute_per_step_regret(
                reference_values, reference_rewards, reference_dones,
                stored_max_return=reference_stored_max_return,
            )

    @property
    def section_title(self) -> str:
        return "PER-STEP REGRET CURVE"

    def analyze(self) -> str:
        lines = []
        ci = self.cand_info
        ds_cand = downsample(ci["regret_curve"], self.max_points)

        lines.append(
            f"{self.label} per-step regret "
            f"(episode length={ci['episode_length']}, "
            f"max_return={ci['max_return']:.3f}, "
            f"mean regret={ci['mean_regret']:.3f}, "
            f"peak={ci['max_regret']:.3f} at step {ci['max_regret_step']}):"
        )
        lines.append(f"  {format_vector(ds_cand)}")

        if self.ref_info is not None:
            ri = self.ref_info
            ds_ref = downsample(ri["regret_curve"], self.max_points)
            lines.append(
                f"\n{self.ref_label} per-step regret "
                f"(episode length={ri['episode_length']}, "
                f"max_return={ri['max_return']:.3f}, "
                f"mean regret={ri['mean_regret']:.3f}, "
                f"peak={ri['max_regret']:.3f} at step {ri['max_regret_step']}):"
            )
            lines.append(f"  {format_vector(ds_ref)}")

            lines.append(
                f"\nRegret comparison: "
                f"{self.label} mean={ci['mean_regret']:.3f} vs "
                f"{self.ref_label} mean={ri['mean_regret']:.3f}."
            )
            if ci["mean_regret"] > ri["mean_regret"] + 0.05:
                lines.append(
                    f"  {self.label} is harder — the agent underestimates "
                    f"its achievable return more."
                )
            elif ri["mean_regret"] > ci["mean_regret"] + 0.05:
                lines.append(
                    f"  {self.ref_label} is harder — the agent struggles "
                    f"more on that maze."
                )
            else:
                lines.append(
                    f"  Both mazes produce similar difficulty for the agent."
                )

        lines.append(
            "\nHigh regret = agent undervalues the state (confused/lost). "
            "Low regret = agent's value estimate is close to actual return (confident)."
        )

        return "\n".join(lines)
