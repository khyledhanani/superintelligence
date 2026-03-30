"""SFL Learnability metric — favors levels at the learning frontier.

Based on "Sampling for Learnability" (Rutherford et al., 2024):
    learnability = p × (1 - p)

where p is the agent's solve rate across multiple rollouts.

Properties:
    - p = 0 (impossible) → learnability = 0
    - p = 1 (mastered)   → learnability = 0
    - p = 0.5 (frontier) → learnability = 0.25 (maximum)

This is the variance of a Bernoulli(p), and greedily maximizing it
is equivalent to maximizing expected improvement.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class LearnabilityInfo:
    """Learnability metrics for a candidate maze.

    Uses SFL learnability: p × (1 - p) where p = solve rate.
    """
    learnability: float = 0.0
    solve_rate: float = 0.0
    n_rollouts: int = 0


def compute_learnability(
    candidate_trajectory: dict,
) -> LearnabilityInfo:
    """Compute SFL learnability from multi-rollout solve rate.

    The candidate_trajectory must come from evaluate_level_multi_rollout()
    so that it contains 'solve_rate' and 'all_returns'.

    Args:
        candidate_trajectory: Dict with 'solve_rate' key (from multi-rollout eval)

    Returns:
        LearnabilityInfo with SFL learnability score
    """
    solve_rate = candidate_trajectory.get("solve_rate", 0.0)
    all_returns = candidate_trajectory.get("all_returns", np.array([]))
    n_rollouts = len(all_returns) if len(all_returns) > 0 else 1

    learnability = solve_rate * (1.0 - solve_rate)

    return LearnabilityInfo(
        learnability=learnability,
        solve_rate=solve_rate,
        n_rollouts=n_rollouts,
    )


def check_learnability(
    info: LearnabilityInfo,
    min_learnability: float,
    issues: List[str],
):
    """Check learnability threshold and append issue if too low.

    Low learnability means the level is either too easy (p ≈ 1) or
    too hard (p ≈ 0) — not at the learning frontier.
    """
    if info.learnability < min_learnability:
        if info.solve_rate < 0.1:
            issues.append(
                f"Maze is too hard for the agent (solve rate = {info.solve_rate:.0%}, "
                f"learnability = {info.learnability:.4f}, "
                f"need > {min_learnability:.4f}).\n"
                f"  The agent almost never solves this level across "
                f"{info.n_rollouts} rollouts.\n"
                f"  The maze needs to be easier — the agent cannot learn from "
                f"a level it never solves."
            )
        elif info.solve_rate > 0.9:
            issues.append(
                f"Maze is too easy for the agent (solve rate = {info.solve_rate:.0%}, "
                f"learnability = {info.learnability:.4f}, "
                f"need > {min_learnability:.4f}).\n"
                f"  The agent solves this level almost every time across "
                f"{info.n_rollouts} rollouts.\n"
                f"  The maze needs to be harder — the agent learns nothing from "
                f"a level it already masters."
            )
        else:
            # Edge case: moderate solve rate but still below threshold
            issues.append(
                f"Maze learnability is below threshold "
                f"(learnability = {info.learnability:.4f}, "
                f"solve rate = {info.solve_rate:.0%}, "
                f"need learnability > {min_learnability:.4f}).\n"
                f"  Aim for a maze the agent can solve about half the time."
            )
