"""MaxMC Regret metric — rejects trivially easy mazes.

Computes the same MaxMC regret used in the ACCEL training loop:
    regret = mean_t[ max_return - V(s_t) ]
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np


@dataclass
class RegretInfo:
    """Regret metrics for a candidate maze.

    Uses the same MaxMC regret as the ACCEL training loop:
        score = mean_t[ max_return - V(s_t) ]
    """
    regret: float = 0.0
    max_return: float = 0.0
    episode_length: int = 0
    solved: bool = False


def compute_regret(
    candidate_trajectory: dict,
    max_steps: int = 250,
    stored_max_return: Optional[float] = None,
) -> RegretInfo:
    """Compute MaxMC regret — same formula as the ACCEL training loop.

    Args:
        candidate_trajectory: Dict with 'rewards', 'values', 'dones' keys
        max_steps: Max steps per episode (default 250)
        stored_max_return: Previously stored max_return. None for new levels.

    Returns:
        RegretInfo with MaxMC regret and supporting data
    """
    rewards = candidate_trajectory["rewards"]
    values = candidate_trajectory["values"]
    dones = candidate_trajectory["dones"]

    done_idx = np.where(dones.astype(bool))[0]
    episode_length = int(done_idx[0] + 1) if len(done_idx) > 0 else len(dones)

    ep_rewards = rewards[:episode_length].astype(np.float64)
    ep_values = values[:episode_length].astype(np.float64)

    rollout_return = float(np.sum(ep_rewards))
    if stored_max_return is not None:
        max_return = max(stored_max_return, rollout_return)
    else:
        max_return = rollout_return

    if episode_length > 0:
        regret = float(np.mean(max_return - ep_values))
    else:
        regret = 0.0

    solved = max_return > 0.001

    return RegretInfo(
        regret=regret,
        max_return=max_return,
        episode_length=episode_length,
        solved=solved,
    )


def check_regret(regret_info: RegretInfo, min_regret: float, issues: List[str]):
    """Check regret threshold and append issue if too low."""
    if regret_info.regret < min_regret:
        if not regret_info.solved:
            issues.append(
                f"Maze may be unsolvable or trivial — agent did not reach the goal "
                f"in {regret_info.episode_length} steps.\n"
                f"  Make sure the goal is reachable through a non-trivial path."
            )
        else:
            issues.append(
                f"Maze is too easy (MaxMC regret = {regret_info.regret:.3f}, "
                f"need > {min_regret:.3f}).\n"
                f"  The agent's value predictions nearly match its actual return "
                f"({regret_info.max_return:.3f}), meaning it is confident and correct.\n"
                f"  Agent solved in {regret_info.episode_length} steps.\n"
                f"  Add more dead ends, deceptive branches, or hidden corridors "
                f"to confuse the agent and increase regret."
            )
