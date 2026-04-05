"""Enrich ReferenceMaze objects with trajectory-based per-maze and pairwise metrics.

Takes ReferenceMaze objects (from BufferStatsExtractor) and adds detailed metrics
computed from agent rollout trajectories. Used by both test_generator.py and
optionally by injector.py when rich reference context is desired.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from llm.prompt_builder import (
    ReferenceMaze,
    MetricEntry,
    PairwiseMetricEntry,
    overlay_path_on_grid,
)
from metrics.standalone.per_step_entropy import compute_per_step_entropy
from metrics.standalone.per_step_regret import compute_per_step_regret
from metrics.standalone.per_step_action import compute_per_step_action
from metrics.standalone.regret import compute_regret
from metrics.standalone.learnability import compute_learnability
from metrics.standalone.value_error import compute_value_error
from metrics.pairwise.pos_dtw import position_trace_dtw
from metrics.utils import downsample, format_vector, truncate_at_done

logger = logging.getLogger(__name__)


def enrich_references_with_metrics(
    references: List[ReferenceMaze],
    trajectories: Optional[List[dict]] = None,
    downsample_points: int = 20,
    prompt_metrics: Optional[dict] = None,
    pairwise_metrics_cfg: Optional[dict] = None,
) -> Tuple[List[ReferenceMaze], List[PairwiseMetricEntry]]:
    """Enrich ReferenceMaze objects with trajectory-based metrics.

    Replaces each reference's metrics list with rich trajectory metrics when
    trajectory data is available. When trajectories is None, returns references
    unchanged with an empty pairwise list.

    Args:
        references: List of ReferenceMaze objects (from BufferStatsExtractor).
        trajectories: List of trajectory dicts from AgentEvaluator, one per reference.
        downsample_points: Max points when downsampling time-series vectors.
        prompt_metrics: Dict of metric_key -> bool controlling which per-maze
            metrics to include. None = all enabled.
        pairwise_metrics_cfg: Dict of metric_key -> bool controlling which
            pairwise metrics to include. None = all enabled.

    Returns:
        (enriched_references, pairwise_metrics) tuple.
    """
    if trajectories is None:
        return references, []

    pm = prompt_metrics or {}
    pw = pairwise_metrics_cfg or {}

    def _enabled(cfg_dict, key):
        return cfg_dict.get(key, True)

    enriched = []
    pairwise = []

    for i, ref in enumerate(references):
        if i >= len(trajectories):
            enriched.append(ref)
            continue

        traj = trajectories[i]
        metrics = list(ref.metrics)  # keep existing metrics (e.g. Regret Score)

        # Per-step entropy
        if _enabled(pm, "per_step_entropy"):
            ent_info = compute_per_step_entropy(traj["entropy"], traj["dones"])
            ds_entropy = downsample(ent_info["entropy"], downsample_points)
            metrics.append(MetricEntry(
                name="Per-Step Entropy",
                value=format_vector(ds_entropy),
                description=(
                    f"Policy uncertainty at each step "
                    f"(mean={ent_info['mean']:.3f}, max={ent_info['max']:.3f} "
                    f"at step {ent_info['max_step']}, ep_len={ent_info['episode_length']})"
                ),
                higher_is="more uncertain (harder decision points)",
                metric_key="per_step_entropy",
            ))

        # Per-step regret
        if _enabled(pm, "per_step_regret"):
            reg_info = compute_per_step_regret(
                traj["values"], traj["rewards"], traj["dones"]
            )
            ds_regret = downsample(reg_info["regret_curve"], downsample_points)
            metrics.append(MetricEntry(
                name="Per-Step Regret",
                value=format_vector(ds_regret),
                description=(
                    f"Difficulty at each step (max_return - V(s_t)), "
                    f"mean={reg_info['mean_regret']:.3f}, "
                    f"ep_len={reg_info['episode_length']}"
                ),
                higher_is="harder (agent expects lower return)",
                metric_key="per_step_regret",
            ))

        # Scalar regret
        if _enabled(pm, "scalar_regret"):
            regret_info = compute_regret(traj)
            metrics.append(MetricEntry(
                name="Scalar Regret",
                value=regret_info.regret,
                description=(
                    f"MaxMC regret (mean gap between best return and value estimate), "
                    f"solved={regret_info.solved}, ep_len={regret_info.episode_length}"
                ),
                higher_is="more learning potential",
                metric_key="scalar_regret",
            ))

        # SFL Learnability
        if _enabled(pm, "learnability") and "solve_rate" in traj:
            learn_info = compute_learnability(traj)
            metrics.append(MetricEntry(
                name="SFL Learnability",
                value=learn_info.learnability,
                description=(
                    f"p*(1-p) where p=solve_rate={learn_info.solve_rate:.0%} "
                    f"across {learn_info.n_rollouts} rollouts "
                    f"(max 0.25 at p=0.5)"
                ),
                higher_is="more at learning frontier (peak at p=0.5)",
                metric_key="learnability",
            ))

        # Action sequence
        if _enabled(pm, "action_sequence"):
            act_info = compute_per_step_action(traj["actions"], traj["dones"])
            ds_actions = downsample(act_info["actions"].astype(np.float64), downsample_points)
            metrics.append(MetricEntry(
                name="Action Sequence",
                value=format_vector(ds_actions, decimals=0),
                description=(
                    f"Agent's action at each step "
                    f"({act_info['num_unique_actions']} unique, "
                    f"dominant=action {act_info['dominant_action']} "
                    f"at {act_info['dominant_fraction']:.0%})"
                ),
                metric_key="action_sequence",
            ))

        # Value error profile
        if _enabled(pm, "value_error"):
            ve_info = compute_value_error(traj["values"], traj["rewards"], traj["dones"])
            ds_error = downsample(ve_info["error_curve"], downsample_points)
            metrics.append(MetricEntry(
                name="Value Error",
                value=format_vector(ds_error),
                description=(
                    f"Signed V(s_t)-G_t: positive=overconfident, negative=underconfident "
                    f"(mean={ve_info['mean_error']:.3f}, "
                    f"overconfident {ve_info['overconfident_frac']:.0%} of steps, "
                    f"ep_len={ve_info['episode_length']})"
                ),
                higher_is="more overconfident (agent expects more than reality)",
                metric_key="value_error",
            ))

        # Position vector
        if _enabled(pm, "position_vector"):
            ep_pos = truncate_at_done(traj["positions"], traj["dones"])
            ds_pos = downsample(ep_pos, downsample_points)
            pos_str = "[" + ", ".join(f"({int(p[0])},{int(p[1])})" for p in ds_pos) + "]"
            metrics.append(MetricEntry(
                name="Position Trace",
                value=pos_str,
                description=(
                    f"Agent (x,y) at each step "
                    f"(ep_len={len(ep_pos)}, downsampled to {len(ds_pos)} points)"
                ),
                metric_key="position_vector",
            ))

        # Path overlay
        path_overlay = ref.path_overlay
        if _enabled(pm, "path_overlay"):
            try:
                ep_pos = truncate_at_done(traj["positions"], traj["dones"])
                path_overlay = overlay_path_on_grid(ref.grid, ep_pos)
            except Exception:
                pass

        enriched.append(ReferenceMaze(
            grid=ref.grid,
            label=ref.label,
            metrics=metrics,
            path_overlay=path_overlay,
        ))

    # Pairwise position DTW
    if _enabled(pw, "position_dtw") and trajectories is not None and len(trajectories) >= 2:
        for i in range(len(trajectories)):
            for j in range(i + 1, len(trajectories)):
                ti, tj = trajectories[i], trajectories[j]
                dtw_result = position_trace_dtw(
                    ti["positions"], ti["dones"],
                    tj["positions"], tj["dones"],
                )
                pairwise.append(PairwiseMetricEntry(
                    maze_a_label=enriched[i].label,
                    maze_b_label=enriched[j].label,
                    name="Position DTW",
                    value=dtw_result["distance"],
                    description="Spatial path similarity (lower = more similar routes)",
                    metric_key="position_dtw",
                ))

    return enriched, pairwise
