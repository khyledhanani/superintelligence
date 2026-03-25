#!/usr/bin/env python3
"""Test the LLM maze generator using the saved replay buffer.

Loads levels from the buffer dump, converts them to ASCII grids,
builds prompts with various metric configurations, and generates new mazes.

Usage:
    # Generate 5 mazes with full top-5 metrics (default)
    python -m llm.test_generator

    # Dry run — build prompts with metrics, skip LLM calls
    python -m llm.test_generator --dry-run

    # Without agent rollouts (just buffer scores)
    python -m llm.test_generator --no-inject-metrics

    # With diversity feedback loop
    python -m llm.test_generator --feedback
"""

import argparse
import logging
import sys
import os
import json
import yaml
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from jaxued.environments.maze import Level
from vae.vae_level_utils import tokens_to_level, GRID_SIZE

from llm.maze_generator import MazeGenerator, GenerationConfig, GenerationResult
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
from metrics.pairwise.mode_transition import mode_transition_divergence, compute_baseline_stats
from metrics.pairwise.td_error_distribution import td_error_divergence
from metrics.utils import downsample, format_vector

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# --- Buffer loading ---

def load_buffer(path: str) -> dict:
    """Load a buffer dump .npz file.

    Returns dict with keys: tokens (N, 52), scores (N,), size (int), etc.
    """
    data = np.load(path)
    info = {k: data[k] for k in data.files}
    size = int(info.get("size", len(info["tokens"])))
    logger.info(f"Loaded buffer: {size} levels, tokens shape {info['tokens'].shape}")
    return info


def tokens_to_ascii(tokens: np.ndarray) -> str:
    """Convert a 52-token sequence to ASCII maze grid via Level.to_str()."""
    tokens_jax = jnp.array(tokens, dtype=jnp.int32)
    level = tokens_to_level(tokens_jax)
    return level.to_str()


def tokens_to_level_obj(tokens: np.ndarray):
    """Convert a 52-token sequence to a Level object."""
    tokens_jax = jnp.array(tokens, dtype=jnp.int32)
    return tokens_to_level(tokens_jax)


# --- Reference maze selection ---

def select_references(
    tokens: np.ndarray,
    scores: np.ndarray,
    size: int,
    n: int = 3,
    strategy: str = "top_regret",
) -> list:
    """Select reference mazes from the buffer.

    Args:
        tokens: (capacity, 52) token array
        scores: (capacity,) score array
        size: number of active levels
        n: number of references to select
        strategy: "top_regret", "random", or "diverse"

    Returns:
        List of (index, tokens, score) tuples
    """
    active_tokens = tokens[:size]
    active_scores = scores[:size]

    if strategy == "top_regret":
        # Top n by score (regret)
        top_indices = np.argsort(active_scores)[::-1][:n]
    elif strategy == "random":
        top_indices = np.random.choice(size, min(n, size), replace=False)
    elif strategy == "diverse":
        # Spread across score range: pick from quartiles
        sorted_idx = np.argsort(active_scores)
        step = max(1, len(sorted_idx) // n)
        top_indices = sorted_idx[::step][:n]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    refs = []
    for idx in top_indices:
        refs.append((int(idx), active_tokens[idx], float(active_scores[idx])))
    return refs


def select_references_diverse(
    tokens: np.ndarray,
    scores: np.ndarray,
    size: int,
    evaluator,
    n: int = 3,
    pool_size: int = 20,
    metric: str = "td_error_emd",
) -> list:
    """Select maximally diverse references using pairwise trajectory metrics.

    Picks a candidate pool (top by regret), runs agent rollouts, computes
    pairwise diversity, and greedily selects N references that maximize
    the minimum pairwise distance.

    Args:
        tokens: (capacity, 52) token array
        scores: (capacity,) score array
        size: number of active levels
        evaluator: AgentEvaluator instance (already loaded)
        n: number of references to select
        pool_size: candidate pool size (top by regret)
        metric: pairwise metric to maximize. One of:
            "td_error_emd" — Earth Mover's Distance between TD error distributions
            "experience_divergence" — KL divergence between mode transition matrices
            "position_dtw" — spatial path DTW distance

    Returns:
        List of (index, tokens, score) tuples
    """
    from metrics.pairwise.td_error_distribution import td_error_divergence
    from metrics.pairwise.mode_transition import (
        mode_transition_divergence,
        compute_baseline_stats,
    )
    from metrics.pairwise.pos_dtw import position_trace_dtw

    active_scores = scores[:size]
    pool_k = min(pool_size, size)

    # Candidate pool: top by regret
    pool_indices = np.argsort(active_scores)[::-1][:pool_k]
    logger.info(f"Diverse selection: pool of {pool_k} candidates, metric={metric}")

    # Roll out agent on candidate pool
    pool_levels = [tokens_to_level_obj(tokens[i]) for i in pool_indices]
    logger.info(f"Rolling out agent on {pool_k} candidate levels...")
    pool_trajectories = evaluator.evaluate_levels(pool_levels)

    # Compute baseline stats for mode transition if needed
    baseline_stats = None
    if metric == "experience_divergence":
        baseline_stats = compute_baseline_stats(pool_trajectories)
        logger.info(
            f"Mode baseline: error_threshold={baseline_stats['error_threshold']:.3f}, "
            f"entropy_threshold={baseline_stats['entropy_threshold']:.3f}"
        )

    # Compute pairwise distance matrix
    dist = np.zeros((pool_k, pool_k))
    for i in range(pool_k):
        for j in range(i + 1, pool_k):
            ti, tj = pool_trajectories[i], pool_trajectories[j]
            if metric == "td_error_emd":
                result = td_error_divergence(ti, ti["dones"], tj, tj["dones"])
                d = result["emd"]
            elif metric == "experience_divergence":
                result = mode_transition_divergence(
                    ti, ti["dones"], tj, tj["dones"],
                    entropy_a=ti.get("entropy"), entropy_b=tj.get("entropy"),
                    baseline_stats=baseline_stats,
                )
                d = result["kl_divergence"]
            elif metric == "position_dtw":
                result = position_trace_dtw(
                    ti["positions"], ti["dones"],
                    tj["positions"], tj["dones"],
                )
                d = result["distance"]
            else:
                raise ValueError(f"Unknown diverse metric: {metric}")
            dist[i, j] = d
            dist[j, i] = d

    # Greedy selection: maximize minimum pairwise distance
    # Start with the pair that has the highest distance
    best_pair = np.unravel_index(np.argmax(dist), dist.shape)
    selected = [best_pair[0], best_pair[1]]

    while len(selected) < n and len(selected) < pool_k:
        best_next = -1
        best_min_dist = -1
        for candidate in range(pool_k):
            if candidate in selected:
                continue
            # Min distance from candidate to any already-selected
            min_d = min(dist[candidate, s] for s in selected)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_next = candidate
        if best_next < 0:
            break
        selected.append(best_next)

    # Log selection
    for i, s in enumerate(selected):
        idx = int(pool_indices[s])
        logger.info(
            f"  Diverse ref {i+1}: buffer idx={idx}, "
            f"regret={float(active_scores[idx]):.4f}, "
            f"min_dist={min(dist[s, o] for o in selected if o != s):.4f}"
        )

    refs = []
    for s in selected:
        idx = int(pool_indices[s])
        refs.append((idx, tokens[idx], float(active_scores[idx])))
    return refs


def build_references_with_metrics(
    ref_data: list,
    trajectories: list = None,
    inject_regret: bool = True,
    inject_dtw: bool = False,
    downsample_points: int = 20,
    prompt_metrics: dict = None,
    pairwise_metrics_cfg: dict = None,
) -> tuple:
    """Build ReferenceMaze objects with configurable metric injection.

    Args:
        ref_data: List of (index, tokens, score) tuples
        trajectories: List of trajectory dicts from AgentEvaluator (or None)
        inject_regret: Include regret score as a metric (fallback to buffer score)
        inject_dtw: Unused (kept for CLI compat)
        downsample_points: Max points when downsampling vectors
        prompt_metrics: Dict of metric_key -> bool controlling which per-maze
            metrics to include. None = all enabled. Keys:
            per_step_entropy, per_step_regret, scalar_regret, action_sequence, path_overlay
        pairwise_metrics_cfg: Dict of metric_key -> bool controlling which
            pairwise metrics to include. None = all enabled. Keys: position_dtw

    Returns:
        Tuple of (references, pairwise_metrics):
            references: List of ReferenceMaze objects with per-maze metrics
            pairwise_metrics: List of PairwiseMetricEntry
    """
    # Default: all metrics enabled
    pm = prompt_metrics or {}
    pw = pairwise_metrics_cfg or {}

    def _enabled(cfg_dict, key):
        return cfg_dict.get(key, True)

    references = []
    pairwise_metrics = []

    for i, (idx, tokens, score) in enumerate(ref_data):
        grid = tokens_to_ascii(tokens)
        label = f"Maze {chr(65 + i)}"  # A, B, C, ...

        metrics = []

        if trajectories is not None and i < len(trajectories):
            traj = trajectories[i]

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

            # SFL Learnability (requires solve_rate from multi-rollout eval)
            if _enabled(pm, "learnability") and "solve_rate" in traj:
                learn_info = compute_learnability(traj)
                metrics.append(MetricEntry(
                    name="SFL Learnability",
                    value=learn_info.learnability,
                    description=(
                        f"p×(1-p) where p=solve_rate={learn_info.solve_rate:.0%} "
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
                from metrics.utils import truncate_at_done
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
            path_overlay = None
            if _enabled(pm, "path_overlay"):
                try:
                    from metrics.utils import truncate_at_done
                    ep_pos = truncate_at_done(traj["positions"], traj["dones"])
                    path_overlay = overlay_path_on_grid(grid, ep_pos)
                except Exception:
                    pass

            references.append(ReferenceMaze(
                grid=grid,
                label=label,
                metrics=metrics,
                path_overlay=path_overlay,
            ))
        else:
            # Fallback: just buffer score
            if inject_regret:
                metrics.append(MetricEntry(
                    name="Regret Score",
                    value=score,
                    description="Agent's learning potential on this maze",
                    higher_is="more to learn",
                    metric_key="scalar_regret",
                ))
            references.append(ReferenceMaze(
                grid=grid,
                label=label,
                metrics=metrics,
            ))

    # Pairwise position DTW between all reference pairs
    if _enabled(pw, "position_dtw") and trajectories is not None and len(trajectories) >= 2:
        for i in range(len(trajectories)):
            for j in range(i + 1, len(trajectories)):
                ti, tj = trajectories[i], trajectories[j]
                dtw_result = position_trace_dtw(
                    ti["positions"], ti["dones"],
                    tj["positions"], tj["dones"],
                )
                pairwise_metrics.append(PairwiseMetricEntry(
                    maze_a_label=references[i].label,
                    maze_b_label=references[j].label,
                    name="Position DTW",
                    value=dtw_result["distance"],
                    description="Spatial path similarity (lower = more similar routes)",
                    metric_key="position_dtw",
                ))

    # Pairwise mode transition divergence between all reference pairs
    if _enabled(pw, "mode_transition") and trajectories is not None and len(trajectories) >= 2:
        baseline = compute_baseline_stats(trajectories)
        logger.info(
            f"Mode baseline: error_threshold={baseline['error_threshold']:.3f}, "
            f"entropy_threshold={baseline['entropy_threshold']:.3f}"
        )
        for i in range(len(trajectories)):
            for j in range(i + 1, len(trajectories)):
                ti, tj = trajectories[i], trajectories[j]
                div_result = mode_transition_divergence(
                    ti, ti["dones"],
                    tj, tj["dones"],
                    entropy_a=ti.get("entropy"),
                    entropy_b=tj.get("entropy"),
                    baseline_stats=baseline,
                )
                pairwise_metrics.append(PairwiseMetricEntry(
                    maze_a_label=references[i].label,
                    maze_b_label=references[j].label,
                    name="Experience Divergence",
                    value=div_result["kl_divergence"],
                    description="Mode transition KL divergence (higher = more different agent experiences)",
                    metric_key="mode_transition",
                ))

    # Pairwise TD error distribution divergence
    if _enabled(pw, "td_error") and trajectories is not None and len(trajectories) >= 2:
        for i in range(len(trajectories)):
            for j in range(i + 1, len(trajectories)):
                ti, tj = trajectories[i], trajectories[j]
                td_result = td_error_divergence(ti, ti["dones"], tj, tj["dones"])
                pairwise_metrics.append(PairwiseMetricEntry(
                    maze_a_label=references[i].label,
                    maze_b_label=references[j].label,
                    name="TD Error EMD",
                    value=td_result["emd"],
                    description="Earth Mover's Distance between TD error distributions (higher = more different learning signals)",
                    metric_key="td_error",
                ))

    return references, pairwise_metrics


# --- Output directory ---

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_levels")


# --- Visualization ---

def print_maze(grid: str, label: str = ""):
    """Pretty-print a maze grid."""
    if label:
        print(f"\n{'=' * 20} {label} {'=' * 20}")
    for row in grid.split('\n'):
        print(f"  {row}")
    print()


def print_result(result: GenerationResult, idx: int):
    """Print a generation result with details."""
    print(f"\n{'#' * 50}")
    print(f"  Generation {idx + 1}: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"  Attempts: {result.attempts}")
    print(f"  Latency: {result.latency_ms:.0f}ms")
    if result.diversity_attempts > 0:
        print(f"  Diversity checks: {result.diversity_attempts}")
    if result.gate_metrics:
        print(f"  Gate metrics:")
        for k, v in result.gate_metrics.items():
            print(f"    {k}: {v:.4f}" if isinstance(v, (int, float)) else f"    {k}: {v}")
    if result.diversity_issues:
        print(f"  Unresolved diversity issues:")
        for issue in result.diversity_issues:
            print(f"    - {issue}")
    if result.errors:
        print(f"  Errors:")
        for e in result.errors:
            print(f"    - {e}")
    if result.success and result.grid:
        print_maze(result.grid, "Generated Maze")
    print(f"{'#' * 50}")


def grid_to_image(grid_str: str) -> np.ndarray:
    """Convert ASCII maze grid to an RGB image array.

    Colors: Wall (#) dark gray, Floor (.) white, Agent (>v<^) blue, Goal (G) green.
    """
    rows = grid_str.strip().split('\n')
    h, w = len(rows), len(rows[0])
    img = np.ones((h, w, 3), dtype=np.float32)

    char_colors = {
        '#': [0.2, 0.2, 0.2],
        '.': [1.0, 1.0, 1.0],
        '>': [0.2, 0.4, 0.9],
        'v': [0.2, 0.4, 0.9],
        '<': [0.2, 0.4, 0.9],
        '^': [0.2, 0.4, 0.9],
        'G': [0.2, 0.8, 0.3],
    }

    for y, row in enumerate(rows):
        for x, c in enumerate(row):
            if c in char_colors:
                img[y, x] = char_colors[c]
    return img


def plot_maze_with_path(ax, grid_str: str, positions=None, dones=None,
                        color='blue', title='', title_color='dimgray',
                        title_bold=False):
    """Plot a maze grid with optional agent trajectory overlay (deep-dive style).

    Path is drawn as line segments with time-gradient alpha (faint early, solid late),
    with circle marker at start and square at end.
    """
    img = grid_to_image(grid_str)
    ax.imshow(img, origin='upper')

    if positions is not None:
        from metrics.utils import truncate_at_done
        if dones is not None:
            ep_pos = truncate_at_done(positions, dones)
        else:
            ep_pos = positions

        # If agent solved the level, extend path to the goal position
        # (positions record pre-step state, so the goal arrival is missing)
        solved = dones is not None and np.any(dones)
        if solved:
            rows = grid_str.strip().split('\n')
            for gy, row in enumerate(rows):
                for gx, c in enumerate(row):
                    if c == 'G':
                        ep_pos = np.concatenate([ep_pos, [[gx, gy]]], axis=0)
                        break

        n = len(ep_pos)
        if n > 0:
            for t in range(n - 1):
                alpha = 0.3 + 0.7 * (t / max(n - 1, 1))
                ax.plot([ep_pos[t, 0], ep_pos[t+1, 0]],
                        [ep_pos[t, 1], ep_pos[t+1, 1]],
                        color=color, alpha=alpha, linewidth=2)
            ax.plot(ep_pos[0, 0], ep_pos[0, 1], 'o', color=color,
                    markersize=8, label='start')
            if n > 1:
                end_label = 'goal' if solved else 'end'
                ax.plot(ep_pos[-1, 0], ep_pos[-1, 1], 's', color=color,
                        markersize=8, label=end_label)
            ax.legend(fontsize=7, loc='upper right')
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(12.5, -0.5)
    ax.set_title(title, fontsize=9,
                 color=title_color,
                 fontweight='bold' if title_bold else 'normal')


def save_results(
    results: list,
    references: list,
    model: str,
    run_dir: str,
    ref_trajectories: list = None,
    gen_trajectories: list = None,
    embedding_metric: str = "td_error_emd",
):
    """Save generated mazes as text files, JSON metadata, and a PNG visualization.

    Output structure in run_dir:
        maze_001.txt          — ASCII grid for each successful maze
        maze_002.txt
        ...
        metadata.json         — Full run metadata (model, config, per-maze stats)
        visualization.png     — Grid of all generated mazes + references
    """
    os.makedirs(run_dir, exist_ok=True)

    successful = [(i, r) for i, r in enumerate(results) if r.success]

    # --- Save individual maze text files ---
    for seq, (orig_idx, result) in enumerate(successful):
        maze_path = os.path.join(run_dir, f"maze_{seq + 1:03d}.txt")
        with open(maze_path, 'w') as f:
            f.write(result.grid)
        logger.info(f"Saved {maze_path}")

    # --- Save metadata JSON ---
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "num_requested": len(results),
        "num_successful": len(successful),
        "success_rate": len(successful) / len(results) if results else 0,
        "total_latency_ms": sum(r.latency_ms for r in results),
        "reference_mazes": [
            {"label": ref.label, "grid": ref.grid}
            for ref in references
        ],
        "generated_mazes": [],
    }
    # Save system prompt once (same for all generations)
    first_result = results[0] if results else None
    if first_result and first_result.system_prompt:
        metadata["system_prompt"] = first_result.system_prompt

    for seq, (orig_idx, result) in enumerate(successful):
        entry = {
            "index": seq + 1,
            "grid": result.grid,
            "attempts": result.attempts,
            "latency_ms": result.latency_ms,
            "errors": result.errors,
            "user_prompt": result.user_prompt,
            "raw_responses": result.raw_responses,
            "thinking_logs": [t for t in result.thinking_logs if t is not None] or None,
        }
        if entry["thinking_logs"] is None:
            del entry["thinking_logs"]
        if result.feedback_prompts:
            entry["feedback_prompts"] = result.feedback_prompts
        if result.gate_metrics:
            entry["gate_metrics"] = {
                k: (round(v, 6) if isinstance(v, (int, float)) else v)
                for k, v in result.gate_metrics.items()
            }
            entry["diversity_attempts"] = result.diversity_attempts
        if result.diversity_issues:
            entry["diversity_issues"] = result.diversity_issues
        # Multi-rollout stats
        gt = gen_trajectories[seq] if gen_trajectories and seq < len(gen_trajectories) else None
        if gt and "solve_rate" in gt:
            entry["multi_rollout"] = {
                "n_rollouts": 100,
                "solve_rate": round(gt["solve_rate"], 3),
                "best_return": round(gt["best_return"], 6),
                "mean_return": round(float(np.mean(gt["all_returns"])), 6),
                "std_return": round(float(np.std(gt["all_returns"])), 6),
            }
        metadata["generated_mazes"].append(entry)
    # Also record failures
    metadata["failed_mazes"] = []
    for i, r in enumerate(results):
        if not r.success:
            metadata["failed_mazes"].append({
                "original_index": i + 1,
                "attempts": r.attempts,
                "latency_ms": r.latency_ms,
                "errors": r.errors,
                "raw_responses": r.raw_responses,
                "thinking_logs": [t for t in r.thinking_logs if t is not None] or None,
                "feedback_prompts": r.feedback_prompts,
            })
            if metadata["failed_mazes"][-1]["thinking_logs"] is None:
                del metadata["failed_mazes"][-1]["thinking_logs"]

    # Record rejected candidates (gate failures during feedback loop)
    metadata["rejected_candidates"] = []
    for i, r in enumerate(results):
        for j, rc in enumerate(r.rejected_candidates):
            entry = {
                "generation_index": i + 1,
                "attempt": j + 1,
                "grid": rc.grid,
                "gate_summary": {
                    k: (round(v, 6) if isinstance(v, (int, float)) else v)
                    for k, v in rc.gate_summary.items()
                },
                "issues": rc.issues,
            }
            metadata["rejected_candidates"].append(entry)

    # Save rejected candidate grids as text files
    all_rejected = []
    for r in results:
        all_rejected.extend(r.rejected_candidates)
    for k, rc in enumerate(all_rejected):
        rej_path = os.path.join(run_dir, f"rejected_{k + 1:03d}.txt")
        with open(rej_path, 'w') as f:
            f.write(rc.grid)
        logger.info(f"Saved {rej_path}")

    meta_path = os.path.join(run_dir, "metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved {meta_path}")

    # --- Render visualization PNG ---
    max_cols = 3  # max mazes per row

    ref_grids = [(ref.label, ref.grid) for ref in references]
    gen_grids = [(f"Generated {orig_idx + 1}", r.grid) for orig_idx, r in successful]

    # Collect all rejected candidates across all generations
    # Color scheme: blue=reference, green=accepted, yellow=rejected-diversity,
    #               red=rejected-difficulty, orange=rejected-both
    all_rejected = []
    for r in results:
        all_rejected.extend(r.rejected_candidates)

    n_refs = len(ref_grids)
    n_gens = len(gen_grids)
    n_rejs = len(all_rejected)

    if n_refs == 0 and n_gens == 0 and n_rejs == 0:
        logger.warning("No mazes to visualize")
        return run_dir

    # Layout: refs fill rows, then rejected, then accepted
    import math
    ref_rows = math.ceil(n_refs / max_cols) if n_refs > 0 else 0
    rej_rows = math.ceil(n_rejs / max_cols) if n_rejs > 0 else 0
    gen_rows = math.ceil(n_gens / max_cols) if n_gens > 0 else 0
    cols = min(max(n_refs, n_gens, n_rejs, 1), max_cols)
    rows = ref_rows + rej_rows + gen_rows
    if rows == 0:
        rows = 1

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.5 * rows))
    fig.suptitle(
        f"LLM Maze Generation — {model}\n"
        f"{n_gens}/{len(results)} accepted, {n_rejs} rejected, "
        f"{sum(r.latency_ms for r in results) / 1000:.0f}s total",
        fontsize=12, fontweight='bold',
    )

    # Ensure axes is always 2D
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for ax_row in axes:
        for ax in ax_row:
            ax.axis('off')

    # Reference mazes (blue paths), filling rows of max_cols
    for i, (label, grid) in enumerate(ref_grids):
        row = i // cols
        col = i % cols
        if row >= ref_rows:
            break
        ax = axes[row, col]
        rt = ref_trajectories[i] if ref_trajectories and i < len(ref_trajectories) else None
        ref_title = label
        if rt:
            ri = compute_regret(rt)
            ref_title += f"\nregret={ri.regret:.3f} solved={ri.solved}"
            if "solve_rate" in rt:
                ref_title += f" solve={rt['solve_rate']:.0%}"
        plot_maze_with_path(
            ax, grid,
            positions=rt["positions"] if rt else None,
            dones=rt["dones"] if rt else None,
            color='blue', title=ref_title,
        )

    # Rejected mazes — color by failure reason
    for i, rc in enumerate(all_rejected):
        row = ref_rows + (i // cols)
        col = i % cols
        if row >= ref_rows + rej_rows:
            break
        ax = axes[row, col]
        rt = rc.trajectory

        # Color + reason tag by failure type
        if rc.failed_difficulty and rc.failed_diversity:
            path_color = 'orange'
            title_color = 'orangered'
            reason = "difficulty+diversity"
        elif rc.failed_difficulty:
            path_color = 'red'
            title_color = 'firebrick'
            reason = "difficulty"
        else:
            path_color = 'gold'
            title_color = 'darkgoldenrod'
            reason = "diversity"

        # Build title with rejection reason + metrics
        gate_sum = rc.gate_summary
        title = f"Rejected {i + 1} [{reason}]"
        if rt and "solve_rate" in rt:
            title += f"\nsolve={rt['solve_rate']:.0%}"
        if "learnability" in gate_sum:
            title += f" learn={gate_sum['learnability']:.4f}"
        elif "regret" in gate_sum:
            title += f" regret={gate_sum['regret']:.3f}"
        if "mean_diversity" in gate_sum:
            title += f" div={gate_sum['min_diversity']:.4f}"

        plot_maze_with_path(
            ax, rc.grid,
            positions=rt["positions"] if rt else None,
            dones=rt["dones"] if rt else None,
            color=path_color, title=title,
            title_color=title_color,
        )

    # Accepted generated mazes (green paths), filling rows after rejected
    for i, (orig_idx, result) in enumerate(successful):
        row = ref_rows + rej_rows + (i // cols)
        col = i % cols
        if row >= rows:
            break

        ax = axes[row, col]
        gt = gen_trajectories[i] if gen_trajectories and i < len(gen_trajectories) else None
        force_accepted = bool(result.diversity_issues)

        # Build title with regret + diversity + solve rate
        title = f"Force-Accepted {i + 1}" if force_accepted else f"Accepted {i + 1}"
        if gt:
            gi = compute_regret(gt)
            solve_str = f"solve={gt['solve_rate']:.0%}" if "solve_rate" in gt else ""
            title += f"\n{solve_str} regret={gi.regret:.3f}"
        if result.gate_metrics:
            diversity_val = result.gate_metrics.get('min_diversity')
            if diversity_val is not None:
                title += f" div={diversity_val:.4f}"

        plot_maze_with_path(
            ax, result.grid,
            positions=gt["positions"] if gt else None,
            dones=gt["dones"] if gt else None,
            color='green', title=title,
            title_color='darkorange' if force_accepted else 'darkgreen',
            title_bold=True,
        )

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Reference'),
    ]
    if n_rejs > 0:
        # Only add legend entries for failure types that actually occurred
        has_diff = any(rc.failed_difficulty and not rc.failed_diversity for rc in all_rejected)
        has_div = any(rc.failed_diversity and not rc.failed_difficulty for rc in all_rejected)
        has_both = any(rc.failed_difficulty and rc.failed_diversity for rc in all_rejected)
        if has_div:
            legend_elements.append(Line2D([0], [0], color='gold', lw=2, label='Rejected (diversity)'))
        if has_diff:
            legend_elements.append(Line2D([0], [0], color='red', lw=2, label='Rejected (difficulty)'))
        if has_both:
            legend_elements.append(Line2D([0], [0], color='orange', lw=2, label='Rejected (both)'))
    if n_gens > 0:
        has_force = any(r.diversity_issues for _, r in successful)
        has_clean = any(not r.diversity_issues for _, r in successful)
        if has_clean:
            legend_elements.append(Line2D([0], [0], color='green', lw=2, label='Accepted'))
        if has_force:
            legend_elements.append(Line2D([0], [0], color='green', lw=2, alpha=0.6,
                                          label='Force-Accepted', linestyle='--'))
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(legend_elements),
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout()
    viz_path = os.path.join(run_dir, "visualization.png")
    fig.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved {viz_path}")

    # --- Diversity embedding plot ---
    # Compute pairwise distances between all refs + rejected + accepted, embed with t-SNE
    _emb_metric = embedding_metric
    all_trajs = []
    all_labels = []
    all_colors = []
    all_markers = []
    n_ref_in_emb = 0
    if ref_trajectories:
        for i, rt in enumerate(ref_trajectories):
            if rt is not None:
                all_trajs.append(rt)
                all_labels.append(references[i].label if i < len(references) else f"Ref {i+1}")
                all_colors.append('blue')
                all_markers.append('o')
                n_ref_in_emb += 1
    # Rejected candidates — colored by failure reason
    for i, rc in enumerate(all_rejected):
        rt = rc.trajectory
        if rt is not None and "dones" in rt:
            all_trajs.append(rt)
            all_labels.append(f"Rej {i+1}")
            if rc.failed_difficulty and rc.failed_diversity:
                all_colors.append('orange')
            elif rc.failed_difficulty:
                all_colors.append('red')
            else:
                all_colors.append('gold')
            all_markers.append('X')
    # Accepted generated (green, or green+red-edge for force-accepted)
    all_force_accepted = []  # track which embedding indices are force-accepted
    if gen_trajectories:
        for i, gt in enumerate(gen_trajectories):
            if gt is not None:
                all_trajs.append(gt)
                force = bool(successful[i][1].diversity_issues) if i < len(successful) else False
                label = f"Force-Accepted {i+1}" if force else f"Accepted {i+1}"
                all_labels.append(label)
                all_colors.append('green')
                all_markers.append('*')
                all_force_accepted.append(force)

    if len(all_trajs) >= 3:
        try:
            from sklearn.manifold import TSNE

            def _pairwise_distance(t1, t2, metric):
                """Compute pairwise distance between two trajectories."""
                if metric == "td_error_emd":
                    from metrics.pairwise.td_error_distribution import td_error_divergence
                    return td_error_divergence(t1, t1["dones"], t2, t2["dones"])["emd"]
                elif metric == "experience_divergence":
                    from metrics.pairwise.mode_transition import mode_transition_divergence
                    return mode_transition_divergence(
                        t1, t1["dones"], t2, t2["dones"],
                        entropy_a=t1.get("entropy"), entropy_b=t2.get("entropy"),
                    )["kl_divergence"]
                elif metric == "position_dtw":
                    from metrics.pairwise.pos_dtw import position_trace_dtw
                    return position_trace_dtw(
                        t1["positions"], t1["dones"], t2["positions"], t2["dones"],
                    )["distance"]
                else:
                    from metrics.pairwise.td_error_distribution import td_error_divergence
                    return td_error_divergence(t1, t1["dones"], t2, t2["dones"])["emd"]

            _emb_label = {
                "td_error_emd": "TD Error EMD",
                "experience_divergence": "Experience Divergence",
                "position_dtw": "Position DTW",
            }.get(_emb_metric, _emb_metric)

            n = len(all_trajs)
            dist_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    d = _pairwise_distance(all_trajs[i], all_trajs[j], _emb_metric)
                    dist_matrix[i, j] = d
                    dist_matrix[j, i] = d

            # t-SNE on precomputed distance matrix
            perplexity = min(5, n - 1)
            embedding = TSNE(
                n_components=2, metric="precomputed",
                perplexity=perplexity, random_state=42,
                init="random",
            ).fit_transform(dist_matrix)

            # Build per-point edge colors: red edge for force-accepted
            n_before_accepted = n_ref_in_emb + sum(1 for rc in all_rejected if rc.trajectory is not None and "dones" in rc.trajectory)
            edge_colors = []
            for i in range(n):
                if i >= n_before_accepted and all_markers[i] == '*':
                    fa_idx = i - n_before_accepted
                    if fa_idx < len(all_force_accepted) and all_force_accepted[fa_idx]:
                        edge_colors.append('red')
                    else:
                        edge_colors.append('black')
                else:
                    edge_colors.append('black')

            fig_emb, ax_emb = plt.subplots(1, 1, figsize=(7, 6))
            for i in range(n):
                ax_emb.scatter(
                    embedding[i, 0], embedding[i, 1],
                    c=all_colors[i], s=150 if all_markers[i] == '*' else 100,
                    marker=all_markers[i], zorder=5,
                    edgecolors=edge_colors[i],
                    linewidths=2.0 if edge_colors[i] == 'red' else 0.5,
                )
                ax_emb.annotate(
                    all_labels[i], (embedding[i, 0], embedding[i, 1]),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=7, color=all_colors[i], fontweight='bold',
                )

            # Plot reference centroid as black dot
            if n_ref_in_emb >= 2:
                ref_emb = embedding[:n_ref_in_emb]
                centroid = ref_emb.mean(axis=0)
                ax_emb.scatter(
                    centroid[0], centroid[1],
                    c='black', s=200, marker='D', zorder=6,
                    edgecolors='white', linewidths=1.5,
                )
                ax_emb.annotate(
                    "Centroid", (centroid[0], centroid[1]),
                    textcoords="offset points", xytext=(6, -10),
                    fontsize=7, color='black', fontweight='bold',
                )

            # Draw edges with distance labels for nearest pairs
            for i in range(n):
                for j in range(i + 1, n):
                    ax_emb.plot(
                        [embedding[i, 0], embedding[j, 0]],
                        [embedding[i, 1], embedding[j, 1]],
                        'gray', alpha=0.15, linewidth=0.5,
                    )

            ax_emb.set_title(
                f"Diversity Embedding ({_emb_label})",
                fontsize=11, fontweight='bold',
            )
            ax_emb.set_xlabel("t-SNE dim 1", fontsize=9)
            ax_emb.set_ylabel("t-SNE dim 2", fontsize=9)
            ax_emb.grid(alpha=0.2)

            # Legend
            from matplotlib.lines import Line2D
            emb_legend = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                       markersize=8, markeredgecolor='black', label='Reference'),
                Line2D([0], [0], marker='D', color='w', markerfacecolor='black',
                       markersize=8, markeredgecolor='white', label='Ref Centroid'),
            ]
            # Only add legend entries for failure types present
            has_diff = any(rc.failed_difficulty and not rc.failed_diversity for rc in all_rejected)
            has_div = any(rc.failed_diversity and not rc.failed_difficulty for rc in all_rejected)
            has_both = any(rc.failed_difficulty and rc.failed_diversity for rc in all_rejected)
            if has_div:
                emb_legend.append(Line2D([0], [0], marker='X', color='w', markerfacecolor='gold',
                                         markersize=8, markeredgecolor='black', label='Rej (diversity)'))
            if has_diff:
                emb_legend.append(Line2D([0], [0], marker='X', color='w', markerfacecolor='red',
                                         markersize=8, markeredgecolor='black', label='Rej (difficulty)'))
            if has_both:
                emb_legend.append(Line2D([0], [0], marker='X', color='w', markerfacecolor='orange',
                                         markersize=8, markeredgecolor='black', label='Rej (both)'))
            if n_gens > 0:
                has_force_emb = any(all_force_accepted)
                has_clean_emb = any(not f for f in all_force_accepted)
                if has_clean_emb:
                    emb_legend.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='green',
                                             markersize=10, markeredgecolor='black', label='Accepted'))
                if has_force_emb:
                    emb_legend.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='green',
                                             markersize=10, markeredgecolor='red',
                                             markeredgewidth=2, label='Force-Accepted'))
            ax_emb.legend(handles=emb_legend, loc='best', fontsize=8, framealpha=0.9)

            emb_path = os.path.join(run_dir, "diversity_embedding.png")
            fig_emb.savefig(emb_path, dpi=150, bbox_inches='tight')
            plt.close(fig_emb)
            logger.info(f"Saved {emb_path}")
        except ImportError:
            logger.warning("sklearn not installed — skipping diversity embedding")
        except Exception as e:
            logger.warning(f"Diversity embedding failed: {e}")

    return run_dir


# --- Main test ---

def run_test(args):
    """Run the maze generation test."""

    # Load buffer
    logger.info(f"Loading buffer from {args.buffer_path}...")
    buf = load_buffer(args.buffer_path)
    size = int(buf["size"])
    tokens = buf["tokens"]
    scores = buf["scores"]

    # Load agent early if needed for diverse selection or metrics
    evaluator = None
    if args.inject_metrics or args.strategy == "diverse":
        from llm.agent_evaluator import AgentEvaluator
        logger.info(f"Loading agent from {args.agent_dir} for metric computation...")
        evaluator = AgentEvaluator.from_checkpoint(args.agent_dir, num_steps=args.num_steps)

    # Select reference mazes
    logger.info(f"Selecting {args.num_refs} reference mazes (strategy={args.strategy})...")
    if args.strategy == "diverse" and evaluator is not None:
        ref_data = select_references_diverse(
            tokens, scores, size, evaluator,
            n=args.num_refs,
            pool_size=args.diverse_pool_size,
            metric=args.diverse_metric,
        )
    else:
        ref_data = select_references(tokens, scores, size, n=args.num_refs, strategy=args.strategy)

    # Roll out agent on selected reference levels to get trajectory data
    ref_trajectories = None
    if args.inject_metrics and evaluator is not None:
        ref_levels = []
        for idx, tok, score in ref_data:
            ref_levels.append(tokens_to_level_obj(tok))

        logger.info(f"Rolling out agent on {len(ref_levels)} reference levels...")
        ref_trajectories = evaluator.evaluate_levels(ref_levels)
        logger.info("Reference trajectories collected")

    # Build references with metrics (configurable via prompt_metrics/pairwise_metrics)
    references, pairwise_metrics = build_references_with_metrics(
        ref_data,
        trajectories=ref_trajectories,
        inject_regret=args.inject_regret,
        downsample_points=args.downsample_points,
        prompt_metrics=args.prompt_metrics,
        pairwise_metrics_cfg=args.pairwise_metrics_cfg,
    )

    # Print reference mazes
    print("\n" + "=" * 60)
    print("  REFERENCE MAZES FROM BUFFER")
    print("=" * 60)
    for ref in references:
        print_maze(ref.grid, ref.label)
        if ref.metrics:
            for m in ref.metrics:
                print(f"  {m.format()}")

    if pairwise_metrics:
        print("\n  PAIRWISE METRICS:")
        for pm in pairwise_metrics:
            print(f"  {pm.format()}")

    # Build global metrics
    global_metrics = None
    if args.inject_buffer_stats:
        active_scores = scores[:size]
        global_metrics = [
            MetricEntry(
                name="Buffer Size",
                value=size,
                description="Number of levels in the replay buffer",
            ),
            MetricEntry(
                name="Mean Regret",
                value=float(np.mean(active_scores)),
                description="Average regret across all buffer levels",
            ),
            MetricEntry(
                name="Max Regret",
                value=float(np.max(active_scores)),
                description="Highest regret level in buffer",
            ),
            MetricEntry(
                name="Score Std Dev",
                value=float(np.std(active_scores)),
                description="Spread of regret scores",
            ),
        ]

    # Build prompt and show it
    from llm.prompt_builder import build_generation_prompt, SYSTEM_PROMPT
    user_prompt = build_generation_prompt(
        references=references,
        pairwise_metrics=pairwise_metrics,
        global_metrics=global_metrics,
        instruction=args.instruction,
    )

    print("\n" + "=" * 60)
    print("  SYSTEM PROMPT")
    print("=" * 60)
    print(SYSTEM_PROMPT)

    print("\n" + "=" * 60)
    print("  USER PROMPT")
    print("=" * 60)
    print(user_prompt)

    if args.dry_run:
        print("\n[DRY RUN] Skipping LLM calls.")
        return

    # Configure generator
    config = GenerationConfig(
        provider=args.provider,
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        thinking=args.thinking,
        thinking_budget=args.thinking_budget,
        thinking_in_output=args.thinking_in_output,
        max_retries=args.max_retries,
        timeout=args.timeout,
        min_walls=args.min_walls,
        min_path_distance=args.min_path_distance,
        validate_solvable=args.validate_solvable,
    )
    generator = MazeGenerator(config)

    # Generate mazes
    if args.feedback:
        print("\n" + "=" * 60)
        print(f"  GENERATING {args.n} MAZES WITH METRIC FEEDBACK LOOP")
        print("=" * 60)

        from llm.decision_gate import DiversityThresholds

        # Reuse evaluator/trajectories if already loaded for metrics
        if not args.inject_metrics:
            from llm.agent_evaluator import AgentEvaluator
            evaluator = AgentEvaluator.from_checkpoint(args.agent_dir, num_steps=args.num_steps)
            ref_levels = [tokens_to_level_obj(tok) for _, tok, _ in ref_data]
            ref_trajectories = evaluator.evaluate_levels(ref_levels)

        ref_labels = [ref.label for ref in references]

        thresholds = DiversityThresholds(
            difficulty_threshold=args.difficulty_threshold,
            difficulty_metric=args.difficulty_metric,
            min_diversity=args.min_diversity,
            diversity_metric=args.diversity_metric,
        )

        # Fit CENIE GMM on buffer trajectories if using CENIE diversity metric
        cenie_model = None
        if args.diversity_metric == "cenie" and evaluator is not None:
            from metrics.standalone.cenie import fit_cenie_model
            # Use all reference trajectories (already computed) to fit the GMM.
            # For a fuller model, roll out on more buffer levels.
            cenie_trajs = ref_trajectories
            if "hstates" not in ref_trajectories[0]:
                logger.warning("Reference trajectories missing hstates — CENIE requires them")
            else:
                # Optionally expand to more buffer levels for better coverage
                n_cenie_levels = min(size, 50)  # fit on up to 50 buffer levels
                if n_cenie_levels > len(ref_trajectories):
                    logger.info(f"Rolling out agent on {n_cenie_levels} buffer levels for CENIE GMM...")
                    cenie_indices = np.argsort(-scores[:size])[:n_cenie_levels]
                    cenie_levels = [tokens_to_level_obj(tokens[i]) for i in cenie_indices]
                    cenie_trajs = evaluator.evaluate_levels(cenie_levels)
                    logger.info(f"CENIE trajectories collected ({n_cenie_levels} levels)")
                cenie_model = fit_cenie_model(cenie_trajs)

        results = generator.generate_batch_with_feedback(
            n=args.n,
            agent_evaluator=evaluator,
            reference_trajectories=ref_trajectories,
            reference_labels=ref_labels,
            references=references,
            pairwise_metrics=pairwise_metrics,
            global_metrics=global_metrics,
            instruction=args.instruction,
            diversity_thresholds=thresholds,
            max_diversity_retries=args.max_diversity_retries,
            n_rollouts=args.n_rollouts,
            cenie_model=cenie_model,
        )
    else:
        print("\n" + "=" * 60)
        print(f"  GENERATING {args.n} MAZES")
        print("=" * 60)

        results = generator.generate_batch(
            n=args.n,
            references=references,
            pairwise_metrics=pairwise_metrics,
            global_metrics=global_metrics,
            instruction=args.instruction,
        )

    for i, result in enumerate(results):
        print_result(result, i)

    # Summary
    successes = sum(1 for r in results if r.success)
    total_attempts = sum(r.attempts for r in results)
    total_latency = sum(r.latency_ms for r in results)
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Success rate: {successes}/{args.n} ({100 * successes / args.n:.0f}%)")
    print(f"  Total attempts: {total_attempts}")
    print(f"  Total latency: {total_latency:.0f}ms")
    if successes > 0:
        print(f"  Avg latency per success: {total_latency / successes:.0f}ms")

    # Show all successful mazes side by side
    if successes > 0:
        print("\n" + "=" * 60)
        print("  ALL GENERATED MAZES")
        print("=" * 60)
        for i, result in enumerate(results):
            if result.success:
                print_maze(result.grid, f"Generated {i + 1}")

    # Roll out agent on successful generated mazes for path overlay
    gen_trajectories = None
    if args.inject_metrics and successes > 0:
        gen_levels = []
        for result in results:
            if result.success:
                try:
                    level = Level.from_str(result.grid)
                    gen_levels.append(level)
                except Exception as e:
                    logger.warning(f"Could not parse generated maze for rollout: {e}")
                    gen_levels.append(None)
        # Multi-rollout evaluation for generated levels (100 rollouts each)
        gen_trajectories = []
        for i, lv in enumerate(gen_levels):
            if lv is not None:
                logger.info(f"Multi-rollout (100x) on generated maze {i+1}...")
                traj = evaluator.evaluate_level_multi_rollout(lv, n_rollouts=args.n_rollouts)
                logger.info(
                    f"  solve_rate={traj['solve_rate']:.0%}, "
                    f"best_return={traj['best_return']:.3f}"
                )
                gen_trajectories.append(traj)
            else:
                gen_trajectories.append(None)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model.replace(":", "_").replace("/", "_")
    run_dir = os.path.join(OUTPUT_DIR, f"{timestamp}_{model_short}")
    save_results(
        results, references, args.model, run_dir,
        ref_trajectories=ref_trajectories,
        gen_trajectories=gen_trajectories,
        embedding_metric=args.embedding_metric,
    )
    print(f"\n  Results saved to: {run_dir}/")
    print(f"    - maze_XXX.txt files (ASCII grids)")
    print(f"    - metadata.json (run details)")
    print(f"    - visualization.png (visual grid)")


def load_config(config_path: str) -> dict:
    """Load YAML config file and return as dict.

    Keeps provider sub-dicts so that CLI --provider can override
    which block gets flattened.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def flatten_provider_config(cfg: dict, provider: str) -> None:
    """Flatten provider-specific settings (base_url, api_key_env) into cfg.

    Called after CLI arg parsing so --provider overrides work correctly.
    """
    provider_cfg = cfg.get(provider, {})
    cfg["base_url"] = provider_cfg.get("base_url", "")
    cfg["api_key_env"] = provider_cfg.get("api_key_env", "")
    # Remove provider sub-dicts (not needed downstream)
    cfg.pop("ollama", None)
    cfg.pop("openrouter", None)


def main():
    # Pre-parse to find --config before building the full parser
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None, help="Path to YAML config file")
    pre_args, _ = pre_parser.parse_known_args()

    # Load config defaults
    cfg = {}
    config_path = pre_args.config
    if config_path is None:
        # Auto-detect config.yaml next to this script
        default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        if os.path.exists(default_config):
            config_path = default_config
    if config_path:
        logger.info(f"Loading config from {config_path}")
        cfg = load_config(config_path)

    parser = argparse.ArgumentParser(description="Test LLM maze generator with saved buffer")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    # All defaults come from config.yaml. The cfg.get() calls below have no
    # hardcoded fallbacks — if config.yaml is missing a key, argparse will
    # require it via CLI flag instead of silently using a stale default.

    parser.add_argument("--buffer-path", default=cfg.get("buffer_path"),
                        help="Path to buffer dump .npz file")
    parser.add_argument("--n", type=int, default=cfg.get("n"),
                        help="Number of mazes to generate")
    parser.add_argument("--num-refs", type=int, default=cfg.get("num_refs"),
                        help="Number of reference mazes")
    parser.add_argument("--strategy", choices=["top_regret", "random", "diverse"],
                        default=cfg.get("strategy"),
                        help="Reference selection strategy")
    parser.add_argument("--diverse-metric",
                        choices=["td_error_emd", "experience_divergence", "position_dtw"],
                        default=cfg.get("diverse_metric", "td_error_emd"),
                        help="Pairwise metric for diverse strategy")
    parser.add_argument("--diverse-pool-size", type=int,
                        default=cfg.get("diverse_pool_size", 20),
                        help="Candidate pool size for diverse strategy")

    # Metric injection flags
    parser.add_argument("--inject-metrics", action="store_true",
                        default=cfg.get("inject_metrics"),
                        help="Compute top-5 metrics from agent rollouts")
    parser.add_argument("--no-inject-metrics", action="store_false", dest="inject_metrics")
    parser.add_argument("--inject-regret", action="store_true",
                        default=cfg.get("inject_regret"),
                        help="Include regret scores in prompt")
    parser.add_argument("--no-inject-regret", action="store_false", dest="inject_regret")
    parser.add_argument("--inject-buffer-stats", action="store_true",
                        default=cfg.get("inject_buffer_stats"),
                        help="Include buffer-wide statistics")
    parser.add_argument("--no-inject-buffer-stats", action="store_false", dest="inject_buffer_stats")

    # LLM settings
    parser.add_argument("--provider", choices=["ollama", "openrouter", "claude-code"],
                        default=cfg.get("provider"),
                        help="API provider")
    parser.add_argument("--base-url", default=None,
                        help="API base URL (auto-resolved from provider if not set)")
    parser.add_argument("--model", default=cfg.get("model"),
                        help="Model name")
    parser.add_argument("--api-key", default=None,
                        help="API key (auto-loaded from env var specified in config)")
    parser.add_argument("--temperature", type=float,
                        default=cfg.get("temperature"))
    parser.add_argument("--max-tokens", type=int,
                        default=cfg.get("max_tokens", 4096),
                        help="Max output tokens for LLM response")
    parser.add_argument("--thinking", action="store_true",
                        default=cfg.get("thinking", False),
                        help="Enable extended thinking (reasoning models like Opus)")
    parser.add_argument("--thinking-in-output", action="store_true",
                        default=cfg.get("thinking_in_output", False),
                        help="Prompt the LLM to include reasoning before the grid")
    parser.add_argument("--thinking-budget", type=int,
                        default=cfg.get("thinking_budget", 10000),
                        help="Token budget for extended thinking")
    parser.add_argument("--max-retries", type=int,
                        default=cfg.get("max_retries"))
    parser.add_argument("--timeout", type=int, default=cfg.get("timeout"),
                        help="API request timeout in seconds")
    parser.add_argument("--min-walls", type=int, default=cfg.get("min_walls"),
                        help="Minimum wall cells for valid maze")
    parser.add_argument("--min-path-distance", type=int,
                        default=cfg.get("min_path_distance"),
                        help="Minimum Manhattan distance agent-to-goal")
    parser.add_argument("--validate-solvable", action="store_true",
                        default=cfg.get("validate_solvable", True),
                        help="BFS solvability check on generated mazes")
    parser.add_argument("--no-validate-solvable", action="store_false",
                        dest="validate_solvable")

    # Custom instruction
    parser.add_argument("--instruction", default=cfg.get("instruction", ""),
                        help="Custom generation instruction")

    # Feedback loop
    parser.add_argument("--feedback", action="store_true",
                        default=cfg.get("feedback"),
                        help="Enable metric feedback loop (requires agent checkpoint)")
    parser.add_argument("--agent-dir", default=cfg.get("agent_dir"),
                        help="Path to agent checkpoint directory")
    parser.add_argument("--num-steps", type=int, default=cfg.get("num_steps"),
                        help="Max rollout steps per episode")
    parser.add_argument("--n-rollouts", type=int, default=cfg.get("n_rollouts"),
                        help="Agent rollouts per maze for robust regret")
    parser.add_argument("--downsample-points", type=int,
                        default=cfg.get("downsample_points"),
                        help="Max points when downsampling metric vectors for LLM prompt")
    parser.add_argument("--max-diversity-retries", type=int,
                        default=cfg.get("max_diversity_retries"),
                        help="Max diversity gate retries per maze")
    # Gate thresholds (read from gate: sub-dict in config, or flat keys for backwards compat)
    gate_cfg = cfg.get("gate", {})
    parser.add_argument("--difficulty-threshold", type=float,
                        default=gate_cfg.get("difficulty_threshold"),
                        help="Min difficulty score to accept (null = disabled)")
    parser.add_argument("--difficulty-metric",
                        choices=["regret", "sfl"],
                        default=gate_cfg.get("difficulty_metric", "regret"),
                        help="Difficulty metric: 'regret' (MaxMC) or 'sfl' (learnability p*(1-p))")
    parser.add_argument("--min-diversity", type=float,
                        default=gate_cfg.get("min_diversity", cfg.get("min_diversity")),
                        help="Min mean pairwise diversity vs references (null = disabled)")
    parser.add_argument("--diversity-metric",
                        choices=["td_error_emd", "experience_divergence", "position_dtw", "cenie"],
                        default=gate_cfg.get("diversity_metric", cfg.get("diversity_metric", "td_error_emd")),
                        help="Diversity metric: pairwise (td_error_emd, experience_divergence, position_dtw) or buffer-wide (cenie)")
    parser.add_argument("--embedding-metric",
                        choices=["td_error_emd"],
                        default=cfg.get("embedding_metric", "td_error_emd"),
                        help="Pairwise metric for t-SNE diversity embedding plot")

    # Mode
    parser.add_argument("--dry-run", action="store_true",
                        default=cfg.get("dry_run", False),
                        help="Only build prompts, skip LLM calls")

    args = parser.parse_args()

    # Flatten provider config using the final --provider value (CLI overrides config)
    provider = args.provider or cfg.get("provider", "ollama")
    flatten_provider_config(cfg, provider)
    if args.base_url is None:
        args.base_url = cfg.get("base_url", "")
    if args.api_key is None:
        api_key_env = cfg.get("api_key_env", "")
        if api_key_env:
            args.api_key = os.environ.get(api_key_env, "")

    # Attach metric config dicts (not CLI-overridable, config-only)
    args.prompt_metrics = cfg.get("prompt_metrics", None)
    args.pairwise_metrics_cfg = cfg.get("pairwise_metrics", None)

    run_test(args)


if __name__ == "__main__":
    main()
