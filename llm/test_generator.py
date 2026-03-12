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
from metrics.pairwise.pos_dtw import position_trace_dtw
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


def build_references_with_metrics(
    ref_data: list,
    trajectories: list = None,
    inject_regret: bool = True,
    inject_dtw: bool = False,
    downsample_points: int = 20,
) -> tuple:
    """Build ReferenceMaze objects with top-5 metric injection from trajectories.

    When trajectories are provided, computes the full top-5 metrics:
      1. Per-step entropy (standalone)
      2. Position DTW (pairwise — returned separately)
      3. Per-step regret (standalone)
      4. Scalar MaxMC regret (standalone)
      5. Per-step action (standalone)

    Args:
        ref_data: List of (index, tokens, score) tuples
        trajectories: List of trajectory dicts from AgentEvaluator (or None)
        inject_regret: Include regret score as a metric (fallback to buffer score)
        inject_dtw: Unused (kept for CLI compat)

    Returns:
        Tuple of (references, pairwise_metrics):
            references: List of ReferenceMaze objects with per-maze metrics
            pairwise_metrics: List of PairwiseMetricEntry (position DTW between refs)
    """
    references = []
    pairwise_metrics = []

    for i, (idx, tokens, score) in enumerate(ref_data):
        grid = tokens_to_ascii(tokens)
        label = f"Maze {chr(65 + i)}"  # A, B, C, ...

        metrics = []

        if trajectories is not None and i < len(trajectories):
            traj = trajectories[i]

            # 1. Per-step entropy (top metric)
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
            ))

            # 3. Per-step regret
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
            ))

            # 4. Scalar regret
            regret_info = compute_regret(traj)
            metrics.append(MetricEntry(
                name="Scalar Regret",
                value=regret_info.regret,
                description=(
                    f"MaxMC regret (mean gap between best return and value estimate), "
                    f"solved={regret_info.solved}, ep_len={regret_info.episode_length}"
                ),
                higher_is="more learning potential",
            ))

            # 5. Per-step action
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
            ))

            # Path overlay
            try:
                from metrics.utils import truncate_at_done
                ep_pos = truncate_at_done(traj["positions"], traj["dones"])
                path_overlay = overlay_path_on_grid(grid, ep_pos)
            except Exception:
                path_overlay = None

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
                ))
            references.append(ReferenceMaze(
                grid=grid,
                label=label,
                metrics=metrics,
            ))

    # 2. Pairwise position DTW between all reference pairs
    if trajectories is not None and len(trajectories) >= 2:
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
            print(f"    {k}: {v:.4f}")
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
    for seq, (orig_idx, result) in enumerate(successful):
        entry = {
            "index": seq + 1,
            "grid": result.grid,
            "attempts": result.attempts,
            "latency_ms": result.latency_ms,
            "errors": result.errors,
            "raw_responses": result.raw_responses,
        }
        if result.feedback_prompts:
            entry["feedback_prompts"] = result.feedback_prompts
        if result.gate_metrics:
            entry["gate_metrics"] = {
                k: round(v, 6) for k, v in result.gate_metrics.items()
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
                "feedback_prompts": r.feedback_prompts,
            })

    meta_path = os.path.join(run_dir, "metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved {meta_path}")

    # --- Render visualization PNG ---
    ref_grids = [(ref.label, ref.grid) for ref in references]
    gen_grids = [(f"Generated {orig_idx + 1}", r.grid) for orig_idx, r in successful]
    has_gate_metrics = any(r.gate_pair_metrics for _, r in successful)

    n_refs = len(ref_grids)
    n_gens = len(gen_grids)

    if n_refs == 0 and n_gens == 0:
        logger.warning("No mazes to visualize")
        return run_dir

    if has_gate_metrics:
        # Layout with DTW profiles:
        # Row 0: reference mazes
        # For each generated maze: maze + pos DTW profiles + val DTW profiles
        cols = max(n_refs, 2)  # maze, pos_dtw plot
        rows = 1 + n_gens
        fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows))
        fig.suptitle(
            f"LLM Maze Generation — {model}\n"
            f"{n_gens}/{len(results)} successful, "
            f"{sum(r.latency_ms for r in results) / 1000:.0f}s total",
            fontsize=13, fontweight='bold',
        )

        # Ensure axes is always 2D
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes[np.newaxis, :]
        elif cols == 1:
            axes = axes[:, np.newaxis]

        # Turn off all axes
        for ax_row in axes:
            for ax in ax_row:
                ax.axis('off')

        # Row 0: Reference mazes
        for i, (label, grid) in enumerate(ref_grids):
            if i < cols:
                ax = axes[0, i]
                rt = ref_trajectories[i] if ref_trajectories and i < len(ref_trajectories) else None
                plot_maze_with_path(
                    ax, grid,
                    positions=rt["positions"] if rt else None,
                    dones=rt["dones"] if rt else None,
                    color='blue', title=label,
                )

        # Rows 1+: each generated maze with DTW profiles
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
        for gen_idx, (orig_idx, result) in enumerate(successful):
            row = 1 + gen_idx
            label = f"Generated {gen_idx + 1}"

            # Col 0: maze grid
            ax_maze = axes[row, 0]
            gt = gen_trajectories[gen_idx] if gen_trajectories and gen_idx < len(gen_trajectories) else None
            summary_str = ""
            if gt and "solve_rate" in gt:
                summary_str += f"\nsolve={gt['solve_rate']:.0%} best_ret={gt['best_return']:.3f}"
            if result.gate_metrics:
                summary_str += (
                    f"\npos_dtw_min={result.gate_metrics.get('min_pos_dtw', 0):.2f}"
                    f" pos_dtw_mean={result.gate_metrics.get('mean_pos_dtw', 0):.2f}"
                )
            plot_maze_with_path(
                ax_maze, result.grid,
                positions=gt["positions"] if gt else None,
                dones=gt["dones"] if gt else None,
                color='red', title=label + summary_str,
                title_color='darkgreen', title_bold=True,
            )

            if not result.gate_pair_metrics:
                continue

            # Col 1: Position DTW local_costs profiles (one line per reference)
            ax_pos = axes[row, 1]
            ax_pos.axis('on')
            for pi, pair in enumerate(result.gate_pair_metrics):
                if pair.pos_dtw_local_costs is not None:
                    c = colors[pi % len(colors)]
                    ax_pos.plot(pair.pos_dtw_local_costs, color=c, linewidth=1.2,
                                label=f"vs {pair.ref_label} (d={pair.pos_dtw_distance:.2f})",
                                alpha=0.8)
                    ax_pos.fill_between(range(len(pair.pos_dtw_local_costs)),
                                         pair.pos_dtw_local_costs, alpha=0.15, color=c)
            ax_pos.set_title("Position DTW Profile", fontsize=9)
            ax_pos.set_xlabel("Warping path step", fontsize=8)
            ax_pos.set_ylabel("Local cost (L2)", fontsize=8)
            ax_pos.legend(fontsize=7, loc='upper right')
            ax_pos.grid(alpha=0.3)
            ax_pos.tick_params(labelsize=7)

    else:
        # Simple layout without DTW profiles
        cols = max(n_refs, n_gens, 1)
        rows = 2 if n_gens > 0 else 1

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        fig.suptitle(
            f"LLM Maze Generation — {model}\n"
            f"{n_gens}/{len(results)} successful, "
            f"{sum(r.latency_ms for r in results) / 1000:.0f}s total",
            fontsize=12, fontweight='bold',
        )

        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes[np.newaxis, :]
        elif cols == 1:
            axes = axes[:, np.newaxis]

        for ax_row in axes:
            for ax in ax_row:
                ax.axis('off')

        for i, (label, grid) in enumerate(ref_grids):
            if i < cols:
                rt = ref_trajectories[i] if ref_trajectories and i < len(ref_trajectories) else None
                plot_maze_with_path(
                    axes[0, i], grid,
                    positions=rt["positions"] if rt else None,
                    dones=rt["dones"] if rt else None,
                    color='blue', title=label,
                )

        if n_gens > 0:
            for i, (label, grid) in enumerate(gen_grids):
                if i < cols:
                    gt = gen_trajectories[i] if gen_trajectories and i < len(gen_trajectories) else None
                    subtitle = label
                    if gt and "solve_rate" in gt:
                        subtitle += f"\nsolve={gt['solve_rate']:.0%} best_ret={gt['best_return']:.3f}"
                    plot_maze_with_path(
                        axes[1, i], grid,
                        positions=gt["positions"] if gt else None,
                        dones=gt["dones"] if gt else None,
                        color='red', title=subtitle,
                        title_color='darkgreen', title_bold=True,
                    )

    plt.tight_layout()
    viz_path = os.path.join(run_dir, "visualization.png")
    fig.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved {viz_path}")

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

    # Select reference mazes
    logger.info(f"Selecting {args.num_refs} reference mazes (strategy={args.strategy})...")
    ref_data = select_references(tokens, scores, size, n=args.num_refs, strategy=args.strategy)

    # Load agent and roll out on reference levels to get trajectory data
    ref_trajectories = None
    if args.inject_metrics:
        from llm.agent_evaluator import AgentEvaluator
        logger.info(f"Loading agent from {args.agent_dir} for metric computation...")
        evaluator = AgentEvaluator(args.agent_dir, num_steps=args.num_steps)

        ref_levels = []
        for idx, tok, score in ref_data:
            ref_levels.append(tokens_to_level_obj(tok))

        logger.info(f"Rolling out agent on {len(ref_levels)} reference levels...")
        ref_trajectories = evaluator.evaluate_levels(ref_levels)
        logger.info("Reference trajectories collected")

    # Build references with metrics (top-5 if trajectories available)
    references, pairwise_metrics = build_references_with_metrics(
        ref_data,
        trajectories=ref_trajectories,
        inject_regret=args.inject_regret,
        downsample_points=args.downsample_points,
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
            evaluator = AgentEvaluator(args.agent_dir, num_steps=args.num_steps)
            ref_levels = [tokens_to_level_obj(tok) for _, tok, _ in ref_data]
            ref_trajectories = evaluator.evaluate_levels(ref_levels)

        ref_labels = [ref.label for ref in references]

        thresholds = DiversityThresholds(
            min_pos_dtw=args.min_pos_dtw,
            min_regret=args.min_regret,
        )

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
    )
    print(f"\n  Results saved to: {run_dir}/")
    print(f"    - maze_XXX.txt files (ASCII grids)")
    print(f"    - metadata.json (run details)")
    print(f"    - visualization.png (visual grid)")


def load_config(config_path: str) -> dict:
    """Load YAML config file and return as dict.

    Flattens provider-specific settings into top-level keys
    (base_url, api_key_env) based on the active provider.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    # Flatten provider-specific block
    provider = cfg.get("provider", "ollama")
    provider_cfg = cfg.get(provider, {})
    cfg.setdefault("base_url", provider_cfg.get("base_url", ""))
    cfg.setdefault("api_key_env", provider_cfg.get("api_key_env", ""))

    # Remove provider sub-dicts (not needed downstream)
    cfg.pop("ollama", None)
    cfg.pop("openrouter", None)

    return cfg


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

    # Resolve API key: config api_key_env -> env var -> empty
    api_key_default = ""
    api_key_env = cfg.get("api_key_env", "")
    if api_key_env:
        api_key_default = os.environ.get(api_key_env, "")

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
    parser.add_argument("--provider", choices=["ollama", "openrouter"],
                        default=cfg.get("provider"),
                        help="API provider")
    parser.add_argument("--base-url", default=cfg.get("base_url"),
                        help="API base URL")
    parser.add_argument("--model", default=cfg.get("model"),
                        help="Model name")
    parser.add_argument("--api-key", default=api_key_default or None,
                        help="API key (auto-loaded from env var specified in config)")
    parser.add_argument("--temperature", type=float,
                        default=cfg.get("temperature"))
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
    parser.add_argument("--min-pos-dtw", type=float,
                        default=cfg.get("min_pos_dtw"),
                        help="Min position trace DTW for diversity gate")
    parser.add_argument("--min-regret", type=float,
                        default=cfg.get("min_regret"),
                        help="Min regret to accept (null = disabled)")

    # Mode
    parser.add_argument("--dry-run", action="store_true",
                        default=cfg.get("dry_run", False),
                        help="Only build prompts, skip LLM calls")

    args = parser.parse_args()
    run_test(args)


if __name__ == "__main__":
    main()
