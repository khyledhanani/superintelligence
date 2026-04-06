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
from llm.buffer_stats import npz_to_sampler, BufferStatsExtractor
from llm.reference_metrics import enrich_references_with_metrics
from metrics.standalone.regret import compute_regret
from metrics.utils import downsample, format_vector

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

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
    embedding_metric: str = "embedding_l2",
    buf_embeddings: np.ndarray = None,
    buf_scores: np.ndarray = None,
    buf_tokens: np.ndarray = None,
    visualisation_plot: str = "tsne",
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
            if rc.thinking:
                entry["thinking"] = rc.thinking
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
        has_traj = rt is not None and "dones" in rt
        logger.info(f"Rejected {i+1}: trajectory={'yes' if has_traj else 'NO'}, "
                     f"diff={rc.failed_difficulty}, div={rc.failed_diversity}")
        if has_traj:
            all_trajs.append(rt)
            all_labels.append(f"Rej {i+1}")
            if rc.failed_difficulty and rc.failed_diversity:
                all_colors.append('darkorange')
            elif rc.failed_difficulty:
                all_colors.append('red')
            else:
                all_colors.append('magenta')
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

    if len(all_trajs) < 3:
        logger.info(f"Skipping diversity plot: need >= 3 trajectories, have {len(all_trajs)} "
                     f"(use --inject-metrics to include reference trajectories)")
    if len(all_trajs) >= 3:
        try:
            from sklearn.manifold import TSNE, MDS
            from metrics.standalone.cenie import extract_state_action_pairs

            _emb_label = {
                "embedding_l2": "Embedding L2",
                "td_error_emd": "TD Error EMD",
                "experience_divergence": "Experience Divergence",
                "position_dtw": "Position DTW",
            }.get(_emb_metric, _emb_metric)

            # Build foreground embedding matrix (257D mean state-action vectors)
            # Prefer mean_embedding (averaged across all rollouts) when available;
            # fall back to single-rollout extract_state_action_pairs
            n_fg = len(all_trajs)
            fg_embeddings = np.zeros((n_fg, 257), dtype=np.float32)
            for i, t in enumerate(all_trajs):
                if "mean_embedding" in t:
                    fg_embeddings[i] = t["mean_embedding"]
                else:
                    pairs = extract_state_action_pairs(t)
                    if pairs is not None and len(pairs) > 0:
                        fg_embeddings[i] = pairs.mean(axis=0)

            # Load buffer embeddings as background (all non-zero embeddings)
            n_buf = 0
            buf_emb_matrix = None
            buf_difficulty_pass = None
            if buf_embeddings is not None:
                norms = np.sqrt(np.sum(buf_embeddings ** 2, axis=1))
                valid_mask = norms > 1e-6
                buf_emb_matrix = buf_embeddings[valid_mask]
                n_buf = len(buf_emb_matrix)
                logger.info(f"Loaded {n_buf} buffer embeddings for plot")

                # Difficulty filter for coloring buffer dots
                if buf_scores is not None:
                    valid_scores = buf_scores[valid_mask]
                    mean_score = float(buf_scores[valid_mask].mean())
                    buf_difficulty_pass = valid_scores >= mean_score
                    logger.info(f"Buffer plot: {int(buf_difficulty_pass.sum())}/{n_buf} pass difficulty filter")

            # Build combined distance matrix: [fg; buf] x [fg; buf]
            n_total = n_fg + n_buf
            if n_buf > 0:
                all_emb = np.vstack([fg_embeddings, buf_emb_matrix])
            else:
                all_emb = fg_embeddings

            # Foreground pairwise L2 distances (for edge annotations)
            fg_dist_matrix = np.zeros((n_fg, n_fg), dtype=np.float32)
            for i in range(n_fg):
                diff = fg_embeddings[i] - fg_embeddings
                fg_dist_matrix[i] = np.sqrt(np.sum(diff ** 2, axis=1))

            # Dimensionality reduction
            primary = visualisation_plot.lower()
            perplexity = min(40, n_total - 1)
            if primary == "mds":
                # MDS needs full precomputed distance matrix
                full_dist = np.zeros((n_total, n_total), dtype=np.float32)
                chunk = max(1, min(500, n_total))
                for start in range(0, n_total, chunk):
                    end = min(start + chunk, n_total)
                    diff = all_emb[start:end, np.newaxis, :] - all_emb[np.newaxis, :, :]
                    full_dist[start:end] = np.sqrt(np.sum(diff ** 2, axis=2))
                embedding_2d = MDS(
                    n_components=2, dissimilarity="precomputed",
                    random_state=42, normalized_stress='auto',
                ).fit_transform(full_dist)
                method_name = "MDS"
            else:
                embedding_2d = TSNE(
                    n_components=2, perplexity=perplexity,
                    random_state=42, init="pca", learning_rate="auto",
                    max_iter=1000,
                ).fit_transform(all_emb)
                method_name = "t-SNE"

            # Build per-point edge colors: red edge for force-accepted
            n_before_accepted = n_ref_in_emb + sum(1 for rc in all_rejected if rc.trajectory is not None and "dones" in rc.trajectory)
            edge_colors = []
            for i in range(n_fg):
                if i >= n_before_accepted and all_markers[i] == '*':
                    fa_idx = i - n_before_accepted
                    if fa_idx < len(all_force_accepted) and all_force_accepted[fa_idx]:
                        edge_colors.append('red')
                    else:
                        edge_colors.append('black')
                else:
                    edge_colors.append('black')

            # Build legend (shared by both plots)
            from matplotlib.lines import Line2D
            emb_legend = []
            if n_buf > 0:
                if buf_difficulty_pass is not None:
                    n_pass = int(np.sum(buf_difficulty_pass))
                    emb_legend.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='lightsteelblue',
                                             markersize=6, markeredgecolor='none', label=f'Buffer pass ({n_pass})'))
                    emb_legend.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='khaki',
                                             markersize=6, markeredgecolor='none', label=f'Buffer fail ({n_buf - n_pass})'))
                else:
                    emb_legend.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='lightsteelblue',
                                             markersize=6, markeredgecolor='none', label=f'Buffer ({n_buf})'))
            emb_legend.extend([
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                       markersize=8, markeredgecolor='black', label='Reference'),
                Line2D([0], [0], marker='D', color='w', markerfacecolor='black',
                       markersize=8, markeredgecolor='white', label='Ref Centroid'),
            ])
            has_diff = any(rc.failed_difficulty and not rc.failed_diversity for rc in all_rejected)
            has_div = any(rc.failed_diversity and not rc.failed_difficulty for rc in all_rejected)
            has_both = any(rc.failed_difficulty and rc.failed_diversity for rc in all_rejected)
            if has_div:
                emb_legend.append(Line2D([0], [0], marker='X', color='w', markerfacecolor='magenta',
                                         markersize=8, markeredgecolor='black', label='Rej (diversity)'))
            if has_diff:
                emb_legend.append(Line2D([0], [0], marker='X', color='w', markerfacecolor='red',
                                         markersize=8, markeredgecolor='black', label='Rej (difficulty)'))
            if has_both:
                emb_legend.append(Line2D([0], [0], marker='X', color='w', markerfacecolor='darkorange',
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

            def _plot_embedding(ax, emb_2d, method_name):
                """Plot buffer background + foreground points on a single axes."""
                # Buffer background
                if n_buf > 0:
                    buf_emb_2d = emb_2d[n_fg:]
                    if buf_difficulty_pass is not None:
                        fail_mask = ~buf_difficulty_pass
                        if np.any(fail_mask):
                            ax.scatter(buf_emb_2d[fail_mask, 0], buf_emb_2d[fail_mask, 1],
                                       c='khaki', s=10, marker='o', zorder=1, alpha=0.4, edgecolors='none')
                        pass_mask = buf_difficulty_pass
                        if np.any(pass_mask):
                            ax.scatter(buf_emb_2d[pass_mask, 0], buf_emb_2d[pass_mask, 1],
                                       c='lightsteelblue', s=12, marker='o', zorder=2, alpha=0.5, edgecolors='none')
                    else:
                        ax.scatter(buf_emb_2d[:, 0], buf_emb_2d[:, 1],
                                   c='lightsteelblue', s=12, marker='o', zorder=1, alpha=0.5, edgecolors='none')

                # Foreground points
                for i in range(n_fg):
                    ax.scatter(emb_2d[i, 0], emb_2d[i, 1],
                               c=all_colors[i], s=150 if all_markers[i] == '*' else 100,
                               marker=all_markers[i], zorder=5,
                               edgecolors=edge_colors[i],
                               linewidths=2.0 if edge_colors[i] == 'red' else 0.5)
                    ax.annotate(all_labels[i], (emb_2d[i, 0], emb_2d[i, 1]),
                                textcoords="offset points", xytext=(6, 6),
                                fontsize=7, color=all_colors[i], fontweight='bold')

                # Reference centroid
                if n_ref_in_emb >= 2:
                    ref_emb_2d = emb_2d[:n_ref_in_emb]
                    centroid = ref_emb_2d.mean(axis=0)
                    ax.scatter(centroid[0], centroid[1], c='black', s=200, marker='D',
                               zorder=6, edgecolors='white', linewidths=1.5)
                    ax.annotate("Centroid", (centroid[0], centroid[1]),
                                textcoords="offset points", xytext=(6, -10),
                                fontsize=7, color='black', fontweight='bold')

                # Distance edges from rejected/accepted to closest reference
                for i in range(n_ref_in_emb, n_fg):
                    closest_ref, closest_dist = -1, float('inf')
                    for j in range(n_ref_in_emb):
                        if fg_dist_matrix[i, j] < closest_dist:
                            closest_dist = fg_dist_matrix[i, j]
                            closest_ref = j
                    if closest_ref >= 0:
                        mid_x = (emb_2d[i, 0] + emb_2d[closest_ref, 0]) / 2
                        mid_y = (emb_2d[i, 1] + emb_2d[closest_ref, 1]) / 2
                        ax.plot([emb_2d[i, 0], emb_2d[closest_ref, 0]],
                                [emb_2d[i, 1], emb_2d[closest_ref, 1]],
                                'gray', alpha=0.4, linewidth=1.0, linestyle='--')
                        ax.annotate(f"{closest_dist:.4f}", (mid_x, mid_y),
                                    fontsize=5.5, color='dimgray', ha='center',
                                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                              alpha=0.8, edgecolor='none'))

                ax.set_title(f"Diversity Embedding ({_emb_label})\n"
                             f"{method_name} layout — distances on edges are actual L2 values",
                             fontsize=10, fontweight='bold')
                ax.set_xlabel(f"{method_name} dim 1", fontsize=9)
                ax.set_ylabel(f"{method_name} dim 2", fontsize=9)
                ax.grid(alpha=0.2)
                ax.legend(handles=emb_legend, loc='best', fontsize=8, framealpha=0.9)

            # Generate the selected plot
            fig_emb, ax_emb = plt.subplots(1, 1, figsize=(8, 7) if n_buf > 0 else (7, 6))
            _plot_embedding(ax_emb, embedding_2d, method_name)
            emb_path = os.path.join(run_dir, "diversity_embedding.png")
            fig_emb.savefig(emb_path, dpi=150, bbox_inches='tight')
            plt.close(fig_emb)
            logger.info(f"Saved {emb_path}")

        except ImportError:
            logger.warning("sklearn not installed — skipping diversity embedding")
        except Exception as e:
            logger.warning(f"Diversity embedding failed: {e}")

    # --- Structural t-SNE plot (173D: wall map + agent/goal positions) ---
    if buf_tokens is not None:
        try:
            from sklearn.manifold import TSNE
            from vae.plot_tsne_training_evolution import tokens_to_structural_features
            from vae.vae_level_utils import level_to_tokens

            # Foreground: refs + rejected + accepted → convert grids to tokens to features
            fg_grids = []
            fg_labels_s = []
            fg_colors_s = []
            fg_markers_s = []
            for ref in references:
                fg_grids.append(ref.grid)
                fg_labels_s.append(ref.label)
                fg_colors_s.append('blue')
                fg_markers_s.append('o')
            for i, rc in enumerate(all_rejected):
                fg_grids.append(rc.grid)
                fg_labels_s.append(f"Rej {i+1}")
                if rc.failed_difficulty and rc.failed_diversity:
                    fg_colors_s.append('darkorange')
                elif rc.failed_difficulty:
                    fg_colors_s.append('red')
                else:
                    fg_colors_s.append('magenta')
                fg_markers_s.append('X')
            for i, (orig_idx, result) in enumerate(successful):
                fg_grids.append(result.grid)
                force = bool(result.diversity_issues)
                fg_labels_s.append(f"Force-Accepted {i+1}" if force else f"Accepted {i+1}")
                fg_colors_s.append('green')
                fg_markers_s.append('*')

            # Convert foreground grids to tokens then to structural features
            fg_tokens = []
            for grid in fg_grids:
                try:
                    level = Level.from_str(grid)
                    tok = np.asarray(level_to_tokens(level))
                    fg_tokens.append(tok)
                except Exception:
                    fg_tokens.append(np.zeros(52, dtype=np.int32))
            fg_tokens = np.stack(fg_tokens)
            fg_struct = tokens_to_structural_features(fg_tokens)

            # Buffer background
            buf_struct = tokens_to_structural_features(buf_tokens)
            n_buf_s = len(buf_struct)

            # Combined embedding
            all_struct = np.vstack([fg_struct, buf_struct])
            n_fg_s = len(fg_struct)
            perp_s = min(40, len(all_struct) - 1)
            struct_2d = TSNE(
                n_components=2, perplexity=perp_s,
                random_state=42, init="pca", learning_rate="auto",
                max_iter=1000,
            ).fit_transform(all_struct)

            # Plot
            fig_s, ax_s = plt.subplots(1, 1, figsize=(8, 7))

            # Buffer background (colored by difficulty)
            buf_2d = struct_2d[n_fg_s:]
            if buf_scores is not None and len(buf_scores) >= n_buf_s:
                mean_score = float(buf_scores[:n_buf_s].mean())
                pass_mask = buf_scores[:n_buf_s] >= mean_score
                fail_mask = ~pass_mask
                if np.any(fail_mask):
                    ax_s.scatter(buf_2d[fail_mask, 0], buf_2d[fail_mask, 1],
                                 c='khaki', s=10, marker='o', zorder=1, alpha=0.4, edgecolors='none')
                if np.any(pass_mask):
                    ax_s.scatter(buf_2d[pass_mask, 0], buf_2d[pass_mask, 1],
                                 c='lightsteelblue', s=12, marker='o', zorder=2, alpha=0.5, edgecolors='none')
            else:
                ax_s.scatter(buf_2d[:, 0], buf_2d[:, 1],
                             c='lightsteelblue', s=12, marker='o', zorder=1, alpha=0.5, edgecolors='none')

            # Foreground points
            fg_2d = struct_2d[:n_fg_s]
            for i in range(n_fg_s):
                ax_s.scatter(fg_2d[i, 0], fg_2d[i, 1],
                             c=fg_colors_s[i], s=150 if fg_markers_s[i] == '*' else 100,
                             marker=fg_markers_s[i], zorder=5,
                             edgecolors='black', linewidths=0.5)
                ax_s.annotate(fg_labels_s[i], (fg_2d[i, 0], fg_2d[i, 1]),
                              textcoords="offset points", xytext=(6, 6),
                              fontsize=7, color=fg_colors_s[i], fontweight='bold')

            ax_s.set_title("Structural Embedding (173D: wall map + positions)\n"
                           "t-SNE layout — maze topology only, no agent behavior",
                           fontsize=10, fontweight='bold')
            ax_s.set_xlabel("t-SNE dim 1", fontsize=9)
            ax_s.set_ylabel("t-SNE dim 2", fontsize=9)
            ax_s.grid(alpha=0.2)

            struct_path = os.path.join(run_dir, "structural_embedding.png")
            fig_s.savefig(struct_path, dpi=150, bbox_inches='tight')
            plt.close(fig_s)
            logger.info(f"Saved {struct_path}")

        except Exception as e:
            logger.warning(f"Structural embedding plot failed: {e}")

    return run_dir


# --- Main test ---

def run_test(args):
    """Run the maze generation test."""

    # Load buffer
    import time as _time
    _t_setup = _time.time()
    logger.info(f"Loading buffer from {args.buffer_path}...")
    sampler = npz_to_sampler(args.buffer_path)
    size = int(np.asarray(sampler["size"]))
    scores = np.asarray(sampler["scores"])
    print(f"[TIMING] Buffer load: {_time.time() - _t_setup:.1f}s", flush=True)

    # Load agent early if needed for embedding-based selection, metrics, or fresh embeddings
    buffer_state = getattr(args, 'buffer_state', 'stale')
    needs_agent = (args.inject_metrics
                   or args.strategy in ("greedy", "hybrid-greedy", "kmedoid", "hybrid-kmedoid")
                   or buffer_state == "fresh")
    evaluator = None
    if needs_agent:
        from llm.agent_evaluator import AgentEvaluator
        logger.info(f"Loading agent from {args.agent_dir} for metric computation...")
        _t_agent = _time.time()
        evaluator = AgentEvaluator.from_checkpoint(args.agent_dir, num_steps=args.num_steps, checkpoint_step=args.checkpoint_step)
        print(f"[TIMING] Agent load: {_time.time() - _t_agent:.1f}s", flush=True)

    # Compute buffer embeddings — fresh (rollout current agent) or stale (from npz)
    buffer_embeddings = None
    needs_fresh = (buffer_state == "fresh"
                   or args.strategy in ("greedy", "hybrid-greedy", "kmedoid", "hybrid-kmedoid"))
    if needs_fresh and evaluator is not None:
        logger.info(f"Computing fresh buffer embeddings for {size} levels (5 rollouts averaged)...")
        _t_emb = _time.time()
        all_levels = [jax.tree_util.tree_map(lambda x: x[i], sampler["levels"]) for i in range(size)]
        buffer_embeddings = evaluator.compute_embeddings(all_levels, n_rollouts=5)
        print(f"[TIMING] Fresh buffer embeddings ({size} levels, 5 rollouts): {_time.time() - _t_emb:.1f}s", flush=True)
    else:
        # Load stale embeddings from npz if available (for plot background)
        npz_data = np.load(args.buffer_path, allow_pickle=True)
        if "embeddings" in npz_data:
            buffer_embeddings = npz_data["embeddings"]
            logger.info(f"Loaded stale buffer embeddings ({buffer_embeddings.shape}) from npz")

    # Select reference mazes via BufferStatsExtractor
    extractor = BufferStatsExtractor(
        n_references=args.num_refs,
        strategy=args.strategy,
        hybrid_difficulty_percentile=getattr(args, 'hybrid_difficulty_percentile', 50.0),
        inject_regret=args.inject_regret,
    )
    if buffer_embeddings is not None:
        extractor._buffer_embeddings = buffer_embeddings

    logger.info(f"Selecting {args.num_refs} reference mazes (strategy={args.strategy})...")
    references, ref_levels = extractor.extract_references_with_levels(sampler)

    # Roll out agent on selected reference levels to get trajectory data
    # Uses multi-rollout (50) for proper SFL solve_rate + mean_embedding
    ref_trajectories = None
    if args.inject_metrics and evaluator is not None and ref_levels:
        logger.info(f"Rolling out agent on {len(ref_levels)} reference levels (50 rollouts each)...")
        _t_ref = _time.time()
        ref_trajectories = []
        for lv in ref_levels:
            traj = evaluator.evaluate_level_multi_rollout(lv, n_rollouts=args.n_rollouts)
            ref_trajectories.append(traj)
        print(f"[TIMING] Reference rollouts ({len(ref_levels)} levels): {_time.time() - _t_ref:.1f}s", flush=True)
        logger.info("Reference trajectories collected")

    # Enrich references with trajectory-based metrics
    references, pairwise_metrics = enrich_references_with_metrics(
        references,
        trajectories=ref_trajectories,
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
        active_scores = np.asarray(scores)
        score_label = "Learnability (SFL)" if args.difficulty_metric == "sfl" else "Regret"
        global_metrics = [
            MetricEntry(
                name="Buffer Size",
                value=size,
                description="Number of levels in the replay buffer",
            ),
            MetricEntry(
                name=f"Mean {score_label}",
                value=float(np.mean(active_scores)),
                description=f"Average {score_label.lower()} across all buffer levels",
            ),
            MetricEntry(
                name=f"Max {score_label}",
                value=float(np.max(active_scores)),
                description=f"Highest {score_label.lower()} level in buffer",
            ),
            MetricEntry(
                name=f"{score_label} Std Dev",
                value=float(np.std(active_scores)),
                description=f"Spread of {score_label.lower()} scores",
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
        effort=args.effort,
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
            evaluator = AgentEvaluator.from_checkpoint(args.agent_dir, num_steps=args.num_steps, checkpoint_step=args.checkpoint_step)
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
                    cenie_indices = np.argsort(-np.asarray(scores))[:n_cenie_levels]
                    cenie_levels = [jax.tree_util.tree_map(lambda x: x[i], sampler["levels"]) for i in cenie_indices]
                    cenie_trajs = evaluator.evaluate_levels(cenie_levels)
                    logger.info(f"CENIE trajectories collected ({n_cenie_levels} levels)")
                cenie_model = fit_cenie_model(cenie_trajs)

        # Compute reference embeddings for embedding_l2 diversity metric
        ref_embeddings = None
        if args.diversity_metric == "embedding_l2" and ref_trajectories:
            from metrics.standalone.cenie import extract_state_action_pairs
            ref_embs = []
            for traj in ref_trajectories:
                pairs = extract_state_action_pairs(traj)
                if pairs is not None and len(pairs) > 0:
                    ref_embs.append(pairs.mean(axis=0))
            if ref_embs:
                ref_embeddings = np.stack(ref_embs)
                logger.info(f"Computed {len(ref_embs)} reference embeddings for L2 diversity")
            # buffer_embeddings already computed above (fresh or stale)

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
            ref_embeddings=ref_embeddings,
            buffer_embeddings=buffer_embeddings,
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
    # Load buffer tokens for structural embedding plot
    _buf_tokens = None
    try:
        _npz = np.load(args.buffer_path, allow_pickle=True)
        if "tokens" in _npz:
            _buf_tokens = _npz["tokens"][:size]
    except Exception:
        pass

    save_results(
        results, references, args.model, run_dir,
        ref_trajectories=ref_trajectories,
        gen_trajectories=gen_trajectories,
        embedding_metric=args.embedding_metric,
        buf_embeddings=buffer_embeddings,
        buf_scores=np.asarray(scores) if buffer_embeddings is not None else None,
        buf_tokens=_buf_tokens,
        visualisation_plot=getattr(args, 'visualisation_plot', 'tsne'),
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
    parser.add_argument("--strategy", choices=["hardest", "top_regret", "random", "greedy", "hybrid-greedy", "kmedoid", "hybrid-kmedoid"],
                        default=cfg.get("strategy", "hardest"),
                        help="Reference selection strategy")
    parser.add_argument("--hybrid-difficulty-percentile", type=float,
                        default=cfg.get("hybrid_difficulty_percentile", 50.0),
                        help="Percentile threshold for hybrid-kmedoid difficulty filter")

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
    parser.add_argument("--effort", default=cfg.get("effort", "low"),
                        choices=["low", "medium", "high"],
                        help="claude-code --effort flag (default: from config.yaml)")

    # Custom instruction
    parser.add_argument("--instruction", default=cfg.get("instruction", ""),
                        help="Custom generation instruction")

    # Feedback loop
    parser.add_argument("--feedback", action="store_true",
                        default=cfg.get("feedback"),
                        help="Enable metric feedback loop (requires agent checkpoint)")
    parser.add_argument("--agent-dir", default=cfg.get("agent_dir"),
                        help="Path to agent checkpoint directory")
    parser.add_argument("--checkpoint-step", type=int, default=cfg.get("checkpoint_step", -1),
                        help="Agent checkpoint step (-1 for latest)")
    parser.add_argument("--num-steps", type=int, default=cfg.get("num_steps"),
                        help="Max rollout steps per episode")
    # Gate thresholds (read from gate: sub-dict in config, or flat keys for backwards compat)
    gate_cfg = cfg.get("gate", {})
    parser.add_argument("--n-rollouts", type=int, default=gate_cfg.get("n_rollouts", cfg.get("n_rollouts", 50)),
                        help="Agent rollouts per maze for robust regret")
    parser.add_argument("--downsample-points", type=int,
                        default=cfg.get("downsample_points"),
                        help="Max points when downsampling metric vectors for LLM prompt")
    parser.add_argument("--max-diversity-retries", type=int,
                        default=gate_cfg.get("max_diversity_retries", cfg.get("max_diversity_retries", 3)),
                        help="Max diversity gate retries per maze")
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
                        choices=["td_error_emd", "experience_divergence", "position_dtw", "cenie", "embedding_l2"],
                        default=gate_cfg.get("diversity_metric", cfg.get("diversity_metric", "td_error_emd")),
                        help="Diversity metric: pairwise (td_error_emd, experience_divergence, position_dtw), buffer-wide (cenie), or embedding_l2")
    parser.add_argument("--embedding-metric",
                        choices=["embedding_l2", "td_error_emd", "experience_divergence", "position_dtw"],
                        default=cfg.get("embedding_metric", "embedding_l2"),
                        help="Pairwise metric for diversity embedding plot")
    parser.add_argument("--visualisation-plot", choices=["tsne", "mds"],
                        default=cfg.get("visualisation_plot", "tsne"),
                        help="Dimensionality reduction for diversity plot")
    parser.add_argument("--buffer-state", choices=["stale", "fresh"],
                        default=cfg.get("buffer_state", "stale"),
                        help="'stale' uses saved embeddings from buffer dump, "
                             "'fresh' recomputes all buffer embeddings with current agent")

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
