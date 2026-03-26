#!/usr/bin/env python3
"""Visualize cached LLM-generated maze levels as PNG grids.

Reads .npy wall_map files + .json sidecars from a level cache directory
and produces one PNG per injection step showing all accepted mazes in a grid.

Usage:
    # Visualize a specific run's cache
    python scripts/visualize_llm_levels.py results/accel-llm-v3/llm_levels/0

    # Visualize a single step
    python scripts/visualize_llm_levels.py results/accel-llm-v3/llm_levels/0 --step 3000
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_level(npy_path: Path):
    """Load a wall_map and its JSON sidecar metadata."""
    wall_map = np.load(str(npy_path))
    json_path = npy_path.with_suffix(".json")
    meta = {}
    if json_path.exists():
        meta = json.loads(json_path.read_text())
    return wall_map, meta


def render_maze_ax(ax, wall_map, meta=None, title=None):
    """Render a single maze wall_map on a matplotlib axes."""
    H, W = wall_map.shape
    # Create RGB image: walls=black, open=white
    img = np.ones((H, W, 3), dtype=np.float32)
    img[wall_map] = [0.2, 0.2, 0.2]  # dark gray walls

    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def visualize_step(levels, step, output_path):
    """Create a grid PNG for all mazes at a given injection step."""
    n = len(levels)
    if n == 0:
        return

    cols = min(n, 6)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx, (wall_map, meta) in enumerate(levels):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        sfl = meta.get("gate_scores", {}).get("solve_rate", None)
        div = meta.get("gate_scores", {}).get("td_error_emd", None)
        label_parts = [f"#{idx}"]
        if sfl is not None:
            label_parts.append(f"sfl={sfl:.3f}")
        if div is not None:
            label_parts.append(f"div={div:.3f}")
        render_maze_ax(ax, wall_map, meta, " ".join(label_parts))

    # Hide empty axes
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    fig.suptitle(f"Step {step} — {n} accepted mazes", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def visualize_cache_dir(cache_dir, step_filter=None):
    """Visualize all steps (or a single step) in a cache directory."""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        print(f"Cache directory not found: {cache_dir}")
        return

    # Group .npy files by step
    steps = defaultdict(list)
    for npy_path in sorted(cache_dir.glob("step_*_idx_*.npy")):
        # Parse step from filename: step_03000_idx_001.npy
        parts = npy_path.stem.split("_")
        step = int(parts[1])
        if step_filter is not None and step != step_filter:
            continue
        wall_map, meta = load_level(npy_path)
        steps[step].append((wall_map, meta))

    if not steps:
        print(f"No levels found in {cache_dir}" +
              (f" for step {step_filter}" if step_filter else ""))
        return

    # Create output dir
    out_dir = cache_dir / "viz"
    out_dir.mkdir(exist_ok=True)

    for step in sorted(steps):
        out_path = out_dir / f"step_{step:05d}.png"
        print(f"Step {step}: {len(steps[step])} mazes")
        visualize_step(steps[step], step, out_path)

    print(f"\nAll visualizations saved to {out_dir}/")


def visualize_single_step(cache_dir, wall_maps, step):
    """Called from injector: visualize wall_maps for a single step.

    Args:
        cache_dir: Path to the level cache directory
        wall_maps: list of numpy bool arrays (wall maps)
        step: injection step number
    """
    cache_dir = Path(cache_dir)
    out_dir = cache_dir / "viz"
    out_dir.mkdir(exist_ok=True)

    levels = [(wm, {}) for wm in wall_maps]
    out_path = out_dir / f"step_{step:05d}.png"
    visualize_step(levels, step, out_path)
    return str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize cached LLM maze levels")
    parser.add_argument("cache_dir", type=str, help="Path to llm_levels/<seed> directory")
    parser.add_argument("--step", type=int, default=None, help="Visualize only this step")
    args = parser.parse_args()
    visualize_cache_dir(args.cache_dir, step_filter=args.step)
