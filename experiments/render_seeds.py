"""Render harvested LLM seed mazes to PNG images.

Loads seeds from seeds_levels.pkl or seeds.npz, renders each maze using
MazeRenderer, and saves individual PNGs + a grid overview.

Usage:
    python experiments/render_seeds.py \
        --seeds_dir /path/to/seeds_10k \
        --output_dir /path/to/seeds_10k/renders
"""
import argparse
import os
import pickle
import sys

import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in [
    os.path.join(_project_root, "src"),
    os.path.join(_project_root, "vae"),
    _project_root,
]:
    if p not in sys.path:
        sys.path.insert(0, p)

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jaxued.environments import Maze
from jaxued.environments.maze import Level
from jaxued.environments.maze.renderer import MazeRenderer


def render_seeds(args):
    # Load seeds
    pkl_path = os.path.join(args.seeds_dir, "seeds_levels.pkl")
    npz_path = os.path.join(args.seeds_dir, "seeds.npz")

    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            levels = pickle.load(f)
        print(f"Loaded {len(levels)} levels from {pkl_path}")
    elif os.path.exists(npz_path):
        from vae_level_utils import tokens_to_level
        data = np.load(npz_path, allow_pickle=True)
        tokens = data["tokens"]
        levels = [tokens_to_level(jnp.array(t)) for t in tokens]
        print(f"Loaded {len(levels)} levels from {npz_path}")
    else:
        print(f"No seeds found in {args.seeds_dir}")
        return

    # Set up renderer
    env = Maze(max_height=13, max_width=13, agent_view_size=5, normalize_obs=True)
    renderer = MazeRenderer(env, tile_size=args.tile_size)
    env_params = env.default_params

    os.makedirs(args.output_dir, exist_ok=True)

    # Render each seed individually
    images = []
    for i, level in enumerate(levels):
        img = np.asarray(renderer.render_level(level, env_params))
        images.append(img)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img)
        ax.set_title(f"Seed {i}")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(args.output_dir, f"seed_{i:02d}.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved seed_{i:02d}.png")

    # Render grid overview
    n = len(images)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if nrows == 1:
        axes = [axes] if ncols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i, (img, ax) in enumerate(zip(images, axes)):
        ax.imshow(img)
        ax.set_title(f"Seed {i}", fontsize=10)
        ax.axis("off")
    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"LLM-Generated Seed Mazes ({n} seeds)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "seeds_overview.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved seeds_overview.png")
    print(f"\nAll renders saved to {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render seed mazes to PNG")
    parser.add_argument("--seeds_dir", type=str, required=True,
                        help="Directory containing seeds_levels.pkl or seeds.npz")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for renders (default: seeds_dir/renders)")
    parser.add_argument("--tile_size", type=int, default=32)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join(args.seeds_dir, "renders")
    render_seeds(args)
