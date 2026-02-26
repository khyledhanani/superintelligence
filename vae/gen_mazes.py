import numpy as np
from maze_dataset.generation import LatticeMazeGenerators
import random
import yaml
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import jax
jax.config.update("jax_platform_name", "cpu")

MAX_WALLS = 50
SEQ_LEN   = 52   # 50 walls + goal + agent
GRID_SIZE = 13

GENERATORS = [
    ("DFS",    lambda: LatticeMazeGenerators.gen_dfs(grid_shape=(7, 7))),
    ("Wilson", lambda: LatticeMazeGenerators.gen_wilson(grid_shape=(7, 7))),
]


def maze_to_inner_grid(maze) -> np.ndarray:
    """Convert LatticeMaze to 13x13 binary wall grid (255=wall, 0=path)."""
    full_grid = maze.as_pixels().astype(int)[:, :, 0]
    full_grid = np.where(full_grid == 255, 0, 255)  # flip: maze-dataset → CLUTR convention
    return full_grid[1:14, 1:14]                    # drop static border


def prune_walls(wall_indices: list, max_walls: int = MAX_WALLS) -> list:
    """Randomly remove walls above the limit. Returns sorted pruned list."""
    if len(wall_indices) <= max_walls:
        return wall_indices
    return sorted(random.sample(wall_indices, max_walls))


def grid_from_walls(wall_indices: list) -> np.ndarray:
    """Reconstruct a 13x13 grid from 1-based wall indices."""
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for idx in wall_indices:
        r, c = divmod(idx - 1, GRID_SIZE)
        grid[r, c] = 255
    return grid


def encode_sequence(wall_indices: list, agent_idx: int, goal_idx: int) -> np.ndarray:
    """Encode as a zero-padded sequence of length SEQ_LEN."""
    seq = sorted(wall_indices) + [goal_idx, agent_idx]
    pad = SEQ_LEN - len(seq)
    return np.array([0] * pad + seq, dtype=np.int32)


def generate_one_maze(gen_fn):
    """
    Generate one maze, prune to MAX_WALLS, place agent & goal.
    Returns: original_grid, pruned_grid, n_walls_before, n_walls_after,
             agent_idx (1-based), goal_idx (1-based)
    """
    maze = gen_fn()
    inner_grid = maze_to_inner_grid(maze)

    flat = inner_grid.flatten()
    all_wall_indices = [idx + 1 for idx, v in enumerate(flat) if v == 255]

    pruned = prune_walls(all_wall_indices)
    pruned_grid = grid_from_walls(pruned)

    # Place agent & goal on original path cells — valid in both original and pruned views
    path_indices_2d = np.argwhere(inner_grid == 0)
    start_2d, goal_2d = random.sample(list(path_indices_2d), 2)
    agent_idx = int(start_2d[0] * GRID_SIZE + start_2d[1]) + 1
    goal_idx  = int(goal_2d[0]  * GRID_SIZE + goal_2d[1])  + 1

    return (inner_grid, pruned_grid,
            len(all_wall_indices), len(pruned),
            agent_idx, goal_idx)


def generate_diverse_mazes(num_mazes: int) -> np.ndarray:
    """
    Generate num_mazes algorithmically diverse mazes pruned to <= MAX_WALLS.
    Returns numpy array of shape (num_mazes, SEQ_LEN).
    """
    sequences = []
    for i in range(num_mazes):
        _, gen_fn = GENERATORS[i % len(GENERATORS)]
        _, pruned_grid, _, _, agent_idx, goal_idx = generate_one_maze(gen_fn)
        flat = pruned_grid.flatten()
        wall_indices = [idx + 1 for idx, v in enumerate(flat) if v == 255]
        sequences.append(encode_sequence(wall_indices, agent_idx, goal_idx))
    return np.stack(sequences)


def visualize_sequences(n_show: int = 16, cols: int = 4, save_path: str = None):
    """
    Generate n_show mazes, encode them as VAE sequences, and render them
    using render_maze() from lpca.py — exactly what the VAE sees.
    This lets you verify the sequences are structurally valid.
    """
    from lpca import render_maze

    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            name, gen_fn = GENERATORS[idx % len(GENERATORS)]
            _, pruned_grid, n_before, n_after, agent_idx, goal_idx = \
                generate_one_maze(gen_fn)

            flat = pruned_grid.flatten()
            wall_indices = [idx2 + 1 for idx2, v in enumerate(flat) if v == 255]
            seq = encode_sequence(wall_indices, agent_idx, goal_idx)

            title = f"{name} | {n_before}→{n_after}w"
            render_maze(axes[i, j], seq, title=title)

    fig.suptitle(
        f"Encoded sequences rendered via MiniGrid  —  walls capped at {MAX_WALLS}",
        fontsize=10, fontweight='bold'
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Viz] saved → {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_maze_grid(n_show: int = 16, cols: int = 4, save_path: str = None):
    """
    Show a grid of mazes: each pair of columns = original | pruned.
    Black = wall kept, orange = wall removed, red dot = agent, green star = goal.
    """
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 4, rows * 2.2))
    axes = np.array(axes).reshape(rows, cols * 2)

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            name, gen_fn = GENERATORS[idx % len(GENERATORS)]
            orig_grid, pruned_grid, n_before, n_after, agent_idx, goal_idx = \
                generate_one_maze(gen_fn)

            ar, ac = divmod(agent_idx - 1, GRID_SIZE)
            gr, gc = divmod(goal_idx  - 1, GRID_SIZE)
            removed_mask = (orig_grid == 255) & (pruned_grid == 0)

            # ── Original ────────────────────────────────────────────
            ax_o = axes[i, j * 2]
            rgb_o = np.stack([orig_grid] * 3, axis=-1).astype(float) / 255.0
            ax_o.imshow(1 - rgb_o)   # invert so walls=black, path=white
            ax_o.scatter(ac, ar, color='red',  s=30, zorder=3)
            ax_o.scatter(gc, gr, color='lime', s=30, marker='*', zorder=3)
            ax_o.set_title(f"{name} {n_before}w", fontsize=7)
            ax_o.axis("off")

            # ── Pruned ──────────────────────────────────────────────
            ax_p = axes[i, j * 2 + 1]
            rgb_p = np.ones((GRID_SIZE, GRID_SIZE, 3), dtype=float)
            rgb_p[pruned_grid == 255] = [0, 0, 0]       # kept walls → black
            rgb_p[removed_mask]       = [1.0, 0.55, 0.0]  # removed walls → orange
            ax_p.imshow(rgb_p)
            ax_p.scatter(ac, ar, color='red',  s=30, zorder=3)
            ax_p.scatter(gc, gr, color='lime', s=30, marker='*', zorder=3)
            ax_p.set_title(f"→ {n_after}w", fontsize=7)
            ax_p.axis("off")

    legend = [
        mpatches.Patch(color='black',       label=f'Wall kept (≤{MAX_WALLS})'),
        mpatches.Patch(color=(1, .55, 0),   label='Wall removed'),
        mpatches.Patch(color='white',       label='Path'),
        plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='red',  label='Agent', markersize=6),
        plt.Line2D([0],[0], marker='*', color='w', markerfacecolor='lime', label='Goal',  markersize=8),
    ]
    fig.legend(handles=legend, loc='lower center', ncol=5, fontsize=8, frameon=False)
    fig.suptitle(
        f"Algorithmic mazes (DFS / Wilson, 7×7)  —  walls capped at {MAX_WALLS}  —  orange = removed",
        fontsize=10, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Viz] saved → {save_path}")
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--num_mazes", type=int, default=10000)
    p.add_argument("--output",    type=str, default="algo_mazes.npy")
    p.add_argument("--visualize", action="store_true",
                   help="Save a raw grid preview (original vs pruned)")
    p.add_argument("--validate",  action="store_true",
                   help="Render encoded sequences via MiniGrid to verify validity")
    p.add_argument("--n_show",    type=int, default=16)
    args = p.parse_args()

    if args.visualize:
        print(f"[Viz] generating {args.n_show} sample mazes …")
        visualize_maze_grid(n_show=args.n_show, save_path="algo_maze_preview.png")

    if args.validate:
        print(f"[Validate] rendering {args.n_show} encoded sequences via MiniGrid …")
        visualize_sequences(n_show=args.n_show, save_path="algo_maze_sequences.png")

    print(f"[Gen] generating {args.num_mazes} mazes …")
    mazes = generate_diverse_mazes(args.num_mazes)
    print(f"[Gen] shape: {mazes.shape}  dtype: {mazes.dtype}")

    with open('vae_train_config.yml', "r") as f:
        CONFIG = yaml.safe_load(f)

    target_dir = os.path.join(CONFIG['working_path'], CONFIG['vae_folder'], 'datasets')
    os.makedirs(target_dir, exist_ok=True)
    save_path = os.path.join(target_dir, args.output)
    np.save(save_path, mazes)
    print(f"[Save] {save_path}")
