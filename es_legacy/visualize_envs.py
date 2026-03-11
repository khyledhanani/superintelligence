"""
Visualize CLUTTR environments as 2D grids.

Converts 52-element integer sequences to 13x13 grid visualizations.
Can visualize random environments, evolved environments, or compare them.

Usage:
    # Generate and visualize random environments
    python visualize_envs.py --random --num_envs 4

    # Visualize evolved environments from a .npy file
    python visualize_envs.py --evolved evolved/evolved_envs.npy --num_envs 4

    # Compare random vs evolved side-by-side
    python visualize_envs.py --compare --num_envs 4
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import os
import sys

# Add parent directory to path to import from vae folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from vae.sample_envs import generate_cluttr_batch_jax


def sequence_to_grid(seq, inner_dim=13):
    """Convert a 52-element CLUTTR sequence to a 2D grid visualization.

    Args:
        seq: Integer array of shape (52,).
             [obstacles (50), goal_idx (1), agent_idx (1)]
        inner_dim: Grid dimension (default 13).

    Returns:
        grid: 2D array of shape (inner_dim, inner_dim) with:
            0 = empty
            1 = obstacle
            2 = goal
            3 = agent
    """
    grid = np.zeros((inner_dim, inner_dim), dtype=int)

    # Place obstacles
    obstacles = seq[:50]
    for obs_idx in obstacles:
        if obs_idx > 0:  # skip padding (0)
            row = (obs_idx - 1) // inner_dim
            col = (obs_idx - 1) % inner_dim
            if 0 <= row < inner_dim and 0 <= col < inner_dim:
                grid[row, col] = 1

    # Place goal
    goal_idx = seq[50]
    if goal_idx > 0:
        row = (goal_idx - 1) // inner_dim
        col = (goal_idx - 1) % inner_dim
        if 0 <= row < inner_dim and 0 <= col < inner_dim:
            grid[row, col] = 2

    # Place agent
    agent_idx = seq[51]
    if agent_idx > 0:
        row = (agent_idx - 1) // inner_dim
        col = (agent_idx - 1) % inner_dim
        if 0 <= row < inner_dim and 0 <= col < inner_dim:
            grid[row, col] = 3

    return grid


def visualize_grid(ax, grid, title=""):
    """Draw a single CLUTTR grid on a matplotlib axis.

    Args:
        ax: Matplotlib axis.
        grid: 2D array (inner_dim, inner_dim) with values:
            0=empty, 1=obstacle, 2=goal, 3=agent
        title: Title for the subplot.
    """
    inner_dim = grid.shape[0]

    # Create color map: white=empty, black=obstacle, green=goal, red=agent
    cmap_colors = ['white', 'black', 'green', 'red']
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(cmap_colors)

    ax.imshow(grid, cmap=cmap, vmin=0, vmax=3, interpolation='nearest')
    ax.set_xticks(np.arange(inner_dim))
    ax.set_yticks(np.arange(inner_dim))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, which='both', color='gray', linewidth=0.5, alpha=0.3)
    ax.set_title(title, fontsize=10)

    # Add border
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)


def compute_stats(seq, inner_dim=13):
    """Compute statistics for a CLUTTR environment sequence.

    Returns:
        dict with keys: num_obstacles, manhattan_dist, valid
    """
    obstacles = seq[:50]
    goal_idx = seq[50]
    agent_idx = seq[51]

    num_obs = int((obstacles > 0).sum())

    agent_row = (agent_idx - 1) // inner_dim
    agent_col = (agent_idx - 1) % inner_dim
    goal_row = (goal_idx - 1) // inner_dim
    goal_col = (goal_idx - 1) % inner_dim
    manhattan = int(abs(agent_row - goal_row) + abs(agent_col - goal_col))

    valid = (goal_idx >= 1 and goal_idx <= inner_dim**2 and
             agent_idx >= 1 and agent_idx <= inner_dim**2 and
             goal_idx != agent_idx)

    return {
        'num_obstacles': num_obs,
        'manhattan_dist': manhattan,
        'valid': valid
    }


def visualize_environments(sequences, title_prefix="Env", inner_dim=13, figsize=(12, 10)):
    """Visualize a batch of CLUTTR environments as a grid of subplots.

    Args:
        sequences: Array of shape (num_envs, 52).
        title_prefix: Prefix for subplot titles.
        inner_dim: Grid dimension.
        figsize: Figure size.

    Returns:
        fig: Matplotlib figure.
    """
    num_envs = len(sequences)
    ncols = min(4, num_envs)
    nrows = (num_envs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if num_envs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, seq in enumerate(sequences):
        grid = sequence_to_grid(seq, inner_dim)
        stats = compute_stats(seq, inner_dim)

        title = (f"{title_prefix} {i}\n"
                 f"Obs: {stats['num_obstacles']}, "
                 f"Dist: {stats['manhattan_dist']}")

        visualize_grid(axes[i], grid, title)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize CLUTTR environments")
    parser.add_argument("--random", action="store_true", help="Generate and visualize random environments")
    parser.add_argument("--evolved", type=str, help="Path to evolved environments .npy file")
    parser.add_argument("--compare", action="store_true", help="Compare random vs evolved")
    parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to visualize")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, help="Save figure to this path (optional)")
    args = parser.parse_args()

    inner_dim = 13

    if args.compare:
        # Generate random
        key = jax.random.PRNGKey(args.seed)
        _, random_seqs = generate_cluttr_batch_jax(key, args.num_envs)
        random_seqs = np.array(random_seqs)

        # Load evolved
        if not os.path.exists("evolved/evolved_envs.npy"):
            print("Error: No evolved environments found. Run evolve_envs.py first.")
            return
        evolved_seqs = np.load("evolved/evolved_envs.npy")[:args.num_envs]

        # Create comparison figure
        fig = plt.figure(figsize=(16, 10))

        # Random on top row
        for i in range(args.num_envs):
            ax = plt.subplot(2, args.num_envs, i + 1)
            grid = sequence_to_grid(random_seqs[i], inner_dim)
            stats = compute_stats(random_seqs[i], inner_dim)
            title = f"Random {i}\nObs: {stats['num_obstacles']}, Dist: {stats['manhattan_dist']}"
            visualize_grid(ax, grid, title)

        # Evolved on bottom row
        for i in range(min(args.num_envs, len(evolved_seqs))):
            ax = plt.subplot(2, args.num_envs, args.num_envs + i + 1)
            grid = sequence_to_grid(evolved_seqs[i], inner_dim)
            stats = compute_stats(evolved_seqs[i], inner_dim)
            title = f"Evolved {i}\nObs: {stats['num_obstacles']}, Dist: {stats['manhattan_dist']}"
            visualize_grid(ax, grid, title)

        plt.suptitle("Random (top) vs Evolved (bottom) Environments", fontsize=14, fontweight='bold')
        plt.tight_layout()

    elif args.random:
        key = jax.random.PRNGKey(args.seed)
        _, sequences = generate_cluttr_batch_jax(key, args.num_envs)
        sequences = np.array(sequences)
        fig = visualize_environments(sequences, "Random", inner_dim)
        plt.suptitle("Randomly Generated CLUTTR Environments", fontsize=14, fontweight='bold')

    elif args.evolved:
        if not os.path.exists(args.evolved):
            print(f"Error: File not found: {args.evolved}")
            return
        sequences = np.load(args.evolved)[:args.num_envs]
        fig = visualize_environments(sequences, "Evolved", inner_dim)
        plt.suptitle("Evolved CLUTTR Environments", fontsize=14, fontweight='bold')

        # Print stats summary
        print(f"\nLoaded {len(sequences)} evolved environments from {args.evolved}")
        stats_list = [compute_stats(seq, inner_dim) for seq in sequences]
        avg_obs = np.mean([s['num_obstacles'] for s in stats_list])
        avg_dist = np.mean([s['manhattan_dist'] for s in stats_list])
        print(f"Average obstacles: {avg_obs:.1f}")
        print(f"Average Manhattan distance: {avg_dist:.1f}")
        print(f"All valid: {all(s['valid'] for s in stats_list)}")

    else:
        print("Error: Specify --random, --evolved <path>, or --compare")
        return

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
