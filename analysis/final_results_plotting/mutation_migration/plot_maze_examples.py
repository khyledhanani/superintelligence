"""Plot example mazes at different tile flip counts from LLM ancestors.

Rows = different LLM ancestor seeds (same ancestor per row).
Columns = original, then each flip count.
Changed cells highlighted in red.

Usage:
    python analysis/final_results_plotting/mutation_migration/plot_maze_examples.py --seed 3
"""
import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'examples'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def tokens_to_grid(tok):
    """Convert tokens to a 13x13 grid. 0=empty, 1=wall, 2=agent, 3=goal."""
    grid = np.zeros((13, 13), dtype=int)
    for w in tok[:50]:
        r, c = divmod(int(w), 13)
        if r < 13 and c < 13:
            grid[r, c] = 1
    agent_r, agent_c = divmod(int(tok[50]), 13)
    goal_r, goal_c = divmod(int(tok[51]), 13)
    if agent_r < 13 and agent_c < 13:
        grid[agent_r, agent_c] = 2
    if goal_r < 13 and goal_c < 13:
        grid[goal_r, goal_c] = 3
    return grid


def render_maze(ax, grid, title=None, ancestor_grid=None):
    """Render a maze grid on a matplotlib axis.

    If ancestor_grid is provided, highlight cells that differ in red.
    """
    rgb = np.ones((13, 13, 3))
    for r in range(13):
        for c in range(13):
            if grid[r, c] == 1:
                rgb[r, c] = [0.2, 0.2, 0.2]
            elif grid[r, c] == 2:
                rgb[r, c] = [0.2, 0.4, 1.0]
            elif grid[r, c] == 3:
                rgb[r, c] = [0.2, 0.8, 0.2]

    ax.imshow(rgb, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str,
                        default="analysis/tsne_evolution_investigation/llm_injection_fresh/cache_solved")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--flip_counts", type=int, nargs='+', default=[2, 20, 80])
    parser.add_argument("--n_rows", type=int, default=3,
                        help="Number of ancestor rows to show")
    parser.add_argument("--tolerance", type=int, default=2,
                        help="Tolerance for matching flip counts")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = "analysis/final_results_plotting/mutation_migration/maze_examples.png"

    # Load LLM originals
    inj_path = os.path.join(args.cache_dir, f"emb_solved_s{args.seed}_t2500.npz")
    inj = np.load(inj_path, allow_pickle=True)
    seed_tokens = {}
    for aid, tok in zip(inj['ancestor_ids'][inj['origins'] == 1], inj['tokens'][inj['origins'] == 1]):
        seed_tokens[int(aid)] = tok

    # Pool descendants across all timesteps
    files = sorted(glob.glob(os.path.join(args.cache_dir, f"emb_solved_s{args.seed}_t*.npz")))
    # ancestor_id -> flip_count -> list of token arrays
    pool = defaultdict(lambda: defaultdict(list))

    for f in files:
        d = np.load(f, allow_pickle=True)
        td = d['tile_diffs']
        origins = d['origins']
        aids = d['ancestor_ids']
        tokens = d['tokens']

        for fc in args.flip_counts:
            mask = (origins > 0) & (td >= 0) & (np.abs(td - fc) <= args.tolerance)
            for idx in np.where(mask)[0]:
                aid = int(aids[idx])
                if aid >= 0 and aid in seed_tokens:
                    pool[aid][fc].append(tokens[idx])

    # Find ancestors that have descendants at ALL flip counts
    flip_counts = args.flip_counts
    valid_ancestors = []
    for aid in sorted(pool.keys()):
        if all(len(pool[aid][fc]) > 0 for fc in flip_counts):
            valid_ancestors.append(aid)

    if len(valid_ancestors) == 0:
        print("No ancestors found with descendants at all flip counts")
        return

    n_rows = min(args.n_rows, len(valid_ancestors))
    chosen_ancestors = valid_ancestors[:n_rows]
    print(f"Using ancestors: {chosen_ancestors}")

    # Layout: rows = ancestors, cols = original + flip_counts
    n_cols = 1 + len(flip_counts)  # original + each flip count

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(2.5 * n_cols, 2.5 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]
    if n_cols == 1:
        axes = axes[:, None]

    rng = np.random.RandomState(0)

    # Column headers
    axes[0, 0].set_title('LLM Original', fontsize=9, fontweight='bold')
    for ci, fc in enumerate(flip_counts):
        axes[0, ci + 1].set_title(f'{fc} tile flips', fontsize=9, fontweight='bold')

    for row_idx, aid in enumerate(chosen_ancestors):
        # Original
        anc_grid = tokens_to_grid(seed_tokens[aid])
        render_maze(axes[row_idx, 0], anc_grid)
        pass  # no row label

        # Descendants at each flip count
        for ci, fc in enumerate(flip_counts):
            candidates = pool[aid][fc]
            tok = candidates[rng.randint(len(candidates))]
            desc_grid = tokens_to_grid(tok)
            render_maze(axes[row_idx, ci + 1], desc_grid, ancestor_grid=anc_grid)

    plt.suptitle('LLM Maze Mutations', fontsize=13, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
