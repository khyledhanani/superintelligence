"""Analyze PLR replay buffer dumps across conditions.

Computes maze structural metrics (wall count, path length, reachability)
and score distributions to understand WHY CMA-ES+PCA outperforms ACCEL.

Usage:
    python scripts/analyze_buffers.py [--snapshot 50k] [--save_plots]
"""
import sys, os
import numpy as np
from collections import defaultdict
from pathlib import Path

# Add vae/ to path for clutr_to_grid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vae'))
from cnn_vae_data import clutr_to_grid


# ── Maze structural metrics ─────────────────────────────────────────────

def compute_maze_metrics(tokens_batch):
    """Compute structural metrics for a batch of CLUTR token sequences.

    Args:
        tokens_batch: (N, 52) int32 array of CLUTR sequences

    Returns:
        dict of metric arrays, each (N,)
    """
    N = len(tokens_batch)
    wall_counts = np.zeros(N, dtype=np.int32)
    path_lengths = np.zeros(N, dtype=np.float32)
    reachable_fracs = np.zeros(N, dtype=np.float32)
    goal_reachable = np.zeros(N, dtype=bool)
    open_cells = np.zeros(N, dtype=np.int32)

    for i in range(N):
        seq = tokens_batch[i]
        grid = clutr_to_grid(np.array(seq))

        walls = grid[:, :, 0]  # (13, 13) binary
        n_walls = int(walls.sum())
        wall_counts[i] = n_walls
        open_cells[i] = 169 - n_walls

        # Agent and goal positions
        agent_pos = int(seq[51]) - 1
        goal_pos = int(seq[50]) - 1
        agent_r, agent_c = agent_pos // 13, agent_pos % 13
        goal_r, goal_c = goal_pos // 13, goal_pos % 13

        # BFS from agent
        dist, visited = _bfs(walls, agent_r, agent_c)
        n_reachable = visited.sum()
        reachable_fracs[i] = n_reachable / max(169 - n_walls, 1)

        if visited[goal_r, goal_c]:
            goal_reachable[i] = True
            path_lengths[i] = dist[goal_r, goal_c]
        else:
            goal_reachable[i] = False
            path_lengths[i] = np.inf

    # Second pass: topology metrics (need BFS results)
    dead_ends = np.zeros(N, dtype=np.int32)
    branch_points = np.zeros(N, dtype=np.int32)
    max_corridor = np.zeros(N, dtype=np.int32)  # longest dead-end corridor
    path_optimality = np.zeros(N, dtype=np.float32)  # shortest_path / reachable_cells
    n_connected_components = np.zeros(N, dtype=np.int32)

    for i in range(N):
        seq = tokens_batch[i]
        grid = clutr_to_grid(np.array(seq))
        walls = grid[:, :, 0]

        # Count neighbors for each open cell
        de = 0
        bp = 0
        for r in range(13):
            for c in range(13):
                if walls[r, c] > 0.5:
                    continue
                neighbors = 0
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < 13 and 0 <= nc < 13 and walls[nr, nc] < 0.5:
                        neighbors += 1
                if neighbors == 1:
                    de += 1  # dead end: only 1 open neighbor
                elif neighbors >= 3:
                    bp += 1  # branch point: 3+ open neighbors
        dead_ends[i] = de
        branch_points[i] = bp

        # Max corridor: longest chain of dead-end cells
        # (BFS from each dead end until branch point)
        agent_pos = int(seq[51]) - 1
        agent_r, agent_c = agent_pos // 13, agent_pos % 13
        dist, visited = _bfs(walls, agent_r, agent_c)
        n_reachable = int(visited.sum())

        # Path optimality: how much of the reachable space must be traversed
        if np.isfinite(path_lengths[i]) and n_reachable > 0:
            path_optimality[i] = path_lengths[i] / n_reachable
        else:
            path_optimality[i] = 0.0

        # Connected components (open cells)
        comp_visited = np.zeros((13, 13), dtype=bool)
        n_comp = 0
        for r in range(13):
            for c in range(13):
                if walls[r, c] < 0.5 and not comp_visited[r, c]:
                    n_comp += 1
                    _, vis = _bfs(walls, r, c)
                    comp_visited |= vis
        n_connected_components[i] = n_comp

    return {
        'wall_count': wall_counts,
        'path_length': path_lengths,
        'reachable_frac': reachable_fracs,
        'goal_reachable': goal_reachable,
        'open_cells': open_cells,
        'dead_ends': dead_ends,
        'branch_points': branch_points,
        'path_optimality': path_optimality,
        'n_components': n_connected_components,
    }


def _bfs(walls, start_r, start_c):
    """BFS on 13x13 grid. Returns distance array and visited mask."""
    dist = np.full((13, 13), -1, dtype=np.int32)
    visited = np.zeros((13, 13), dtype=bool)

    if walls[start_r, start_c] > 0.5:
        return dist, visited

    from collections import deque
    queue = deque()
    queue.append((start_r, start_c))
    dist[start_r, start_c] = 0
    visited[start_r, start_c] = True

    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 13 and 0 <= nc < 13 and not visited[nr, nc] and walls[nr, nc] < 0.5:
                visited[nr, nc] = True
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))

    return dist, visited


# ── Data loading ─────────────────────────────────────────────────────────

# Map condition name -> (base_dir, group_name)
CONDITIONS = {
    'pca-staged-accel': ('buffer_dumps_barnacle', 'pca-staged-accel'),
    'clutr-staged-accel': ('buffer_dumps_albacore', 'clutr-staged-accel'),
    'accel-baseline': ('buffer_dumps_tpu', 'accel-baseline'),
}


def load_dump(base_dir, group, seed, snapshot):
    """Load a single buffer dump .npz file."""
    root = Path(__file__).parent.parent
    path = root / base_dir / group / str(seed) / f"buffer_dump_{snapshot}.npz"
    if not path.exists():
        return None
    return dict(np.load(path))


def find_seeds(base_dir, group):
    """Find available seed directories."""
    root = Path(__file__).parent.parent
    group_dir = root / base_dir / group
    if not group_dir.exists():
        return []
    return sorted([int(d.name) for d in group_dir.iterdir() if d.is_dir() and d.name.isdigit()])


# ── Analysis ─────────────────────────────────────────────────────────────

def analyze_condition(base_dir, group, snapshot, max_seeds=3):
    """Analyze all seeds for a condition at a given snapshot."""
    seeds = find_seeds(base_dir, group)[:max_seeds]
    if not seeds:
        print(f"  [SKIP] No seeds found for {group} in {base_dir}")
        return None

    all_metrics = defaultdict(list)
    all_scores = []
    total_levels = 0

    for seed in seeds:
        data = load_dump(base_dir, group, seed, snapshot)
        if data is None:
            print(f"  [SKIP] {group}/seed {seed}/{snapshot} not found")
            continue

        tokens = data['tokens']
        scores = data['scores']
        size = int(data['size'])
        tokens = tokens[:size]
        scores = scores[:size]

        metrics = compute_maze_metrics(tokens)
        for k, v in metrics.items():
            all_metrics[k].append(v)
        all_scores.append(scores)
        total_levels += size

    if total_levels == 0:
        return None

    # Concatenate across seeds
    result = {}
    for k in all_metrics:
        result[k] = np.concatenate(all_metrics[k])
    result['scores'] = np.concatenate(all_scores)
    result['n_seeds'] = len(seeds)
    result['total_levels'] = total_levels
    return result


def print_summary(name, result):
    """Print summary statistics for a condition."""
    if result is None:
        print(f"\n{'='*60}")
        print(f"  {name}: NO DATA")
        return

    print(f"\n{'='*60}")
    print(f"  {name}  ({result['n_seeds']} seeds, {result['total_levels']} levels)")
    print(f"{'='*60}")

    sc = result['scores']
    print(f"  Scores:       mean={sc.mean():.3f}  std={sc.std():.3f}  "
          f"median={np.median(sc):.3f}  max={sc.max():.3f}")

    wc = result['wall_count']
    print(f"  Wall count:   mean={wc.mean():.1f}  std={wc.std():.1f}  "
          f"min={wc.min()}  max={wc.max()}")

    pl = result['path_length']
    finite = pl[np.isfinite(pl)]
    print(f"  Path length:  mean={finite.mean():.1f}  std={finite.std():.1f}  "
          f"max={finite.max():.0f}  (inf: {(~np.isfinite(pl)).sum()}/{len(pl)})")

    rf = result['reachable_frac']
    print(f"  Reachable %:  mean={rf.mean()*100:.1f}%  std={rf.std()*100:.1f}%")

    gr = result['goal_reachable']
    print(f"  Goal reach:   {gr.sum()}/{len(gr)} ({gr.mean()*100:.1f}%)")

    oc = result['open_cells']
    print(f"  Open cells:   mean={oc.mean():.1f}  std={oc.std():.1f}")

    de = result['dead_ends']
    print(f"  Dead ends:    mean={de.mean():.1f}  std={de.std():.1f}")

    bp = result['branch_points']
    print(f"  Branch pts:   mean={bp.mean():.1f}  std={bp.std():.1f}")

    po = result['path_optimality']
    print(f"  Path optim:   mean={po.mean():.3f}  std={po.std():.3f}  "
          f"(shortest_path / reachable_cells — higher = more linear maze)")

    nc = result['n_components']
    print(f"  Components:   mean={nc.mean():.2f}  (>1 means disconnected regions)")

    # Level diversity: std of wall positions across buffer
    print(f"\n  Wall count distribution:")
    for lo, hi in [(0, 20), (20, 35), (35, 45), (45, 50), (50, 51)]:
        count = ((wc >= lo) & (wc < hi)).sum()
        pct = count / len(wc) * 100
        print(f"    [{lo:2d}-{hi:2d}): {count:5d} ({pct:5.1f}%)")

    # Score-weighted metrics: what do HIGH-REGRET levels look like?
    top_k = max(1, len(sc) // 10)  # top 10%
    top_idx = np.argsort(sc)[-top_k:]
    bot_idx = np.argsort(sc)[:top_k]

    print(f"\n  --- Top 10% regret levels (n={top_k}) ---")
    print(f"  Walls: {wc[top_idx].mean():.1f}  Path: {pl[top_idx][np.isfinite(pl[top_idx])].mean():.1f}"
          f"  DeadEnds: {de[top_idx].mean():.1f}  BranchPts: {bp[top_idx].mean():.1f}"
          f"  PathOpt: {po[top_idx].mean():.3f}")

    print(f"  --- Bottom 10% regret levels (n={top_k}) ---")
    pl_bot = pl[bot_idx]
    finite_bot = pl_bot[np.isfinite(pl_bot)]
    pl_bot_mean = finite_bot.mean() if len(finite_bot) > 0 else float('inf')
    print(f"  Walls: {wc[bot_idx].mean():.1f}  Path: {pl_bot_mean:.1f}"
          f"  DeadEnds: {de[bot_idx].mean():.1f}  BranchPts: {bp[bot_idx].mean():.1f}"
          f"  PathOpt: {po[bot_idx].mean():.3f}")


def print_evolution(base_dir, group, name, snapshots=None):
    """Print how metrics evolve over training snapshots."""
    if snapshots is None:
        snapshots = ['10k', '20k', '30k', '40k', '50k']

    print(f"\n{'='*60}")
    print(f"  {name} — Evolution over training")
    print(f"{'='*60}")
    print(f"  {'Snap':>6}  {'Score':>8}  {'Walls':>6}  {'Path':>6}  "
          f"{'DeadEnd':>7}  {'Branch':>7}  {'PathOpt':>8}  {'Comp':>5}  {'Levels':>7}")
    print(f"  {'-'*70}")

    for snap in snapshots:
        result = analyze_condition(base_dir, group, snap)
        if result is None:
            print(f"  {snap:>6}  {'---':>8}")
            continue

        sc = result['scores']
        wc = result['wall_count']
        pl = result['path_length']
        finite = pl[np.isfinite(pl)]
        de = result['dead_ends']
        bp = result['branch_points']
        po = result['path_optimality']
        nc = result['n_components']

        pl_mean = finite.mean() if len(finite) > 0 else float('inf')
        print(f"  {snap:>6}  {sc.mean():>8.3f}  {wc.mean():>6.1f}  "
              f"{pl_mean:>6.1f}  {de.mean():>7.1f}  {bp.mean():>7.1f}  "
              f"{po.mean():>8.3f}  {nc.mean():>5.2f}  {result['total_levels']:>7}")


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', default='50k', help='Snapshot to compare (e.g. 10k, 50k, final)')
    parser.add_argument('--evolution', action='store_true', help='Show evolution over all snapshots')
    args = parser.parse_args()

    print(f"\n*** Buffer Analysis — Snapshot: {args.snapshot} ***\n")

    # Cross-condition comparison at a single snapshot
    for name, (base_dir, group) in CONDITIONS.items():
        result = analyze_condition(base_dir, group, args.snapshot)
        print_summary(name, result)

    # Evolution over training
    if args.evolution:
        print(f"\n\n*** Evolution Over Training ***")
        for name, (base_dir, group) in CONDITIONS.items():
            print_evolution(base_dir, group, name)

    print("\n\nDone.")
