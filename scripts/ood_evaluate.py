#!/usr/bin/env python
"""
Out-of-distribution evaluation of trained agents on novel mazes.

Two modes:
  generate — Create OOD eval sets (random + perfect mazes at various sizes)
  evaluate — Load agent checkpoints and evaluate on OOD mazes

Usage:
    # Generate eval sets (run once)
    python scripts/ood_evaluate.py generate --sizes 13 15 17 --n_mazes 100 --types random perfect

    # Evaluate on specific training steps
    python scripts/ood_evaluate.py evaluate \
        --eval_sets eval_levels/random_13x13.npz eval_levels/perfect_17x17.npz \
        --eval_steps 10000 20000 30000 \
        --num_attempts 10

    # Evaluate all available checkpoints (full learning curve)
    python scripts/ood_evaluate.py evaluate --eval_steps all

    # Evaluate specific conditions only
    python scripts/ood_evaluate.py evaluate \
        --conditions accel-llm-gpt41mini \
        --eval_steps 30000
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Condition registry — edit this when adding new experiment conditions
# ---------------------------------------------------------------------------
CONDITIONS = {
    "final-gpt41mini":      {"seeds": [0, 1, 2, 3, 4], "results_dir": "results/final-gpt41mini"},
    "final-accel-baseline": {"seeds": [0, 1, 2, 3, 4], "results_dir": "results/final-accel-baseline"},
}

# ===================================================================
#  GENERATE MODE
# ===================================================================

def generate_random_mazes(size, n_mazes, n_walls=25, seed=42):
    """Generate random solvable mazes using JAX level generator + solvability check."""
    import jax
    import jax.numpy as jnp

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from jaxued.environments.maze.util import make_level_generator
    from jaxued.environments.maze.env_solved import MazeSolved
    from jaxued.environments.maze.level import Level

    generator = make_level_generator(size, size, n_walls)
    checker = MazeSolved(max_height=size, max_width=size, agent_view_size=5, normalize_obs=True)
    env_params = checker.default_params

    # JIT the generate + check pipeline
    @jax.jit
    def gen_and_check(rng):
        level = generator(rng)
        state = checker.init_state_from_level(level)
        solvable = checker.is_solveable(state, env_params)
        return level, solvable

    rng = jax.random.PRNGKey(seed)
    levels = []
    attempts = 0

    print(f"  Generating {n_mazes} random solvable {size}x{size} mazes (n_walls={n_walls})...")
    while len(levels) < n_mazes:
        rng, rng_gen = jax.random.split(rng)
        level, solvable = gen_and_check(rng_gen)
        attempts += 1
        if solvable:
            levels.append(level)
            if len(levels) % 25 == 0:
                print(f"    {len(levels)}/{n_mazes} collected ({attempts} attempts)")

    print(f"  Done: {n_mazes} mazes from {attempts} attempts ({n_mazes/attempts:.0%} solvable)")

    # Stack into batched Level
    stacked = Level.stack(levels)
    return stacked, levels


def generate_perfect_mazes(size, n_mazes, seed=42):
    """Generate perfect mazes using mazelib's RecursiveBacktracker."""
    from mazelib import Maze as MazelibMaze
    from mazelib.generate.BacktrackingGenerator import BacktrackingGenerator

    import jax.numpy as jnp

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from jaxued.environments.maze.level import Level

    # mazelib uses (h, w) where output grid is (2*h+1, 2*w+1)
    # For 13x13 output: h=w=6 -> 2*6+1=13
    cell_h = (size - 1) // 2
    cell_w = (size - 1) // 2

    rng = np.random.RandomState(seed)
    levels = []

    print(f"  Generating {n_mazes} perfect {size}x{size} mazes (cell grid {cell_h}x{cell_w})...")

    for i in range(n_mazes):
        m = MazelibMaze(seed=int(rng.randint(0, 2**31)))
        m.generator = BacktrackingGenerator(cell_h, cell_w)
        m.generate()

        grid = np.array(m.grid, dtype=bool)  # True = wall, False = passage
        actual_h, actual_w = grid.shape

        # Ensure grid matches target size (pad or crop if needed)
        if actual_h != size or actual_w != size:
            new_grid = np.ones((size, size), dtype=bool)  # all walls
            h_copy = min(actual_h, size)
            w_copy = min(actual_w, size)
            new_grid[:h_copy, :w_copy] = grid[:h_copy, :w_copy]
            grid = new_grid

        # Find all passable cells
        passable = np.argwhere(~grid)
        if len(passable) < 2:
            continue

        # Place agent and goal at maximally separated passable cells (BFS distance)
        agent_idx, goal_idx = _find_max_distance_pair(grid, passable)
        agent_pos = passable[agent_idx]  # (y, x)
        goal_pos = passable[goal_idx]    # (y, x)

        # Create Level: positions are (x, y) in Level format
        wall_map = jnp.array(grid)
        level = Level(
            wall_map=wall_map,
            goal_pos=jnp.array([goal_pos[1], goal_pos[0]], dtype=jnp.uint32),  # (x, y)
            agent_pos=jnp.array([agent_pos[1], agent_pos[0]], dtype=jnp.uint32),  # (x, y)
            agent_dir=jnp.array(0, dtype=jnp.uint8),  # facing right
            width=size,
            height=size,
        )
        levels.append(level)

        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{n_mazes} generated")

    print(f"  Done: {len(levels)} perfect mazes generated")

    stacked = Level.stack(levels)
    return stacked, levels


def _find_max_distance_pair(grid, passable):
    """Find the pair of passable cells with maximum BFS distance."""
    from collections import deque

    h, w = grid.shape

    # BFS from corner-most passable cell
    start = passable[0]
    dist = _bfs_distances(grid, start, h, w)

    # Find the farthest cell from start
    max_dist = -1
    farthest_idx = 0
    for i, (y, x) in enumerate(passable):
        if dist[y, x] > max_dist and dist[y, x] < np.inf:
            max_dist = dist[y, x]
            farthest_idx = i

    # BFS from farthest to find the true diameter endpoints
    far_cell = passable[farthest_idx]
    dist2 = _bfs_distances(grid, far_cell, h, w)

    max_dist2 = -1
    goal_idx = 0
    for i, (y, x) in enumerate(passable):
        if dist2[y, x] > max_dist2 and dist2[y, x] < np.inf:
            max_dist2 = dist2[y, x]
            goal_idx = i

    return farthest_idx, goal_idx


def _bfs_distances(grid, start, h, w):
    """BFS from start cell, returns distance array."""
    from collections import deque

    dist = np.full((h, w), np.inf)
    sy, sx = int(start[0]), int(start[1])
    dist[sy, sx] = 0
    queue = deque([(sy, sx)])

    while queue:
        y, x = queue.popleft()
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not grid[ny, nx] and dist[ny, nx] == np.inf:
                dist[ny, nx] = dist[y, x] + 1
                queue.append((ny, nx))

    return dist


def save_eval_set(stacked_levels, output_path, metadata):
    """Save a stacked Level as NPZ."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.savez_compressed(
        output_path,
        wall_map=np.array(stacked_levels.wall_map),
        goal_pos=np.array(stacked_levels.goal_pos),
        agent_pos=np.array(stacked_levels.agent_pos),
        agent_dir=np.array(stacked_levels.agent_dir),
        width=np.array(stacked_levels.width),
        height=np.array(stacked_levels.height),
        **{f"meta_{k}": np.array(v) if isinstance(v, (list, np.ndarray)) else np.array([v])
           for k, v in metadata.items()},
    )
    print(f"  Saved: {output_path} ({np.array(stacked_levels.wall_map).shape[0]} levels)")


def load_eval_set(npz_path):
    """Load a stacked Level from NPZ."""
    import jax.numpy as jnp
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from jaxued.environments.maze.level import Level

    data = np.load(npz_path, allow_pickle=True)
    level = Level(
        wall_map=jnp.array(data["wall_map"]),
        goal_pos=jnp.array(data["goal_pos"]),
        agent_pos=jnp.array(data["agent_pos"]),
        agent_dir=jnp.array(data["agent_dir"]),
        width=jnp.array(data["width"]),
        height=jnp.array(data["height"]),
    )
    n_levels = data["wall_map"].shape[0]
    maze_size = data["wall_map"].shape[1]
    level_names = None
    if "names" in data.files:
        level_names = [str(n) for n in data["names"]]
    return level, n_levels, maze_size, level_names


def cmd_generate(args):
    """Generate OOD evaluation maze sets."""
    output_dir = args.output_dir

    for size in args.sizes:
        for maze_type in args.types:
            print(f"\n{'='*60}")
            print(f"Generating {maze_type} {size}x{size} mazes")
            print(f"{'='*60}")

            output_path = os.path.join(output_dir, f"{maze_type}_{size}x{size}.npz")

            if os.path.exists(output_path) and not args.force:
                print(f"  Already exists: {output_path} (use --force to overwrite)")
                continue

            if maze_type == "random":
                stacked, levels = generate_random_mazes(
                    size, args.n_mazes, n_walls=args.n_walls, seed=args.seed,
                )
            elif maze_type == "perfect":
                stacked, levels = generate_perfect_mazes(
                    size, args.n_mazes, seed=args.seed,
                )
            else:
                print(f"  Unknown type: {maze_type}")
                continue

            # Verify a few
            print(f"  Sample maze 0:\n{levels[0].to_str()}")

            metadata = {
                "maze_type": maze_type,
                "maze_size": size,
                "n_mazes": args.n_mazes,
                "seed": args.seed,
            }
            save_eval_set(stacked, output_path, metadata)

    print(f"\nAll eval sets saved to {output_dir}/")


# ===================================================================
#  EVALUATE MODE
# ===================================================================

def resolve_checkpoint_steps(checkpoint_dir, eval_steps_arg, eval_freq=250):
    """Resolve requested eval steps to available orbax checkpoint steps.

    Args:
        checkpoint_dir: Path to seed checkpoint dir (has config.json + models/)
        eval_steps_arg: List of strings like ["10000", "20000"] or ["all"] or ["latest"]
        eval_freq: Eval frequency from config

    Returns:
        List of (orbax_step, training_updates) tuples
    """
    import orbax.checkpoint as ocp

    models_dir = os.path.join(checkpoint_dir, "models")
    if not os.path.exists(models_dir):
        print(f"  [WARN] No models dir: {models_dir}")
        return []

    checkpoint_manager = ocp.CheckpointManager(
        models_dir, item_handlers=ocp.StandardCheckpointHandler()
    )
    available = sorted(checkpoint_manager.all_steps())
    if not available:
        return []

    if eval_steps_arg == ["all"]:
        return [(s, (s + 1) * eval_freq) for s in available]

    if eval_steps_arg == ["latest"]:
        s = available[-1]
        return [(s, (s + 1) * eval_freq)]

    # Specific training update numbers — snap to nearest available
    results = []
    for step_str in eval_steps_arg:
        target_updates = int(step_str)
        target_orbax = (target_updates // eval_freq) - 1

        # Find closest
        closest = min(available, key=lambda s: abs(s - target_orbax))
        actual_updates = (closest + 1) * eval_freq
        gap = abs(actual_updates - target_updates)
        if gap > 2500:
            print(f"  [WARN] Step {target_updates}: closest is {actual_updates} (gap={gap}), skipping")
            continue
        results.append((closest, actual_updates))

    return results


def _bfs_shortest_paths(levels):
    """BFS from agent to goal for each stacked level. Returns (N,) float array
    (np.inf if unreachable)."""
    from collections import deque
    wall_maps = np.asarray(levels.wall_map)
    agent_pos = np.asarray(levels.agent_pos)
    goal_pos = np.asarray(levels.goal_pos)
    n = wall_maps.shape[0]
    out = np.full(n, np.inf, dtype=np.float64)
    for i in range(n):
        wm = wall_maps[i]
        ax, ay = int(agent_pos[i, 0]), int(agent_pos[i, 1])
        gx, gy = int(goal_pos[i, 0]), int(goal_pos[i, 1])
        h, w = wm.shape
        if wm[ay, ax] or wm[gy, gx]:
            continue
        q = deque([(ax, ay, 0)])
        visited = {(ax, ay)}
        while q:
            cx, cy, d = q.popleft()
            if cx == gx and cy == gy:
                out[i] = float(d)
                break
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h and not wm[ny, nx] and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append((nx, ny, d + 1))
    return out


def _iqm(values):
    """Interquartile mean of an array."""
    values = np.asarray(values, dtype=np.float64)
    if values.size < 4:
        return float(values.mean())
    q25, q75 = np.percentile(values, [25, 75])
    mask = (values >= q25) & (values <= q75)
    return float(values[mask].mean()) if mask.any() else float(values.mean())


def evaluate_agent_on_levels(train_state, env_params, levels, maze_size,
                              num_attempts, max_steps=None):
    """Evaluate an agent on OOD levels.

    Returns a dict with per-level + aggregate metrics matching the friend's
    eval protocol (vae/eval_on_test_levels.py):
        solve_rates, mean_returns, optimal_returns, optimality_gaps,
        shortest_paths, ep_length_means, path_ratio_means,
        mean_solve_rate, iqm_solve_rate, mean_return_agg,
        mean_optimality_gap, mean_path_ratio.
    """
    import jax
    import jax.numpy as jnp

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from jaxued.environments import Maze
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))
    from maze_plr import ActorCritic, evaluate_rnn

    # Create eval env at the correct maze size
    eval_env = Maze(
        max_height=maze_size, max_width=maze_size,
        agent_view_size=5, normalize_obs=True,
    )
    eval_params = eval_env.default_params

    if max_steps is None:
        # Scale max steps proportionally to maze area
        base_steps = 250
        max_steps = int(base_steps * (maze_size ** 2) / (13 ** 2))
        max_steps = max(max_steps, base_steps)

    num_levels = jax.tree_util.tree_flatten(levels)[0][0].shape[0]

    # BFS shortest paths (once). Reward schedule in Maze: at step t reaching
    # the goal gives `1 - 0.9 * (t+1) / max_steps`; matches the friend's
    # optimality-gap convention.
    shortest_paths = _bfs_shortest_paths(levels)
    optimal_returns = np.where(
        np.isfinite(shortest_paths),
        1.0 - 0.9 * (shortest_paths + 1) / max_steps,
        0.0,
    )

    all_cum_rewards = []
    all_ep_lengths = []

    for attempt_idx in range(num_attempts):
        rng = jax.random.PRNGKey(attempt_idx + 1000)
        rng, rng_eval, rng_reset = jax.random.split(rng, 3)

        init_obs, init_env_state = jax.vmap(eval_env.reset_to_level, (0, 0, None))(
            jax.random.split(rng_reset, num_levels), levels, eval_params
        )

        states, rewards, episode_lengths = evaluate_rnn(
            rng_eval, eval_env, eval_params, train_state,
            ActorCritic.initialize_carry((num_levels,)),
            init_obs, init_env_state, max_steps,
        )

        mask = np.arange(max_steps)[:, None] < np.asarray(episode_lengths)[None, :]
        cum_rewards = (np.asarray(rewards) * mask).sum(axis=0)
        all_cum_rewards.append(cum_rewards)
        all_ep_lengths.append(np.asarray(episode_lengths))

    all_cum_rewards = np.stack(all_cum_rewards)   # (num_attempts, num_levels)
    all_ep_lengths = np.stack(all_ep_lengths)      # (num_attempts, num_levels)

    # Per-level metrics
    solve_matrix = (all_cum_rewards > 0).astype(float)
    solve_rates = solve_matrix.mean(axis=0)
    mean_returns = all_cum_rewards.mean(axis=0)

    # Optimality gap: V*(s0) − mean V^pi(s0) (return-based proxy).
    optimality_gaps = optimal_returns - mean_returns

    # Episode lengths + path ratio (solved rollouts only)
    solved_mask = all_cum_rewards > 0
    with np.errstate(all="ignore"):
        ep_solved = np.where(solved_mask, all_ep_lengths, np.nan)
        ep_length_means = np.nanmean(ep_solved, axis=0)
        sp_broadcast = shortest_paths[None, :]
        path_ratios = np.where(solved_mask, all_ep_lengths / sp_broadcast, np.nan)
        path_ratio_means = np.nanmean(path_ratios, axis=0)

    return {
        "solve_rates": solve_rates,
        "mean_returns": mean_returns,
        "optimal_returns": optimal_returns,
        "optimality_gaps": optimality_gaps,
        "shortest_paths": shortest_paths,
        "ep_length_means": ep_length_means,
        "path_ratio_means": path_ratio_means,
        "mean_solve_rate": float(solve_rates.mean()),
        "iqm_solve_rate": _iqm(solve_rates),
        "mean_return_agg": float(np.nanmean(mean_returns)),
        "mean_optimality_gap": float(np.nanmean(optimality_gaps)),
        "mean_path_ratio": float(np.nanmean(path_ratio_means)),
    }


def cmd_evaluate(args):
    """Evaluate agents from all conditions on OOD maze sets."""
    import jax

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))
    from cross_evaluate import load_agent

    # Determine which conditions to evaluate
    conditions = args.conditions if args.conditions else list(CONDITIONS.keys())

    # Determine eval sets
    if args.eval_sets:
        eval_set_paths = args.eval_sets
    else:
        # Default: all NPZ files in eval_levels/
        eval_dir = args.eval_levels_dir
        if os.path.exists(eval_dir):
            eval_set_paths = sorted([
                os.path.join(eval_dir, f) for f in os.listdir(eval_dir) if f.endswith(".npz")
            ])
        else:
            print(f"No eval sets found in {eval_dir}. Run 'generate' first.")
            return

    if not eval_set_paths:
        print("No eval sets specified. Use --eval_sets or run 'generate' first.")
        return

    eval_steps_arg = args.eval_steps if args.eval_steps else ["latest"]

    all_results = []
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    for eval_set_path in eval_set_paths:
        eval_name = os.path.basename(eval_set_path).replace(".npz", "")
        print(f"\n{'='*70}")
        print(f"Eval Set: {eval_name}")
        print(f"{'='*70}")

        levels, n_levels, maze_size, level_names = load_eval_set(eval_set_path)
        print(f"  {n_levels} levels, {maze_size}x{maze_size}")

        for cond_name in conditions:
            if cond_name not in CONDITIONS:
                print(f"  [WARN] Unknown condition: {cond_name}")
                continue

            cond = CONDITIONS[cond_name]

            for seed in cond["seeds"]:
                checkpoint_dir = os.path.abspath(os.path.join(cond["results_dir"], "checkpoints", str(seed)))
                if not os.path.exists(checkpoint_dir):
                    print(f"  [SKIP] No checkpoint dir: {checkpoint_dir}")
                    continue

                # Load config to get eval_freq
                config_path = os.path.join(checkpoint_dir, "config.json")
                if not os.path.exists(config_path):
                    print(f"  [SKIP] No config.json: {config_path}")
                    continue
                with open(config_path) as f:
                    config = json.load(f)
                eval_freq = config.get("eval_freq", 250)

                # Resolve checkpoint steps
                steps = resolve_checkpoint_steps(checkpoint_dir, eval_steps_arg, eval_freq)
                if not steps:
                    print(f"  [SKIP] No valid checkpoints for {cond_name}/seed{seed}")
                    continue

                for orbax_step, training_updates in steps:
                    t0 = time.time()
                    print(f"\n  {cond_name}/seed{seed} @ {training_updates} updates (orbax step {orbax_step})")

                    train_state, _, _, _ = load_agent(checkpoint_dir, orbax_step)
                    if train_state is None:
                        print(f"    SKIP: checkpoint load failed")
                        continue

                    metrics = evaluate_agent_on_levels(
                        train_state, None, levels, maze_size,
                        args.num_attempts,
                    )

                    solve_rates = metrics["solve_rates"]
                    elapsed = time.time() - t0
                    mean_sr = metrics["mean_solve_rate"]
                    print(
                        f"    SR={mean_sr:.1%}  IQM={metrics['iqm_solve_rate']:.1%}  "
                        f"Return={metrics['mean_return_agg']:.3f}  "
                        f"OptGap={metrics['mean_optimality_gap']:.3f}  "
                        f"PathRatio={metrics['mean_path_ratio']:.2f}  "
                        f"elapsed={elapsed:.1f}s"
                    )

                    def _nan2none(xs):
                        return [None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
                                for v in xs]

                    result = {
                        "eval_set": eval_name,
                        "maze_size": maze_size,
                        "condition": cond_name,
                        "seed": seed,
                        "training_updates": training_updates,
                        "orbax_step": orbax_step,
                        "mean_solve_rate": mean_sr,
                        "median_solve_rate": float(np.median(solve_rates)),
                        "iqm_solve_rate": metrics["iqm_solve_rate"],
                        "mean_return": metrics["mean_return_agg"],
                        "mean_optimality_gap": metrics["mean_optimality_gap"],
                        "mean_path_ratio": metrics["mean_path_ratio"],
                        "unsolved_count": int((solve_rates == 0).sum()),
                        "fully_solved_count": int((solve_rates == 1.0).sum()),
                        "n_levels": n_levels,
                        "num_attempts": args.num_attempts,
                        "elapsed_s": elapsed,
                        "per_level_solve_rates": [float(x) for x in solve_rates],
                        "per_level_mean_returns": [float(x) for x in metrics["mean_returns"]],
                        "per_level_optimal_returns": [float(x) for x in metrics["optimal_returns"]],
                        "per_level_optimality_gaps": [float(x) for x in metrics["optimality_gaps"]],
                        "per_level_shortest_paths": _nan2none(metrics["shortest_paths"]),
                        "per_level_ep_length_means": _nan2none(metrics["ep_length_means"]),
                        "per_level_path_ratio_means": _nan2none(metrics["path_ratio_means"]),
                        "level_names": level_names,
                    }
                    all_results.append(result)

                    # Clear JAX caches between evaluations to avoid OOM
                    jax.clear_caches()

    if not all_results:
        print("\nNo results collected.")
        return

    # Print summary tables
    _print_summary(all_results)

    # Save CSV
    csv_path = os.path.join(output_dir, "ood_eval_summary.csv")
    fieldnames = [
        "eval_set", "maze_size", "condition", "seed", "training_updates",
        "mean_solve_rate", "iqm_solve_rate", "median_solve_rate",
        "mean_return", "mean_optimality_gap", "mean_path_ratio",
        "unsolved_count", "fully_solved_count",
        "n_levels", "num_attempts", "elapsed_s",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"\nCSV saved: {csv_path}")

    # Save JSON
    json_path = os.path.join(output_dir, "ood_eval_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"JSON saved: {json_path}")


def _print_summary(results):
    """Print formatted summary tables grouped by eval set."""
    # Group by eval_set
    by_eval_set = defaultdict(list)
    for r in results:
        by_eval_set[r["eval_set"]].append(r)

    for eval_set, rs in by_eval_set.items():
        # Get unique training steps
        steps = sorted(set(r["training_updates"] for r in rs))
        step_labels = [f"{s//1000}k" if s >= 1000 else str(s) for s in steps]

        maze_size = rs[0]["maze_size"]
        n_levels = rs[0]["n_levels"]
        n_attempts = rs[0]["num_attempts"]

        print(f"\n{'='*70}")
        print(f"OOD Evaluation — {eval_set} ({maze_size}x{maze_size}, {n_levels} levels, {n_attempts} attempts)")
        print(f"{'='*70}")

        # Header
        step_header = "".join(f"{sl:>10s}" for sl in step_labels)
        print(f"{'Condition':>30s} {'Seed':>6s}{step_header}")
        print("-" * (36 + 10 * len(steps)))

        # Group by condition
        by_cond = defaultdict(list)
        for r in rs:
            by_cond[r["condition"]].append(r)

        for cond_name in sorted(by_cond.keys()):
            cond_rs = by_cond[cond_name]
            # Group by seed
            by_seed = defaultdict(dict)
            for r in cond_rs:
                by_seed[r["seed"]][r["training_updates"]] = r

            for seed in sorted(by_seed.keys()):
                seed_data = by_seed[seed]
                vals = []
                for step in steps:
                    if step in seed_data:
                        vals.append(f"{seed_data[step]['mean_solve_rate']:>9.1%}")
                    else:
                        vals.append(f"{'—':>9s}")
                print(f"{cond_name:>30s} {seed:>6d}{''.join(vals)}")

            # Condition mean
            if len(by_seed) > 1:
                vals = []
                for step in steps:
                    step_rates = [
                        by_seed[s][step]["mean_solve_rate"]
                        for s in by_seed if step in by_seed[s]
                    ]
                    if step_rates:
                        vals.append(f"{np.mean(step_rates):>9.1%}")
                    else:
                        vals.append(f"{'—':>9s}")
                print(f"{'(mean)':>30s} {'':>6s}{''.join(vals)}")

        print()


# ===================================================================
#  MAIN
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="OOD evaluation of trained maze agents"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Generate subcommand
    gen_parser = subparsers.add_parser("generate", help="Generate OOD eval maze sets")
    gen_parser.add_argument("--sizes", nargs="+", type=int, default=[13, 15, 17],
                            help="Maze sizes to generate (default: 13 15 17)")
    gen_parser.add_argument("--n_mazes", type=int, default=100,
                            help="Number of mazes per set (default: 100)")
    gen_parser.add_argument("--types", nargs="+", default=["random", "perfect"],
                            choices=["random", "perfect"],
                            help="Types of mazes to generate (default: random perfect)")
    gen_parser.add_argument("--n_walls", type=int, default=25,
                            help="Number of walls for random mazes (default: 25)")
    gen_parser.add_argument("--seed", type=int, default=42,
                            help="Random seed (default: 42)")
    gen_parser.add_argument("--output_dir", type=str, default="eval_levels",
                            help="Output directory (default: eval_levels)")
    gen_parser.add_argument("--force", action="store_true",
                            help="Overwrite existing files")

    # Evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate agents on OOD mazes")
    eval_parser.add_argument("--eval_sets", nargs="+", type=str, default=None,
                             help="Paths to eval set NPZ files (default: all in eval_levels/)")
    eval_parser.add_argument("--eval_levels_dir", type=str, default="eval_levels",
                             help="Directory containing eval set NPZ files (default: eval_levels)")
    eval_parser.add_argument("--eval_steps", nargs="+", type=str, default=None,
                             help="Training update steps to evaluate: numbers, 'all', or 'latest' (default: latest)")
    eval_parser.add_argument("--conditions", nargs="+", type=str, default=None,
                             help="Conditions to evaluate (default: all in registry)")
    eval_parser.add_argument("--num_attempts", type=int, default=10,
                             help="Rollout attempts per level (default: 10)")
    eval_parser.add_argument("--output_dir", type=str, default="results/ood_eval",
                             help="Output directory for results (default: results/ood_eval)")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
