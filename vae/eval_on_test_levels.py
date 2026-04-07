"""Evaluate agents from multiple methods/seeds on held-out test levels.

Downloads agent checkpoints from GCS at a specified update step, evaluates
on prefab test levels, and produces a summary CSV + bar chart.

Uses the same YAML config format as analyze_buffer_stats.py.

Usage:
    # 21x21 methods on 21x21 test levels, final checkpoint (30k updates)
    python vae/eval_on_test_levels.py \
        --config vae/buffer_stats_21x21_all.yaml \
        --eval_updates 30000 \
        --num_attempts 100 \
        --eval_levels PerfectMaze21_1 PerfectMaze21_2 PerfectMaze21_3 PerfectMaze21_4 \
                      Rooms21_1 Rooms21_2 Labyrinth21_1 Labyrinth21_2

    # 13x13 methods on standard 13x13 test levels
    python vae/eval_on_test_levels.py \
        --config vae/buffer_stats_13x13_all.yaml \
        --eval_updates 30000 \
        --num_attempts 100
"""
import argparse
import csv
import json
import os
import re
import sys
import tempfile
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

GCS_BUCKET = "ucl-ued-project-bucket"
GCS_PROJECT = "open-endedness-ued-project"

_gcs_client = None
_gcs_bucket_obj = None


def _get_bucket():
    global _gcs_client, _gcs_bucket_obj
    if _gcs_bucket_obj is None:
        from google.cloud import storage
        _gcs_client = storage.Client(project=GCS_PROJECT)
        _gcs_bucket_obj = _gcs_client.bucket(GCS_BUCKET)
    return _gcs_bucket_obj


def gcs_download(gcs_path, local_path):
    if os.path.exists(local_path):
        return True
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        blob = _get_bucket().blob(gcs_path)
        if not blob.exists():
            return False
        blob.download_to_filename(local_path)
        return True
    except Exception as e:
        print(f"  [GCS] Failed: {gcs_path}: {e}")
        return False


def gcs_download_dir(gcs_prefix, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    try:
        bucket = _get_bucket()
        blobs = list(bucket.list_blobs(prefix=gcs_prefix + "/"))
        if not blobs:
            return False
        for blob in blobs:
            rel_path = blob.name[len(gcs_prefix):].lstrip("/")
            if not rel_path:
                continue
            local_file = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(local_file), exist_ok=True)
            blob.download_to_filename(local_file)
        return True
    except Exception as e:
        print(f"  [GCS] Failed dir: {gcs_prefix}: {e}")
        return False


def find_checkpoint_on_gcs(run_cfg, eval_step):
    """Find config.json + models/{step} on GCS. Returns (config_gcs_dir, models_gcs_prefix) or None."""
    seed = run_cfg.get("seed", 0)
    run_id = run_cfg.get("run_id", run_cfg["name"])
    gcs_prefix = run_cfg.get("gcs_prefix")
    if not gcs_prefix:
        return None

    bucket = _get_bucket()

    # Try several layouts
    for layout in [f"{gcs_prefix}/checkpoints/{run_id}/{seed}",
                   f"{gcs_prefix}/checkpoints/{seed}",
                   f"{gcs_prefix}/checkpoints"]:
        config_gcs = f"{layout}/config.json"
        if bucket.blob(config_gcs).exists():
            models_gcs = f"{layout}/models/{eval_step}"
            # Check model dir exists (probe for any blob under it)
            probes = list(bucket.list_blobs(prefix=models_gcs + "/", max_results=1))
            if probes:
                return layout, models_gcs
    return None


def download_checkpoint(run_cfg, eval_step, local_dir):
    """Download agent checkpoint to local_dir. Returns local checkpoint dir or None."""
    result = find_checkpoint_on_gcs(run_cfg, eval_step)
    if result is None:
        return None

    config_gcs_dir, models_gcs = result
    ckpt_dir = local_dir

    # Download config.json
    if not gcs_download(f"{config_gcs_dir}/config.json", os.path.join(ckpt_dir, "config.json")):
        return None

    # Download models/{step}/
    models_local = os.path.join(ckpt_dir, "models", str(eval_step))
    if not gcs_download_dir(models_gcs, models_local):
        return None

    return ckpt_dir


def load_agent_for_eval(checkpoint_dir, checkpoint_step, maze_height, maze_width):
    """Load agent from checkpoint, using correct grid size."""
    import jax
    import jax.numpy as jnp
    import optax
    import orbax.checkpoint as ocp

    from jaxued.environments import Maze
    from jaxued.environments.maze import Level, make_level_generator
    from jaxued.wrappers import AutoReplayWrapper
    from jaxued.level_sampler import LevelSampler
    from maze_plr import ActorCritic, TrainState, evaluate_rnn

    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    mh = config.get("maze_height", maze_height)
    mw = config.get("maze_width", maze_width)
    avs = config.get("agent_view_size", 5)
    n_walls = config.get("n_walls", 25)
    n_envs = config.get("num_train_envs", 32)

    base_env = Maze(max_height=mh, max_width=mw, agent_view_size=avs, normalize_obs=True)
    env = AutoReplayWrapper(base_env)
    env_params = env.default_params
    sample_random_level = make_level_generator(mh, mw, n_walls)

    level_sampler = LevelSampler(
        capacity=config.get("level_buffer_capacity", 4000),
        replay_prob=config.get("replay_prob", 0.8),
        staleness_coeff=config.get("staleness_coeff", 0.3),
        minimum_fill_ratio=config.get("minimum_fill_ratio", 0.5),
        prioritization=config.get("prioritization", "rank"),
        prioritization_params={"temperature": config.get("temperature", 0.3),
                               "k": config.get("topk_k", 4)},
        duplicate_check=config.get("buffer_duplicate_check", True),
    )

    rng = jax.random.PRNGKey(0)
    obs, _ = env.reset_to_level(rng, sample_random_level(rng), env_params)
    obs_batch = jax.tree_util.tree_map(
        lambda x: jnp.repeat(jnp.repeat(x[None, ...], n_envs, axis=0)[None, ...], 256, axis=0), obs)
    init_x = (obs_batch, jnp.zeros((256, n_envs)))
    network = ActorCritic(env.action_space(env_params).n)
    network_params = network.init(rng, init_x, ActorCritic.initialize_carry((n_envs,)))

    tx = optax.chain(
        optax.clip_by_global_norm(config.get("max_grad_norm", 0.5)),
        optax.adam(learning_rate=config.get("lr", 1e-4), eps=1e-5),
    )

    pholder_level = sample_random_level(jax.random.PRNGKey(0))
    sampler = level_sampler.initialize(pholder_level, {"max_return": -jnp.inf})
    pholder_level_batch = jax.tree_util.tree_map(
        lambda x: jnp.array([x]).repeat(n_envs, axis=0), pholder_level)

    template_state = TrainState.create(
        apply_fn=network.apply, params=network_params, tx=tx,
        sampler=sampler, update_state=0, es_state=None,
        num_dr_updates=0, num_replay_updates=0, num_mutation_updates=0,
        dr_last_level_batch=pholder_level_batch,
        replay_last_level_batch=pholder_level_batch,
        replay_last_level_inds=jnp.full(n_envs, -1, dtype=jnp.int32),
        mutation_last_level_batch=pholder_level_batch,
    )

    models_dir = os.path.join(checkpoint_dir, "models")
    checkpoint_manager = ocp.CheckpointManager(
        models_dir, item_handlers=ocp.StandardCheckpointHandler())

    available = sorted(checkpoint_manager.all_steps())
    if checkpoint_step in available:
        step = checkpoint_step
    elif available:
        step = min(available, key=lambda s: abs(s - checkpoint_step))
        print(f"  [Agent] Step {checkpoint_step} not found, using {step}")
    else:
        print(f"  [Agent] No checkpoints in {models_dir}")
        return None, config, None, None

    try:
        restored = checkpoint_manager.restore(step)
    except Exception as e:
        print(f"  [Agent] Restore failed: {e}")
        return None, config, None, None

    train_state = template_state.replace(params=restored["params"])
    return train_state, config, base_env, env_params


def evaluate_on_prefab_levels(train_state, config, base_env, env_params,
                              level_names, num_attempts, max_steps_override=None):
    """Evaluate agent on named prefab levels. Returns dict of {level_name: solve_rate}."""
    import jax
    import jax.numpy as jnp
    from jaxued.environments.maze.level import Level, prefabs
    from maze_plr import ActorCritic, evaluate_rnn

    mh = config.get("maze_height", 13)
    mw = config.get("maze_width", 13)

    # Load and stack prefab levels (padding to max grid size)
    level_objects = []
    valid_names = []
    for name in level_names:
        if name not in prefabs:
            print(f"  WARNING: prefab '{name}' not found, skipping")
            continue
        lvl = Level.from_str(prefabs[name])
        lh, lw = lvl.wall_map.shape
        if lh > mh or lw > mw:
            lvl = lvl.pad_to_shape(max(lw, mw), max(lh, mh))
        else:
            lvl = lvl.pad_to_shape(mw, mh)
        level_objects.append(lvl)
        valid_names.append(name)

    if not level_objects:
        return {}

    levels = Level.stack(level_objects)
    num_levels = len(valid_names)
    # Build eval env matching agent's grid size (padded levels fit into it)
    eval_h = max(mh, levels.wall_map.shape[-2])
    eval_w = max(mw, levels.wall_map.shape[-1])
    from jaxued.environments import Maze
    from jaxued.environments.maze.env import EnvParams
    eval_env = Maze(max_height=eval_h, max_width=eval_w,
                    agent_view_size=config.get("agent_view_size", 5),
                    normalize_obs=True)
    if max_steps_override is not None:
        eval_params = EnvParams(max_steps_in_episode=max_steps_override)
    else:
        eval_params = eval_env.default_params
    max_steps = eval_params.max_steps_in_episode

    all_attempt_solves = []
    for attempt_idx in range(num_attempts):
        rng = jax.random.PRNGKey(attempt_idx + 2000)
        rng, rng_reset, rng_eval = jax.random.split(rng, 3)

        init_obs, init_state = jax.vmap(eval_env.reset_to_level, (0, 0, None))(
            jax.random.split(rng_reset, num_levels), levels, eval_params)

        states, rewards, episode_lengths = evaluate_rnn(
            rng_eval, eval_env, eval_params, train_state,
            ActorCritic.initialize_carry((num_levels,)),
            init_obs, init_state, max_steps)

        mask = np.arange(max_steps)[:, None] < np.asarray(episode_lengths)[None, :]
        cum_rewards = (np.asarray(rewards) * mask).sum(axis=0)
        solved = (cum_rewards > 0).astype(float)
        all_attempt_solves.append(solved)

    solve_rates = np.stack(all_attempt_solves).mean(axis=0)
    return {name: float(sr) for name, sr in zip(valid_names, solve_rates)}


def method_name_from_run(run_name):
    return re.sub(r'\s+s\d+$', '', run_name)


def main():
    import yaml
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, required=True,
                        help="YAML config (same format as analyze_buffer_stats.py)")
    parser.add_argument("--eval_updates", type=int, default=30000,
                        help="PPO update step to load agent from (default: 30000)")
    parser.add_argument("--num_attempts", type=int, default=100,
                        help="Rollouts per level per agent (averaged for solve rate)")
    parser.add_argument("--eval_levels", nargs="+", default=None,
                        help="Prefab level names. Default: standard 13x13 or 21x21 set based on grid_size")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Max episode steps for eval (default: use env default, typically 250)")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    runs = cfg["runs"]
    grid_size = cfg.get("grid_size", 13)

    # Default eval levels based on grid size
    if args.eval_levels is None:
        if grid_size <= 13:
            args.eval_levels = [
                "SixteenRooms", "SixteenRooms2", "Labyrinth", "LabyrinthFlipped",
                "Labyrinth2", "StandardMaze", "StandardMaze2", "StandardMaze3",
            ]
        else:
            args.eval_levels = [
                "PerfectMaze21_1", "PerfectMaze21_2", "PerfectMaze21_3", "PerfectMaze21_4",
                "Rooms21_1", "Rooms21_2", "Labyrinth21_1", "Labyrinth21_2",
            ]

    config_stem = os.path.splitext(os.path.basename(args.config))[0]
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.config),
                                       "eval_results", config_stem)
    os.makedirs(args.output_dir, exist_ok=True)

    # Convert update count to eval_step (orbax step)
    eval_freq = 250  # standard
    eval_step = (args.eval_updates // eval_freq) - 1
    print(f"=== Eval on test levels ===")
    print(f"  Config: {args.config}")
    print(f"  Grid: {grid_size}x{grid_size}")
    print(f"  Target: {args.eval_updates} updates -> eval_step {eval_step}")
    print(f"  Levels: {args.eval_levels}")
    print(f"  Attempts: {args.num_attempts}")
    print(f"  Output: {args.output_dir}")
    print()

    # --- Evaluate each run ---
    all_rows = []
    for ri, run in enumerate(runs):
        run_name = run["name"]
        run_id = run.get("run_id", run_name)
        seed = run.get("seed", 0)

        print(f"--- {run_name} (run_id={run_id}, seed={seed}) ---")

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_dir = download_checkpoint(run, eval_step, tmp_dir)
            if ckpt_dir is None:
                print(f"  SKIP: checkpoint not found on GCS")
                continue

            train_state, config, base_env, env_params = load_agent_for_eval(
                ckpt_dir, eval_step, grid_size, grid_size)
            if train_state is None:
                print(f"  SKIP: agent load failed")
                continue

            results = evaluate_on_prefab_levels(
                train_state, config, base_env, env_params,
                args.eval_levels, args.num_attempts,
                max_steps_override=args.max_steps)

            row = {
                "run_name": run_name,
                "method": method_name_from_run(run_name),
                "run_id": run_id,
                "seed": seed,
                "eval_updates": args.eval_updates,
                "num_attempts": args.num_attempts,
                "mean_solve_rate": float(np.mean(list(results.values()))) if results else 0.0,
            }
            row.update(results)
            all_rows.append(row)

            sr_str = "  ".join(f"{n}={results[n]:.0%}" for n in args.eval_levels if n in results)
            print(f"  Mean: {row['mean_solve_rate']:.1%} | {sr_str}")

        print()

    if not all_rows:
        print("No results collected.")
        return

    # --- Save CSV ---
    csv_path = os.path.join(args.output_dir, "eval_test_levels.csv")
    fieldnames = list(all_rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved CSV: {csv_path}")

    # --- Aggregate by method ---
    methods = list(dict.fromkeys(r["method"] for r in all_rows))
    level_names = [l for l in args.eval_levels if l in all_rows[0]]

    print(f"\n{'='*70}")
    print("SUMMARY (mean ± std across seeds)")
    print(f"{'='*70}")
    header = f"{'Method':>25s} {'Mean':>8s}"
    for ln in level_names:
        short = ln[:12]
        header += f" {short:>12s}"
    print(header)
    print("-" * len(header))

    method_means = {}
    for m in methods:
        m_rows = [r for r in all_rows if r["method"] == m]
        mean_srs = [r["mean_solve_rate"] for r in m_rows]
        method_means[m] = np.mean(mean_srs)

        line = f"{m:>25s} {np.mean(mean_srs):>7.1%}"
        for ln in level_names:
            vals = [r[ln] for r in m_rows if ln in r]
            if vals:
                line += f" {np.mean(vals):>11.1%}"
            else:
                line += f" {'N/A':>11s}"
        print(line)

    # --- Bar chart ---
    fig, ax = plt.subplots(figsize=(max(10, len(methods) * 2), 6))

    x = np.arange(len(methods))
    width = 0.8 / max(len(level_names), 1)

    for li, ln in enumerate(level_names):
        means, stds = [], []
        for m in methods:
            m_rows = [r for r in all_rows if r["method"] == m]
            vals = [r[ln] for r in m_rows if ln in r]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if len(vals) > 1 else 0)
        offset = (li - len(level_names) / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=stds, label=ln[:15],
               alpha=0.8, capsize=3)

    ax.set_xlabel("Method")
    ax.set_ylabel("Solve Rate")
    ax.set_title(f"Eval on Held-Out Test Levels ({grid_size}x{grid_size}, {args.eval_updates} updates, {args.num_attempts} attempts)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=7, ncol=min(4, len(level_names)), loc="upper right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(args.output_dir, "eval_test_levels.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved plot: {plot_path}")

    # --- Also save a simple method-level summary ---
    summary_path = os.path.join(args.output_dir, "eval_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "n_seeds", "mean_solve_rate", "std_solve_rate"] + level_names)
        for m in methods:
            m_rows = [r for r in all_rows if r["method"] == m]
            mean_srs = [r["mean_solve_rate"] for r in m_rows]
            per_level = []
            for ln in level_names:
                vals = [r[ln] for r in m_rows if ln in r]
                per_level.append(f"{np.mean(vals):.4f}" if vals else "")
            writer.writerow([m, len(m_rows), f"{np.mean(mean_srs):.4f}",
                             f"{np.std(mean_srs):.4f}"] + per_level)
    print(f"Saved summary: {summary_path}")
    print(f"\nDone. {len(all_rows)} agent evaluations, {len(methods)} methods.")


if __name__ == "__main__":
    main()
