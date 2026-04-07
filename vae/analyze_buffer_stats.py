"""Analyze buffer statistics across training checkpoints.

Computes per-checkpoint stats from buffer dumps:
  - Average wall count (all levels)
  - Average wall count (top-K by SFL score)
  - Score distribution stats
  - Provenance breakdown (if origins available)
  - Average agent path length (optional, requires GPU + agent checkpoint)

Outputs a CSV + wandb line plots showing how these evolve over training.

Works with any run on GCS — specify via YAML config (same format as
plot_tsne_compare_runs.py).

Usage:
    # Buffer stats only (CPU, no agent rollout)
    python vae/analyze_buffer_stats.py --config vae/compare_accel_v2.yaml

    # Include agent path lengths (GPU)
    python vae/analyze_buffer_stats.py --config vae/compare_accel_v2.yaml --compute_paths

    # Custom top-K
    python vae/analyze_buffer_stats.py --config vae/compare_accel_v2.yaml --top_k 50
"""
import argparse
import csv
import os
import sys
import numpy as np
import yaml

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
    if os.path.exists(local_dir) and len(os.listdir(local_dir)) > 0:
        return True
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
        print(f"  [GCS] Failed: {gcs_prefix}: {e}")
        return False


def load_buffer_dump(path):
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    size = int(d["size"]) if "size" in d else len(d["tokens"])
    result = {
        "tokens": d["tokens"][:size],
        "scores": d["scores"][:size],
        "size": size,
    }
    if "origins" in d:
        result["origins"] = d["origins"][:size]
    return result


def compute_buffer_stats(buf, top_k=100, grid_size=13):
    """Compute stats from a buffer dump (no agent needed)."""
    tokens = buf["tokens"]
    scores = buf["scores"]
    size = buf["size"]
    max_walls = tokens.shape[1] - 2

    # Wall counts
    walls = (tokens[:, :max_walls] > 0).sum(axis=1).astype(float)

    # Top-K by SFL score
    k = min(top_k, size)
    top_idx = np.argsort(-scores)[:k]
    top_walls = walls[top_idx]
    top_scores = scores[top_idx]

    stats = {
        "size": size,
        "walls_mean": float(walls.mean()),
        "walls_std": float(walls.std()),
        "walls_min": float(walls.min()),
        "walls_max": float(walls.max()),
        "top_k_walls_mean": float(top_walls.mean()),
        "top_k_walls_std": float(top_walls.std()),
        "top_k_score_mean": float(top_scores.mean()),
        "top_k_score_min": float(top_scores.min()),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "score_max": float(scores.max()),
        "n_zero_score": int((scores == 0).sum()),
        "n_max_learnable": int((scores > 0.24).sum()),  # SFL > 0.24 ≈ p near 0.5
    }

    # Provenance breakdown
    if "origins" in buf:
        origins = buf["origins"]
        for code, label in [(0, "dr"), (1, "cmaes"), (2, "dr_mut"), (3, "cmaes_mut")]:
            stats[f"n_{label}"] = int((origins == code).sum())
            stats[f"pct_{label}"] = float((origins == code).sum() / size * 100)

    return stats


def compute_path_lengths(tokens, checkpoint_dir, ckpt_step, grid_size=13,
                         batch_size=256, num_rollouts=5):
    """Compute mean episode length by rolling out agent on buffer levels."""
    from cross_evaluate import load_agent, tokens_to_levels_batch
    from maze_plr import ActorCritic, evaluate_rnn
    from jaxued.environments import Maze
    from jaxued.wrappers import AutoReplayWrapper
    import jax
    import jax.numpy as jnp

    train_state, config, env, env_params = load_agent(checkpoint_dir, checkpoint_step=ckpt_step)
    if train_state is None:
        return None, None

    maze_h = config.get("maze_height", grid_size)
    maze_w = config.get("maze_width", grid_size)
    eval_env = Maze(max_height=maze_h, max_width=maze_w,
                    agent_view_size=config["agent_view_size"], normalize_obs=True)
    max_steps = env_params.max_steps_in_episode

    levels = tokens_to_levels_batch(tokens)
    n_levels = len(tokens)
    all_lengths = np.zeros(n_levels, dtype=np.float32)
    all_solved = np.zeros(n_levels, dtype=np.float32)

    for rollout_idx in range(num_rollouts):
        for start in range(0, n_levels, batch_size):
            end = min(start + batch_size, n_levels)
            chunk_levels = jax.tree_util.tree_map(lambda x: x[start:end], levels)
            n_chunk = end - start

            rng = jax.random.PRNGKey(rollout_idx * 1000 + start)
            rng, rng_reset, rng_eval = jax.random.split(rng, 3)

            init_obs, init_state = jax.vmap(
                eval_env.reset_to_level, (0, 0, None)
            )(jax.random.split(rng_reset, n_chunk), chunk_levels, env_params)

            states, rewards, ep_lengths = evaluate_rnn(
                rng_eval, eval_env, env_params, train_state,
                ActorCritic.initialize_carry((n_chunk,)),
                init_obs, init_state, max_steps,
            )
            all_lengths[start:end] += np.array(ep_lengths).astype(np.float32)
            all_solved[start:end] += np.array((rewards.sum(axis=0) > 0).astype(np.float32))

    all_lengths /= num_rollouts
    all_solved /= num_rollouts

    return all_lengths, all_solved


def find_gcs_buffer_path(run_cfg, timestep):
    """Find the GCS path for a buffer dump without downloading."""
    seed = run_cfg.get("seed", 0)
    run_id = run_cfg.get("run_id", run_cfg["name"])
    if "gcs_prefix" not in run_cfg:
        return None
    gcs_prefix = run_cfg["gcs_prefix"]
    candidates = [
        f"{gcs_prefix}/buffer_dumps/{run_id}/{seed}/buffer_dump_{timestep}.npz",
        f"{gcs_prefix}/buffer_dumps/buffer_dump_{timestep}.npz",
        f"{gcs_prefix}/{seed}/buffer_dump_{timestep}.npz",
    ]
    bucket = _get_bucket()
    for c in candidates:
        if bucket.blob(c).exists():
            return c
    return None


def download_buffer_to_tmp(gcs_path, tmp_dir):
    """Download a single buffer dump to a temp location, return local path."""
    local_path = os.path.join(tmp_dir, "buffer_dump.npz")
    try:
        _get_bucket().blob(gcs_path).download_to_filename(local_path)
        return local_path
    except Exception as e:
        print(f"  [GCS] Download failed: {e}")
        return None


def resolve_checkpoint_dir(run_cfg, ckpt_step, local_data_root):
    seed = run_cfg.get("seed", 0)
    run_id = run_cfg.get("run_id", run_cfg["name"])
    safe_id = f"{run_id}_s{seed}".replace(" ", "_").replace("/", "_")

    if "gcs_prefix" in run_cfg:
        gcs_prefix = run_cfg["gcs_prefix"]
        ckpt_dir = os.path.join(local_data_root, safe_id, "checkpoints")

        for layout in [f"{gcs_prefix}/checkpoints", gcs_prefix]:
            for config_gcs in [
                f"{layout}/{run_id}/{seed}/config.json",
                f"{layout}/{seed}/config.json",
                f"{layout}/config.json",
            ]:
                config_local = os.path.join(ckpt_dir, "config.json")
                if gcs_download(config_gcs, config_local):
                    config_gcs_dir = config_gcs.rsplit("/config.json", 1)[0]
                    models_gcs = f"{config_gcs_dir}/models/{ckpt_step}"
                    models_local = os.path.join(ckpt_dir, "models", str(ckpt_step))
                    gcs_download_dir(models_gcs, models_local)
                    if os.path.exists(os.path.join(ckpt_dir, "config.json")):
                        return ckpt_dir
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, required=True,
                        help="YAML config (same format as plot_tsne_compare_runs.py)")
    parser.add_argument("--top_k", type=int, default=100,
                        help="Top-K levels by SFL for wall count analysis")
    parser.add_argument("--compute_paths", action="store_true",
                        help="Compute agent path lengths (requires GPU + checkpoints)")
    parser.add_argument("--num_rollouts", type=int, default=5,
                        help="Rollouts per level for path length computation")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--local_data_root", type=str,
                        default="/cs/student/project_msc/2025/csml/rhautier/run_comparison_data")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    runs = cfg["runs"]
    timesteps = cfg.get("timesteps", [250, 1000, 5000, 10000, 20000, 30000])
    title = cfg.get("title", "Buffer Stats")
    grid_size = cfg.get("grid_size", 13)

    # Output dir defaults to buffer_stats/<config_stem> to avoid overwriting
    config_stem = os.path.splitext(os.path.basename(args.config))[0]
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.config), "buffer_stats", config_stem)
    os.makedirs(args.output_dir, exist_ok=True)

    import re
    import tempfile

    # --- Step 1: Verify GCS availability, filter incomplete runs ---
    print("=== Checking buffer availability on GCS ===")
    run_gcs_paths = {}  # (run_idx, timestep) -> gcs_path
    run_available = {}  # run_idx -> set of available timesteps

    for ri, run in enumerate(runs):
        run_name = run["name"]
        available = set()
        # Check first and last timestep to detect missing runs fast
        for ts in [timesteps[0], timesteps[-1]]:
            gcs_path = find_gcs_buffer_path(run, ts)
            if gcs_path:
                available.add(ts)
                run_gcs_paths[(ri, ts)] = gcs_path

        if len(available) < 2:
            print(f"  {run_name}: MISSING (skipping entirely)")
            continue

        # Check all timesteps
        for ts in timesteps:
            if ts in available:
                continue
            gcs_path = find_gcs_buffer_path(run, ts)
            if gcs_path:
                available.add(ts)
                run_gcs_paths[(ri, ts)] = gcs_path

        run_available[ri] = available
        n_missing = len(timesteps) - len(available)
        status = "COMPLETE" if n_missing == 0 else f"{n_missing} MISSING"
        print(f"  {run_name}: {len(available)}/{len(timesteps)} timesteps ({status})")

    # Filter: only keep methods where ALL seeds have ALL timesteps
    def method_name_from_run(run_name):
        return re.sub(r'\s+s\d+$', '', run_name)

    method_runs = {}  # method -> [run_indices]
    for ri, run in enumerate(runs):
        if ri not in run_available:
            continue
        m = method_name_from_run(run["name"])
        method_runs.setdefault(m, []).append(ri)

    valid_runs = set()
    print("\n=== Method completeness ===")
    for m, indices in method_runs.items():
        all_complete = all(len(run_available.get(ri, set())) == len(timesteps) for ri in indices)
        if all_complete:
            valid_runs.update(indices)
            print(f"  {m}: {len(indices)} seeds, ALL COMPLETE")
        else:
            for ri in indices:
                n = len(run_available.get(ri, set()))
                print(f"  {m} (run {ri}): {n}/{len(timesteps)} — {'OK' if n == len(timesteps) else 'INCOMPLETE'}")
            # Include if at least some data
            for ri in indices:
                if len(run_available.get(ri, set())) > len(timesteps) * 0.8:
                    valid_runs.add(ri)

    # --- Step 2: Download, compute stats, delete — one buffer at a time ---
    print(f"\n=== Computing stats ({len(valid_runs)} runs x {len(timesteps)} timesteps) ===")
    all_rows = []

    for ri in sorted(valid_runs):
        run = runs[ri]
        run_name = run["name"]
        run_id = run.get("run_id", run_name)
        seed = run.get("seed", 0)

        for ts in timesteps:
            key = (ri, ts)
            if key not in run_gcs_paths:
                continue

            with tempfile.TemporaryDirectory() as tmp_dir:
                local_path = download_buffer_to_tmp(run_gcs_paths[key], tmp_dir)
                if not local_path:
                    continue

                buf = load_buffer_dump(local_path)
                if buf is None:
                    continue

                stats = compute_buffer_stats(buf, top_k=args.top_k, grid_size=grid_size)
                stats["run_name"] = run_name
                stats["run_id"] = run_id
                stats["seed"] = seed
                stats["update"] = ts
                all_rows.append(stats)

        print(f"  {run_name}: {sum(1 for r in all_rows if r['run_name'] == run_name)} timesteps collected")

    if not all_rows:
        print("No data collected.")
        return

    # Save CSV
    csv_path = os.path.join(args.output_dir, "buffer_stats.csv")
    fieldnames = list(all_rows[0].keys())
    # Ensure all rows have all keys
    for row in all_rows:
        for k in fieldnames:
            row.setdefault(k, "")
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved CSV: {csv_path}")

    # --- Aggregate by method (strip seed suffix from name) ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def method_name(run_name):
        return method_name_from_run(run_name)

    methods = list(dict.fromkeys(method_name(r["run_name"]) for r in all_rows))
    colors = {m: f"C{i}" for i, m in enumerate(methods)}

    metric_configs = [
        ("walls_mean", "Avg Walls (all buffer)", "Wall count"),
        ("top_k_walls_mean", f"Avg Walls (top-{args.top_k} by SFL)", "Wall count"),
        ("score_mean", "Mean SFL Score", "SFL = p(1-p)"),
        ("n_max_learnable", "Levels with SFL > 0.24", "Count"),
    ]
    if args.compute_paths:
        metric_configs += [
            ("path_length_mean", "Avg Path Length (all buffer)", "Steps"),
            ("top_k_path_length_mean", f"Avg Path Length (top-{args.top_k})", "Steps"),
            ("solve_rate_buffer", "Buffer Solve Rate", "Solve rate"),
        ]

    # Provenance plots
    prov_keys = [k for k in all_rows[0] if k.startswith("pct_")]
    if prov_keys:
        metric_configs.append(("__provenance__", "Provenance Breakdown", "% of buffer"))

    n_plots = len(metric_configs)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots), squeeze=False)

    for plot_idx, (metric, plot_title, ylabel) in enumerate(metric_configs):
        ax = axes[plot_idx][0]

        if metric == "__provenance__":
            # Stacked area per method (averaged over seeds)
            for mi, m in enumerate(methods):
                m_rows = [r for r in all_rows if method_name(r["run_name"]) == m]
                ts_set = sorted(set(r["update"] for r in m_rows))
                for pk in prov_keys:
                    label_name = pk.replace("pct_", "").upper()
                    means = []
                    for t in ts_set:
                        vals = [r[pk] for r in m_rows if r["update"] == t and pk in r and r[pk] != ""]
                        means.append(np.mean(vals) if vals else 0)
                    ax.plot(ts_set, means, label=f"{m} — {label_name}", alpha=0.7)
            ax.set_title(plot_title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Updates")
            ax.legend(fontsize=6, ncol=2)
            ax.grid(alpha=0.3)
            continue

        for m in methods:
            m_rows = [r for r in all_rows if method_name(r["run_name"]) == m]
            ts_set = sorted(set(r["update"] for r in m_rows))

            means, stds = [], []
            for t in ts_set:
                vals = [r[metric] for r in m_rows if r["update"] == t
                        and metric in r and r[metric] != ""]
                if vals:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
                else:
                    means.append(np.nan)
                    stds.append(0)

            means = np.array(means)
            stds = np.array(stds)
            ax.plot(ts_set, means, label=m, color=colors[m], linewidth=2)
            ax.fill_between(ts_set, means - stds, means + stds,
                            color=colors[m], alpha=0.15)

        ax.set_title(plot_title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Updates")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()

    plot_path = os.path.join(args.output_dir, "buffer_stats.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {plot_path}")

    print(f"\nDone. {len(all_rows)} data points, {len(methods)} methods, {len(timesteps)} timesteps.")


if __name__ == "__main__":
    main()
