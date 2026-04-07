"""Compare buffer embedding evolution across different training runs via t-SNE.

Works with any checkpoint + buffer dump — no assumptions about LLM injection
or provenance tracking. Can compare ACCEL vs CMA-ES vs mixed strategies etc.

Grid: rows = runs (methods), columns = training timesteps.
Each cell shows t-SNE of the buffer levels using fresh 257D embeddings from
the current agent at that timestep.

If the buffer dump contains an `origins` field, levels are colored by
provenance. Otherwise all levels are shown in a single color per run.

Runs are specified via a YAML config file:

    # runs.yaml
    runs:
      - name: "ACCEL baseline"
        color: "tab:blue"
        checkpoint_dir: /path/to/checkpoints/run_name/seed/
        buffer_dir: /path/to/buffer_dumps/
        # OR fetch from GCS:
        gcs_prefix: llm-exp/training/run_name
        seed: 0

    timesteps: [250, 1000, 3000, 5000, 7000, 10000]

Usage:
    python vae/plot_tsne_compare_runs.py --config runs.yaml
    python vae/plot_tsne_compare_runs.py --config runs.yaml --cache_only
"""
import argparse
import os
import sys
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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


def updates_to_ckpt_step(updates, eval_freq=250):
    return (updates // eval_freq) - 1


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


def compute_embeddings(tokens, checkpoint_dir, ckpt_step, batch_size=256, num_rollouts=5):
    """Compute 257D behavioral embeddings and fresh SFL learnability scores.

    Returns (embeddings, sfl) where:
        embeddings: (N, 257) averaged over num_rollouts
        sfl: (N,) = p*(1-p) where p = solve rate over num_rollouts
    """
    from cross_evaluate import load_agent, tokens_to_levels_batch
    from maze_plr import ActorCritic, sample_trajectories_rnn, compute_insertion_embeddings
    from jaxued.environments import Maze
    from jaxued.wrappers import AutoReplayWrapper
    import jax
    import jax.numpy as jnp

    train_state, config, env, env_params = load_agent(checkpoint_dir, checkpoint_step=ckpt_step)
    if train_state is None:
        return None, None

    maze_h = config.get("maze_height", 13)
    maze_w = config.get("maze_width", 13)
    eval_env = Maze(max_height=maze_h, max_width=maze_w,
                    agent_view_size=config["agent_view_size"], normalize_obs=True)
    wrapped_env = AutoReplayWrapper(eval_env)
    max_steps = env_params.max_steps_in_episode

    levels = tokens_to_levels_batch(tokens)
    n_levels = len(tokens)
    all_embeddings = np.zeros((n_levels, 257), dtype=np.float32)
    all_solved = []  # list of (N,) bool arrays, one per rollout

    for rollout_idx in range(num_rollouts):
        rollout_embs = []
        rollout_solved = []
        for start in range(0, n_levels, batch_size):
            end = min(start + batch_size, n_levels)
            chunk_levels = jax.tree_util.tree_map(lambda x: x[start:end], levels)
            n_chunk = end - start

            rng = jax.random.PRNGKey(rollout_idx * 1000 + start)
            rng, rng_reset, rng_eval = jax.random.split(rng, 3)

            init_obs, init_state = jax.vmap(
                wrapped_env.reset_to_level, (0, 0, None)
            )(jax.random.split(rng_reset, n_chunk), chunk_levels, env_params)

            init_hstate = ActorCritic.initialize_carry((n_chunk,))

            (_, _, _, _, _, _), traj = sample_trajectories_rnn(
                rng_eval, wrapped_env, env_params, train_state,
                init_hstate, init_obs, init_state,
                n_chunk, max_steps,
            )
            _, actions, rewards, dones, _, _, _, _, hstates = traj
            embeddings = compute_insertion_embeddings(hstates, actions, dones)
            rollout_embs.append(np.array(embeddings))
            # Solved if cumulative reward > 0
            rollout_solved.append(np.array((rewards.sum(axis=0) > 0).astype(np.float32)))

        all_embeddings += np.concatenate(rollout_embs, axis=0)
        all_solved.append(np.concatenate(rollout_solved, axis=0))

    all_embeddings /= num_rollouts
    # SFL = p * (1-p) where p = solve rate across rollouts
    solve_rate = np.stack(all_solved, axis=0).mean(axis=0)  # (N,)
    sfl = solve_rate * (1 - solve_rate)

    return all_embeddings, sfl


def resolve_run_paths(run_cfg, timestep, local_data_root):
    """Resolve checkpoint dir and buffer dump path for a run at a given timestep.

    Supports three modes:
    1. Local paths provided directly (checkpoint_dir + buffer_dir)
    2. GCS prefix provided (auto-download)
    3. Mix: local checkpoint_dir + GCS buffer or vice versa
    """
    ckpt_step = updates_to_ckpt_step(timestep)
    seed = run_cfg.get("seed", 0)
    run_id = run_cfg.get("run_id", run_cfg["name"])
    safe_id = f"{run_id}_s{seed}".replace(" ", "_").replace("/", "_")

    # Checkpoint dir
    ckpt_dir = run_cfg.get("checkpoint_dir")
    if ckpt_dir and os.path.exists(os.path.join(ckpt_dir, "config.json")):
        pass  # local path exists
    elif "gcs_prefix" in run_cfg:
        gcs_prefix = run_cfg["gcs_prefix"]
        ckpt_dir = os.path.join(local_data_root, safe_id, "checkpoints")

        # Try to find the checkpoint structure
        # Layout 1: {gcs_prefix}/checkpoints/{run_id}/{seed}/
        # Layout 2: {gcs_prefix}/{seed}/ (simpler)
        for layout in [
            f"{gcs_prefix}/checkpoints",
            gcs_prefix,
        ]:
            # Find config.json
            config_candidates = [
                f"{layout}/config.json",
                f"{layout}/{seed}/config.json",
            ]
            # Also try run_id subdirs
            if run_id:
                config_candidates.insert(0, f"{layout}/{run_id}/{seed}/config.json")

            for config_gcs in config_candidates:
                config_local = os.path.join(ckpt_dir, "config.json")
                if gcs_download(config_gcs, config_local):
                    # Found config, now get the models dir relative to it
                    config_gcs_dir = config_gcs.rsplit("/config.json", 1)[0]
                    models_gcs = f"{config_gcs_dir}/models/{ckpt_step}"
                    models_local = os.path.join(ckpt_dir, "models", str(ckpt_step))
                    gcs_download_dir(models_gcs, models_local)
                    break
            if os.path.exists(os.path.join(ckpt_dir, "config.json")):
                break

    # Buffer dump path
    buffer_dir = run_cfg.get("buffer_dir")
    buf_path = None
    if buffer_dir:
        buf_path = os.path.join(buffer_dir, f"buffer_dump_{timestep}.npz")
    elif "gcs_prefix" in run_cfg:
        gcs_prefix = run_cfg["gcs_prefix"]
        local_buf_dir = os.path.join(local_data_root, safe_id, "buffer_dumps")

        # Try common layouts
        buf_candidates = [
            f"{gcs_prefix}/buffer_dumps/{run_id}/{seed}/buffer_dump_{timestep}.npz",
            f"{gcs_prefix}/buffer_dumps/buffer_dump_{timestep}.npz",
            f"{gcs_prefix}/{seed}/buffer_dump_{timestep}.npz",
        ]
        for buf_gcs in buf_candidates:
            buf_local = os.path.join(local_buf_dir, f"buffer_dump_{timestep}.npz")
            if gcs_download(buf_gcs, buf_local):
                buf_path = buf_local
                break

    return ckpt_dir, buf_path, ckpt_step


# ── Origin-based coloring (when origins available) ──────────────────────────

# Origin color schemes — auto-detected based on which codes appear
# LLM injection: 0=organic ACCEL, 1=LLM original, 2=LLM mutation
ORIGIN_COLORS_LLM = {
    0: ("lightgrey", 0.25, 3, "o", "Organic"),
    1: ("blue", 0.9, 35, "*", "LLM orig"),
    2: ("green", 0.5, 8, "o", "LLM mut"),
}
# CMA-ES runs: 0=DR generation, 1=CMA-ES generation, 2=DR mutation, 3=CMA-ES mutation
ORIGIN_COLORS_CMAES = {
    0: ("tab:blue", 0.4, 5, "o", "DR gen"),
    1: ("tab:orange", 0.4, 5, "o", "CMA-ES gen"),
    2: ("tab:cyan", 0.3, 4, "o", "DR mut"),
    3: ("tab:red", 0.3, 4, "o", "CMA-ES mut"),
}


def _pick_origin_scheme(origins):
    """Auto-detect which origin color scheme to use."""
    unique = set(np.unique(origins).tolist())
    if 3 in unique or (1 in unique and 0 not in unique):
        return ORIGIN_COLORS_CMAES
    return ORIGIN_COLORS_LLM


def plot_cell_with_origins(ax, coords, origins):
    scheme = _pick_origin_scheme(origins)
    # Plot in order: background first, distinctive last
    for origin_val in sorted(scheme.keys()):
        mask = origins == origin_val
        if mask.sum() == 0:
            continue
        color, alpha, size, marker, label = scheme[origin_val]
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, s=size, alpha=alpha, marker=marker,
                   edgecolors='black' if marker == '*' else 'none',
                   linewidths=0.3 if marker == '*' else 0,
                   rasterized=True, label=f"{label} ({mask.sum()})")


def plot_cell_uniform(ax, coords, color, label):
    ax.scatter(coords[:, 0], coords[:, 1],
               c=color, s=5, alpha=0.4, edgecolors='none',
               rasterized=True, label=f"{label} ({len(coords)})")


def plot_cell_learnability(ax, coords, sfl):
    """Color points by SFL learnability (0=easy/hard, 0.25=maximally learnable)."""
    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=sfl, cmap='viridis', s=5, alpha=0.5,
                    edgecolors='none', rasterized=True,
                    vmin=0, vmax=0.25)
    return sc


# Marker styles for origin+learnability combined mode
ORIGIN_MARKERS_LLM = {
    0: ("o", 3, "Organic"),
    1: ("*", 30, "LLM orig"),
    2: ("^", 6, "LLM mut"),
}
ORIGIN_MARKERS_CMAES = {
    0: ("o", 3, "DR gen"),
    1: ("D", 6, "CMA-ES gen"),
    2: ("o", 3, "DR mut"),
    3: ("D", 6, "CMA-ES mut"),
}


def plot_cell_origin_learnability(ax, coords, origins, sfl):
    """Marker by origin, color by SFL learnability."""
    unique = set(np.unique(origins).tolist())
    markers = ORIGIN_MARKERS_CMAES if (3 in unique or (1 in unique and 0 not in unique)) else ORIGIN_MARKERS_LLM
    last_sc = None
    for origin_val in sorted(markers.keys()):
        mask = origins == origin_val
        if mask.sum() == 0:
            continue
        marker, size, label = markers[origin_val]
        sc = ax.scatter(coords[mask, 0], coords[mask, 1],
                        c=sfl[mask], cmap='viridis', s=size, alpha=0.5,
                        marker=marker, edgecolors='none', rasterized=True,
                        vmin=0, vmax=0.25, label=f"{label} ({mask.sum()})")
        last_sc = sc
    return last_sc


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, required=True,
                        help="YAML config file defining runs to compare")
    parser.add_argument("--output", type=str, default=None,
                        help="Output PNG path (default: auto-generated)")
    parser.add_argument("--tsne_perplexity", type=float, default=40)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--local_data_root", type=str,
                        default="/cs/student/project_msc/2025/csml/rhautier/run_comparison_data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_rollouts", type=int, default=5)
    parser.add_argument("--cache_only", action="store_true")
    parser.add_argument("--gcs_cache_prefix", type=str, default=None,
                        help="GCS prefix for cache. Upload immediately and delete local file.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    runs = cfg["runs"]
    timesteps = cfg.get("timesteps", [250, 1000, 3000, 5000, 7000, 10000])
    title = cfg.get("title", "Buffer Embedding Comparison")
    # color_by: "origin", "learnability", or "both" (produces 2 plots)
    color_by = cfg.get("color_by", "origin")
    # backward compat
    if "color_by_origin" in cfg and "color_by" not in cfg:
        color_by = "origin" if cfg["color_by_origin"] else "uniform"

    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)

    n_rows = len(runs)
    n_cols = len(timesteps)

    # --- Load / compute embeddings ---
    data = {}  # (run_idx, timestep) -> {embeddings, origins (optional), color, label}

    for ri, run in enumerate(runs):
        run_name = run["name"]
        run_color = run.get("color", f"C{ri}")
        run_id = run.get("run_id", run_name)
        seed = run.get("seed", 0)
        cache_id = f"{run_id}_s{seed}".replace(" ", "_").replace("/", "_")

        for ts in timesteps:
            key = (ri, ts)
            print(f"\n  [{run_name}] update {ts}:")

            # Check cache (local first, then GCS)
            if args.cache_dir:
                cache_path = os.path.join(args.cache_dir, f"emb_{cache_id}_t{ts}.npz")
                if not os.path.exists(cache_path) and args.gcs_cache_prefix:
                    gcs_download(f"{args.gcs_cache_prefix}/emb_{cache_id}_t{ts}.npz", cache_path)
                if os.path.exists(cache_path):
                    print(f"    Loading from cache")
                    cached = np.load(cache_path, allow_pickle=True)
                    d = {
                        "embeddings": cached["embeddings"],
                        "scores": cached["scores"],
                        "color": run_color,
                        "label": run_name,
                    }
                    if "origins" in cached:
                        d["origins"] = cached["origins"]
                    if "sfl" in cached:
                        d["sfl"] = cached["sfl"]
                    data[key] = d
                    continue
                elif args.cache_only:
                    print(f"    SKIP: not cached")
                    continue

            # Resolve paths
            ckpt_dir, buf_path, ckpt_step = resolve_run_paths(run, ts, args.local_data_root)

            if not buf_path or not os.path.exists(buf_path):
                print(f"    SKIP: buffer dump not found")
                continue

            buf = load_buffer_dump(buf_path)
            if buf is None:
                print(f"    SKIP: could not load buffer")
                continue

            if not ckpt_dir or not os.path.exists(os.path.join(ckpt_dir, "config.json")):
                print(f"    SKIP: checkpoint not found")
                continue

            print(f"    {buf['size']} levels, computing embeddings + SFL (step {ckpt_step}, {args.num_rollouts} rollouts)...")
            embeddings, sfl = compute_embeddings(
                buf["tokens"], ckpt_dir, ckpt_step,
                batch_size=args.batch_size,
                num_rollouts=args.num_rollouts,
            )
            if embeddings is None:
                print(f"    SKIP: could not load agent")
                continue

            d = {
                "embeddings": embeddings,
                "scores": buf["scores"],
                "sfl": sfl,
                "color": run_color,
                "label": run_name,
            }
            if "origins" in buf:
                d["origins"] = buf["origins"]
            data[key] = d

            # Cache locally, upload to GCS, then delete local copy
            if args.cache_dir:
                save_dict = {
                    "embeddings": embeddings,
                    "scores": buf["scores"],
                    "sfl": sfl,
                }
                if "origins" in buf:
                    save_dict["origins"] = buf["origins"]
                np.savez_compressed(cache_path, **save_dict)
                # Upload immediately to GCS and remove local file
                if args.gcs_cache_prefix:
                    try:
                        blob = _get_bucket().blob(f"{args.gcs_cache_prefix}/{os.path.basename(cache_path)}")
                        blob.upload_from_filename(cache_path)
                        os.remove(cache_path)
                        print(f"    Uploaded to GCS, local removed")
                    except Exception as e:
                        print(f"    Cached locally (GCS upload failed: {e})")
                else:
                    print(f"    Cached locally")

    if not data:
        print("No data loaded.")
        return

    # --- Pre-compute per-cell t-SNE coordinates (shared across color modes) ---
    print(f"\n=== Computing per-cell t-SNE ===")
    tsne_coords = {}
    n_done = 0
    n_total = sum(1 for k in data)
    for ri, run in enumerate(runs):
        for j, ts in enumerate(timesteps):
            key = (ri, ts)
            if key not in data:
                continue
            n_done += 1
            emb = data[key]["embeddings"]
            print(f"  t-SNE: {run['name']}, {ts} upd ({len(emb)} pts) [{n_done}/{n_total}]")
            perp = min(args.tsne_perplexity, len(emb) - 1)
            tsne = TSNE(n_components=2, perplexity=perp,
                        random_state=42, max_iter=1000, learning_rate='auto', init='pca')
            tsne_coords[key] = tsne.fit_transform(emb)

    # --- Determine which plots to produce ---
    mode_map = {
        "origin": ["origin"],
        "learnability": ["learnability"],
        "combined": ["combined"],
        "both": ["origin", "learnability"],
        "all": ["origin", "learnability", "combined"],
    }
    color_modes = mode_map.get(color_by, ["origin"])

    base_output = args.output or os.path.join(os.path.dirname(args.config), "tsne_comparison.png")

    for mode in color_modes:
        print(f"\n=== Plotting {n_rows}x{n_cols} grid (color_by={mode}) ===")
        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(3.5 * n_cols, 3.5 * n_rows),
                                  squeeze=False)
        last_sc = None

        for ri, run in enumerate(runs):
            for j, ts in enumerate(timesteps):
                ax = axes[ri][j]
                key = (ri, ts)

                if key not in data:
                    ax.text(0.5, 0.5, "N/A", ha='center', va='center',
                            transform=ax.transAxes, fontsize=12, color='red')
                    ax.set_title(f"{run['name']}\n{ts} upd", fontsize=8)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                d = data[key]
                coords = tsne_coords[key]

                if mode == "combined" and "origins" in d and "sfl" in d:
                    last_sc = plot_cell_origin_learnability(ax, coords, d["origins"], d["sfl"])
                elif mode == "learnability" and "sfl" in d:
                    last_sc = plot_cell_learnability(ax, coords, d["sfl"])
                elif mode == "origin" and "origins" in d:
                    plot_cell_with_origins(ax, coords, d["origins"])
                else:
                    plot_cell_uniform(ax, coords, d.get("color", "grey"), d.get("label", ""))

                ax.set_title(f"{run['name']}\n{ts} upd ({len(d['embeddings'])} lvls)", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])

                if ri == 0 and j == 0 and mode in ("origin", "combined"):
                    ax.legend(fontsize=5, loc='lower left', framealpha=0.7)

            axes[ri][0].set_ylabel(run["name"], fontsize=10, rotation=90, labelpad=10)

        subtitles = {
            "origin": "colored by origin",
            "learnability": "colored by SFL learnability",
            "combined": "marker=origin, color=SFL learnability",
        }
        plt.suptitle(f"{title}\n({args.num_rollouts}-rollout, {subtitles[mode]})",
                     fontsize=13, y=1.01)

        if mode in ("learnability", "combined") and last_sc is not None:
            fig.colorbar(last_sc, ax=axes, label="SFL = p(1-p)", shrink=0.6, pad=0.02)

        plt.tight_layout()

        stem, ext = os.path.splitext(base_output)
        out_path = f"{stem}_{mode}{ext}" if len(color_modes) > 1 else base_output

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
