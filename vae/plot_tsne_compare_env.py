"""Compare buffer structural (environment) embeddings across training runs via t-SNE.

Same interface as plot_tsne_compare_runs.py but uses 173D structural features
(wall map + agent/goal positions) instead of 257D agent behavioral embeddings.
No GPU needed, no agent checkpoints needed — only buffer dumps.

Grid: rows = runs (or seeds), columns = training timesteps.

Uses the same YAML config format as plot_tsne_compare_runs.py.

Usage:
    python vae/plot_tsne_compare_env.py --config vae/compare_accel.yaml
    python vae/plot_tsne_compare_env.py --config vae/compare_accel.yaml --cache_only
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


def tokens_to_structural_features(tokens_batch, grid_size=13):
    """Convert (N, seq_len) token array to (N, grid_size^2 + 4) structural features.

    Features: flattened wall map (grid_size^2 D bool) + agent_pos (x,y) + goal_pos (x,y).
    """
    N = len(tokens_batch)
    n_cells = grid_size * grid_size
    features = np.zeros((N, n_cells + 4), dtype=np.float32)

    for i in range(N):
        tokens = tokens_batch[i]
        wall_tokens = tokens[:-2]
        goal_idx = tokens[-2]
        agent_idx = tokens[-1]

        wall_flat = np.zeros(n_cells, dtype=np.float32)
        for w in wall_tokens:
            if 0 < w <= n_cells:
                wall_flat[int(w) - 1] = 1.0
        features[i, :n_cells] = wall_flat

        if agent_idx > 0:
            a0 = int(agent_idx) - 1
            features[i, n_cells] = (a0 % grid_size) / max(grid_size - 1, 1)
            features[i, n_cells + 1] = (a0 // grid_size) / max(grid_size - 1, 1)

        if goal_idx > 0:
            g0 = int(goal_idx) - 1
            features[i, n_cells + 2] = (g0 % grid_size) / max(grid_size - 1, 1)
            features[i, n_cells + 3] = (g0 // grid_size) / max(grid_size - 1, 1)

    return features


def resolve_buffer_path(run_cfg, timestep, local_data_root):
    """Resolve buffer dump path for a run at a given timestep."""
    seed = run_cfg.get("seed", 0)
    run_name = run_cfg["name"]
    safe_name = run_name.replace(" ", "_").replace("/", "_")

    buffer_dir = run_cfg.get("buffer_dir")
    if buffer_dir:
        return os.path.join(buffer_dir, f"buffer_dump_{timestep}.npz")

    if "gcs_prefix" in run_cfg:
        gcs_prefix = run_cfg["gcs_prefix"]
        run_id = run_cfg.get("run_id", "")
        local_buf_dir = os.path.join(local_data_root, safe_name, "buffer_dumps")

        buf_candidates = [
            f"{gcs_prefix}/buffer_dumps/{run_id}/{seed}/buffer_dump_{timestep}.npz",
            f"{gcs_prefix}/buffer_dumps/buffer_dump_{timestep}.npz",
            f"{gcs_prefix}/{seed}/buffer_dump_{timestep}.npz",
        ]
        for buf_gcs in buf_candidates:
            buf_local = os.path.join(local_buf_dir, f"buffer_dump_{timestep}.npz")
            if gcs_download(buf_gcs, buf_local):
                return buf_local

    return None


# ── Origin coloring (same as plot_tsne_compare_runs.py) ─────────────────────

ORIGIN_COLORS_LLM = {
    0: ("lightgrey", 0.25, 3, "o", "Organic"),
    1: ("blue", 0.9, 35, "*", "LLM orig"),
    2: ("green", 0.5, 8, "o", "LLM mut"),
}
ORIGIN_COLORS_CMAES = {
    0: ("tab:blue", 0.4, 5, "o", "DR gen"),
    1: ("tab:orange", 0.4, 5, "o", "CMA-ES gen"),
    2: ("tab:cyan", 0.3, 4, "o", "DR mut"),
    3: ("tab:red", 0.3, 4, "o", "CMA-ES mut"),
}


def _pick_origin_scheme(origins):
    unique = set(np.unique(origins).tolist())
    if 3 in unique or (1 in unique and 0 not in unique):
        return ORIGIN_COLORS_CMAES
    return ORIGIN_COLORS_LLM


def plot_cell_with_origins(ax, coords, origins):
    scheme = _pick_origin_scheme(origins)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, required=True,
                        help="YAML config file (same format as plot_tsne_compare_runs.py)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--tsne_perplexity", type=float, default=40)
    parser.add_argument("--grid_size", type=int, default=13,
                        help="Maze grid size for structural features")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--local_data_root", type=str,
                        default="/cs/student/project_msc/2025/csml/rhautier/run_comparison_data")
    parser.add_argument("--cache_only", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    runs = cfg["runs"]
    timesteps = cfg.get("timesteps", [250, 1000, 3000, 5000, 7000, 10000])
    title = cfg.get("title", "Buffer Structural Comparison")
    color_by_origin = cfg.get("color_by_origin", False)

    grid_size = cfg.get("grid_size", args.grid_size)
    n_features = grid_size * grid_size + 4

    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)

    n_rows = len(runs)
    n_cols = len(timesteps)

    # --- Load / compute structural features ---
    data = {}

    for ri, run in enumerate(runs):
        run_name = run["name"]
        safe_name = run_name.replace(" ", "_").replace("/", "_")

        for ts in timesteps:
            key = (ri, ts)
            print(f"\n  [{run_name}] update {ts}:")

            # Check cache
            if args.cache_dir:
                cache_path = os.path.join(args.cache_dir, f"env_{safe_name}_t{ts}.npz")
                if os.path.exists(cache_path):
                    print(f"    Loading from cache")
                    cached = np.load(cache_path, allow_pickle=True)
                    d = {"features": cached["features"], "scores": cached["scores"],
                         "color": run.get("color", f"C{ri}"), "label": run_name}
                    if "origins" in cached:
                        d["origins"] = cached["origins"]
                    data[key] = d
                    continue
                elif args.cache_only:
                    print(f"    SKIP: not cached")
                    continue

            # Resolve buffer path
            buf_path = resolve_buffer_path(run, ts, args.local_data_root)
            if not buf_path or not os.path.exists(buf_path):
                print(f"    SKIP: buffer dump not found")
                continue

            buf = load_buffer_dump(buf_path)
            if buf is None:
                print(f"    SKIP: could not load buffer")
                continue

            print(f"    {buf['size']} levels, computing structural features (grid={grid_size})...")
            features = tokens_to_structural_features(buf["tokens"], grid_size=grid_size)

            d = {"features": features, "scores": buf["scores"],
                 "color": run.get("color", f"C{ri}"), "label": run_name}
            if "origins" in buf:
                d["origins"] = buf["origins"]
            data[key] = d

            # Cache
            if args.cache_dir:
                save_dict = {"features": features, "scores": buf["scores"]}
                if "origins" in buf:
                    save_dict["origins"] = buf["origins"]
                np.savez_compressed(cache_path, **save_dict)
                print(f"    Cached")

    if not data:
        print("No data loaded.")
        return

    # --- Plot ---
    print(f"\n=== Plotting {n_rows}x{n_cols} grid ===")
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.5 * n_cols, 3.5 * n_rows),
                              squeeze=False)

    n_done = 0
    n_total = sum(1 for k in data)

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
            n_done += 1
            print(f"  t-SNE: {run['name']}, {ts} upd ({len(d['features'])} pts) [{n_done}/{n_total}]")

            perp = min(args.tsne_perplexity, len(d["features"]) - 1)
            tsne = TSNE(n_components=2, perplexity=perp,
                        random_state=42, max_iter=1000, learning_rate='auto', init='pca')
            coords = tsne.fit_transform(d["features"])

            if color_by_origin and "origins" in d:
                plot_cell_with_origins(ax, coords, d["origins"])
            else:
                plot_cell_uniform(ax, coords, d["color"], d["label"])

            ax.set_title(f"{run['name']}\n{ts} upd ({len(d['features'])} lvls)", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

            if ri == 0 and j == 0:
                ax.legend(fontsize=5, loc='lower left', framealpha=0.7)

        axes[ri][0].set_ylabel(run["name"], fontsize=10, rotation=90, labelpad=10)

    plt.suptitle(f"{title} — Structural (env) space\n"
                 f"({n_features}D: {grid_size}x{grid_size} wall map + positions, per-cell t-SNE)",
                 fontsize=13, y=1.01)
    plt.tight_layout()

    if args.output:
        out_path = args.output
    else:
        out_path = os.path.join(os.path.dirname(args.config), "tsne_env_comparison.png")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
