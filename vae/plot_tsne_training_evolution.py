"""Visualize buffer embedding evolution during training via t-SNE.

For each (seed, timestep), loads the agent checkpoint and buffer dump,
recomputes fresh 257D embeddings by rolling out the current agent on
all buffer levels, then plots t-SNE colored by origin:
  grey = organic ACCEL, green = LLM mutation, blue = LLM original.

Grid: rows = seeds, columns = training timesteps.

Data is fetched from GCS if not available locally.

Usage:
    python vae/plot_tsne_training_evolution.py \
        --inject_pct 10pct \
        --seeds 0,1,2 \
        --timesteps 250,500,750,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000
"""
import argparse
import os
import sys
import json
import numpy as np
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
    """Download a file from GCS if it doesn't exist locally."""
    if os.path.exists(local_path):
        return True
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        blob = _get_bucket().blob(gcs_path)
        if not blob.exists():
            print(f"  [GCS] Not found: {gcs_path}")
            return False
        blob.download_to_filename(local_path)
        return True
    except Exception as e:
        print(f"  [GCS] Failed to download {gcs_path}: {e}")
        return False


def gcs_download_dir(gcs_prefix, local_dir):
    """Download all files under a GCS prefix to a local directory."""
    if os.path.exists(local_dir) and len(os.listdir(local_dir)) > 0:
        return True
    os.makedirs(local_dir, exist_ok=True)
    try:
        bucket = _get_bucket()
        blobs = list(bucket.list_blobs(prefix=gcs_prefix + "/"))
        if not blobs:
            print(f"  [GCS] No files found under {gcs_prefix}/")
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
        print(f"  [GCS] Failed to download {gcs_prefix}: {e}")
        return False


def updates_to_ckpt_step(updates, eval_freq=250):
    """Convert update count to checkpoint step index."""
    return (updates // eval_freq) - 1


def load_buffer_dump(local_dir, updates):
    """Load buffer dump npz for a given update count."""
    path = os.path.join(local_dir, f"buffer_dump_{updates}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    size = int(d["size"]) if "size" in d else len(d["tokens"])
    origins = d["origins"][:size] if "origins" in d else np.zeros(size, dtype=np.int32)
    return {
        "tokens": d["tokens"][:size],
        "origins": origins,
        "scores": d["scores"][:size],
        "size": size,
    }


def compute_embeddings(tokens, checkpoint_dir, ckpt_step, batch_size=256, num_rollouts=5):
    """Compute fresh 257D embeddings using agent at a specific checkpoint step.

    Averages over num_rollouts rollouts per level to reduce stochasticity
    from action sampling.
    """
    from cross_evaluate import load_agent, tokens_to_levels_batch
    from maze_plr import ActorCritic, sample_trajectories_rnn, compute_insertion_embeddings
    from jaxued.environments import Maze
    from jaxued.wrappers import AutoReplayWrapper
    import jax
    import jax.numpy as jnp

    train_state, config, env, env_params = load_agent(checkpoint_dir, checkpoint_step=ckpt_step)
    if train_state is None:
        return None

    eval_env = Maze(max_height=13, max_width=13,
                    agent_view_size=config["agent_view_size"], normalize_obs=True)
    wrapped_env = AutoReplayWrapper(eval_env)
    max_steps = env_params.max_steps_in_episode

    levels = tokens_to_levels_batch(tokens)
    n_levels = len(tokens)
    all_embeddings = np.zeros((n_levels, 257), dtype=np.float32)

    for rollout_idx in range(num_rollouts):
        rollout_embs = []
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
            _, actions, _, dones, _, _, _, _, hstates = traj
            embeddings = compute_insertion_embeddings(hstates, actions, dones)
            rollout_embs.append(np.array(embeddings))

        all_embeddings += np.concatenate(rollout_embs, axis=0)

    all_embeddings /= num_rollouts
    return all_embeddings


def run_single_pct(inject_pct, seeds, timesteps, args):
    """Generate one t-SNE grid plot for a single injection percentage."""
    print(f"\n{'='*60}")
    print(f"  Processing {inject_pct} injection")
    print(f"{'='*60}")

    n_rows = len(seeds)
    n_cols = len(timesteps)

    # --- Step 1: Download data from GCS and compute embeddings ---
    print("=== Step 1: Loading data and computing embeddings ===")
    data = {}  # (seed, timestep) -> {embeddings, origins, scores}

    for seed in seeds:
        run_name = f"inject_llm_{inject_pct}_seed{seed}"
        gcs_training_prefix = f"llm-exp/training/{run_name}"

        run_local = os.path.join(args.local_data_root, run_name)
        ckpt_local = os.path.join(run_local, "checkpoints")
        buffer_local = os.path.join(run_local, "buffer_dumps")

        for ts in timesteps:
            key = (seed, ts)
            print(f"\n  Seed {seed}, update {ts}:")

            # Check embedding cache
            if args.cache_dir:
                cache_path = os.path.join(args.cache_dir,
                                          f"emb_{inject_pct}_s{seed}_t{ts}.npz")
                if os.path.exists(cache_path):
                    print(f"    Loading from cache: {cache_path}")
                    cached = np.load(cache_path)
                    data[key] = {
                        "embeddings": cached["embeddings"],
                        "origins": cached["origins"],
                        "scores": cached["scores"],
                    }
                    continue
                elif args.cache_only:
                    print(f"    SKIP: not cached (--cache_only)")
                    continue

            # Download buffer dump from GCS
            buf_gcs = f"{gcs_training_prefix}/buffer_dumps/{run_name}/{seed}/buffer_dump_{ts}.npz"
            buf_local_path = os.path.join(buffer_local, f"buffer_dump_{ts}.npz")
            if not gcs_download(buf_gcs, buf_local_path):
                print(f"    SKIP: buffer dump not found")
                continue

            buf = load_buffer_dump(buffer_local, ts)
            if buf is None:
                print(f"    SKIP: could not load buffer dump")
                continue

            n_org = (buf["origins"] == 0).sum()
            n_orig = (buf["origins"] == 1).sum()
            n_mut = (buf["origins"] == 2).sum()
            print(f"    {buf['size']} levels: organic={n_org}, LLM orig={n_orig}, LLM mut={n_mut}")

            # Download checkpoint from GCS
            ckpt_step = updates_to_ckpt_step(ts)
            ckpt_run_dir = os.path.join(ckpt_local, run_name, str(seed))

            config_gcs = f"{gcs_training_prefix}/checkpoints/{run_name}/{seed}/config.json"
            config_local = os.path.join(ckpt_run_dir, "config.json")
            if not gcs_download(config_gcs, config_local):
                print(f"    SKIP: config.json not found on GCS")
                continue

            models_gcs = f"{gcs_training_prefix}/checkpoints/{run_name}/{seed}/models/{ckpt_step}"
            models_local = os.path.join(ckpt_run_dir, "models", str(ckpt_step))
            if not gcs_download_dir(models_gcs, models_local):
                print(f"    SKIP: checkpoint step {ckpt_step} not found on GCS")
                continue

            print(f"    Computing embeddings (ckpt step {ckpt_step}, {args.num_rollouts} rollouts)...")
            embeddings = compute_embeddings(
                buf["tokens"], ckpt_run_dir, ckpt_step,
                batch_size=args.batch_size,
                num_rollouts=args.num_rollouts,
            )
            if embeddings is None:
                print(f"    SKIP: could not load agent")
                continue

            data[key] = {
                "embeddings": embeddings,
                "origins": buf["origins"],
                "scores": buf["scores"],
            }

            if args.cache_dir:
                np.savez_compressed(cache_path,
                                    embeddings=embeddings,
                                    origins=buf["origins"],
                                    scores=buf["scores"])
                print(f"    Cached to {cache_path}")

    if not data:
        print(f"No data loaded for {inject_pct}. Skipping.")
        return None

    # --- Step 2 + 3: Fit per-cell t-SNE and plot ---
    print(f"\n=== Step 2: Fitting per-cell t-SNE (perplexity={args.tsne_perplexity}) ===")

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.5 * n_cols, 3.5 * n_rows),
                              squeeze=False)

    n_done = 0
    n_total = sum(1 for s in seeds for t in timesteps if (s, t) in data)

    for i, seed in enumerate(seeds):
        for j, ts in enumerate(timesteps):
            ax = axes[i][j]
            key = (seed, ts)

            if key not in data:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center',
                        transform=ax.transAxes, fontsize=12, color='red')
                ax.set_title(f"Seed {seed}, {ts} upd", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            n_done += 1
            emb = data[key]["embeddings"]
            origins = data[key]["origins"]
            print(f"  t-SNE for seed {seed}, {ts} upd ({len(emb)} pts) [{n_done}/{n_total}]...")

            tsne = TSNE(n_components=2, perplexity=min(args.tsne_perplexity, len(emb) - 1),
                        random_state=42, max_iter=1000, learning_rate='auto', init='pca')
            coords = tsne.fit_transform(emb)

            is_organic = origins == 0
            is_original = origins == 1
            is_mutation = origins == 2

            # Organic background (grey)
            if is_organic.sum() > 0:
                ax.scatter(coords[is_organic, 0], coords[is_organic, 1],
                           c='lightgrey', s=3, alpha=0.25, edgecolors='none',
                           rasterized=True, label=f'Organic ({is_organic.sum()})')

            # LLM mutations (green)
            if is_mutation.sum() > 0:
                ax.scatter(coords[is_mutation, 0], coords[is_mutation, 1],
                           c='green', s=8, alpha=0.5, edgecolors='none',
                           rasterized=True, label=f'LLM mut ({is_mutation.sum()})')

            # LLM originals (blue stars)
            if is_original.sum() > 0:
                ax.scatter(coords[is_original, 0], coords[is_original, 1],
                           c='blue', s=35, marker='*', alpha=0.9,
                           edgecolors='black', linewidths=0.3,
                           label=f'LLM orig ({is_original.sum()})')

            n_llm = is_original.sum() + is_mutation.sum()
            ax.set_title(f"Seed {seed}, {ts} upd\n"
                         f"({n_llm} LLM / {len(origins)} total)", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0 and j == 0:
                ax.legend(fontsize=6, loc='lower left', framealpha=0.7)

    pct_label = inject_pct.replace("pct", "%")
    plt.suptitle(f"Buffer t-SNE Evolution — {pct_label} injection\n"
                 f"(per-cell t-SNE, {args.num_rollouts}-rollout embeddings from current agent)",
                 fontsize=13, y=1.01)
    plt.tight_layout()

    out_path = os.path.join(args.output_dir, f"tsne_evolution_{inject_pct}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")
    return out_path


ALL_PCTS = ["5pct", "10pct", "15pct", "20pct", "25pct"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inject_pct", type=str, default="all",
                        help="Injection percentage tag (e.g., 10pct) or 'all' for 5/10/15/20/25%%")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--timesteps", type=str,
                        default="250,500,750,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000")
    parser.add_argument("--tsne_perplexity", type=float, default=40)
    parser.add_argument("--output_dir", type=str, default="vae/plots/tsne_training_evolution")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Cache embeddings to avoid recomputation")
    parser.add_argument("--local_data_root", type=str,
                        default="/cs/student/project_msc/2025/csml/rhautier/injection_training_data",
                        help="Local dir to cache GCS downloads")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_rollouts", type=int, default=5,
                        help="Number of rollouts to average per level for stable embeddings")
    parser.add_argument("--cache_only", action="store_true",
                        help="Only plot cached embeddings, skip GCS downloads and computation")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    timesteps = [int(t) for t in args.timesteps.split(",")]

    os.makedirs(args.output_dir, exist_ok=True)
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)

    pcts = ALL_PCTS if args.inject_pct == "all" else [args.inject_pct]

    saved = []
    for pct in pcts:
        out = run_single_pct(pct, seeds, timesteps, args)
        if out:
            saved.append(out)

    print(f"\n{'='*60}")
    print(f"Done. {len(saved)} plots saved:")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
