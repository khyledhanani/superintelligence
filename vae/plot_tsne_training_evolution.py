"""Visualize buffer embedding evolution during training via t-SNE.

Supports two modes:
  - behavioral: 257D embeddings from agent rollouts (requires GPU + checkpoints)
  - structural: 173D env features from token layout (CPU only, no checkpoints)

For each (seed, timestep), loads the buffer dump (and agent checkpoint in
behavioral mode), computes embeddings, then plots t-SNE colored by origin:
  grey = organic ACCEL, green = LLM mutation, blue = LLM original.

Grid: rows = seeds, columns = training timesteps.

Data is fetched from GCS if not available locally.

Usage:
    # Behavioral (default, needs GPU)
    python vae/plot_tsne_training_evolution.py \
        --inject_pct 10pct --mode behavioral

    # Structural (CPU only)
    python vae/plot_tsne_training_evolution.py \
        --inject_pct 10pct --mode structural --cache_dir vae/plots/tsne_env_cache
"""
import argparse
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

GRID_SIZE = 13  # default, overridable via --grid_size
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


def tokens_to_structural_features(tokens_batch, grid_size=13):
    """Convert (N, seq_len) token array to (N, grid_size^2 + 4) structural features."""
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
    ancestor_ids = d["ancestor_ids"][:size] if "ancestor_ids" in d else np.full(size, -1, dtype=np.int32)
    return {
        "tokens": d["tokens"][:size],
        "origins": origins,
        "scores": d["scores"][:size],
        "ancestor_ids": ancestor_ids,
        "size": size,
    }


EVAL_LEVEL_NAMES = [
    "SixteenRooms", "SixteenRooms2", "Labyrinth", "LabyrinthFlipped",
    "Labyrinth2", "StandardMaze", "StandardMaze2", "StandardMaze3",
]

EVAL_LEVEL_SHORT = {
    "SixteenRooms": "16R", "SixteenRooms2": "16R2",
    "Labyrinth": "Lab", "LabyrinthFlipped": "LabF",
    "Labyrinth2": "Lab2", "StandardMaze": "SM",
    "StandardMaze2": "SM2", "StandardMaze3": "SM3",
}


def get_eval_level_tokens():
    """Get token arrays for the 8 eval benchmark levels."""
    from jaxued.environments.maze.level import prefabs, Level
    from vae_level_utils import level_to_tokens

    tokens_list = []
    for name in EVAL_LEVEL_NAMES:
        level = Level.from_str(prefabs[name])
        tok = np.asarray(level_to_tokens(level))
        tokens_list.append(tok)
    return np.stack(tokens_list)


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

    mode = args.mode
    is_structural = (mode == "structural")
    cache_prefix = "env" if is_structural else "emb"

    # --- Step 1: Download data and compute embeddings/features ---
    print(f"=== Step 1: Loading data and computing {'structural features' if is_structural else 'behavioral embeddings'} ===")
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

            # Check cache
            if args.cache_dir:
                cache_path = os.path.join(args.cache_dir,
                                          f"{cache_prefix}_{inject_pct}_s{seed}_t{ts}.npz")
                if os.path.exists(cache_path):
                    print(f"    Loading from cache")
                    cached = np.load(cache_path)
                    data[key] = {
                        "embeddings": cached["embeddings"],
                        "origins": cached["origins"],
                        "scores": cached["scores"],
                        "ancestor_ids": cached["ancestor_ids"] if "ancestor_ids" in cached else np.full(len(cached["origins"]), -1, dtype=np.int32),
                    }
                    if "eval_embeddings" in cached:
                        data[key]["eval_embeddings"] = cached["eval_embeddings"]
                    elif args.show_eval_levels and not args.cache_only:
                        # Eval embeddings not in cache — need to compute them
                        # Fall through to computation path below
                        print(f"    Eval embeddings not cached, will compute...")
                        pass
                    else:
                        continue
                    if "eval_embeddings" in data[key] or not args.show_eval_levels:
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

            if is_structural:
                print(f"    Computing structural features (grid={args.grid_size})...")
                embeddings = tokens_to_structural_features(buf["tokens"], grid_size=args.grid_size)
            else:
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

                print(f"    Computing behavioral embeddings (ckpt step {ckpt_step}, {args.num_rollouts} rollouts)...")
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
                "ancestor_ids": buf["ancestor_ids"],
            }

            # Compute eval level embeddings if requested (behavioral mode only)
            if args.show_eval_levels and not is_structural:
                eval_tokens = get_eval_level_tokens()
                print(f"    Computing eval level embeddings ({len(eval_tokens)} levels)...")
                eval_embeddings = compute_embeddings(
                    eval_tokens, ckpt_run_dir, ckpt_step,
                    batch_size=args.batch_size,
                    num_rollouts=args.num_rollouts,
                )
                if eval_embeddings is not None:
                    data[key]["eval_embeddings"] = eval_embeddings

            if args.cache_dir:
                save_data = {
                    "embeddings": embeddings,
                    "origins": buf["origins"],
                    "scores": buf["scores"],
                    "ancestor_ids": buf["ancestor_ids"],
                }
                if "eval_embeddings" in data[key]:
                    save_data["eval_embeddings"] = data[key]["eval_embeddings"]
                np.savez_compressed(cache_path, **save_data)
                print(f"    Cached")

    if not data:
        print(f"No data loaded for {inject_pct}. Skipping.")
        return None

    # --- Step 1b: Collect seed (original) tokens and ensure embeddings at every timestep ---
    # Extract raw tokens for each LLM seed from buffer dumps where they appear as originals
    seed_tokens_by_ancestor = {}  # ancestor_id -> token array (1, seq_len)
    for seed_run in seeds:
        run_name = f"inject_llm_{inject_pct}_seed{seed_run}"
        buffer_local = os.path.join(args.local_data_root, run_name, "buffer_dumps")
        for ts in timesteps:
            buf = load_buffer_dump(buffer_local, ts)
            if buf is None:
                continue
            orig_mask = buf["origins"] == 1
            for idx in np.where(orig_mask)[0]:
                aid = int(buf["ancestor_ids"][idx])
                if aid not in seed_tokens_by_ancestor:
                    seed_tokens_by_ancestor[aid] = buf["tokens"][idx:idx+1]

    if seed_tokens_by_ancestor:
        # Stack all seed tokens into a single array for batch embedding
        sorted_seed_aids = sorted(seed_tokens_by_ancestor.keys())
        seed_tokens_all = np.concatenate([seed_tokens_by_ancestor[a] for a in sorted_seed_aids], axis=0)
        print(f"\n  Found {len(sorted_seed_aids)} LLM seed originals: {sorted_seed_aids}")

        # For each (seed_run, timestep), compute seed embeddings if not already in data
        for seed_run in seeds:
            run_name = f"inject_llm_{inject_pct}_seed{seed_run}"
            ckpt_local = os.path.join(args.local_data_root, run_name, "checkpoints")

            for ts in timesteps:
                key = (seed_run, ts)
                if key not in data:
                    continue

                # Check if seed embeddings already cached
                if args.cache_dir:
                    cache_path = os.path.join(args.cache_dir,
                                              f"{cache_prefix}_{inject_pct}_s{seed_run}_t{ts}.npz")
                    if os.path.exists(cache_path):
                        cached = np.load(cache_path)
                        if "seed_embeddings" in cached and "seed_ancestor_ids" in cached:
                            data[key]["seed_embeddings"] = cached["seed_embeddings"]
                            data[key]["seed_ancestor_ids"] = cached["seed_ancestor_ids"]
                            continue

                if is_structural:
                    seed_emb = tokens_to_structural_features(seed_tokens_all, grid_size=args.grid_size)
                else:
                    ckpt_step = updates_to_ckpt_step(ts)
                    ckpt_run_dir = os.path.join(ckpt_local, run_name, str(seed_run))
                    if not os.path.exists(os.path.join(ckpt_run_dir, "config.json")):
                        continue
                    seed_rollouts = args.seed_num_rollouts
                    print(f"    Computing seed embeddings at t={ts} (ckpt {ckpt_step}, {seed_rollouts} rollouts)...")
                    seed_emb = compute_embeddings(
                        seed_tokens_all, ckpt_run_dir, ckpt_step,
                        batch_size=args.batch_size,
                        num_rollouts=seed_rollouts,
                    )
                    if seed_emb is None:
                        continue

                data[key]["seed_embeddings"] = seed_emb
                data[key]["seed_ancestor_ids"] = np.array(sorted_seed_aids, dtype=np.int32)

                # Update cache
                if args.cache_dir and os.path.exists(cache_path):
                    cached = dict(np.load(cache_path))
                    cached["seed_embeddings"] = seed_emb
                    cached["seed_ancestor_ids"] = np.array(sorted_seed_aids, dtype=np.int32)
                    np.savez_compressed(cache_path, **cached)

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

            # Optionally include eval benchmark levels in the t-SNE
            eval_emb = data[key].get("eval_embeddings", None)
            n_eval = len(eval_emb) if eval_emb is not None else 0

            # Include seed original embeddings (even if evicted from buffer)
            seed_emb = data[key].get("seed_embeddings", None)
            seed_aids = data[key].get("seed_ancestor_ids", None)
            n_seeds = len(seed_emb) if seed_emb is not None else 0

            parts = [emb]
            extra_label = ""
            if n_eval > 0:
                parts.append(eval_emb)
                extra_label += f"+{n_eval} eval"
            if n_seeds > 0:
                parts.append(seed_emb)
                extra_label += f"+{n_seeds} seeds"
            combined = np.concatenate(parts, axis=0) if len(parts) > 1 else emb
            print(f"  t-SNE for seed {seed}, {ts} upd ({len(emb)}{extra_label} pts) [{n_done}/{n_total}]...")

            tsne = TSNE(n_components=2, perplexity=min(args.tsne_perplexity, len(combined) - 1),
                        random_state=42, max_iter=1000, learning_rate='auto', init='pca')
            all_coords = tsne.fit_transform(combined)
            coords = all_coords[:len(emb)]
            offset = len(emb)
            if n_eval > 0:
                eval_coords = all_coords[offset:offset + n_eval]
                offset += n_eval
            else:
                eval_coords = None
            if n_seeds > 0:
                seed_coords = all_coords[offset:offset + n_seeds]
            else:
                seed_coords = None

            is_organic = origins == 0
            is_original = origins == 1
            is_mutation = origins == 2
            scores = data[key]["scores"]

            if args.show_difficulty:
                # Color by SFL learnability: yellow (0) -> red (0.25+)
                cmap = mcolors.LinearSegmentedColormap.from_list(
                    "sfl", ["yellow", "red"])
                norm = mcolors.Normalize(vmin=0, vmax=0.25)

                # Organic (light/small)
                if is_organic.sum() > 0:
                    ax.scatter(coords[is_organic, 0], coords[is_organic, 1],
                               c=scores[is_organic], cmap=cmap, norm=norm,
                               s=3, alpha=0.25, edgecolors='none',
                               rasterized=True)

                # LLM mutations (bold circles with green edge)
                if is_mutation.sum() > 0:
                    ax.scatter(coords[is_mutation, 0], coords[is_mutation, 1],
                               c=scores[is_mutation], cmap=cmap, norm=norm,
                               s=15, alpha=0.8, edgecolors='green', linewidths=0.5,
                               rasterized=True, zorder=5)

                # LLM originals (stars, larger with bright edge to stand out)
                if is_original.sum() > 0:
                    ax.scatter(coords[is_original, 0], coords[is_original, 1],
                               c=scores[is_original], cmap=cmap, norm=norm,
                               s=60, marker='*', alpha=0.95,
                               edgecolors='blue', linewidths=0.6, zorder=8)

                # Plot eval benchmark levels (small diamonds)
                if eval_coords is not None:
                    ax.scatter(eval_coords[:, 0], eval_coords[:, 1],
                               c='cyan', s=20, marker='D', alpha=0.9,
                               edgecolors='black', linewidths=0.5, zorder=10)
                    for ei, name in enumerate(EVAL_LEVEL_NAMES[:n_eval]):
                        ax.annotate(EVAL_LEVEL_SHORT.get(name, name),
                                    (eval_coords[ei, 0], eval_coords[ei, 1]),
                                    fontsize=3, ha='center', va='bottom',
                                    xytext=(0, 3), textcoords='offset points')

                # Add legend on first panel, colorbar on last panel
                if i == 0 and j == 0:
                    from matplotlib.lines import Line2D
                    legend_els = [
                        Line2D([0], [0], marker='o', color='w', markerfacecolor='grey',
                               markersize=4, alpha=0.5, label='Organic'),
                        Line2D([0], [0], marker='o', color='w', markerfacecolor='grey',
                               markersize=6, markeredgecolor='green', markeredgewidth=0.5,
                               label='LLM mutation'),
                        Line2D([0], [0], marker='*', color='w', markerfacecolor='grey',
                               markersize=10, markeredgecolor='blue', markeredgewidth=0.6,
                               label='LLM original'),
                    ]
                    if eval_coords is not None:
                        legend_els.append(
                            Line2D([0], [0], marker='D', color='w', markerfacecolor='cyan',
                                   markersize=5, markeredgecolor='black', markeredgewidth=0.5,
                                   label='Eval benchmark'))
                    ax.legend(handles=legend_els, fontsize=5, loc='upper left',
                              framealpha=0.7)
                if i == n_rows - 1 and j == n_cols - 1:
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
                    cb.set_label("SFL", fontsize=7)
                    cb.ax.tick_params(labelsize=6)
            else:
                ancestor_ids = data[key]["ancestor_ids"]

                # Organic background (grey)
                if is_organic.sum() > 0:
                    ax.scatter(coords[is_organic, 0], coords[is_organic, 1],
                               c='lightgrey', s=3, alpha=0.25, edgecolors='none',
                               rasterized=True, label=f'Organic ({is_organic.sum()})')

                # Build a consistent color palette for ancestor IDs (once)
                if not hasattr(run_single_pct, '_lineage_cmap'):
                    all_anc = set()
                    for dk in data.values():
                        llm_m = dk["origins"] > 0
                        all_anc.update(dk["ancestor_ids"][llm_m].tolist())
                        if "seed_ancestor_ids" in dk:
                            all_anc.update(dk["seed_ancestor_ids"].tolist())
                    all_anc.discard(-1)
                    sorted_anc = sorted(all_anc)
                    cmap_name = 'tab10' if len(sorted_anc) <= 10 else 'tab20'
                    cmap_obj = plt.get_cmap(cmap_name)
                    run_single_pct._lineage_colors = {}
                    for idx, aid in enumerate(sorted_anc):
                        run_single_pct._lineage_colors[aid] = cmap_obj(idx % cmap_obj.N)
                    run_single_pct._lineage_colors[-1] = (0.5, 0.5, 0.5, 1.0)
                    run_single_pct._lineage_cmap = True

                lineage_colors = run_single_pct._lineage_colors

                # Color LLM levels by ancestor lineage
                llm_mask = origins > 0
                if llm_mask.sum() > 0:
                    unique_ancestors = sorted(set(ancestor_ids[llm_mask].tolist()))

                    for aid in unique_ancestors:
                        color = lineage_colors.get(aid, (0.5, 0.5, 0.5, 1.0))
                        anc_mask = (ancestor_ids == aid)

                        # Mutations for this lineage (circles)
                        mut_mask = anc_mask & is_mutation
                        if mut_mask.sum() > 0:
                            ax.scatter(coords[mut_mask, 0], coords[mut_mask, 1],
                                       c=[color], s=8, alpha=0.6, edgecolors='none',
                                       rasterized=True)

                # Always plot seed originals from seed_coords (even if evicted)
                if seed_coords is not None and seed_aids is not None:
                    for si, aid in enumerate(seed_aids):
                        color = lineage_colors.get(int(aid), (0.5, 0.5, 0.5, 1.0)) if hasattr(run_single_pct, '_lineage_colors') else 'blue'
                        ax.scatter(seed_coords[si, 0], seed_coords[si, 1],
                                   c=[color], s=50, marker='*', alpha=0.95,
                                   edgecolors='black', linewidths=0.4, zorder=8)

                # Plot eval benchmark levels
                if eval_coords is not None:
                    ax.scatter(eval_coords[:, 0], eval_coords[:, 1],
                               c='cyan', s=60, marker='D', alpha=0.95,
                               edgecolors='black', linewidths=0.8, zorder=10,
                               label='Eval benchmark')
                    for ei, name in enumerate(EVAL_LEVEL_NAMES[:n_eval]):
                        ax.annotate(EVAL_LEVEL_SHORT.get(name, name),
                                    (eval_coords[ei, 0], eval_coords[ei, 1]),
                                    fontsize=4, ha='center', va='bottom',
                                    xytext=(0, 4), textcoords='offset points')

                if i == n_rows - 1 and j == n_cols - 1:
                    from matplotlib.lines import Line2D
                    legend_els = [
                        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgrey',
                               markersize=4, alpha=0.5, label='Organic'),
                    ]
                    lineage_colors = run_single_pct._lineage_colors if hasattr(run_single_pct, '_lineage_colors') else {}
                    for aid in sorted(lineage_colors.keys()):
                        c = lineage_colors[aid]
                        if aid == -1:
                            legend_els.append(
                                Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                                       markersize=5, label='Unknown anc.'))
                        else:
                            legend_els.append(
                                Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                                       markersize=5, label=f'Seed {aid}'))
                    legend_els.append(
                        Line2D([0], [0], marker='*', color='w', markerfacecolor='grey',
                               markersize=8, markeredgecolor='black', markeredgewidth=0.4,
                               label='LLM original'))
                    fig.legend(handles=legend_els, fontsize=6, loc='lower center',
                               framealpha=0.7, ncol=len(legend_els),
                               bbox_to_anchor=(0.5, -0.02))

            n_llm = is_original.sum() + is_mutation.sum()
            ax.set_title(f"Seed {seed}, {ts} upd\n"
                         f"({n_llm} LLM / {len(origins)} total)", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

    pct_label = inject_pct.replace("pct", "%")
    if is_structural:
        n_feat = args.grid_size * args.grid_size + 4
        subtitle = f"({n_feat}D structural: {args.grid_size}x{args.grid_size} wall map + positions)"
    else:
        subtitle = f"({args.num_rollouts}-rollout behavioral embeddings from current agent)"
    plt.suptitle(f"Buffer t-SNE Evolution — {pct_label} injection\n{subtitle}",
                 fontsize=13, y=1.01)
    plt.tight_layout()

    mode_tag = "env" if is_structural else "behav"
    out_path = os.path.join(args.output_dir, f"tsne_{mode_tag}_{inject_pct}.png")
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
    parser.add_argument("--seed_num_rollouts", type=int, default=50,
                        help="Number of rollouts for seed original embeddings (higher for stability)")
    parser.add_argument("--cache_only", action="store_true",
                        help="Only plot cached embeddings, skip GCS downloads and computation")
    parser.add_argument("--mode", type=str, default="behavioral", choices=["behavioral", "structural"],
                        help="behavioral: 257D agent rollout embeddings (GPU). structural: env features (CPU)")
    parser.add_argument("--grid_size", type=int, default=13,
                        help="Maze grid size for structural features (default 13)")
    parser.add_argument("--show_difficulty", action="store_true",
                        help="Color points by SFL learnability (yellow=0, red=0.25) "
                             "instead of by origin type")
    parser.add_argument("--show_eval_levels", action="store_true",
                        help="Plot eval benchmark mazes (SixteenRooms, Labyrinth, etc.) "
                             "as diamond markers on each panel")
    parser.add_argument("--upload_gcs", action="store_true",
                        help="Upload cache and plots to GCS after completion")
    args = parser.parse_args()

    # Default cache to project_msc if not specified
    if args.cache_dir is None:
        mode_tag = "env" if args.mode == "structural" else "behavioral"
        args.cache_dir = f"/cs/student/project_msc/2025/csml/rhautier/embedding_caches/injection_{mode_tag}"

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

    # Upload to GCS
    if args.upload_gcs and (saved or args.cache_dir):
        print(f"\n=== Uploading to GCS ===")
        import glob
        mode_tag = "env" if args.mode == "structural" else "behavioral"
        gcs_base = f"llm-exp/embedding_caches/injection_{mode_tag}"
        bucket = _get_bucket()

        # Upload cache
        if args.cache_dir:
            cache_files = sorted(glob.glob(os.path.join(args.cache_dir, "*.npz")))
            for f in cache_files:
                bucket.blob(f"{gcs_base}/{os.path.basename(f)}").upload_from_filename(f)
            print(f"  Uploaded {len(cache_files)} cache files to gs://{GCS_BUCKET}/{gcs_base}/")

        # Upload plots
        for p in saved:
            fname = os.path.basename(p)
            bucket.blob(f"{gcs_base}/plots/{fname}").upload_from_filename(p)
            print(f"  Uploaded {fname}")


if __name__ == "__main__":
    main()
