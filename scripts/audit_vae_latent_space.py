#!/usr/bin/env python3
"""VAE Latent Space Audit — runs on CPU (no GPU needed).

Samples N random latent vectors, decodes through VAE, and measures:
  1. Valid fraction (solvable + complex enough)
  2. Solvable fraction (flood fill reachable)
  3. Complex fraction (min_obstacles >= 5, min_distance >= 3)
  4. Structural diversity (unique wall patterns, obstacle count distribution)
  5. Shortest path length distribution (proxy for difficulty)
  6. Latent region analysis: where do valid/invalid levels cluster?

Usage:
    python scripts/audit_vae_latent_space.py [--n_samples 10000] [--output audit_results/]
"""

import os
import sys
import argparse
import pickle
import time
import numpy as np
from collections import Counter

# Force CPU — this script doesn't need GPU
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

# Add accel_training to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "accel_training"))

from vae_decoder import load_vae_params, extract_decoder_params, decode_latent_to_env, repair_cluttr_sequence
from env_bridge import cluttr_sequence_to_level, flood_fill_solvable


def compute_shortest_path(wall_map, agent_pos, goal_pos):
    """BFS shortest path length. Returns -1 if unsolvable."""
    H, W = wall_map.shape
    # Convert from Level convention [x,y]=[col,row] to grid [row,col]
    start = (int(agent_pos[1]), int(agent_pos[0]))
    goal = (int(goal_pos[1]), int(goal_pos[0]))

    if start == goal:
        return 0

    visited = set()
    visited.add(start)
    queue = [(start, 0)]
    head = 0

    while head < len(queue):
        (r, c), dist = queue[head]
        head += 1
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited and not wall_map[nr, nc]:
                if (nr, nc) == goal:
                    return dist + 1
                visited.add((nr, nc))
                queue.append(((nr, nc), dist + 1))
    return -1  # unsolvable


def main():
    parser = argparse.ArgumentParser(description="VAE Latent Space Audit")
    parser.add_argument("--n_samples", type=int, default=10000, help="Number of random latent vectors")
    parser.add_argument("--output", type=str, default="audit_results", help="Output directory")
    parser.add_argument("--vae_checkpoint", type=str, default="vae/model/checkpoint_final.pkl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.25, help="VAE decode temperature")
    parser.add_argument("--batch_size", type=int, default=500, help="Decode batch size")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    rng = jax.random.PRNGKey(args.seed)
    np_rng = np.random.default_rng(args.seed)
    N = args.n_samples

    print(f"VAE Latent Space Audit")
    print(f"  Samples: {N}")
    print(f"  Checkpoint: {args.vae_checkpoint}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Output: {args.output}/")
    print(f"  Running on CPU (no GPU needed)")
    print("=" * 60)

    # --- Load VAE decoder ---
    print("\nLoading VAE decoder...")
    full_params = load_vae_params(args.vae_checkpoint)
    decoder_params = extract_decoder_params(full_params)
    print("  Decoder loaded.")

    # --- Sample latent vectors ---
    print(f"\nSampling {N} latent vectors from N(0, I)...")
    rng, rng_z = jax.random.split(rng)
    all_z = jax.random.normal(rng_z, (N, 64))

    # --- Decode in batches ---
    print(f"Decoding (batch_size={args.batch_size})...")
    all_sequences = []
    t0 = time.time()

    for i in range(0, N, args.batch_size):
        batch_z = all_z[i:i + args.batch_size]
        rng, rng_dec = jax.random.split(rng)
        raw_seqs = decode_latent_to_env(decoder_params, batch_z, rng_key=rng_dec,
                                        temperature=args.temperature)
        # Repair each sequence
        for j in range(batch_z.shape[0]):
            repaired = repair_cluttr_sequence(raw_seqs[j])
            all_sequences.append(np.asarray(repaired))

        if (i + args.batch_size) % 2000 == 0 or i + args.batch_size >= N:
            elapsed = time.time() - t0
            done = min(i + args.batch_size, N)
            print(f"  Decoded {done}/{N} ({elapsed:.1f}s)")

    all_sequences = np.array(all_sequences)  # (N, 52)
    print(f"  Decoding done in {time.time() - t0:.1f}s")

    # --- Analyze each level (pure numpy — no JAX in the loop) ---
    print(f"\nAnalyzing {N} levels (pure numpy, no JAX recompilation)...")
    t0 = time.time()

    solvable = np.zeros(N, dtype=bool)
    complex_enough = np.zeros(N, dtype=bool)
    valid = np.zeros(N, dtype=bool)
    n_obstacles = np.zeros(N, dtype=int)
    agent_goal_dist = np.zeros(N, dtype=int)
    shortest_paths = np.full(N, -1, dtype=int)
    wall_hashes = []
    latent_norms = np.linalg.norm(np.asarray(all_z), axis=1)

    H, W = 13, 13

    for i in range(N):
        seq = all_sequences[i]

        # --- Build wall_map, agent_pos, goal_pos in pure numpy ---
        obstacles = seq[:50]
        goal_idx = int(seq[50])
        agent_idx = int(seq[51])

        # Obstacle count
        obs_mask = obstacles > 0
        obs_count = int(obs_mask.sum())
        n_obstacles[i] = obs_count

        # Build wall map
        wall_map = np.zeros((H, W), dtype=bool)
        for ob in obstacles:
            if ob > 0:
                r, c = divmod(int(ob) - 1, W)
                if 0 <= r < H and 0 <= c < W:
                    wall_map[r, c] = True

        # Agent / goal positions ([col, row] convention)
        goal_col, goal_row = (goal_idx - 1) % W, (goal_idx - 1) // W
        agent_col, agent_row = (agent_idx - 1) % W, (agent_idx - 1) // W

        # Clear walls at agent/goal
        if 0 <= goal_row < H and 0 <= goal_col < W:
            wall_map[goal_row, goal_col] = False
        if 0 <= agent_row < H and 0 <= agent_col < W:
            wall_map[agent_row, agent_col] = False

        # Manhattan distance
        manhattan = abs(agent_col - goal_col) + abs(agent_row - goal_row)
        agent_goal_dist[i] = manhattan

        # Complexity check
        complex_enough[i] = obs_count >= 5 and manhattan >= 3

        # Solvability + shortest path via BFS (pure python)
        agent_pos_arr = np.array([agent_col, agent_row])
        goal_pos_arr = np.array([goal_col, goal_row])
        sp = compute_shortest_path(wall_map, agent_pos_arr, goal_pos_arr)
        is_solvable = sp >= 0
        solvable[i] = is_solvable
        if is_solvable:
            shortest_paths[i] = sp

        valid[i] = is_solvable and complex_enough[i]

        # Wall pattern hash for uniqueness
        wall_hashes.append(wall_map.tobytes())

        if (i + 1) % 2000 == 0 or i == N - 1:
            elapsed = time.time() - t0
            print(f"  Analyzed {i+1}/{N} ({elapsed:.1f}s)")

    # --- Compute stats ---
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    n_solvable = solvable.sum()
    n_complex = complex_enough.sum()
    n_valid = valid.sum()
    n_unique = len(set(wall_hashes))
    n_unique_valid = len(set(h for h, v in zip(wall_hashes, valid) if v))

    print(f"\n--- Validity Breakdown ({N} samples) ---")
    print(f"  Solvable:       {n_solvable:6d} ({100*n_solvable/N:.1f}%)")
    print(f"  Complex enough: {n_complex:6d} ({100*n_complex/N:.1f}%)")
    print(f"  Valid (both):   {n_valid:6d} ({100*n_valid/N:.1f}%)")
    print(f"  Invalid:        {N - n_valid:6d} ({100*(N-n_valid)/N:.1f}%)")

    print(f"\n--- Structural Diversity ---")
    print(f"  Unique wall patterns (all):   {n_unique:6d} / {N} ({100*n_unique/N:.1f}%)")
    print(f"  Unique wall patterns (valid): {n_unique_valid:6d} / {n_valid} ({100*n_unique_valid/max(n_valid,1):.1f}%)")

    print(f"\n--- Obstacle Count Distribution ---")
    for label, mask in [("All", np.ones(N, bool)), ("Valid", valid), ("Invalid", ~valid)]:
        subset = n_obstacles[mask]
        if len(subset) > 0:
            print(f"  {label:8s}: mean={subset.mean():.1f}, median={np.median(subset):.0f}, "
                  f"min={subset.min()}, max={subset.max()}, std={subset.std():.1f}")

    print(f"\n--- Manhattan Distance (Agent-Goal) ---")
    for label, mask in [("All", np.ones(N, bool)), ("Valid", valid)]:
        subset = agent_goal_dist[mask]
        if len(subset) > 0:
            print(f"  {label:8s}: mean={subset.mean():.1f}, median={np.median(subset):.0f}, "
                  f"min={subset.min()}, max={subset.max()}")

    print(f"\n--- Shortest Path Length (solvable only) ---")
    sp_valid = shortest_paths[solvable & (shortest_paths >= 0)]
    if len(sp_valid) > 0:
        print(f"  N={len(sp_valid)}")
        print(f"  mean={sp_valid.mean():.1f}, median={np.median(sp_valid):.0f}")
        print(f"  min={sp_valid.min()}, max={sp_valid.max()}, std={sp_valid.std():.1f}")
        # Distribution buckets
        buckets = [(0, 5, "trivial"), (5, 15, "easy"), (15, 30, "medium"),
                   (30, 50, "hard"), (50, 999, "very hard")]
        for lo, hi, label in buckets:
            count = ((sp_valid >= lo) & (sp_valid < hi)).sum()
            print(f"    {label:10s} (path {lo:2d}-{hi:2d}): {count:5d} ({100*count/len(sp_valid):.1f}%)")

    print(f"\n--- Latent Norm vs Validity ---")
    for label, mask in [("Valid", valid), ("Invalid", ~valid)]:
        subset = latent_norms[mask]
        if len(subset) > 0:
            print(f"  {label:8s}: mean_norm={subset.mean():.2f}, std={subset.std():.2f}")

    # --- Failure mode analysis ---
    unsolvable_complex = (~solvable) & complex_enough
    solvable_simple = solvable & (~complex_enough)
    print(f"\n--- Failure Modes ---")
    print(f"  Unsolvable but complex: {unsolvable_complex.sum():5d} ({100*unsolvable_complex.sum()/N:.1f}%) — walls block all paths")
    print(f"  Solvable but too simple:{solvable_simple.sum():5d} ({100*solvable_simple.sum()/N:.1f}%) — trivial mazes")
    print(f"  Unsolvable and simple:  {((~solvable) & (~complex_enough)).sum():5d}")

    # --- Save raw data ---
    np.savez_compressed(
        os.path.join(args.output, "audit_data.npz"),
        z=np.asarray(all_z),
        sequences=all_sequences,
        solvable=solvable,
        complex_enough=complex_enough,
        valid=valid,
        n_obstacles=n_obstacles,
        agent_goal_dist=agent_goal_dist,
        shortest_paths=shortest_paths,
        latent_norms=latent_norms,
    )
    print(f"\nRaw data saved to {args.output}/audit_data.npz")

    # --- Save summary ---
    summary = {
        "n_samples": N,
        "temperature": args.temperature,
        "vae_checkpoint": args.vae_checkpoint,
        "pct_solvable": 100 * n_solvable / N,
        "pct_complex": 100 * n_complex / N,
        "pct_valid": 100 * n_valid / N,
        "n_unique_valid": n_unique_valid,
        "mean_shortest_path": float(sp_valid.mean()) if len(sp_valid) > 0 else 0,
        "median_shortest_path": float(np.median(sp_valid)) if len(sp_valid) > 0 else 0,
        "mean_obstacles": float(n_obstacles[valid].mean()) if n_valid > 0 else 0,
    }
    with open(os.path.join(args.output, "summary.txt"), "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
    print(f"Summary saved to {args.output}/summary.txt")

    # --- Plots ---
    print("\nGenerating plots...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # PCA via numpy SVD (no sklearn needed)
    def pca_np(X, n_components=2):
        X_c = X - X.mean(axis=0)
        _, S, Vt = np.linalg.svd(X_c, full_matrices=False)
        explained_var = (S ** 2) / (X.shape[0] - 1)
        explained_ratio = explained_var / explained_var.sum()
        return X_c @ Vt[:n_components].T, explained_ratio[:n_components]

    # 1. Sample maze grids: 5 valid (by difficulty) + 5 invalid
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    fig.suptitle("VAE-Decoded Mazes: Valid (top) vs Invalid (bottom)", fontsize=14)

    def seq_to_maze_img(seq_np):
        """Convert a 52-element sequence to a 13x13x3 RGB image (pure numpy)."""
        obstacles = seq_np[:50]
        goal_idx, agent_idx = int(seq_np[50]), int(seq_np[51])
        wm = np.zeros((13, 13), dtype=bool)
        for ob in obstacles:
            if ob > 0:
                r, c = divmod(int(ob) - 1, 13)
                if 0 <= r < 13 and 0 <= c < 13:
                    wm[r, c] = True
        gc, gr = (goal_idx - 1) % 13, (goal_idx - 1) // 13
        ac, ar = (agent_idx - 1) % 13, (agent_idx - 1) // 13
        if 0 <= gr < 13 and 0 <= gc < 13:
            wm[gr, gc] = False
        if 0 <= ar < 13 and 0 <= ac < 13:
            wm[ar, ac] = False
        img = np.ones((13, 13, 3))
        img[wm] = [0.2, 0.2, 0.2]
        if 0 <= ar < 13 and 0 <= ac < 13:
            img[ar, ac] = [0.2, 0.4, 1.0]
        if 0 <= gr < 13 and 0 <= gc < 13:
            img[gr, gc] = [0.2, 0.8, 0.2]
        return img

    # Top row: valid mazes sorted by shortest path (easy → hard)
    valid_idx = np.where(valid)[0]
    if len(valid_idx) > 0:
        sp_of_valid = shortest_paths[valid_idx]
        sorted_by_sp = valid_idx[np.argsort(sp_of_valid)]
        pick = np.linspace(0, len(sorted_by_sp) - 1, 5, dtype=int)
        for col, idx in enumerate(sorted_by_sp[pick]):
            img = seq_to_maze_img(all_sequences[idx])
            axes[0, col].imshow(img, interpolation='nearest')
            axes[0, col].set_title(f"path={shortest_paths[idx]}, obs={n_obstacles[idx]}", fontsize=8)
            axes[0, col].axis("off")
    axes[0, 0].set_ylabel("Valid", fontsize=11)

    # Bottom row: 5 random invalid mazes
    invalid_idx = np.where(~valid)[0]
    if len(invalid_idx) > 0:
        pick_inv = np_rng.choice(invalid_idx, size=min(5, len(invalid_idx)), replace=False)
        for col, idx in enumerate(pick_inv):
            img = seq_to_maze_img(all_sequences[idx])
            reason = "unsolvable" if not solvable[idx] else "too simple"
            axes[1, col].imshow(img, interpolation='nearest')
            axes[1, col].set_title(f"{reason}, obs={n_obstacles[idx]}", fontsize=8)
            axes[1, col].axis("off")
    axes[1, 0].set_ylabel("Invalid", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output, "sample_mazes.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {args.output}/sample_mazes.png")

    # 2. PCA of latent space colored by validity + shortest path
    z_np = np.asarray(all_z)
    z_pca, explained_ratio = pca_np(z_np, 2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 2a. Valid vs invalid
    axes[0].scatter(z_pca[~valid, 0], z_pca[~valid, 1], c='red', alpha=0.15, s=3, label='Invalid')
    axes[0].scatter(z_pca[valid, 0], z_pca[valid, 1], c='blue', alpha=0.25, s=3, label='Valid')
    axes[0].set_title(f"PCA: Valid ({n_valid/N*100:.0f}%) vs Invalid")
    axes[0].legend(markerscale=4)
    axes[0].set_xlabel(f"PC1 ({explained_ratio[0]*100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({explained_ratio[1]*100:.1f}%)")

    # 2b. Colored by shortest path (valid only)
    sp_plot = shortest_paths[valid].copy().astype(float)
    sc = axes[1].scatter(z_pca[valid, 0], z_pca[valid, 1], c=sp_plot, cmap='viridis',
                         alpha=0.4, s=5, vmin=0, vmax=min(50, sp_plot.max()))
    plt.colorbar(sc, ax=axes[1], label='Shortest path')
    axes[1].set_title("PCA: Valid levels colored by difficulty")
    axes[1].set_xlabel(f"PC1 ({explained_ratio[0]*100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({explained_ratio[1]*100:.1f}%)")

    # 2c. Colored by obstacle count
    sc2 = axes[2].scatter(z_pca[:, 0], z_pca[:, 1], c=n_obstacles, cmap='plasma',
                          alpha=0.2, s=3, vmin=0, vmax=30)
    plt.colorbar(sc2, ax=axes[2], label='Obstacle count')
    axes[2].set_title("PCA: All levels colored by obstacle count")
    axes[2].set_xlabel(f"PC1 ({explained_ratio[0]*100:.1f}%)")
    axes[2].set_ylabel(f"PC2 ({explained_ratio[1]*100:.1f}%)")

    fig.tight_layout()
    fig.savefig(os.path.join(args.output, "latent_pca.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {args.output}/latent_pca.png")

    # 3. Histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(shortest_paths[solvable & (shortest_paths >= 0)], bins=30, color='steelblue', edgecolor='white')
    axes[0].set_title("Shortest Path Distribution (solvable)")
    axes[0].set_xlabel("Path length")
    axes[0].axvline(x=np.median(sp_valid), color='red', linestyle='--', label=f'median={np.median(sp_valid):.0f}')
    axes[0].legend()

    axes[1].hist(n_obstacles[valid], bins=30, color='seagreen', edgecolor='white', alpha=0.7, label='Valid')
    axes[1].hist(n_obstacles[~valid], bins=30, color='salmon', edgecolor='white', alpha=0.5, label='Invalid')
    axes[1].set_title("Obstacle Count Distribution")
    axes[1].set_xlabel("Number of obstacles")
    axes[1].legend()

    axes[2].hist(latent_norms[valid], bins=30, color='steelblue', edgecolor='white', alpha=0.7, label='Valid')
    axes[2].hist(latent_norms[~valid], bins=30, color='salmon', edgecolor='white', alpha=0.5, label='Invalid')
    axes[2].set_title("Latent Norm Distribution")
    axes[2].set_xlabel("||z||")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(args.output, "histograms.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved {args.output}/histograms.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
