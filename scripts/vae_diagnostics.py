#!/usr/bin/env python3
"""VAE Latent Space Diagnostics — PCA grid + benchmark encoding overlay.

Builds on audit_vae_latent_space.py results. Produces:
  1. pca_grid_mazes.png  — 4x4 grid of mazes sampled at regular PCA coordinates
  2. benchmark_pca.png   — benchmark levels encoded + overlaid on random sample PCA
  3. benchmark_mazes.png  — side-by-side: original benchmark vs VAE round-trip reconstruction

Requires: saved audit_data.npz from audit_vae_latent_space.py (or generates fresh).
Runs on CPU.

Usage:
    python scripts/vae_diagnostics.py [--audit_dir audit_results/] [--output diag_results/]
"""

import os
import sys
import argparse
import pickle
import time
import numpy as np

# Force CPU
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

# Add paths — accel_training must come BEFORE es_legacy so its vae_decoder.py
# (which supports checkpoint_final.pkl format B) is imported instead of the
# es_legacy version (which only supports format A).
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.join(_script_dir, "..")
sys.path.insert(0, os.path.join(_project_root, "es_legacy"))
sys.path.insert(0, os.path.join(_project_root, "accel_training"))
sys.path.insert(0, _project_root)

from flax import linen as nn

from vae_decoder import load_vae_params, extract_decoder_params, decode_latent_to_env, repair_cluttr_sequence
from env_bridge import level_to_cluttr_sequence
from jaxued.environments.maze.level import Level, prefabs


# ── Encoder: remap checkpoint_final.pkl (Format B) to auto-numbered keys ─────
#
# The encoder architecture uses auto-numbered Flax modules:
#   Embed_0, HighwayStage_0, HighwayStage_1, LSTMCell_0, LSTMCell_1, mean_layer
# But checkpoint_final.pkl stores named keys:
#   embed, enc_hw1, enc_hw2, enc_bilstm/{forward_rnn,backward_rnn}/cell, mean_layer
#
# HighwayStage Dense numbering (by Python evaluation order in @nn.compact):
#   Dense_0 = dense_g, Dense_1 = dense_fg,
#   Dense_2 = dense_q2 (outer), Dense_3 = dense_q1 (inner), Dense_4 = dense_gate

class HighwayStage(nn.Module):
    dim: int = 300
    @nn.compact
    def __call__(self, x):
        g = nn.Dense(self.dim)(x)
        g = nn.relu(g)
        f_g_x = nn.relu(nn.Dense(self.dim)(g))
        q_x = nn.Dense(self.dim)(nn.relu(nn.Dense(self.dim)(x)))
        gate = nn.sigmoid(nn.Dense(self.dim)(x))
        return gate * f_g_x + (1.0 - gate) * q_x


class CluttrEncoderV2(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Embed(170, 300)(x)
        x = HighwayStage(300)(x)
        x = HighwayStage(300)(x)
        outputs = nn.Bidirectional(
            nn.RNN(nn.LSTMCell(300)),
            nn.RNN(nn.LSTMCell(300)),
        )(x)
        h = outputs[:, -1, :]
        mean = nn.Dense(64, name='mean_layer')(h)
        return jnp.tanh(mean) * 4.0


# Dense key mapping: checkpoint named → auto-numbered
_HW_DENSE_MAP = {
    'dense_g': 'Dense_0',
    'dense_fg': 'Dense_1',
    'dense_q2': 'Dense_2',   # outer q (created 3rd by Python eval order)
    'dense_q1': 'Dense_3',   # inner q (created 4th)
    'dense_gate': 'Dense_4',
}


def _remap_highway(hw_params: dict) -> dict:
    """Remap named HighwayStage Dense keys to auto-numbered."""
    return {_HW_DENSE_MAP[k]: v for k, v in hw_params.items()}


def extract_encoder_params_b(full_params: dict) -> dict:
    """Remap Format B checkpoint encoder params to auto-numbered keys.

    Format B (checkpoint_final.pkl):
        embed, enc_hw1, enc_hw2, enc_bilstm/{forward,backward}_rnn/cell, mean_layer
    Auto-numbered (what Flax expects):
        Embed_0, HighwayStage_0, HighwayStage_1, LSTMCell_0, LSTMCell_1, mean_layer
    """
    return {
        'Embed_0': full_params['embed'],
        'HighwayStage_0': _remap_highway(full_params['enc_hw1']),
        'HighwayStage_1': _remap_highway(full_params['enc_hw2']),
        'LSTMCell_0': full_params['enc_bilstm']['forward_rnn']['cell'],
        'LSTMCell_1': full_params['enc_bilstm']['backward_rnn']['cell'],
        'mean_layer': full_params['mean_layer'],
    }


def encode_levels_to_latents_b(encoder_params: dict, sequences: jnp.ndarray) -> jnp.ndarray:
    """Encode CLUTTR sequences (batch, 52) to latent vectors (batch, 64)."""
    return CluttrEncoderV2().apply({'params': encoder_params}, sequences)


# ── Helpers ──────────────────────────────────────────────────────────────────

def pca_np(X, n_components=2):
    """PCA via numpy SVD. Returns projected coords and (Vt, mean) for inverse."""
    mean = X.mean(axis=0)
    X_c = X - mean
    _, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    explained_var = (S ** 2) / (X.shape[0] - 1)
    explained_ratio = explained_var / explained_var.sum()
    coords = X_c @ Vt[:n_components].T
    return coords, explained_ratio[:n_components], Vt[:n_components], mean


def seq_to_maze_img(seq_np, H=13, W=13):
    """Convert a 52-element CLUTTR sequence to a 13x13x3 RGB image."""
    obstacles = seq_np[:50]
    goal_idx, agent_idx = int(seq_np[50]), int(seq_np[51])
    wm = np.zeros((H, W), dtype=bool)
    for ob in obstacles:
        if ob > 0:
            r, c = divmod(int(ob) - 1, W)
            if 0 <= r < H and 0 <= c < W:
                wm[r, c] = True
    gc, gr = (goal_idx - 1) % W, (goal_idx - 1) // W
    ac, ar = (agent_idx - 1) % W, (agent_idx - 1) // W
    if 0 <= gr < H and 0 <= gc < W:
        wm[gr, gc] = False
    if 0 <= ar < H and 0 <= ac < W:
        wm[ar, ac] = False
    img = np.ones((H, W, 3))
    img[wm] = [0.2, 0.2, 0.2]  # walls: dark gray
    if 0 <= ar < H and 0 <= ac < W:
        img[ar, ac] = [0.2, 0.4, 1.0]  # agent: blue
    if 0 <= gr < H and 0 <= gc < W:
        img[gr, gc] = [0.2, 0.8, 0.2]  # goal: green
    return img


def level_to_maze_img(wall_map_np, goal_pos_np, agent_pos_np, H=13, W=13):
    """Convert Level arrays to a 13x13x3 RGB image."""
    img = np.ones((H, W, 3))
    img[wall_map_np.astype(bool)] = [0.2, 0.2, 0.2]
    gc, gr = int(goal_pos_np[0]), int(goal_pos_np[1])
    ac, ar = int(agent_pos_np[0]), int(agent_pos_np[1])
    if 0 <= ar < H and 0 <= ac < W:
        img[ar, ac] = [0.2, 0.4, 1.0]
    if 0 <= gr < H and 0 <= gc < W:
        img[gr, gc] = [0.2, 0.8, 0.2]
    return img


def compute_shortest_path(wall_map, agent_pos, goal_pos):
    """BFS shortest path. Returns -1 if unsolvable."""
    H, W = wall_map.shape
    start = (int(agent_pos[1]), int(agent_pos[0]))
    goal = (int(goal_pos[1]), int(goal_pos[0]))
    if start == goal:
        return 0
    visited = {start}
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
    return -1


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VAE Latent Space Diagnostics")
    parser.add_argument("--audit_dir", type=str, default="audit_results",
                        help="Dir with audit_data.npz from audit_vae_latent_space.py")
    parser.add_argument("--output", type=str, default="audit_results",
                        help="Output directory for diagnostic plots")
    parser.add_argument("--vae_checkpoint", type=str, default="vae/model/checkpoint_final.pkl")
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--grid_size", type=int, default=4, help="PCA grid dimension (4 = 4x4)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── Load audit data ──────────────────────────────────────────────────────
    npz_path = os.path.join(args.audit_dir, "audit_data.npz")
    if not os.path.exists(npz_path):
        print(f"ERROR: {npz_path} not found. Run audit_vae_latent_space.py first.")
        sys.exit(1)

    print(f"Loading audit data from {npz_path}...")
    data = np.load(npz_path)
    z_all = data["z"]           # (N, 64)
    seqs_all = data["sequences"]  # (N, 52)
    valid = data["valid"]       # (N,) bool
    shortest_paths = data["shortest_paths"]  # (N,) int
    N = len(z_all)
    print(f"  Loaded {N} samples, {valid.sum()} valid")

    # ── Load VAE params ──────────────────────────────────────────────────────
    print(f"Loading VAE from {args.vae_checkpoint}...")
    full_params = load_vae_params(args.vae_checkpoint)
    decoder_params = extract_decoder_params(full_params)
    encoder_params = extract_encoder_params_b(full_params)
    print("  Loaded encoder + decoder.")

    # ── PCA ───────────────────────────────────────────────────────────────────
    print("Computing PCA...")
    z_pca, explained, Vt, z_mean = pca_np(z_all, 2)
    print(f"  PC1: {explained[0]*100:.1f}%, PC2: {explained[1]*100:.1f}%")

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. PCA GRID: decode mazes at regular grid points across PC1 x PC2
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n--- PCA Grid ({args.grid_size}x{args.grid_size}) ---")

    # Use percentile range of random samples (5th to 95th) for grid extent
    pc1_lo, pc1_hi = np.percentile(z_pca[:, 0], 5), np.percentile(z_pca[:, 0], 95)
    pc2_lo, pc2_hi = np.percentile(z_pca[:, 1], 5), np.percentile(z_pca[:, 1], 95)
    G = args.grid_size
    pc1_vals = np.linspace(pc1_lo, pc1_hi, G)
    pc2_vals = np.linspace(pc2_lo, pc2_hi, G)

    # Reconstruct 64-dim latent vectors from PCA grid coordinates
    # z = mean + pc1_coord * V[0] + pc2_coord * V[1]
    grid_z = []
    grid_labels = []
    for i, pc2 in enumerate(reversed(pc2_vals)):  # top row = high PC2
        for j, pc1 in enumerate(pc1_vals):
            z_64 = z_mean + pc1 * Vt[0] + pc2 * Vt[1]
            grid_z.append(z_64)
            grid_labels.append(f"({pc1:.1f}, {pc2:.1f})")

    grid_z = np.array(grid_z)  # (G*G, 64)
    print(f"  Decoding {G*G} grid points...")

    # Decode through VAE
    rng = jax.random.PRNGKey(args.seed)
    rng, rng_dec = jax.random.split(rng)
    grid_seqs_raw = decode_latent_to_env(decoder_params, jnp.array(grid_z),
                                          rng_key=rng_dec, temperature=args.temperature)
    grid_seqs = []
    for k in range(G * G):
        grid_seqs.append(np.asarray(repair_cluttr_sequence(grid_seqs_raw[k])))
    grid_seqs = np.array(grid_seqs)

    # Analyze each grid maze
    grid_paths = []
    grid_valid = []
    for k in range(G * G):
        seq = grid_seqs[k]
        obstacles = seq[:50]
        goal_idx, agent_idx = int(seq[50]), int(seq[51])
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
        sp = compute_shortest_path(wm, np.array([ac, ar]), np.array([gc, gr]))
        obs_count = int((obstacles > 0).sum())
        manhattan = abs(ac - gc) + abs(ar - gr)
        is_valid = sp >= 0 and obs_count >= 5 and manhattan >= 3
        grid_paths.append(sp)
        grid_valid.append(is_valid)

    # Plot PCA grid
    fig, axes = plt.subplots(G, G, figsize=(3 * G, 3 * G))
    fig.suptitle(f"VAE Decoded Mazes: {G}x{G} PCA Grid\n"
                 f"PC1 [{pc1_lo:.1f}, {pc1_hi:.1f}] x PC2 [{pc2_lo:.1f}, {pc2_hi:.1f}]  "
                 f"(5th-95th percentile range)",
                 fontsize=13, y=1.02)

    for k in range(G * G):
        row, col = k // G, k % G
        ax = axes[row, col] if G > 1 else axes
        img = seq_to_maze_img(grid_seqs[k])
        ax.imshow(img, interpolation='nearest')
        sp = grid_paths[k]
        v = grid_valid[k]
        color = 'green' if v else 'red'
        sp_str = str(sp) if sp >= 0 else "X"
        ax.set_title(f"path={sp_str}", fontsize=9, color=color)
        ax.set_xlabel(grid_labels[k], fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        # Add border color
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    # Label axes on edges
    if G > 1:
        for i in range(G):
            axes[i, 0].set_ylabel(f"PC2={pc2_vals[G - 1 - i]:.1f}", fontsize=8)
        for j in range(G):
            axes[-1, j].set_xlabel(f"PC1={pc1_vals[j]:.1f}", fontsize=8)

    fig.tight_layout()
    path_out = os.path.join(args.output, "pca_grid_mazes.png")
    fig.savefig(path_out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path_out}")

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. BENCHMARK LEVELS: encode through VAE, project to PCA, overlay
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n--- Benchmark Level Encoding ---")

    # The 8 DCD benchmark levels used for eval
    benchmark_names = [
        "SixteenRooms", "SixteenRooms2",
        "Labyrinth", "LabyrinthFlipped", "Labyrinth2",
        "StandardMaze", "StandardMaze2", "StandardMaze3",
    ]

    # Load levels and convert to CLUTTR sequences
    benchmark_levels = []
    benchmark_seqs = []
    benchmark_sp = []  # shortest paths of originals

    for name in benchmark_names:
        level = Level.from_str(prefabs[name])
        benchmark_levels.append(level)

        # Convert to CLUTTR sequence
        seq = level_to_cluttr_sequence(level.wall_map, level.goal_pos, level.agent_pos)
        benchmark_seqs.append(np.asarray(seq))

        # Shortest path of original
        wm_np = np.asarray(level.wall_map)
        ap = np.array([int(level.agent_pos[0]), int(level.agent_pos[1])])
        gp = np.array([int(level.goal_pos[0]), int(level.goal_pos[1])])
        sp = compute_shortest_path(wm_np, ap, gp)
        benchmark_sp.append(sp)
        print(f"  {name}: shortest_path={sp}")

    benchmark_seqs_arr = jnp.array(np.array(benchmark_seqs))  # (8, 52)

    # Encode through VAE encoder
    print("  Encoding benchmark levels through VAE encoder...")
    z_benchmark = np.asarray(encode_levels_to_latents_b(encoder_params, benchmark_seqs_arr))  # (8, 64)
    print(f"  Encoded to {z_benchmark.shape} latent vectors")

    # Project to PCA space using same basis
    z_bm_pca = (z_benchmark - z_mean) @ Vt.T  # (8, 2)
    print(f"  Benchmark PCA coords:")
    for i, name in enumerate(benchmark_names):
        print(f"    {name:20s}: PC1={z_bm_pca[i,0]:+6.2f}, PC2={z_bm_pca[i,1]:+6.2f}, "
              f"||z||={np.linalg.norm(z_benchmark[i]):.2f}")

    # Round-trip: decode benchmark latents back through decoder
    print("  Decoding benchmark latents (round-trip)...")
    rng, rng_dec2 = jax.random.split(rng)
    recon_seqs_raw = decode_latent_to_env(decoder_params, jnp.array(z_benchmark),
                                           rng_key=rng_dec2, temperature=args.temperature)
    recon_seqs = []
    recon_sp = []
    for k in range(len(benchmark_names)):
        seq_rep = np.asarray(repair_cluttr_sequence(recon_seqs_raw[k]))
        recon_seqs.append(seq_rep)
        # Compute shortest path of reconstruction
        obs = seq_rep[:50]
        gi, ai = int(seq_rep[50]), int(seq_rep[51])
        wm = np.zeros((13, 13), dtype=bool)
        for ob in obs:
            if ob > 0:
                r, c = divmod(int(ob) - 1, 13)
                if 0 <= r < 13 and 0 <= c < 13:
                    wm[r, c] = True
        gc, gr = (gi - 1) % 13, (gi - 1) // 13
        ac, ar = (ai - 1) % 13, (ai - 1) // 13
        if 0 <= gr < 13 and 0 <= gc < 13:
            wm[gr, gc] = False
        if 0 <= ar < 13 and 0 <= ac < 13:
            wm[ar, ac] = False
        sp = compute_shortest_path(wm, np.array([ac, ar]), np.array([gc, gr]))
        recon_sp.append(sp)

    print(f"  Round-trip shortest paths:")
    for i, name in enumerate(benchmark_names):
        orig = benchmark_sp[i]
        recon = recon_sp[i]
        ratio = recon / orig if orig > 0 and recon >= 0 else float('nan')
        status = "OK" if recon >= 0 else "BROKEN"
        print(f"    {name:20s}: orig={orig:3d} -> recon={recon:3d} "
              f"(ratio={ratio:.2f}) [{status}]")

    # ── Plot 2a: PCA scatter with benchmark overlay ──────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Background: random samples colored by validity
    ax.scatter(z_pca[~valid, 0], z_pca[~valid, 1], c='lightcoral', alpha=0.08, s=2, label='Invalid', zorder=1)
    ax.scatter(z_pca[valid, 0], z_pca[valid, 1], c='steelblue', alpha=0.12, s=2, label='Valid', zorder=2)

    # Overlay: benchmark levels as large markers
    colors_bm = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
                  '#ff7f00', '#a65628', '#f781bf', '#999999']
    for i, name in enumerate(benchmark_names):
        ax.scatter(z_bm_pca[i, 0], z_bm_pca[i, 1],
                   c=colors_bm[i], s=200, marker='*', edgecolors='black',
                   linewidths=0.8, zorder=10, label=f"{name} (path={benchmark_sp[i]})")

    # PCA grid points
    grid_pca_coords = (grid_z - z_mean) @ Vt.T  # (G*G, 2)
    for k in range(G * G):
        color = 'green' if grid_valid[k] else 'red'
        ax.plot(grid_pca_coords[k, 0], grid_pca_coords[k, 1],
                's', color=color, markersize=6, markeredgecolor='black',
                markeredgewidth=0.5, zorder=5)

    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% var)", fontsize=12)
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% var)", fontsize=12)
    ax.set_title("VAE Latent PCA: Random Samples + Benchmark Levels (stars)\n"
                 "Grid squares = decoded PCA grid points (green=valid, red=invalid)", fontsize=11)
    ax.legend(fontsize=7, loc='upper right', ncol=2, markerscale=0.8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    path_out = os.path.join(args.output, "benchmark_pca.png")
    fig.savefig(path_out, dpi=150)
    plt.close(fig)
    print(f"  Saved {path_out}")

    # ── Plot 2b: Benchmark originals vs round-trip reconstructions ───────────
    n_bm = len(benchmark_names)
    fig, axes = plt.subplots(2, n_bm, figsize=(2.5 * n_bm, 6))
    fig.suptitle("Benchmark Levels: Original (top) vs VAE Round-Trip Reconstruction (bottom)",
                 fontsize=12, y=1.02)

    for i, name in enumerate(benchmark_names):
        # Original
        level = benchmark_levels[i]
        wm_np = np.asarray(level.wall_map)
        gp_np = np.array([int(level.goal_pos[0]), int(level.goal_pos[1])])
        ap_np = np.array([int(level.agent_pos[0]), int(level.agent_pos[1])])
        img_orig = level_to_maze_img(wm_np, gp_np, ap_np)
        axes[0, i].imshow(img_orig, interpolation='nearest')
        axes[0, i].set_title(f"{name}\npath={benchmark_sp[i]}", fontsize=7)
        axes[0, i].axis('off')

        # Reconstruction
        img_recon = seq_to_maze_img(recon_seqs[i])
        recon_color = 'green' if recon_sp[i] >= 0 else 'red'
        sp_str = str(recon_sp[i]) if recon_sp[i] >= 0 else "X"
        axes[1, i].imshow(img_recon, interpolation='nearest')
        axes[1, i].set_title(f"recon path={sp_str}", fontsize=7, color=recon_color)
        axes[1, i].axis('off')
        for spine in axes[1, i].spines.values():
            spine.set_edgecolor(recon_color)
            spine.set_linewidth(2)

    axes[0, 0].set_ylabel("Original", fontsize=10)
    axes[1, 0].set_ylabel("Reconstruction", fontsize=10)
    fig.tight_layout()
    path_out = os.path.join(args.output, "benchmark_mazes.png")
    fig.savefig(path_out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path_out}")

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. LATENT NORM ANALYSIS: benchmark vs random
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n--- Latent Space Statistics ---")
    z_norms_random = np.linalg.norm(z_all, axis=1)
    z_norms_bm = np.linalg.norm(z_benchmark, axis=1)
    print(f"  Random samples ||z||: mean={z_norms_random.mean():.2f}, "
          f"std={z_norms_random.std():.2f}, range=[{z_norms_random.min():.2f}, {z_norms_random.max():.2f}]")
    print(f"  Benchmark      ||z||: mean={z_norms_bm.mean():.2f}, "
          f"std={z_norms_bm.std():.2f}, range=[{z_norms_bm.min():.2f}, {z_norms_bm.max():.2f}]")

    # Are benchmarks inside or outside the random sample distribution?
    for i, name in enumerate(benchmark_names):
        z_norm = z_norms_bm[i]
        percentile = (z_norms_random < z_norm).mean() * 100
        print(f"    {name:20s}: ||z||={z_norm:.2f} (percentile {percentile:.0f}% of random)")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("KEY FINDINGS")
    print(f"{'='*60}")
    n_grid_valid = sum(grid_valid)
    print(f"  PCA grid: {n_grid_valid}/{G*G} points decode to valid mazes ({100*n_grid_valid/(G*G):.0f}%)")
    n_recon_solvable = sum(1 for sp in recon_sp if sp >= 0)
    print(f"  Benchmark round-trip: {n_recon_solvable}/{n_bm} reconstruct to solvable mazes")
    mean_ratio = np.mean([recon_sp[i] / benchmark_sp[i]
                          for i in range(n_bm) if benchmark_sp[i] > 0 and recon_sp[i] >= 0])
    print(f"  Mean path ratio (recon/orig): {mean_ratio:.2f}")
    if mean_ratio < 0.5:
        print("  ⚠ VAE reconstructions are significantly easier than originals!")
    bm_outside = sum(1 for n in z_norms_bm if n > np.percentile(z_norms_random, 95))
    print(f"  Benchmark levels outside 95th percentile of random norms: {bm_outside}/{n_bm}")

    print(f"\nPlots saved to {args.output}/")
    print("Done.")


if __name__ == "__main__":
    main()
