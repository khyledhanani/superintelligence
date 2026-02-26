"""
latent_similarity_probe.py
==========================
Diagnostic tool: do structurally similar mazes end up close together
in the VAE latent space, and does the VAE preserve structural similarity
through encode → decode?

Key questions answered
----------------------
1. LATENT GEOMETRY         — do similar mazes map to nearby z vectors?
                             (within-group L2 vs across-group L2, ratio)
2. SIMILARITY PRESERVATION — if maze A and B are similar going in,
                             are their reconstructions still similar coming out?
                             (input_sim vs recon_sim scatter, slope ≈ 1 = good)

Usage (notebook)
----------------
    import yaml, importlib
    import train_vae, utils
    importlib.reload(train_vae); importlib.reload(utils)
    import latent_similarity_probe as lsp
    importlib.reload(lsp)

    with open("vae_train_config.yml") as f:
        CONFIG = yaml.safe_load(f)

    results = lsp.run_probe(
        CONFIG,
        ckpt_path="/path/to/checkpoint_445000.pkl",
        n_groups=6, group_size=5,
        similarity_threshold=0.60,
        output_path="latent_similarity_probe.png"
    )
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import jax
import jax.numpy as jnp

from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Wall

# ── Constants ─────────────────────────────────────────────────────────────────
SEQ_LEN    = 169
LATENT_DIM = 128
_INNER_DIM = 13
_FULL_DIM  = 15
_TILE_SIZE = 16

_GROUP_COLOURS = [
    '#e74c3c', '#3498db', '#2ecc71', '#f39c12',
    '#9b59b6', '#1abc9c', '#e67e22', '#34495e'
]

# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_val_data(config: dict) -> np.ndarray:
    path = os.path.join(
        config['working_path'], config['vae_folder'],
        'datasets', config['validation_data_path']
    )
    data = np.load(path)
    print(f"[Data] Loaded val data: {data.shape}  from {path}")
    return data


def load_checkpoint_params(ckpt_path: str) -> dict:
    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)
    params = ckpt['params'] if 'params' in ckpt else ckpt
    print(f"[Checkpoint] Loaded from {ckpt_path}")
    return params


# ══════════════════════════════════════════════════════════════════════════════
# Encoding
# ══════════════════════════════════════════════════════════════════════════════

def encode_dataset(model, params, data: np.ndarray,
                   batch_size: int = 256, seed: int = 42):
    """Encode every maze -> posterior mean z and greedy reconstruction."""
    rng = jax.random.key(seed)

    @jax.jit
    def batch_encode(xs, rng_key):
        def fwd(x):
            _logits, mean, _logvar = model.apply(
                {'params': params}, x[jnp.newaxis, :], rng_key,
                train=False, rngs={'dropout': jax.random.key(0)}
            )
            return jnp.argmax(_logits[0], axis=-1), mean[0]
        return jax.vmap(fwd)(xs)

    all_means, all_recons = [], []
    data_jnp = jnp.array(data)
    for start in range(0, len(data), batch_size):
        batch = data_jnp[start:start + batch_size]
        rng, sub = jax.random.split(rng)
        recons, means = batch_encode(batch, sub)
        all_means.append(np.array(means))
        all_recons.append(np.array(recons))

    means  = np.concatenate(all_means,  axis=0)
    recons = np.concatenate(all_recons, axis=0)
    print(f"[Encode] {len(means)} mazes -> z {means.shape}")
    return means, recons


# ══════════════════════════════════════════════════════════════════════════════
# Similarity metrics
# ══════════════════════════════════════════════════════════════════════════════

def _build_sim_matrix(pool: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """
    Vectorised (N, N) pairwise similarity matrix.
    similarity = 0.6*Jaccard(walls) + 0.2*agent_match + 0.2*goal_match

    Jaccard over sequence positions (which slots are non-zero) — fast,
    fully vectorised, and matches the original mining behaviour exactly.
    """
    N         = len(pool)
    seqs      = pool.reshape(N, -1).astype(np.int32)[:, :seq_len]
    n_cells   = _INNER_DIM * _INNER_DIM   # 169

    # Scatter wall indices into a (N, 169) binary cell indicator — vectorised
    wall_vals = seqs[:, :-2]              # (N, seq_len-2) — wall index columns
    wall_bin  = np.zeros((N, n_cells), dtype=np.uint8)
    rows      = np.repeat(np.arange(N), wall_vals.shape[1])
    cols      = wall_vals.flatten()
    valid     = (cols > 0) & (cols <= n_cells)
    wall_bin[rows[valid], cols[valid] - 1] = 1   # 1-based -> 0-based

    wall_counts = wall_bin.sum(axis=1).astype(np.float32)
    inter       = (wall_bin @ wall_bin.T).astype(np.float32)
    union       = wall_counts[:, None] + wall_counts[None, :] - inter
    jaccard     = np.where(union > 0, inter / union, 1.0)

    agents      = seqs[:, -1]
    goals       = seqs[:, -2]
    agent_match = (agents[:, None] == agents[None, :]).astype(np.float32)
    goal_match  = (goals[:, None]  == goals[None, :]).astype(np.float32)

    return (0.6 * jaccard + 0.2 * agent_match + 0.2 * goal_match).astype(np.float32)
    #N    = len(pool)
    #seqs = pool.reshape(N, -1).astype(np.int32)

    #walls       = (seqs[:, :-2] != 0).astype(np.uint8)   # (N, SEQ_LEN-2)
    #wall_counts = walls.sum(axis=1).astype(np.float32)
    #inter       = (walls @ walls.T).astype(np.float32)
    #union       = wall_counts[:, None] + wall_counts[None, :] - inter
    #jaccard     = np.where(union > 0, inter / union, 1.0)

    #agents      = seqs[:, -1]
    #goals       = seqs[:, -2]
    #agent_match = (agents[:, None] == agents[None, :]).astype(np.float32)
    #goal_match  = (goals[:, None]  == goals[None, :]).astype(np.float32)

    #return (0.6 * jaccard + 0.2 * agent_match + 0.2 * goal_match).astype(np.float32)


def pairwise_l2(z: np.ndarray) -> np.ndarray:
    diff = z[:, None, :] - z[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


# ══════════════════════════════════════════════════════════════════════════════
# Group mining
# ══════════════════════════════════════════════════════════════════════════════

def mine_similarity_groups(
    data: np.ndarray,
    n_groups: int = 6,
    group_size: int = 5,
    similarity_threshold: float = 0.60,
    seed: int = 0,
    max_search: int = 2000
) -> list:
    """
    Find n_groups clusters of group_size mazes where every pair exceeds
    similarity_threshold. Builds the full NxN similarity matrix once
    (vectorised), then greedily picks groups.
    """
    rng      = np.random.default_rng(seed)
    pool_idx = rng.choice(len(data), min(max_search, len(data)), replace=False)
    pool     = data[pool_idx]

    print(f"[Mine] Building {len(pool)}×{len(pool)} similarity matrix ...",
          end=" ", flush=True)
    sim_mat = _build_sim_matrix(pool, SEQ_LEN)
    np.fill_diagonal(sim_mat, 0.0)
    print("done.")

    groups, used = [], set()
    for anchor_pos in rng.permutation(len(pool_idx)):
        if len(groups) >= n_groups:
            break
        if anchor_pos in used:
            continue
        row   = sim_mat[anchor_pos]
        above = np.where(row >= similarity_threshold)[0]
        above = above[np.argsort(-row[above])]
        group_pos = [anchor_pos]
        for j in above:
            if j not in used:
                group_pos.append(j)
            if len(group_pos) == group_size:
                break
        if len(group_pos) == group_size:
            used.update(group_pos)
            groups.append([int(pool_idx[p]) for p in group_pos])

    if len(groups) < n_groups:
        print(f"[Mine] Warning: only found {len(groups)}/{n_groups} groups "
              f"at threshold={similarity_threshold}. Try lowering it.")
    else:
        print(f"[Mine] Found {len(groups)} groups "
              f"(threshold={similarity_threshold}, size={group_size})")
    return groups


# ══════════════════════════════════════════════════════════════════════════════
# MiniGrid rendering
# ══════════════════════════════════════════════════════════════════════════════

def _get_pos(idx: int):
    y = (idx - 1) // _INNER_DIM + 1
    x = (idx - 1) % _INNER_DIM + 1
    return x, y


def seq_to_rgb(seq: np.ndarray) -> np.ndarray:
    s         = seq.flatten().astype(int)
    agent_idx = int(s[-1])
    goal_idx  = int(s[-2])
    grid = Grid(_FULL_DIM, _FULL_DIM)
    grid.wall_rect(0, 0, _FULL_DIM, _FULL_DIM)
    gx, gy = _get_pos(goal_idx)
    grid.set(gx, gy, Goal())
    for o_idx in s[:-2]:
        if o_idx > 0:
            ox, oy = _get_pos(o_idx)
            grid.set(ox, oy, Wall())
    ax_pos, ay_pos = _get_pos(agent_idx)
    return grid.render(_TILE_SIZE, agent_pos=(ax_pos, ay_pos), agent_dir=0)


def render_maze(ax, seq: np.ndarray, title: str = "", border_colour=None):
    try:
        rgb = seq_to_rgb(seq)
    except Exception:
        rgb = np.full((_FULL_DIM * _TILE_SIZE, _FULL_DIM * _TILE_SIZE, 3),
                      200, dtype=np.uint8)
    ax.imshow(rgb, interpolation='nearest', aspect='equal')
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7, pad=2)
    if border_colour is not None:
        for spine in ax.spines.values():
            spine.set_edgecolor(border_colour)
            spine.set_linewidth(2.5)


# ══════════════════════════════════════════════════════════════════════════════
# Statistics
# ══════════════════════════════════════════════════════════════════════════════

def compute_group_stats(groups, means, data, recons):
    within_l2s, within_input_sims, within_recon_sims = [], [], []
    l2_matrices, sim_in_matrices, sim_rc_matrices     = [], [], []
    all_pairs = []

    for g in groups:
        z_g   = means[g]
        d_mat = pairwise_l2(z_g)
        mask  = np.triu(np.ones(len(g), dtype=bool), k=1)
        within_l2s.append(float(d_mat[mask].mean()))
        l2_matrices.append(d_mat)

        sim_in = _build_sim_matrix(data[g], SEQ_LEN)
        sim_rc = _build_sim_matrix(recons[g], SEQ_LEN)
        sim_in_matrices.append(sim_in)
        sim_rc_matrices.append(sim_rc)
        within_input_sims.append(float(sim_in[mask].mean()))
        within_recon_sims.append(float(sim_rc[mask].mean()))

        K = len(g)
        for i in range(K):
            for j in range(i + 1, K):
                all_pairs.append((float(sim_in[i, j]), float(sim_rc[i, j])))

    # Across-group L2 baseline
    flat_idx     = [idx for g in groups for idx in g]
    group_labels = [gi for gi, g in enumerate(groups) for _ in g]
    z_all        = means[flat_idx]
    rng          = np.random.default_rng(99)
    across_l2s   = []
    for _ in range(500):
        i, j = rng.choice(len(flat_idx), 2, replace=False)
        if group_labels[i] != group_labels[j]:
            across_l2s.append(float(np.sqrt(((z_all[i] - z_all[j]) ** 2).sum())))

    return (within_l2s, within_input_sims, within_recon_sims,
            l2_matrices, sim_in_matrices, sim_rc_matrices,
            across_l2s, all_pairs)


# ══════════════════════════════════════════════════════════════════════════════
# Heatmap helper
# ══════════════════════════════════════════════════════════════════════════════

def _draw_heatmap(ax, mat, title, cmap, vmin, vmax, fontsize=6):
    im = ax.imshow(mat, cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
    if title:
        ax.set_title(title, fontsize=fontsize + 1, pad=2)
    K = len(mat)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels([f"M{k+1}" for k in range(K)], fontsize=fontsize)
    ax.set_yticklabels([f"M{k+1}" for k in range(K)], fontsize=fontsize)
    for i in range(K):
        for j in range(K):
            v   = float(mat[i, j])
            col = 'white' if v > vmax * 0.6 else 'black'
            ax.text(j, i, f"{v:.2f}", ha='center', va='center',
                    fontsize=fontsize, color=col)


# ══════════════════════════════════════════════════════════════════════════════
# Main figure  — ALL groups, each row has mazes + 3 heatmaps
# ══════════════════════════════════════════════════════════════════════════════

def plot_probe_results(
    groups, data, recons, means,
    within_l2s, within_input_sims, within_recon_sims,
    l2_matrices, sim_in_matrices, sim_rc_matrices,
    across_l2s, all_pairs,
    output_path="latent_similarity_probe.png"
):
    """
    One row per group:
      Orig_1..K  |  →  |  Recon_1..K  |  InputSim  |  ReconSim  |  L2-in-z

    Bottom summary panel:
      Violin (within vs across L2) | Bar (input vs recon sim) |
      Scatter (preservation) | Text summary
    """
    n_groups   = len(groups)
    K          = len(groups[0])
    global_l2_max = max(m.max() for m in l2_matrices)

    # Column layout per maze row:
    #   K orig | 1 arrow | K recon | 1 gap | 3 heatmaps
    n_maze_cols = K + 1 + K + 1 + 3

    fig_w = max(24, n_maze_cols * 1.5)
    fig_h = n_groups * 2.6 + 4.8

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='#fafafa')
    gs_outer = gridspec.GridSpec(
        n_groups + 1, 1,
        hspace=0.06,
        height_ratios=[2.4] * n_groups + [4.8],
        figure=fig
    )

    # ── One row per group ──────────────────────────────────────────────────
    for gi, group in enumerate(groups):
        colour = _GROUP_COLOURS[gi % len(_GROUP_COLOURS)]
        gs_row = gridspec.GridSpecFromSubplotSpec(
            1, n_maze_cols, subplot_spec=gs_outer[gi], wspace=0.06
        )

        sim_in    = sim_in_matrices[gi]
        sim_rc    = sim_rc_matrices[gi]
        l2_mat    = l2_matrices[gi]
        mask      = np.triu(np.ones((K, K), dtype=bool), k=1)
        delta     = within_recon_sims[gi] - within_input_sims[gi]
        sign      = "+" if delta >= 0 else ""
        preserved = abs(delta) < 0.05

        # Original mazes
        for k, idx in enumerate(group):
            ax = fig.add_subplot(gs_row[0, k])
            render_maze(ax, data[idx],
                        title=f"Orig {k+1}" if gi == 0 else "",
                        border_colour=colour)
            if k == 0:
                ax.set_ylabel(
                    f"G{gi+1}\nin={within_input_sims[gi]:.2f}\n"
                    f"rc={within_recon_sims[gi]:.2f} "
                    f"{'↑' if delta > 0.01 else ('↓' if delta < -0.01 else '≈')}",
                    fontsize=6.5, rotation=0, labelpad=50,
                    va='center', color=colour, fontweight='bold'
                )

        # Arrow
        ax_div = fig.add_subplot(gs_row[0, K])
        ax_div.axis('off')
        ax_div.text(0.5, 0.5, '→', ha='center', va='center',
                    fontsize=13, color='#888', transform=ax_div.transAxes)

        # Reconstructions
        for k, idx in enumerate(group):
            ax = fig.add_subplot(gs_row[0, K + 1 + k])
            render_maze(ax, recons[idx],
                        title=f"Recon {k+1}" if gi == 0 else "",
                        border_colour='#aaaaaa')

        # Gap column (invisible)
        fig.add_subplot(gs_row[0, 2*K + 1]).axis('off')

        # Input Sim heatmap
        ax_in = fig.add_subplot(gs_row[0, 2*K + 2])
        _draw_heatmap(ax_in, sim_in,
                      "Input Sim (1=identical)" if gi == 0 else "",
                      'Greens', 0, 1)

        # Recon Sim heatmap  ← the key new panel
        ax_rc = fig.add_subplot(gs_row[0, 2*K + 3])
        _draw_heatmap(ax_rc, sim_rc,
                      "Recon Sim (1=identical)" if gi == 0 else "",
                      'Greens', 0, 1)
        ax_rc.set_xlabel(
            f"Δ={sign}{delta:.3f}  {'✓' if preserved else '✗'}",
            fontsize=7, labelpad=2, fontweight='bold',
            color='#2ecc71' if preserved else '#e74c3c'
        )

        # L2 in z heatmap
        ax_l2 = fig.add_subplot(gs_row[0, 2*K + 4])
        _draw_heatmap(ax_l2, l2_mat,
                      "L2 in z-space" if gi == 0 else "",
                      'YlOrRd', 0, global_l2_max)
        ax_l2.set_xlabel(f"μ={within_l2s[gi]:.2f}", fontsize=7, labelpad=2)

    # ── Summary row ────────────────────────────────────────────────────────
    gs_sum = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=gs_outer[-1], wspace=0.38
    )

    ratio = (np.mean(within_l2s) / np.mean(across_l2s)
             if across_l2s else float('nan'))

    # 1) Violin: within vs across L2
    ax_viol = fig.add_subplot(gs_sum[0, 0])
    can_violin = len(within_l2s) >= 2 and len(across_l2s) >= 2
    if can_violin:
        vp = ax_viol.violinplot([within_l2s, across_l2s], positions=[1, 2],
                                showmedians=True, showextrema=True)
        for body, c in zip(vp['bodies'], ['#3498db', '#e74c3c']):
            body.set_facecolor(c); body.set_alpha(0.7)
    else:
        ax_viol.bar([1, 2],
                    [np.mean(within_l2s) if within_l2s else 0,
                     np.mean(across_l2s) if across_l2s else 0],
                    color=['#3498db', '#e74c3c'], alpha=0.8, width=0.5)
    ax_viol.set_xticks([1, 2])
    ax_viol.set_xticklabels(['Within\ngroup', 'Across\ngroups'], fontsize=9)
    ax_viol.set_ylabel("L2 in z-space", fontsize=9)
    ax_viol.set_title("Latent Distance\n(within vs across)", fontsize=9, pad=4)
    ax_viol.grid(axis='y', alpha=0.3)
    ax_viol.text(0.97, 0.97, f"ratio={ratio:.2f}\n(ideal<<1)",
                 ha='right', va='top', transform=ax_viol.transAxes, fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # 2) Grouped bar: input sim vs recon sim per group
    ax_bar = fig.add_subplot(gs_sum[0, 1])
    x = np.arange(n_groups)
    w = 0.35
    ax_bar.bar(x - w/2, within_input_sims, width=w,
               label='Input sim', color='#3498db', alpha=0.85)
    ax_bar.bar(x + w/2, within_recon_sims, width=w,
               label='Recon sim', color='#e67e22', alpha=0.85)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"G{i+1}" for i in range(n_groups)], fontsize=8)
    ax_bar.set_ylabel("Mean pairwise similarity", fontsize=9)
    ax_bar.set_title("Input vs Recon Similarity per Group\n(gap = VAE distortion)",
                     fontsize=9, pad=4)
    ax_bar.legend(fontsize=7)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.grid(axis='y', alpha=0.3)

    # 3) Scatter: input_sim vs recon_sim per pair
    ax_scat = fig.add_subplot(gs_sum[0, 2])
    if all_pairs:
        xs = np.array([p[0] for p in all_pairs])
        ys = np.array([p[1] for p in all_pairs])
        ax_scat.scatter(xs, ys, alpha=0.5, s=20, color='#9b59b6', zorder=3)
        if len(xs) > 1:
            m, b = np.polyfit(xs, ys, 1)
            x_line = np.linspace(xs.min(), xs.max(), 50)
            ax_scat.plot(x_line, m * x_line + b, color='#e74c3c',
                         lw=1.5, zorder=4, label=f"slope={m:.2f}")
            ax_scat.legend(fontsize=7)
        lims = [min(xs.min(), ys.min()) - 0.05,
                max(xs.max(), ys.max()) + 0.05]
        ax_scat.plot(lims, lims, 'k--', lw=1, alpha=0.4, label='y=x (perfect)')
        ax_scat.set_xlim(lims); ax_scat.set_ylim(lims)
    ax_scat.set_xlabel("Input similarity", fontsize=9)
    ax_scat.set_ylabel("Reconstruction similarity", fontsize=9)
    ax_scat.set_title("Similarity Preservation\n(slope≈1 = structure kept)",
                      fontsize=9, pad=4)
    ax_scat.grid(alpha=0.3)

    # 4) Summary text
    ax_txt = fig.add_subplot(gs_sum[0, 3])
    ax_txt.axis('off')
    mean_in = np.mean(within_input_sims)
    mean_rc = np.mean(within_recon_sims)
    delta   = mean_rc - mean_in
    slope   = (float(np.polyfit([p[0] for p in all_pairs],
                                 [p[1] for p in all_pairs], 1)[0])
               if len(all_pairs) > 1 else float('nan'))
    vg = ("✓ Good"     if ratio < 0.7 else
          ("~ Marginal" if ratio < 0.9 else "✗ Poor"))
    vp = ("✓ Good"     if abs(delta) < 0.05 else
          ("~ Partial"  if abs(delta) < 0.15 else "✗ Poor"))

    lines = [
        "── Summary ──────────────────────",
        f"  Groups / size:    {n_groups} × {K}",
        "",
        "  LATENT GEOMETRY",
        f"  Within-group L2:  {np.mean(within_l2s):.2f}",
        f"  Across-group L2:  {np.mean(across_l2s):.2f}",
        f"  Ratio (w/a):      {ratio:.3f}",
        f"  Verdict:          {vg}",
        "",
        "  SIMILARITY PRESERVATION",
        f"  Mean input sim:   {mean_in:.3f}",
        f"  Mean recon sim:   {mean_rc:.3f}",
        f"  Delta (rc-in):    {delta:+.3f}",
        f"  Slope (scatter):  {slope:.2f}",
        f"  Verdict:          {vp}",
        "",
        "─────────────────────────────────",
        "  Ideal: ratio<<1, delta≈0,",
        "  slope≈1.",
    ]
    ax_txt.text(0.04, 0.97, '\n'.join(lines),
                va='top', ha='left', transform=ax_txt.transAxes,
                fontsize=8, family='monospace',
                bbox=dict(boxstyle='round,pad=0.5',
                          facecolor='#f0f4f8', alpha=0.9))

    fig.suptitle(
        "VAE Latent Similarity Probe — geometry & similarity preservation",
        fontsize=13, fontweight='bold', y=1.001
    )
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Plot] Saved -> {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Top-level entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_probe(
    config: dict,
    ckpt_path: str,
    val_path: str = None,
    n_groups: int = 6,
    group_size: int = 5,
    similarity_threshold: float = 0.60,
    batch_size: int = 256,
    output_path: str = "latent_similarity_probe.png",
    seed: int = 42,
    max_search: int = 2000,
    n_val_cap: int = 3000
) -> dict:
    """
    Full pipeline. Returns dict with groups, means, recons, data, and all stats.

    similarity_threshold : lower = easier to form groups (try 0.5 if 0.6 fails)
    max_search           : pool size for mining (2000 fast, 5000 thorough)
    n_val_cap            : cap on val mazes encoded
    """
    import train_vae
    from train_vae import CluttrVAE

    data_raw = np.load(val_path) if val_path is not None else load_val_data(config)
    data     = data_raw[:n_val_cap]

    params        = load_checkpoint_params(ckpt_path)
    model         = CluttrVAE()
    means, recons = encode_dataset(model, params, data,
                                   batch_size=batch_size, seed=seed)

    groups = mine_similarity_groups(
        data, n_groups=n_groups, group_size=group_size,
        similarity_threshold=similarity_threshold,
        seed=seed, max_search=min(max_search, len(data))
    )

    if not groups:
        print("[Probe] No groups found. Lower similarity_threshold and retry.")
        return {}
    if len(groups) < 2:
        print(f"[Probe] Only {len(groups)} group(s) found — stats will be thin.")

    (within_l2s, within_input_sims, within_recon_sims,
     l2_matrices, sim_in_matrices, sim_rc_matrices,
     across_l2s, all_pairs) = compute_group_stats(groups, means, data, recons)

    ratio = (np.mean(within_l2s) / np.mean(across_l2s)
             if across_l2s else float('nan'))
    delta = np.mean(within_recon_sims) - np.mean(within_input_sims)
    slope = (float(np.polyfit([p[0] for p in all_pairs],
                               [p[1] for p in all_pairs], 1)[0])
             if len(all_pairs) > 1 else float('nan'))

    print(f"\n── Latent Similarity Probe ───────────────────────────────")
    print(f"  Groups:               {len(groups)} × {group_size}")
    print(f"  Mean input sim:       {np.mean(within_input_sims):.3f}")
    print(f"  Mean recon sim:       {np.mean(within_recon_sims):.3f}  "
          f"(delta={delta:+.3f})")
    print(f"  Scatter slope:        {slope:.3f}  (ideal: 1.0)")
    print(f"  Within-group L2:      {np.mean(within_l2s):.2f}")
    print(f"  Across-group L2:      {np.mean(across_l2s):.2f}")
    print(f"  Ratio (w/a):          {ratio:.3f}  (ideal: << 1.0)")
    print(f"──────────────────────────────────────────────────────────\n")

    plot_probe_results(
        groups=groups, data=data, recons=recons, means=means,
        within_l2s=within_l2s,
        within_input_sims=within_input_sims,
        within_recon_sims=within_recon_sims,
        l2_matrices=l2_matrices,
        sim_in_matrices=sim_in_matrices,
        sim_rc_matrices=sim_rc_matrices,
        across_l2s=across_l2s,
        all_pairs=all_pairs,
        output_path=output_path
    )

    return {
        "groups":             groups,
        "data":               data,
        "means":              means,
        "recons":             recons,
        "within_l2s":         within_l2s,
        "across_l2s":         across_l2s,
        "within_input_sims":  within_input_sims,
        "within_recon_sims":  within_recon_sims,
        "sim_in_matrices":    sim_in_matrices,
        "sim_rc_matrices":    sim_rc_matrices,
        "all_pairs":          all_pairs,
        "ratio":              ratio,
        "delta":              delta,
        "slope":              slope,
    }