"""
latent_pca_explorer.py
======================
PCA analysis and interactive perturbation of the VAE latent space.

Answers:
  1. What structure exists in the latent space? (PCA variance, 2D scatter)
  2. What does moving along each principal component decode to?
     (PC traversal — decode points along PC1, PC2, ... axes)
  3. What happens when I perturb a specific encoded maze in z-space?
     (perturbation grid — vary two dimensions at a time)

All paths and model dimensions are read from vae_train_config.yml.

Usage (notebook)
----------------
    import yaml, importlib
    import train_vae, utils
    importlib.reload(train_vae); importlib.reload(utils)
    import latent_pca_explorer as lpca
    importlib.reload(lpca)

    with open("vae_train_config.yml") as f:
        CONFIG = yaml.safe_load(f)

    # 1. Fit PCA on the full validation set
    results = lpca.run_pca(
        CONFIG,
        ckpt_path="/path/to/checkpoint_445000.pkl",
        output_dir="pca_results/"
    )

    # 2. Traverse principal components (what does each PC encode?)
    lpca.plot_pc_traversal(
        results, CONFIG,
        n_components=6,       # how many PCs to traverse
        n_steps=7,            # steps from -3σ to +3σ
        output_path="pca_results/pc_traversal.png"
    )

    # 3. Perturb a specific maze along two chosen dimensions
    lpca.plot_perturbation_grid(
        results, CONFIG,
        maze_idx=42,          # index in val data to use as anchor
        dim_x=0,              # latent dim for x-axis (or 'pc0' for PC0)
        dim_y=1,              # latent dim for y-axis
        n_steps=5,            # grid size (n_steps x n_steps)
        sigma=2.0,            # perturbation range in std devs
        output_path="pca_results/perturbation_grid.png"
    )
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA

import jax
import jax.numpy as jnp
from flax import linen as nn

from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Wall


# ══════════════════════════════════════════════════════════════════════════════
# Config helpers — all dimensions read from CONFIG
# ══════════════════════════════════════════════════════════════════════════════

def _get_dims(config: dict) -> dict:
    """Extract all relevant dimensions from the config dict."""
    return {
        "seq_len":    config.get("seq_len",    config.get("SEQ_LEN",    52)),
        "latent_dim": config.get("latent_dim", config.get("LATENT_DIM", 64)),
        "inner_dim":  config.get("inner_dim",  13),
        "full_dim":   config.get("full_dim",   15),
        "tile_size":  config.get("tile_size",  16),
    }


def _val_data_path(config: dict) -> str:
    return os.path.join(
        config["working_path"], config["vae_folder"], config["validation_data_path"]
    )


def _ckpt_dir(config: dict) -> str:
    return os.path.join(
        config["working_path"], config["vae_folder"],
        config["checkpoint_dir"]
    )


# ══════════════════════════════════════════════════════════════════════════════
# Data & model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_val_data(config: dict, n_cap: int = 5000) -> np.ndarray:
    path = _val_data_path(config)
    data = np.load(path)[:n_cap]
    print(f"[Data] {data.shape}  from {path}")
    return data


def load_params(ckpt_path: str) -> dict:
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    params = ckpt["params"] if "params" in ckpt else ckpt
    print(f"[Checkpoint] Loaded from {ckpt_path}")
    return params


# ══════════════════════════════════════════════════════════════════════════════
# Encoding
# ══════════════════════════════════════════════════════════════════════════════

def encode_dataset(model, params, data: np.ndarray,
                   batch_size: int = 256, seed: int = 42):
    """Returns means (N, D) and recons (N, seq_len)."""
    rng = jax.random.key(seed)

    @jax.jit
    def batch_encode(xs, rng_key):
        def fwd(x):
            logits, mean, _lv = model.apply(
                {"params": params}, x[jnp.newaxis, :], rng_key,
                train=False, rngs={"dropout": jax.random.key(0)}
            )
            return jnp.argmax(logits[0], axis=-1), mean[0]
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
    print(f"[Encode] {len(means)} mazes  z-shape: {means.shape}")
    return means, recons


# ══════════════════════════════════════════════════════════════════════════════
# Decoding a raw z vector
# ══════════════════════════════════════════════════════════════════════════════
# We use CluttrVAE.decode() directly via model.apply(..., method=CluttrVAE.decode).
# This is correct because:
#   - CluttrVAE.decode() is already implemented in train_vae.py and handles
#     both batched (batch, latent_dim) and unbatched (latent_dim,) inputs.
#   - The full checkpoint contains ALL params (encoder + decoder) so no
#     weight remapping or standalone-decoder extraction is needed.
#   - We avoid re-implementing the decoder architecture here, preventing
#     any mismatch with the actual trained model.

def repair_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Post-process a decoded sequence to enforce CLUTTR validity constraints:
      1. All values clamped to [0, 169]
      2. Goal (index -2) and agent (index -1) clamped to [1, 169]
      3. If goal == agent, agent is shifted by +1 (wrapping within [1, 169])
      4. Obstacles colliding with goal/agent are zeroed out
      5. Obstacle positions sorted ascending (zeros first)
    """
    seq   = jnp.clip(jnp.array(seq), 0, 169)
    goal  = jnp.clip(seq[-2], 1, 169)
    agent = jnp.clip(seq[-1], 1, 169)
    agent = jnp.where(goal == agent, (agent % 169) + 1, agent)

    obs = seq[:-2]
    obs = jnp.where(obs == goal,  0, obs)
    obs = jnp.where(obs == agent, 0, obs)
    obs = jnp.sort(obs)

    return np.array(jnp.concatenate([obs, jnp.array([goal, agent])]))


def _decode(z: np.ndarray, results: dict) -> np.ndarray:
    """
    Decode a single latent vector z (latent_dim,) to a repaired token sequence.

    Uses CluttrVAE.decode() via model.apply() with the full checkpoint params.
    z is passed as a 1-D array; CluttrVAE.decode() internally unsqueezes the
    batch dim and re-squeezes it before returning, so we get (seq_len, vocab).
    """
    from train_vae import CluttrVAE

    model  = results["model"]
    params = results["params"]

    # CluttrVAE.decode() accepts (latent_dim,) directly — it handles the squeeze
    logits = model.apply(
        {"params": params},
        jnp.array(z),
        method=CluttrVAE.decode,
    )  # → (seq_len, vocab_size)

    seq = np.array(jnp.argmax(logits, axis=-1))  # (seq_len,)
    return repair_sequence(seq)


# ══════════════════════════════════════════════════════════════════════════════
# MiniGrid rendering
# ══════════════════════════════════════════════════════════════════════════════

def _get_pos(idx: int, inner_dim: int = 13):
    y = (idx - 1) // inner_dim + 1
    x = (idx - 1) % inner_dim + 1
    return x, y


def seq_to_rgb(seq: np.ndarray, inner_dim: int = 13,
               full_dim: int = 15, tile_size: int = 16) -> np.ndarray:
    s         = seq.flatten().astype(int)
    agent_idx = int(s[-1])
    goal_idx  = int(s[-2])
    grid      = Grid(full_dim, full_dim)
    grid.wall_rect(0, 0, full_dim, full_dim)
    gx, gy = _get_pos(goal_idx, inner_dim)
    grid.set(gx, gy, Goal())
    for o_idx in s[:-2]:
        if o_idx > 0:
            ox, oy = _get_pos(o_idx, inner_dim)
            grid.set(ox, oy, Wall())
    ax_pos, ay_pos = _get_pos(agent_idx, inner_dim)
    return grid.render(tile_size, agent_pos=(ax_pos, ay_pos), agent_dir=0)


def render_maze(ax, seq: np.ndarray, title: str = "",
                border_colour=None, dims: dict = None):
    dims = dims or {"inner_dim": 13, "full_dim": 15, "tile_size": 16}
    try:
        rgb = seq_to_rgb(seq, dims["inner_dim"], dims["full_dim"], dims["tile_size"])
    except Exception:
        h = dims["full_dim"] * dims["tile_size"]
        rgb = np.full((h, h, 3), 200, dtype=np.uint8)
    ax.imshow(rgb, interpolation="nearest", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7, pad=2)
    if border_colour:
        for spine in ax.spines.values():
            spine.set_edgecolor(border_colour)
            spine.set_linewidth(2)


# ══════════════════════════════════════════════════════════════════════════════
# PCA fitting
# ══════════════════════════════════════════════════════════════════════════════

def fit_pca(means: np.ndarray, n_components: int = None) -> PCA:
    """Fit PCA on the posterior means. n_components=None keeps all dims."""
    n_components = n_components or means.shape[1]
    pca = PCA(n_components=n_components)
    pca.fit(means)
    print(f"[PCA] Fitted on {len(means)} points, {n_components} components")
    print(f"[PCA] Var explained: "
          + "  ".join(f"PC{i}={v:.1%}" for i, v in
                      enumerate(pca.explained_variance_ratio_[:8])))
    return pca


# ══════════════════════════════════════════════════════════════════════════════
# Plot 1: PCA overview — scree + 2D scatter coloured by wall density
# ══════════════════════════════════════════════════════════════════════════════

def plot_pca_overview(means: np.ndarray, pca: PCA, data: np.ndarray,
                      output_path: str = "pca_overview.png"):
    """
    3-panel figure:
      Left   : Scree plot — how much variance each PC explains
      Middle : PC1 vs PC2 scatter, coloured by wall density
      Right  : PC1 vs PC2 scatter, coloured by agent position (row)
    """
    z_pca       = pca.transform(means)
    wall_counts = (data[:, :-2] != 0).sum(axis=1).astype(float)
    agent_rows  = ((data[:, -1] - 1) // 13).astype(float)   # 0-12

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="#fafafa")

    # Scree
    ax = axes[0]
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    ax.bar(range(1, len(cumvar) + 1), pca.explained_variance_ratio_ * 100,
           color="#3498db", alpha=0.8, label="Per-PC")
    ax.plot(range(1, len(cumvar) + 1), cumvar, "r-o",
            markersize=3, lw=1.5, label="Cumulative")
    ax.axhline(90, color="k", ls="--", lw=1, alpha=0.4, label="90% threshold")
    ax.set_xlabel("Principal Component", fontsize=10)
    ax.set_ylabel("Variance Explained (%)", fontsize=10)
    ax.set_title("PCA Variance Explanation Plot", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    n90 = int(np.searchsorted(cumvar, 90)) + 1
    ax.text(0.97, 0.5, f"{n90} PCs\nexplain 90%",
            ha="right", va="center", transform=ax.transAxes,
            fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # PC1 vs PC2, coloured by wall density
    ax = axes[1]
    sc = ax.scatter(z_pca[:, 0], z_pca[:, 1], c=wall_counts,
                    cmap="viridis", alpha=0.4, s=6)
    plt.colorbar(sc, ax=ax, label="Wall count")
    ax.set_xlabel("PC1", fontsize=10)
    ax.set_ylabel("PC2", fontsize=10)
    ax.set_title("PC1 vs PC2\n(colour = wall density)", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.2)

    # PC1 vs PC2, coloured by agent row
    ax = axes[2]
    sc = ax.scatter(z_pca[:, 0], z_pca[:, 1], c=agent_rows,
                    cmap="plasma", alpha=0.4, s=6)
    plt.colorbar(sc, ax=ax, label="Agent row (0-12)")
    ax.set_xlabel("PC1", fontsize=10)
    ax.set_ylabel("PC2", fontsize=10)
    ax.set_title("PC1 vs PC2\n(colour = agent row)", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.2)

    fig.suptitle("VAE Latent Space — PCA Overview", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Plot] PCA overview -> {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 2: PC traversal — decode along each PC axis
# ══════════════════════════════════════════════════════════════════════════════

def plot_pc_traversal(results: dict, config: dict,
                      n_components: int = 6,
                      n_steps: int = 7,
                      sigma: float = 3.0,
                      output_path: str = "pc_traversal.png"):
    """
    For each of the first n_components PCs, decode n_steps points
    evenly spaced from -sigma*std to +sigma*std along that PC,
    keeping all other dimensions at the mean.

    This shows what each PC "controls" in maze space.
    """
    dims    = _get_dims(config)
    pca     = results["pca"]
    means   = results["means"]

    z_mean  = means.mean(axis=0)
    pc_stds = np.sqrt(pca.explained_variance_)

    fig, axes = plt.subplots(
        n_components, n_steps,
        figsize=(n_steps * 1.8, n_components * 2.0),
        facecolor="#fafafa"
    )
    fig.suptitle(
        f"PC Traversal  —  decoding along each PC axis  (±{sigma}σ)\n"
        "Each row = one PC, columns = steps from -σ to +σ",
        fontsize=11, fontweight="bold"
    )

    alphas = np.linspace(-sigma, sigma, n_steps)

    for pi in range(n_components):
        pc_vec  = pca.components_[pi]
        std     = pc_stds[pi]
        var_pct = pca.explained_variance_ratio_[pi] * 100

        for si, alpha in enumerate(alphas):
            z_perturbed = z_mean + alpha * std * pc_vec
            seq         = _decode(z_perturbed, results)
            ax          = axes[pi, si]
            label       = f"{alpha:+.1f}σ" if pi == 0 else ""
            render_maze(ax, seq, title=label, dims=dims)

            if si == 0:
                axes[pi, 0].set_ylabel(
                    f"PC{pi+1}\n{var_pct:.1f}%",
                    fontsize=8, rotation=0, labelpad=38, va="center",
                    fontweight="bold"
                )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Plot] PC traversal -> {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 3: Perturbation grid — vary two dims around an anchor maze
# ══════════════════════════════════════════════════════════════════════════════

def plot_perturbation_grid(results: dict, config: dict,
                           maze_idx: int = 0,
                           dim_x: int = 0,
                           dim_y: int = 1,
                           n_steps: int = 5,
                           sigma: float = 2.0,
                           use_pc_dims: bool = True,
                           output_path: str = "perturbation_grid.png"):
    """
    Take one encoded maze as anchor, perturb it along two chosen dimensions,
    and decode each point in the resulting n_steps × n_steps grid.

    Parameters
    ----------
    maze_idx    : index into val data to use as the anchor maze
    dim_x       : which dimension to vary on the x-axis
    dim_y       : which dimension to vary on the y-axis
    n_steps     : grid resolution (n_steps × n_steps mazes rendered)
    sigma       : perturbation range in std devs of that dimension
    use_pc_dims : if True, dim_x/dim_y refer to PC indices and perturbation
                  is along PC directions; if False, raw latent dimensions
    """
    dims   = _get_dims(config)
    pca    = results["pca"]
    means  = results["means"]
    recons = results["recons"]
    data   = results["data"]

    z_anchor = means[maze_idx]

    if use_pc_dims:
        vec_x  = pca.components_[dim_x]
        vec_y  = pca.components_[dim_y]
        std_x  = np.sqrt(pca.explained_variance_[dim_x])
        std_y  = np.sqrt(pca.explained_variance_[dim_y])
        xlabel = f"PC{dim_x+1}  ({pca.explained_variance_ratio_[dim_x]*100:.1f}%)"
        ylabel = f"PC{dim_y+1}  ({pca.explained_variance_ratio_[dim_y]*100:.1f}%)"
    else:
        std_x  = float(means[:, dim_x].std())
        std_y  = float(means[:, dim_y].std())
        vec_x  = np.zeros(means.shape[1]); vec_x[dim_x] = 1.0
        vec_y  = np.zeros(means.shape[1]); vec_y[dim_y] = 1.0
        xlabel = f"z[{dim_x}]  (std={std_x:.2f})"
        ylabel = f"z[{dim_y}]  (std={std_y:.2f})"

    alphas = np.linspace(-sigma, sigma, n_steps)

    fig, axes = plt.subplots(
        n_steps, n_steps + 1,
        figsize=((n_steps + 1) * 1.9, n_steps * 1.9),
        facecolor="#fafafa"
    )
    fig.suptitle(
        f"Perturbation Grid  —  anchor: maze idx={maze_idx}\n"
        f"x-axis: {xlabel}    y-axis: {ylabel}    range: ±{sigma}σ",
        fontsize=10, fontweight="bold"
    )

    for ri, ay in enumerate(alphas[::-1]):
        for ci, ax_val in enumerate(alphas):
            z   = z_anchor + ax_val * std_x * vec_x + ay * std_y * vec_y
            seq = _decode(z, results)
            ax  = axes[ri, ci]
            render_maze(ax, seq, dims=dims, border_colour="#dddddd")
            if ri == n_steps - 1:
                ax.set_xlabel(f"{ax_val:+.1f}σ", fontsize=6)
            if ci == 0:
                ax.set_ylabel(f"{ay:+.1f}σ", fontsize=6)

    # Rightmost column: anchor + its reconstruction
    for ri in range(n_steps):
        axes[ri, -1].axis("off")
    mid = n_steps // 2
    render_maze(axes[mid - 1, -1], data[maze_idx],
                title="Anchor\n(orig)", border_colour="#e74c3c", dims=dims)
    render_maze(axes[mid, -1], recons[maze_idx],
                title="Anchor\n(recon)", border_colour="#3498db", dims=dims)

    # Column and row labels
    fig.text(0.5, 1.01, xlabel, ha="center", fontsize=9, color="#555")
    fig.text(-0.01, 0.5, ylabel, va="center", rotation=90, fontsize=9, color="#555")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Plot] Perturbation grid -> {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 4: 2D PCA scatter with decoded thumbnails at selected points
# ══════════════════════════════════════════════════════════════════════════════

def plot_pca_with_thumbnails(results: dict, config: dict,
                             n_thumbnails: int = 12,
                             pc_x: int = 0, pc_y: int = 1,
                             output_path: str = "pca_thumbnails.png"):
    """
    PCA scatter (PC_x vs PC_y) with small maze thumbnails inset at
    selected points — spread evenly across the 2D space so you can
    see what different regions of the latent space decode to.
    """
    dims   = _get_dims(config)
    pca    = results["pca"]
    means  = results["means"]
    recons = results["recons"]

    z_pca = pca.transform(means)
    wall_counts = (results["data"][:, :-2] != 0).sum(axis=1)

    fig, ax_main = plt.subplots(figsize=(14, 10), facecolor="#fafafa")

    sc = ax_main.scatter(z_pca[:, pc_x], z_pca[:, pc_y],
                         c=wall_counts, cmap="viridis",
                         alpha=0.3, s=5, zorder=1)
    plt.colorbar(sc, ax=ax_main, label="Wall count", shrink=0.7)
    ax_main.set_xlabel(
        f"PC{pc_x+1}  ({pca.explained_variance_ratio_[pc_x]*100:.1f}%)", fontsize=11)
    ax_main.set_ylabel(
        f"PC{pc_y+1}  ({pca.explained_variance_ratio_[pc_y]*100:.1f}%)", fontsize=11)
    ax_main.set_title(
        f"Latent Space (PC{pc_x+1} vs PC{pc_y+1}) with Decoded Thumbnails",
        fontsize=13, fontweight="bold"
    )
    ax_main.grid(alpha=0.2)

    # Select n_thumbnails points spread across the 2D space using a grid
    x_vals = z_pca[:, pc_x]
    y_vals = z_pca[:, pc_y]
    x_bins = np.linspace(x_vals.min(), x_vals.max(), int(np.sqrt(n_thumbnails)) + 1)
    y_bins = np.linspace(y_vals.min(), y_vals.max(), int(np.sqrt(n_thumbnails)) + 1)

    selected = []
    for xi in range(len(x_bins) - 1):
        for yi in range(len(y_bins) - 1):
            in_cell = np.where(
                (x_vals >= x_bins[xi]) & (x_vals < x_bins[xi + 1]) &
                (y_vals >= y_bins[yi]) & (y_vals < y_bins[yi + 1])
            )[0]
            if len(in_cell) > 0:
                selected.append(int(in_cell[len(in_cell) // 2]))

    selected = selected[:n_thumbnails]

    # Draw thumbnails as inset axes
    fig_w, fig_h = fig.get_size_inches()
    thumb_w = 0.09   # fraction of figure width
    thumb_h = thumb_w * (fig_w / fig_h)

    ax_pos  = ax_main.get_position()   # Bbox in figure coords

    for idx in selected:
        px = z_pca[idx, pc_x]
        py = z_pca[idx, pc_y]

        # Convert data coords -> figure coords
        disp   = ax_main.transData.transform((px, py))
        fig_pt = fig.transFigure.inverted().transform(disp)

        # Place inset; clip to stay inside figure
        left = np.clip(fig_pt[0] - thumb_w / 2, ax_pos.x0, ax_pos.x1 - thumb_w)
        bot  = np.clip(fig_pt[1] - thumb_h / 2, ax_pos.y0, ax_pos.y1 - thumb_h)

        ax_inset = fig.add_axes([left, bot, thumb_w, thumb_h])
        render_maze(ax_inset, recons[idx], dims=dims, border_colour="#e74c3c")

        # Connector line
        ax_main.plot(px, py, "r+", markersize=6, zorder=5)

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Plot] PCA thumbnails -> {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Top-level entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_pca(
    config: dict,
    ckpt_path: str,
    n_val_cap: int = 5000,
    batch_size: int = 256,
    n_pca_components: int = None,
    output_dir: str = "pca_results/",
    seed: int = 42,
) -> dict:
    """
    Full PCA pipeline:
      1. Load val data and encode with the VAE
      2. Fit PCA on posterior means
      3. Save overview plot (scree + scatter)
      4. Return results dict for further plotting

    Parameters
    ----------
    config           : dict from yaml.safe_load("vae_train_config.yml")
    ckpt_path        : path to checkpoint_*.pkl
    n_val_cap        : how many val mazes to encode (more = better PCA)
    n_pca_components : PCA components to keep (None = keep all latent dims)
    output_dir       : where to save plots

    Returns
    -------
    dict with keys: means, recons, data, pca, z_pca, dims, config,
                    model, params
    """
    from train_vae import CluttrVAE

    os.makedirs(output_dir, exist_ok=True)
    dims = _get_dims(config)
    print(f"[Config] seq_len={dims['seq_len']}  latent_dim={dims['latent_dim']}")

    data        = load_val_data(config, n_cap=n_val_cap)
    full_params = load_params(ckpt_path)

    # Use the full CluttrVAE model for both encoding AND decoding.
    # CluttrVAE.decode() is already correct and tested — no standalone
    # decoder or weight remapping is required.
    model = CluttrVAE()
    print(f"[Model] Using CluttrVAE.decode() for all latent-space decoding")

    means, recons = encode_dataset(model, full_params, data,
                                   batch_size=batch_size, seed=seed)

    n_pca_components = n_pca_components or dims["latent_dim"]
    pca    = fit_pca(means, n_components=n_pca_components)
    z_pca  = pca.transform(means)

    overview_path = os.path.join(output_dir, "pca_overview.png")
    plot_pca_overview(means, pca, data, output_path=overview_path)

    results = {
        "means":   means,
        "recons":  recons,
        "data":    data,
        "pca":     pca,
        "z_pca":   z_pca,
        "dims":    dims,
        "config":  config,
        # Store model + full params so _decode() can call CluttrVAE.decode()
        "model":   model,
        "params":  full_params,
    }

    print(f"\n── PCA Summary ───────────────────────────────────────────")
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    for thresh in [50, 80, 90, 95]:
        n = int(np.searchsorted(cumvar, thresh)) + 1
        print(f"  {thresh}% variance explained by {n} PCs")
    print(f"  Top 5 PCs: "
          + "  ".join(f"PC{i+1}={v:.1f}%"
                      for i, v in enumerate(pca.explained_variance_ratio_[:5] * 100)))
    print(f"──────────────────────────────────────────────────────────\n")

    return results