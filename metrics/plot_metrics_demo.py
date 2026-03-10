"""
Phase 1 demo: Load trained agent + buffer levels from gcs_artifacts and visualize
all 4 diversity metrics with plots.

Uses:
  - Agent: gcs_artifacts/agent/cmaes_vae_beta2.0_seed0_198/
  - Buffer: gcs_artifacts/buffer/buffer_dump_final.npz

Usage:
    python3 metrics/plot_metrics_demo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vae'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

PLOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'test_log', 'phase_1')
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from functools import partial

from jaxued.environments import Maze
from jaxued.environments.maze import Level, make_level_generator
from jaxued.wrappers import AutoReplayWrapper

from trajectory_metrics import (
    compute_pairwise_metrics,
    position_trace_dtw,
    observation_dtw,
    value_trajectory_dtw,
    spatial_footprint_jaccard,
)
from trajectory_cache import TrajectoryCache

# Agent loading
from cross_evaluate import load_agent, tokens_to_levels_batch
from maze_plr import ActorCritic


def collect_trained_agent_trajectories(train_state, env, env_params, levels, num_levels, num_steps, rng):
    """Roll out trained agent on levels, return trajectory data with positions, values, obs, dones."""
    eval_env = Maze(max_height=13, max_width=13, agent_view_size=5, normalize_obs=True)

    rng, rng_reset = jax.random.split(rng)
    init_obs, init_env_state = jax.vmap(eval_env.reset_to_level, (0, 0, None))(
        jax.random.split(rng_reset, num_levels), levels, env_params
    )
    init_hstate = ActorCritic.initialize_carry((num_levels,))

    def step(carry, _):
        rng, hstate, obs, state, done = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        agent_pos = state.agent_pos

        # Network forward pass
        x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, done))
        hstate, pi, value = train_state.apply_fn(train_state.params, x, hstate)
        action = pi.sample(seed=rng_action).squeeze(0)
        value = value.squeeze(0)

        # Step environment
        next_obs, next_state, reward, next_done, _ = jax.vmap(
            eval_env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_levels), state, action, env_params)

        carry = (rng, hstate, next_obs, next_state, next_done)
        return carry, (obs.image, action, reward, done, agent_pos, value)

    _, traj = jax.lax.scan(
        step,
        (rng, init_hstate, init_obs, init_env_state, jnp.zeros(num_levels, dtype=bool)),
        None,
        length=num_steps,
    )
    obs_images, actions, rewards, dones, positions, values = traj
    return {
        "observations": np.asarray(obs_images),   # (T, N, ...)
        "actions": np.asarray(actions),            # (T, N)
        "rewards": np.asarray(rewards),            # (T, N)
        "dones": np.asarray(dones),                # (T, N)
        "positions": np.asarray(positions),        # (T, N, 2)
        "values": np.asarray(values),              # (T, N)
    }


def wall_map_to_grid(wall_map_np):
    """Convert a wall_map (H, W) bool array to a displayable image."""
    h, w = wall_map_np.shape
    img = np.ones((h, w, 3), dtype=np.float32)
    img[wall_map_np] = [0.2, 0.2, 0.2]  # walls = dark
    return img


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    # ---- Load trained agent ----
    agent_dir = os.path.join(PROJECT_ROOT, "gcs_artifacts", "agent", "cmaes_vae_beta2.0_seed0_198")
    print(f"Loading trained agent from {agent_dir}...")
    train_state, config, env, env_params = load_agent(agent_dir, checkpoint_step=198)
    assert train_state is not None, "Failed to load agent checkpoint"

    # ---- Load buffer levels ----
    buffer_path = os.path.join(PROJECT_ROOT, "gcs_artifacts", "buffer", "buffer_dump_final.npz")
    print(f"Loading buffer from {buffer_path}...")
    buffer_data = np.load(buffer_path, allow_pickle=True)
    tokens = buffer_data["tokens"]
    size = int(buffer_data.get("size", len(tokens)))
    tokens = tokens[:size]
    print(f"  Buffer has {size} levels")

    # Convert tokens to levels
    all_levels = tokens_to_levels_batch(tokens)

    # Select 16 levels evenly spaced from the buffer for visualization
    NUM_LEVELS = 16
    indices = np.linspace(0, size - 1, NUM_LEVELS, dtype=int)
    levels = jax.tree_util.tree_map(lambda x: x[indices], all_levels)
    print(f"  Selected {NUM_LEVELS} levels at indices: {indices.tolist()}")

    # ---- Collect trajectories with trained agent ----
    NUM_STEPS = 250  # max episode length for the maze env
    rng = jax.random.PRNGKey(42)
    print(f"Rolling out trained agent on {NUM_LEVELS} buffer levels for {NUM_STEPS} steps...")
    data = collect_trained_agent_trajectories(
        train_state, env, env_params, levels, NUM_LEVELS, NUM_STEPS, rng
    )

    # Build per-level trajectory list
    trajectories = []
    for i in range(NUM_LEVELS):
        trajectories.append({
            "observations": data["observations"][:, i],
            "positions": data["positions"][:, i],
            "values": data["values"][:, i],
            "dones": data["dones"][:, i],
        })

    # Compute pairwise metrics
    print("Computing pairwise metrics...")
    pairwise = compute_pairwise_metrics(trajectories)

    # ============================================================
    # FIGURE 1: Pairwise metric distributions (histograms)
    # ============================================================
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle("Phase 1: Pairwise Diversity Metrics (trained agent, buffer levels)", fontsize=14, fontweight='bold')

    metrics_to_plot = [
        ("obs_dtw_distances", "Obs Sequence DTW", "steelblue"),
        ("pos_dtw_distances", "Position Trace DTW", "darkorange"),
        ("value_dtw_distances", "Value Trajectory DTW", "seagreen"),
        ("jaccard_indices", "Spatial Footprint Jaccard", "crimson"),
    ]

    for idx, (key, title, color) in enumerate(metrics_to_plot):
        ax = axes.flat[idx]
        vals = pairwise[key]
        ax.hist(vals, bins=20, color=color, alpha=0.75, edgecolor='white')
        ax.axvline(vals.mean(), color='black', linestyle='--', linewidth=1.5, label=f'mean={vals.mean():.3f}')
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Pair count")
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    # Last subplots: summary stats table
    for idx in range(len(metrics_to_plot), len(axes.flat)):
        axes.flat[idx].axis('off')
    ax_table = axes.flat[len(metrics_to_plot)]
    table_data = []
    for key, title, _ in metrics_to_plot:
        v = pairwise[key]
        short = title.split(" ")[0] + " " + title.split(" ")[1] if len(title.split(" ")) > 1 else title
        table_data.append([short, f"{v.mean():.3f}", f"{v.std():.3f}", f"{v.min():.3f}", f"{v.max():.3f}"])
    table = ax_table.table(cellText=table_data,
                     colLabels=["Metric", "Mean", "Std", "Min", "Max"],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    plt.tight_layout()
    fig.savefig(f"{PLOT_DIR}/plots_pairwise_distributions.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {PLOT_DIR}/plots_pairwise_distributions.png")

    # ============================================================
    # FIGURE 2: Pairwise distance matrices (heatmaps)
    # ============================================================
    n = NUM_LEVELS
    metric_matrices = {}
    for key in ["obs_dtw_distances", "pos_dtw_distances", "value_dtw_distances", "jaccard_indices"]:
        mat = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                mat[i, j] = pairwise[key][idx]
                mat[j, i] = pairwise[key][idx]
                idx += 1
        metric_matrices[key] = mat

    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
    fig2.suptitle("Pairwise Distance/Similarity Matrices (trained agent, buffer levels)", fontsize=14, fontweight='bold')

    titles = ["Obs Sequence DTW Distance", "Position Trace DTW Distance",
              "Value Trajectory DTW Distance", "Spatial Footprint Jaccard"]
    keys = ["obs_dtw_distances", "pos_dtw_distances", "value_dtw_distances", "jaccard_indices"]
    cmaps = ["YlOrRd", "YlOrRd", "YlOrRd", "YlGnBu"]

    for idx, (key, title, cmap) in enumerate(zip(keys, titles, cmaps)):
        ax = axes2.flat[idx]
        mat = metric_matrices[key].copy()
        # Mask diagonal so it doesn't skew the color scale
        off_diag = mat[~np.eye(n, dtype=bool)]
        vmin, vmax = off_diag.min(), off_diag.max()
        # Set diagonal to NaN so it renders as the 'bad' color
        np.fill_diagonal(mat, np.nan)
        cmap_obj = plt.colormaps.get_cmap(cmap).copy()
        cmap_obj.set_bad(color='black')
        im = ax.imshow(mat, cmap=cmap_obj, aspect='equal', vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Level")
        ax.set_ylabel("Level")
        fig2.colorbar(im, ax=ax, fraction=0.046)

    # Hide unused subplot(s)
    for idx in range(len(keys), len(axes2.flat)):
        axes2.flat[idx].axis('off')

    plt.tight_layout()
    fig2.savefig(f"{PLOT_DIR}/plots_pairwise_matrices.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {PLOT_DIR}/plots_pairwise_matrices.png")

    # ============================================================
    # FIGURES 3a-3d: Trajectory deep-dives, one per metric
    # ============================================================
    pair_indices = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_indices.append((i, j))

    wall_maps = np.asarray(levels.wall_map)

    # Define the 5 metrics to generate deep-dives for
    # For distance metrics: most similar = min, most different = max
    # For correlation: most similar = max, most different = min
    # For Jaccard: most similar = max, most different = min
    metric_configs = [
        ("pos_dtw_distances", "Position DTW", True),       # distance: sim=min, diff=max
        ("obs_dtw_distances", "Observation DTW", True),     # distance: sim=min, diff=max
        ("value_dtw_distances", "Value DTW", True),         # distance: sim=min, diff=max
        ("jaccard_indices", "Spatial Jaccard", False),       # similarity: sim=max, diff=min
    ]

    for metric_key, metric_name, is_distance in metric_configs:
        metric_vals = pairwise[metric_key]
        if is_distance:
            pair_sim = pair_indices[np.argmin(metric_vals)]
            pair_diff = pair_indices[np.argmax(metric_vals)]
        else:
            pair_sim = pair_indices[np.argmax(metric_vals)]
            pair_diff = pair_indices[np.argmin(metric_vals)]

        fig, axes = plt.subplots(2, 6, figsize=(30, 10))
        fig.suptitle(f"Deep-Dive by {metric_name}: Most Similar vs Most Different (trained agent)",
                     fontsize=14, fontweight='bold')

        # Pre-compute all metric results for both pairs
        deepdive_pairs = [(pair_sim, "Most Similar"), (pair_diff, "Most Different")]
        deepdive_results = []
        for pair, label in deepdive_pairs:
            i, j = pair
            ti, tj = trajectories[i], trajectories[j]
            pos_result = position_trace_dtw(ti["positions"], ti["dones"], tj["positions"], tj["dones"])
            obs_result = observation_dtw(ti["observations"], ti["dones"], tj["observations"], tj["dones"])
            val_result = value_trajectory_dtw(ti["values"], ti["dones"], tj["values"], tj["dones"])
            jac_result = spatial_footprint_jaccard(ti["positions"], ti["dones"], tj["positions"], tj["dones"])
            deepdive_results.append((pos_result, obs_result, val_result, jac_result))

        # Shared y-axis limits across both rows
        pos_dtw_ymax = max(r[0]["local_costs"].max() for r in deepdive_results) * 1.1
        obs_dtw_ymax = max(r[1]["local_costs"].max() for r in deepdive_results) * 1.1
        val_dtw_ymax = max(r[2]["local_costs"].max() for r in deepdive_results) * 1.1

        for row, ((pair, label), (pos_result, obs_result, val_result, jac_result)) in enumerate(zip(deepdive_pairs, deepdive_results)):
            i, j = pair
            ti, tj = trajectories[i], trajectories[j]

            # Position traces on maze
            for col, (env_idx, traj_data, color) in enumerate([(i, ti, 'blue'), (j, tj, 'red')]):
                ax = axes[row, col]
                img = wall_map_to_grid(wall_maps[env_idx])
                ax.imshow(img, origin='upper')
                pos = traj_data["positions"]
                dones = traj_data["dones"]
                done_idx = np.where(dones)[0]
                end = done_idx[0] + 1 if len(done_idx) > 0 else len(pos)
                pos_trunc = pos[:end]
                for t in range(len(pos_trunc) - 1):
                    alpha = 0.3 + 0.7 * (t / max(len(pos_trunc) - 1, 1))
                    ax.plot([pos_trunc[t, 0], pos_trunc[t+1, 0]],
                            [pos_trunc[t, 1], pos_trunc[t+1, 1]],
                            color=color, alpha=alpha, linewidth=2)
                ax.plot(pos_trunc[0, 0], pos_trunc[0, 1], 'o', color=color, markersize=8, label='start')
                if end > 1:
                    ax.plot(pos_trunc[-1, 0], pos_trunc[-1, 1], 's', color=color, markersize=8, label='end')
                ax.set_title(f"{label} — Level {env_idx} ({len(pos_trunc)} steps)", fontsize=10)
                ax.legend(fontsize=8)
                ax.set_xlim(-0.5, 12.5)
                ax.set_ylim(12.5, -0.5)

            # Position DTW profile
            ax_pos_dtw = axes[row, 2]
            ax_pos_dtw.plot(pos_result["local_costs"], color='darkorange', linewidth=1.5)
            ax_pos_dtw.fill_between(range(len(pos_result["local_costs"])), pos_result["local_costs"],
                                alpha=0.3, color='darkorange')
            ax_pos_dtw.set_title(f"{label} — Position DTW Profile\n(distance = {pos_result['distance']:.3f})", fontsize=10)
            ax_pos_dtw.set_xlabel("Warping path step")
            ax_pos_dtw.set_ylabel("Local cost (L2)")
            ax_pos_dtw.set_ylim(0, pos_dtw_ymax)
            ax_pos_dtw.grid(alpha=0.3)

            # Observation DTW profile
            ax_obs_dtw = axes[row, 3]
            ax_obs_dtw.plot(obs_result["local_costs"], color='purple', linewidth=1.5)
            ax_obs_dtw.fill_between(range(len(obs_result["local_costs"])), obs_result["local_costs"],
                                alpha=0.3, color='purple')
            ax_obs_dtw.set_title(f"{label} — Obs DTW Profile\n(distance = {obs_result['distance']:.3f})", fontsize=10)
            ax_obs_dtw.set_xlabel("Warping path step")
            ax_obs_dtw.set_ylabel("Local cost (L2)")
            ax_obs_dtw.set_ylim(0, obs_dtw_ymax)
            ax_obs_dtw.grid(alpha=0.3)

            # Value DTW profile
            ax_val_dtw = axes[row, 4]
            ax_val_dtw.plot(val_result["local_costs"], color='seagreen', linewidth=1.5)
            ax_val_dtw.fill_between(range(len(val_result["local_costs"])), val_result["local_costs"],
                                alpha=0.3, color='seagreen')
            ax_val_dtw.set_title(f"{label} — Value DTW Profile\n(distance = {val_result['distance']:.3f})", fontsize=10)
            ax_val_dtw.set_xlabel("Warping path step")
            ax_val_dtw.set_ylabel("Local cost (L1)")
            ax_val_dtw.set_ylim(0, val_dtw_ymax)
            ax_val_dtw.grid(alpha=0.3)

            # Spatial Jaccard overlay
            ax_jac = axes[row, 5]
            img_jac = np.ones((13, 13, 3), dtype=np.float32) * 0.9
            cells_a = jac_result["cells_a"]
            cells_b = jac_result["cells_b"]
            only_a = cells_a - cells_b
            only_b = cells_b - cells_a
            both = cells_a & cells_b
            for (x, y) in only_a:
                if 0 <= y < 13 and 0 <= x < 13:
                    img_jac[y, x] = [0.2, 0.5, 1.0]
            for (x, y) in only_b:
                if 0 <= y < 13 and 0 <= x < 13:
                    img_jac[y, x] = [1.0, 0.3, 0.3]
            for (x, y) in both:
                if 0 <= y < 13 and 0 <= x < 13:
                    img_jac[y, x] = [0.2, 0.8, 0.3]
            ax_jac.imshow(img_jac, origin='upper')
            ax_jac.set_title(f"{label} — Spatial Jaccard\n(J={jac_result['jaccard']:.3f}, "
                             f"|A∩B|={jac_result['intersection_size']}, |A∪B|={jac_result['union_size']})", fontsize=10)
            ax_jac.set_xlim(-0.5, 12.5)
            ax_jac.set_ylim(12.5, -0.5)
            ax_jac.legend(handles=[
                Patch(facecolor=[0.2, 0.5, 1.0], label=f'Only L{i}'),
                Patch(facecolor=[1.0, 0.3, 0.3], label=f'Only L{j}'),
                Patch(facecolor=[0.2, 0.8, 0.3], label='Both'),
            ], fontsize=7, loc='lower right')

        plt.tight_layout()
        safe_name = metric_key.replace("_", "-")
        fig.savefig(f"{PLOT_DIR}/plots_deepdive_{safe_name}.png", dpi=150, bbox_inches='tight')
        print(f"Saved: {PLOT_DIR}/plots_deepdive_{safe_name}.png")
        plt.close(fig)

    # ============================================================
    # FIGURE 4: Spatial footprints
    # ============================================================
    fig4, axes4 = plt.subplots(2, 8, figsize=(20, 5))
    fig4.suptitle("Spatial Footprints: Cells Visited by Trained Agent (buffer levels)", fontsize=13, fontweight='bold')

    for i in range(NUM_LEVELS):
        row = i // 8
        col = i % 8
        ax = axes4[row, col]
        img = wall_map_to_grid(wall_maps[i])

        # Overlay visited cells
        pos = trajectories[i]["positions"]
        dones = trajectories[i]["dones"]
        done_idx = np.where(dones)[0]
        end = done_idx[0] + 1 if len(done_idx) > 0 else len(pos)
        pos_trunc = pos[:end]
        visited = set(map(tuple, pos_trunc.tolist()))
        for (x, y) in visited:
            if 0 <= y < 13 and 0 <= x < 13:
                img[y, x] = [0.2, 0.6, 1.0]  # blue for visited

        ax.imshow(img, origin='upper')
        ax.set_title(f"L{i} ({len(visited)} cells, {len(pos_trunc)} steps)", fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    fig4.savefig(f"{PLOT_DIR}/plots_spatial_footprints.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {PLOT_DIR}/plots_spatial_footprints.png")

    print(f"\nDone! All 4 figures saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
