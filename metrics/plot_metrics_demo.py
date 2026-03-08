"""
Phase 1 demo: Generate trajectory data on random + prefab mazes and visualize
all 4 diversity metrics with plots.

Usage:
    python3 metrics/plot_metrics_demo.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

PLOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'test_log', 'phase_1')

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from jaxued.environments import Maze
from jaxued.environments.maze import Level, make_level_generator
from jaxued.wrappers import AutoReplayWrapper

from trajectory_metrics import (
    compute_pairwise_metrics,
    position_trace_dtw,
    observation_dtw,
    value_trajectory_correlation,
    spatial_footprint_jaccard,
)
from trajectory_cache import TrajectoryCache


def collect_trajectories(env, env_wrapped, env_params, levels, num_levels, num_steps, rng):
    """Roll out random agent on levels, return trajectory data."""
    rng, rng_reset = jax.random.split(rng)
    init_obs, init_env_state = jax.vmap(env_wrapped.reset_to_level, (0, 0, None))(
        jax.random.split(rng_reset, num_levels), levels, env_params
    )

    def sample_step(carry, _):
        rng, obs, env_state, last_done = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)
        agent_pos = env_state.env_state.agent_pos
        action = jax.random.randint(rng_action, (num_levels,), 0, 4)
        next_obs, env_state, reward, done, info = jax.vmap(
            env_wrapped.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_levels), env_state, action, env_params)
        # Fake value: distance-to-goal heuristic
        goal_pos = env_state.env_state.goal_pos
        dist_to_goal = jnp.sqrt(((agent_pos - goal_pos) ** 2).sum(axis=-1).astype(jnp.float32))
        value = 1.0 / (1.0 + dist_to_goal)
        carry = (rng, next_obs, env_state, done)
        return carry, (obs.image, action, reward, done, agent_pos, value)

    _, traj = jax.lax.scan(
        sample_step,
        (rng, init_obs, init_env_state, jnp.zeros(num_levels, dtype=bool)),
        None,
        length=num_steps,
    )
    obs_images, actions, rewards, dones, positions, values = traj
    return {
        "observations": np.asarray(obs_images),   # (T, N, 5, 5, 3)
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
    print("Setting up environment...")
    env = Maze(max_height=13, max_width=13, agent_view_size=5, normalize_obs=True)
    env_wrapped = AutoReplayWrapper(env)
    env_params = env_wrapped.default_params

    NUM_RANDOM = 16
    NUM_STEPS = 80
    rng = jax.random.PRNGKey(123)

    # Generate random levels
    sample_random_level = make_level_generator(13, 13, 25)
    rng, rng_levels = jax.random.split(rng)
    levels = jax.vmap(sample_random_level)(jax.random.split(rng_levels, NUM_RANDOM))

    # Collect trajectories
    print(f"Rolling out random agent on {NUM_RANDOM} levels for {NUM_STEPS} steps...")
    rng, rng_rollout = jax.random.split(rng)
    data = collect_trajectories(env, env_wrapped, env_params, levels, NUM_RANDOM, NUM_STEPS, rng_rollout)

    # Build per-level trajectory list
    trajectories = []
    for i in range(NUM_RANDOM):
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
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Phase 1: Pairwise Diversity Metrics (16 random mazes, random agent)", fontsize=14, fontweight='bold')

    metrics_to_plot = [
        ("obs_dtw_distances", "Obs Sequence DTW", "steelblue"),
        ("pos_dtw_distances", "Position Trace DTW", "darkorange"),
        ("value_correlations", "Value Trajectory Correlation", "seagreen"),
        ("value_l2_distances", "Value Trajectory L2 Distance", "mediumpurple"),
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

    # Last subplot: summary stats table
    ax = axes.flat[5]
    ax.axis('off')
    table_data = []
    for key, title, _ in metrics_to_plot:
        v = pairwise[key]
        table_data.append([title.split(" ")[0], f"{v.mean():.3f}", f"{v.std():.3f}", f"{v.min():.3f}", f"{v.max():.3f}"])
    table = ax.table(cellText=table_data,
                     colLabels=["Metric", "Mean", "Std", "Min", "Max"],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    plt.tight_layout()
    fig.savefig(f"{PLOT_DIR}/plots_pairwise_distributions.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {PLOT_DIR}/plots_pairwise_distributions.png")

    # ============================================================
    # FIGURE 2: Pairwise distance matrices (heatmaps)
    # ============================================================
    n = NUM_RANDOM
    metric_matrices = {}
    for key in ["obs_dtw_distances", "pos_dtw_distances", "value_correlations", "jaccard_indices"]:
        mat = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                mat[i, j] = pairwise[key][idx]
                mat[j, i] = pairwise[key][idx]
                idx += 1
        metric_matrices[key] = mat

    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
    fig2.suptitle("Pairwise Distance/Similarity Matrices (16 random mazes)", fontsize=14, fontweight='bold')

    titles = ["Obs Sequence DTW Distance", "Position Trace DTW Distance",
              "Value Trajectory Correlation", "Spatial Footprint Jaccard"]
    keys = ["obs_dtw_distances", "pos_dtw_distances", "value_correlations", "jaccard_indices"]
    cmaps = ["YlOrRd", "YlOrRd", "RdYlGn", "YlGnBu"]

    for idx, (key, title, cmap) in enumerate(zip(keys, titles, cmaps)):
        ax = axes2.flat[idx]
        im = ax.imshow(metric_matrices[key], cmap=cmap, aspect='equal')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Level")
        ax.set_ylabel("Level")
        fig2.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    fig2.savefig(f"{PLOT_DIR}/plots_pairwise_matrices.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {PLOT_DIR}/plots_pairwise_matrices.png")

    # ============================================================
    # FIGURE 3: Example trajectory pair deep-dive
    # ============================================================
    # Pick the most similar and most different pairs by position DTW
    pos_dists = pairwise["pos_dtw_distances"]
    pair_indices = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            pair_indices.append((i, j))
            idx += 1

    most_similar_idx = np.argmin(pos_dists)
    most_different_idx = np.argmax(pos_dists)
    pair_sim = pair_indices[most_similar_idx]
    pair_diff = pair_indices[most_different_idx]

    fig3, axes3 = plt.subplots(2, 4, figsize=(20, 10))
    fig3.suptitle("Trajectory Deep-Dive: Most Similar vs Most Different Pairs", fontsize=14, fontweight='bold')

    for row, (pair, label) in enumerate([(pair_sim, "Most Similar"), (pair_diff, "Most Different")]):
        i, j = pair
        ti, tj = trajectories[i], trajectories[j]

        # Position traces on maze
        wall_maps = np.asarray(levels.wall_map)
        for col, (env_idx, traj_data, color) in enumerate([(i, ti, 'blue'), (j, tj, 'red')]):
            ax = axes3[row, col]
            img = wall_map_to_grid(wall_maps[env_idx])
            ax.imshow(img, origin='upper')
            pos = traj_data["positions"]
            dones = traj_data["dones"]
            done_idx = np.where(dones)[0]
            end = done_idx[0] + 1 if len(done_idx) > 0 else len(pos)
            pos_trunc = pos[:end]
            # Plot path with time gradient
            for t in range(len(pos_trunc) - 1):
                alpha = 0.3 + 0.7 * (t / max(len(pos_trunc) - 1, 1))
                ax.plot([pos_trunc[t, 0], pos_trunc[t+1, 0]],
                        [pos_trunc[t, 1], pos_trunc[t+1, 1]],
                        color=color, alpha=alpha, linewidth=2)
            ax.plot(pos_trunc[0, 0], pos_trunc[0, 1], 'o', color=color, markersize=8, label='start')
            if end > 1:
                ax.plot(pos_trunc[-1, 0], pos_trunc[-1, 1], 's', color=color, markersize=8, label='end')
            ax.set_title(f"{label} — Level {env_idx}", fontsize=10)
            ax.legend(fontsize=8)
            ax.set_xlim(-0.5, 12.5)
            ax.set_ylim(12.5, -0.5)

        # DTW warping path local costs (similarity profile)
        ax_dtw = axes3[row, 2]
        pos_result = position_trace_dtw(ti["positions"], ti["dones"], tj["positions"], tj["dones"])
        ax_dtw.plot(pos_result["local_costs"], color='darkorange', linewidth=1.5)
        ax_dtw.fill_between(range(len(pos_result["local_costs"])), pos_result["local_costs"],
                            alpha=0.3, color='darkorange')
        ax_dtw.set_title(f"{label} — DTW Similarity Profile\n(pos DTW = {pos_result['distance']:.3f})", fontsize=10)
        ax_dtw.set_xlabel("Warping path step")
        ax_dtw.set_ylabel("Local cost (distance)")
        ax_dtw.grid(alpha=0.3)

        # Value trajectories
        ax_val = axes3[row, 3]
        val_result = value_trajectory_correlation(ti["values"], ti["dones"], tj["values"], tj["dones"])
        ax_val.plot(val_result["resampled_a"], label=f"Level {i}", color='blue', linewidth=1.5)
        ax_val.plot(val_result["resampled_b"], label=f"Level {j}", color='red', linewidth=1.5)
        ax_val.set_title(f"{label} — Value Trajectories\n(corr={val_result['correlation']:.3f}, L2={val_result['l2_distance']:.3f})", fontsize=10)
        ax_val.set_xlabel("Normalized time")
        ax_val.set_ylabel("Value estimate")
        ax_val.legend(fontsize=8)
        ax_val.grid(alpha=0.3)

    plt.tight_layout()
    fig3.savefig(f"{PLOT_DIR}/plots_trajectory_deepdive.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {PLOT_DIR}/plots_trajectory_deepdive.png")

    # ============================================================
    # FIGURE 4: Spatial footprints
    # ============================================================
    fig4, axes4 = plt.subplots(2, 8, figsize=(20, 5))
    fig4.suptitle("Spatial Footprints: Cells Visited by Random Agent (16 levels)", fontsize=13, fontweight='bold')

    wall_maps = np.asarray(levels.wall_map)
    for i in range(NUM_RANDOM):
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
        ax.set_title(f"L{i} ({len(visited)} cells)", fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    fig4.savefig(f"{PLOT_DIR}/plots_spatial_footprints.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {PLOT_DIR}/plots_spatial_footprints.png")

    print(f"\nDone! All 4 figures saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
