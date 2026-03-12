"""
Phase 1 demo: Deep-dive visualization of diversity metrics.

Figures:
  1. Deep-dive by position DTW (most similar vs most different)
  2. Deep-dive by regret DTW (most similar vs most different difficulty)

Deep-dive panels: pos DTW profile, regret DTW profile, per-step regret, per-step entropy.

Usage:
    python3 metrics/scripts/plot_metrics_demo.py
"""

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'examples'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'vae'))
sys.path.insert(0, PROJECT_ROOT)

PLOT_DIR = os.path.join(PROJECT_ROOT, 'test_log', 'phase_1')

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from jaxued.environments import Maze
from jaxued.environments.maze import Level

from metrics.pairwise.pos_dtw import position_trace_dtw
from metrics.pairwise.regret_dtw import regret_curve_dtw
from metrics.pairwise.action_dtw_binary import action_sequence_distance
from metrics.standalone.per_step_entropy import compute_per_step_entropy
from metrics.standalone.per_step_regret import compute_per_step_regret
from metrics.standalone.per_step_action import compute_per_step_action
from metrics.standalone.regret import compute_regret

from cross_evaluate import load_agent, tokens_to_levels_batch
from maze_plr import ActorCritic


def collect_trajectories(train_state, env_params, levels, num_levels, num_steps, rng):
    """Roll out trained agent, return trajectory data with entropy."""
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
        x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, done))
        hstate, pi, value = train_state.apply_fn(train_state.params, x, hstate)
        action = pi.sample(seed=rng_action).squeeze(0)
        value = value.squeeze(0)
        entropy = pi.entropy().squeeze(0)
        next_obs, next_state, reward, next_done, _ = jax.vmap(
            eval_env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_levels), state, action, env_params)
        carry = (rng, hstate, next_obs, next_state, next_done)
        return carry, (obs.image, action, reward, done, agent_pos, value, entropy)

    _, traj = jax.lax.scan(
        step,
        (rng, init_hstate, init_obs, init_env_state, jnp.zeros(num_levels, dtype=bool)),
        None, length=num_steps,
    )
    obs_images, actions, rewards, dones, positions, values, entropy = traj
    return {
        "observations": np.asarray(obs_images),
        "actions": np.asarray(actions),
        "rewards": np.asarray(rewards),
        "dones": np.asarray(dones),
        "positions": np.asarray(positions),
        "values": np.asarray(values),
        "entropy": np.asarray(entropy),
    }


def wall_map_to_grid(wall_map_np):
    h, w = wall_map_np.shape
    img = np.ones((h, w, 3), dtype=np.float32)
    img[wall_map_np] = [0.2, 0.2, 0.2]
    return img


def plot_deepdive(axes_row, pair, pair_label, trajectories, ent_infos, reg_infos, wall_maps):
    """Plot a deep-dive row: 2 grids, pos DTW, action DTW, regret DTW, per-step regret, per-step entropy."""
    li, lj = pair
    ti, tj = trajectories[li], trajectories[lj]

    # Col 0-1: maze grids with paths
    for col, (env_idx, traj_data, color) in enumerate([(li, ti, 'blue'), (lj, tj, 'red')]):
        ax = axes_row[col]
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
        ax.set_title(f"{pair_label} — L{env_idx} ({len(pos_trunc)} steps)", fontsize=10)
        ax.legend(fontsize=8); ax.set_xlim(-0.5, 12.5); ax.set_ylim(12.5, -0.5)

    # Col 2: Position DTW profile
    ax = axes_row[2]
    pos_result = position_trace_dtw(ti["positions"], ti["dones"], tj["positions"], tj["dones"])
    ax.plot(pos_result["local_costs"], color='darkorange', linewidth=1.5)
    ax.fill_between(range(len(pos_result["local_costs"])), pos_result["local_costs"],
                    alpha=0.3, color='darkorange')
    ax.set_title(f"{pair_label} — Pos DTW (d={pos_result['distance']:.3f})", fontsize=10)
    ax.set_xlabel("Warping step"); ax.set_ylabel("Local cost"); ax.grid(alpha=0.3)

    # Col 3: Action DTW profile
    ax = axes_row[3]
    act_dtw = action_sequence_distance(ti["actions"], ti["dones"], tj["actions"], tj["dones"])
    if len(act_dtw["local_costs"]) > 0:
        ax.plot(act_dtw["local_costs"], color='forestgreen', linewidth=1.5)
        ax.fill_between(range(len(act_dtw["local_costs"])), act_dtw["local_costs"],
                        alpha=0.3, color='forestgreen')
    ax.set_title(f"{pair_label} — Action Dist (d={act_dtw['distance']:.3f})", fontsize=10)
    ax.set_xlabel("Warping step"); ax.set_ylabel("Local cost"); ax.grid(alpha=0.3)

    # Col 4: Regret DTW profile
    ax = axes_row[4]
    reg_dtw = regret_curve_dtw(
        ti["values"], ti["rewards"], ti["dones"],
        tj["values"], tj["rewards"], tj["dones"],
    )
    if len(reg_dtw["local_costs"]) > 0:
        ax.plot(reg_dtw["local_costs"], color='firebrick', linewidth=1.5)
        ax.fill_between(range(len(reg_dtw["local_costs"])), reg_dtw["local_costs"],
                        alpha=0.3, color='firebrick')
    ax.set_title(f"{pair_label} — Regret DTW (d={reg_dtw['distance']:.3f})", fontsize=10)
    ax.set_xlabel("Warping step"); ax.set_ylabel("Local cost"); ax.grid(alpha=0.3)

    # Col 5: Per-step regret (both levels overlaid)
    ax = axes_row[5]
    ri, rj = reg_infos[li], reg_infos[lj]
    if ri["episode_length"] > 0:
        ax.plot(ri["regret_curve"], color='firebrick', linewidth=1.2,
                label=f'L{li} (m={ri["mean_regret"]:.3f})', alpha=0.8)
    if rj["episode_length"] > 0:
        ax.plot(rj["regret_curve"], color='purple', linewidth=1.2,
                label=f'L{lj} (m={rj["mean_regret"]:.3f})', alpha=0.8)
    ax.set_title(f"{pair_label} — Per-step Regret", fontsize=10)
    ax.set_xlabel("Step"); ax.set_ylabel("Regret"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Col 6: Per-step entropy (both levels overlaid)
    ax = axes_row[6]
    ei, ej = ent_infos[li], ent_infos[lj]
    if ei["episode_length"] > 0:
        ax.plot(ei["entropy"], color='teal', linewidth=1.2,
                label=f'L{li} (m={ei["mean"]:.3f})', alpha=0.8)
    if ej["episode_length"] > 0:
        ax.plot(ej["entropy"], color='darkorange', linewidth=1.2,
                label=f'L{lj} (m={ej["mean"]:.3f})', alpha=0.8)
    ax.set_title(f"{pair_label} — Per-step Entropy", fontsize=10)
    ax.set_xlabel("Step"); ax.set_ylabel("Entropy"); ax.legend(fontsize=8); ax.grid(alpha=0.3)


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Load agent + buffer
    agent_dir = os.path.join(PROJECT_ROOT, "gcs_artifacts", "agent", "cmaes_vae_beta2.0_seed0_198")
    print(f"Loading agent from {agent_dir}...")
    train_state, config, env, env_params = load_agent(agent_dir, checkpoint_step=198)
    assert train_state is not None

    buffer_path = os.path.join(PROJECT_ROOT, "gcs_artifacts", "buffer", "buffer_dump_final.npz")
    print(f"Loading buffer from {buffer_path}...")
    buffer_data = np.load(buffer_path, allow_pickle=True)
    tokens = buffer_data["tokens"]
    size = int(buffer_data.get("size", len(tokens)))
    tokens = tokens[:size]
    print(f"  {size} levels")

    all_levels = tokens_to_levels_batch(tokens)

    NUM_LEVELS = 16
    indices = np.linspace(0, size - 1, NUM_LEVELS, dtype=int)
    levels = jax.tree_util.tree_map(lambda x: x[indices], all_levels)
    print(f"  Selected {NUM_LEVELS} levels at indices: {indices.tolist()}")

    # Collect trajectories
    NUM_STEPS = 250
    rng = jax.random.PRNGKey(42)
    print(f"Rolling out agent on {NUM_LEVELS} levels...")
    data = collect_trajectories(train_state, env_params, levels, NUM_LEVELS, NUM_STEPS, rng)

    trajectories = []
    for i in range(NUM_LEVELS):
        trajectories.append({
            "positions": data["positions"][:, i],
            "values": data["values"][:, i],
            "dones": data["dones"][:, i],
            "rewards": data["rewards"][:, i],
            "entropy": data["entropy"][:, i],
            "actions": data["actions"][:, i],
        })

    wall_maps = np.asarray(levels.wall_map)

    # Compute per-level metrics
    print("Computing per-level metrics...")
    ent_infos = [compute_per_step_entropy(t["entropy"], t["dones"]) for t in trajectories]
    reg_infos = [compute_per_step_regret(t["values"], t["rewards"], t["dones"]) for t in trajectories]

    # Compute pairwise metrics
    print("Computing pairwise DTW (position, action, regret)...")
    pair_indices = []
    pos_dtw_dists = []
    act_dtw_dists = []
    reg_dtw_dists = []
    for i in range(NUM_LEVELS):
        for j in range(i + 1, NUM_LEVELS):
            pair_indices.append((i, j))
            ti, tj = trajectories[i], trajectories[j]

            r = position_trace_dtw(
                ti["positions"], ti["dones"],
                tj["positions"], tj["dones"],
            )
            pos_dtw_dists.append(r["distance"])

            a = action_sequence_distance(
                ti["actions"], ti["dones"],
                tj["actions"], tj["dones"],
            )
            act_dtw_dists.append(a["distance"])

            rd = regret_curve_dtw(
                ti["values"], ti["rewards"], ti["dones"],
                tj["values"], tj["rewards"], tj["dones"],
            )
            reg_dtw_dists.append(rd["distance"])

    pos_dtw_dists = np.array(pos_dtw_dists)
    act_dtw_dists = np.array(act_dtw_dists)
    reg_dtw_dists = np.array(reg_dtw_dists)

    # ============================================================
    # FIGURE 1: Deep-dive by position DTW
    # ============================================================
    pair_sim_pos = pair_indices[np.argmin(pos_dtw_dists)]
    pair_diff_pos = pair_indices[np.argmax(pos_dtw_dists)]

    fig1, axes1 = plt.subplots(2, 7, figsize=(35, 10))
    fig1.suptitle("Deep-Dive: Most Similar vs Most Different by Position DTW",
                  fontsize=14, fontweight='bold')
    plot_deepdive(axes1[0], pair_sim_pos, "Most Similar (pos)", trajectories, ent_infos, reg_infos, wall_maps)
    plot_deepdive(axes1[1], pair_diff_pos, "Most Different (pos)", trajectories, ent_infos, reg_infos, wall_maps)
    plt.tight_layout()
    fig1.savefig(f"{PLOT_DIR}/plots_deepdive_pos.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {PLOT_DIR}/plots_deepdive_pos.png")
    plt.close(fig1)

    # ============================================================
    # FIGURE 2: Deep-dive by action DTW
    # ============================================================
    pair_sim_act = pair_indices[np.argmin(act_dtw_dists)]
    pair_diff_act = pair_indices[np.argmax(act_dtw_dists)]

    fig2, axes2 = plt.subplots(2, 7, figsize=(35, 10))
    fig2.suptitle("Deep-Dive: Most Similar vs Most Different by Action Distance (Behavioral Diversity)",
                  fontsize=14, fontweight='bold')
    plot_deepdive(axes2[0], pair_sim_act, "Most Similar (action)", trajectories, ent_infos, reg_infos, wall_maps)
    plot_deepdive(axes2[1], pair_diff_act, "Most Different (action)", trajectories, ent_infos, reg_infos, wall_maps)
    plt.tight_layout()
    fig2.savefig(f"{PLOT_DIR}/plots_deepdive_action.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {PLOT_DIR}/plots_deepdive_action.png")
    plt.close(fig2)

    # ============================================================
    # FIGURE 3: Deep-dive by regret DTW
    # ============================================================
    pair_sim_reg = pair_indices[np.argmin(reg_dtw_dists)]
    pair_diff_reg = pair_indices[np.argmax(reg_dtw_dists)]

    fig3, axes3 = plt.subplots(2, 7, figsize=(35, 10))
    fig3.suptitle("Deep-Dive: Most Similar vs Most Different by Regret DTW (Difficulty Profile)",
                  fontsize=14, fontweight='bold')
    plot_deepdive(axes3[0], pair_sim_reg, "Most Similar (regret)", trajectories, ent_infos, reg_infos, wall_maps)
    plot_deepdive(axes3[1], pair_diff_reg, "Most Different (regret)", trajectories, ent_infos, reg_infos, wall_maps)
    plt.tight_layout()
    fig3.savefig(f"{PLOT_DIR}/plots_deepdive_regret.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {PLOT_DIR}/plots_deepdive_regret.png")
    plt.close(fig3)

    print(f"\nDone! All figures saved to {PLOT_DIR}/")
    corr_pa = np.corrcoef(pos_dtw_dists, act_dtw_dists)[0, 1]
    corr_pr = np.corrcoef(pos_dtw_dists, reg_dtw_dists)[0, 1]
    corr_ar = np.corrcoef(act_dtw_dists, reg_dtw_dists)[0, 1]
    print(f"\nStats:")
    print(f"  Pos DTW:    mean={pos_dtw_dists.mean():.3f}, std={pos_dtw_dists.std():.3f}")
    print(f"  Action DTW: mean={act_dtw_dists.mean():.3f}, std={act_dtw_dists.std():.3f}")
    print(f"  Regret DTW: mean={reg_dtw_dists.mean():.3f}, std={reg_dtw_dists.std():.3f}")
    print(f"  Correlations: pos-action r={corr_pa:.3f}, pos-regret r={corr_pr:.3f}, action-regret r={corr_ar:.3f}")


if __name__ == "__main__":
    main()
