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

PLOT_DIR = os.path.join(PROJECT_ROOT, 'metrics', 'plots')

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
from metrics.pairwise.mode_transition import classify_modes, mode_transition_divergence, compute_baseline_stats, MODE_NAMES, NUM_MODES
from metrics.pairwise.td_error_distribution import compute_td_errors, td_error_divergence
from metrics.standalone.per_step_entropy import compute_per_step_entropy
from metrics.standalone.per_step_regret import compute_per_step_regret
from metrics.standalone.per_step_action import compute_per_step_action
from metrics.standalone.value_error import compute_value_error
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


def plot_deepdive_experience(axes_row, pair, pair_label, trajectories, ve_infos, mode_infos, wall_maps):
    """Deep-dive row for value error + mode transition metrics.

    Cols: grid A, grid B, value error curves, mode sequences, mode fractions bar, transition heatmaps.
    """
    li, lj = pair
    ti, tj = trajectories[li], trajectories[lj]

    # Col 0-1: maze grids with paths (same as original)
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

    # Col 2: Signed value error curves (both overlaid)
    ax = axes_row[2]
    vi, vj = ve_infos[li], ve_infos[lj]
    if vi["episode_length"] > 0:
        ax.plot(vi["error_curve"], color='steelblue', linewidth=1.2, alpha=0.8,
                label=f'L{li} (mean={vi["mean_error"]:.3f})')
    if vj["episode_length"] > 0:
        ax.plot(vj["error_curve"], color='firebrick', linewidth=1.2, alpha=0.8,
                label=f'L{lj} (mean={vj["mean_error"]:.3f})')
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_title(f"{pair_label} — Signed Value Error", fontsize=10)
    ax.set_xlabel("Step"); ax.set_ylabel("V(s_t) - G_t")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Col 3: Mode sequences as colored bars
    ax = axes_row[3]
    mode_colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db', '#9b59b6']
    mi, mj = mode_infos[li], mode_infos[lj]
    if mi["episode_length"] > 0:
        for t in range(mi["episode_length"]):
            ax.barh(1, 1, left=t, height=0.4, color=mode_colors[mi["modes"][t]])
    if mj["episode_length"] > 0:
        for t in range(mj["episode_length"]):
            ax.barh(0, 1, left=t, height=0.4, color=mode_colors[mj["modes"][t]])
    ax.set_yticks([0, 1]); ax.set_yticklabels([f'L{lj}', f'L{li}'], fontsize=9)
    ax.set_title(f"{pair_label} — Mode Sequence", fontsize=10)
    ax.set_xlabel("Step")
    # Legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=mode_colors[m], label=MODE_NAMES[m]) for m in range(NUM_MODES)]
    ax.legend(handles=legend_patches, fontsize=6, loc='upper right', ncol=2)

    # Col 4: Mode fractions bar chart (side by side)
    ax = axes_row[4]
    x = np.arange(NUM_MODES)
    w = 0.35
    ax.bar(x - w/2, mi["mode_fractions"], w, color='steelblue', alpha=0.8, label=f'L{li}')
    ax.bar(x + w/2, mj["mode_fractions"], w, color='firebrick', alpha=0.8, label=f'L{lj}')
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace('_', '\n') for n in MODE_NAMES], fontsize=7)
    ax.set_title(f"{pair_label} — Mode Fractions", fontsize=10)
    ax.set_ylabel("Fraction"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    # Col 5: Transition heatmap for level i
    ax = axes_row[5]
    im = ax.imshow(mi["transition_probs"], vmin=0, vmax=1, cmap='YlOrRd', aspect='equal')
    ax.set_xticks(range(NUM_MODES)); ax.set_yticks(range(NUM_MODES))
    short_names = ['CC', 'CW', 'Unc', 'Rec', 'Deg']
    ax.set_xticklabels(short_names, fontsize=8); ax.set_yticklabels(short_names, fontsize=8)
    ax.set_title(f"L{li} Transitions", fontsize=10)
    ax.set_xlabel("To"); ax.set_ylabel("From")
    for r in range(NUM_MODES):
        for c in range(NUM_MODES):
            v = mi["transition_probs"][r, c]
            if v > 0.01:
                ax.text(c, r, f'{v:.2f}', ha='center', va='center', fontsize=7,
                        color='white' if v > 0.5 else 'black')

    # Col 6: Transition heatmap for level j
    ax = axes_row[6]
    im = ax.imshow(mj["transition_probs"], vmin=0, vmax=1, cmap='YlOrRd', aspect='equal')
    ax.set_xticks(range(NUM_MODES)); ax.set_yticks(range(NUM_MODES))
    ax.set_xticklabels(short_names, fontsize=8); ax.set_yticklabels(short_names, fontsize=8)
    ax.set_title(f"L{lj} Transitions", fontsize=10)
    ax.set_xlabel("To"); ax.set_ylabel("From")
    for r in range(NUM_MODES):
        for c in range(NUM_MODES):
            v = mj["transition_probs"][r, c]
            if v > 0.01:
                ax.text(c, r, f'{v:.2f}', ha='center', va='center', fontsize=7,
                        color='white' if v > 0.5 else 'black')


def plot_deepdive_td_error(axes_row, pair, pair_label, trajectories, wall_maps):
    """Deep-dive row for TD error distribution comparison.

    Cols: grid A, grid B, TD error time series, TD error histograms,
          cumulative distributions, per-step regret overlay, per-step entropy overlay.
    """
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

    # Compute TD errors and divergence
    td_result = td_error_divergence(ti, ti["dones"], tj, tj["dones"])
    td_a, td_b = td_result["td_errors_a"], td_result["td_errors_b"]

    # Col 2: TD error time series (both overlaid)
    ax = axes_row[2]
    if len(td_a) > 0:
        ax.plot(td_a, color='steelblue', linewidth=1.2, alpha=0.8,
                label=f'L{li} (μ={td_result["mean_a"]:.3f})')
    if len(td_b) > 0:
        ax.plot(td_b, color='firebrick', linewidth=1.2, alpha=0.8,
                label=f'L{lj} (μ={td_result["mean_b"]:.3f})')
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_title(f"{pair_label} — TD Errors (δ_t)", fontsize=10)
    ax.set_xlabel("Step"); ax.set_ylabel("r + γV(s') - V(s)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Col 3: TD error histograms (overlaid)
    ax = axes_row[3]
    bin_centers = (td_result["bin_edges"][:-1] + td_result["bin_edges"][1:]) / 2
    ax.bar(bin_centers, td_result["histogram_a"], width=bin_centers[1]-bin_centers[0],
           color='steelblue', alpha=0.5, label=f'L{li}')
    ax.bar(bin_centers, td_result["histogram_b"], width=bin_centers[1]-bin_centers[0],
           color='firebrick', alpha=0.5, label=f'L{lj}')
    ax.set_title(f"{pair_label} — TD Error Distribution (EMD={td_result['emd']:.3f})", fontsize=10)
    ax.set_xlabel("TD Error (δ)"); ax.set_ylabel("Density")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Col 4: Cumulative distributions (ECDFs)
    ax = axes_row[4]
    if len(td_a) > 0:
        sorted_a = np.sort(td_a)
        ecdf_a = np.arange(1, len(sorted_a) + 1) / len(sorted_a)
        ax.plot(sorted_a, ecdf_a, color='steelblue', linewidth=1.5, label=f'L{li}')
    if len(td_b) > 0:
        sorted_b = np.sort(td_b)
        ecdf_b = np.arange(1, len(sorted_b) + 1) / len(sorted_b)
        ax.plot(sorted_b, ecdf_b, color='firebrick', linewidth=1.5, label=f'L{lj}')
    ax.set_title(f"{pair_label} — Cumulative Distribution", fontsize=10)
    ax.set_xlabel("TD Error (δ)"); ax.set_ylabel("CDF")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Col 5: Absolute TD error (learning magnitude) comparison
    ax = axes_row[5]
    if len(td_a) > 0:
        ax.plot(np.abs(td_a), color='steelblue', linewidth=1.0, alpha=0.7,
                label=f'L{li} (μ={np.mean(np.abs(td_a)):.3f})')
    if len(td_b) > 0:
        ax.plot(np.abs(td_b), color='firebrick', linewidth=1.0, alpha=0.7,
                label=f'L{lj} (μ={np.mean(np.abs(td_b)):.3f})')
    ax.set_title(f"{pair_label} — |TD Error| (Learning Magnitude)", fontsize=10)
    ax.set_xlabel("Step"); ax.set_ylabel("|δ_t|")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Col 6: Sign of TD error (overestimate vs underestimate)
    ax = axes_row[6]
    if len(td_a) > 0:
        pos_frac_a = np.mean(td_a > 0)
        ax.barh(1, pos_frac_a, height=0.4, color='#e74c3c', alpha=0.7, label='underestimate (δ>0)')
        ax.barh(1, -(1 - pos_frac_a), height=0.4, color='#3498db', alpha=0.7, label='overestimate (δ<0)')
    if len(td_b) > 0:
        pos_frac_b = np.mean(td_b > 0)
        ax.barh(0, pos_frac_b, height=0.4, color='#e74c3c', alpha=0.7)
        ax.barh(0, -(1 - pos_frac_b), height=0.4, color='#3498db', alpha=0.7)
    ax.set_yticks([0, 1]); ax.set_yticklabels([f'L{lj}', f'L{li}'], fontsize=9)
    ax.set_xlim(-1.1, 1.1)
    ax.axvline(0, color='gray', linewidth=0.8)
    ax.set_title(f"{pair_label} — TD Error Sign Balance", fontsize=10)
    ax.set_xlabel("← overestimate | underestimate →")
    ax.legend(fontsize=7, loc='lower right'); ax.grid(alpha=0.3, axis='x')


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

    # ============================================================
    # Compute new metrics: value error + mode transitions
    # ============================================================
    print("Computing pairwise TD error divergence (EMD)...")
    td_emd_dists = []
    for i in range(NUM_LEVELS):
        for j in range(i + 1, NUM_LEVELS):
            ti, tj = trajectories[i], trajectories[j]
            td_result = td_error_divergence(ti, ti["dones"], tj, tj["dones"])
            td_emd_dists.append(td_result["emd"])
    td_emd_dists = np.array(td_emd_dists)

    print("Computing value error profiles...")
    ve_infos = [compute_value_error(t["values"], t["rewards"], t["dones"]) for t in trajectories]

    print("Computing baseline stats from all trajectories...")
    baseline = compute_baseline_stats(trajectories)
    print(f"  error: mean={baseline['error_mean']:.3f}, std={baseline['error_std']:.3f}, "
          f"threshold={baseline['error_threshold']:.3f}")
    print(f"  entropy: mean={baseline['entropy_mean']:.3f}, std={baseline['entropy_std']:.3f}, "
          f"threshold={baseline['entropy_threshold']:.3f}")

    print("Computing mode classifications...")
    mode_infos = [classify_modes(t["values"], t["rewards"], t["dones"],
                                 entropy=t.get("entropy"),
                                 baseline_stats=baseline) for t in trajectories]

    print("Computing pairwise mode transition divergence...")
    mode_div_dists = []
    for i in range(NUM_LEVELS):
        for j in range(i + 1, NUM_LEVELS):
            ti, tj = trajectories[i], trajectories[j]
            div = mode_transition_divergence(
                ti, ti["dones"], tj, tj["dones"],
                entropy_a=ti.get("entropy"), entropy_b=tj.get("entropy"),
                baseline_stats=baseline,
            )
            mode_div_dists.append(div["kl_divergence"])
    mode_div_dists = np.array(mode_div_dists)

    # Value error pairwise: L1 distance of error curves (truncated to min length)
    print("Computing pairwise value error distances...")
    ve_dists = []
    for i in range(NUM_LEVELS):
        for j in range(i + 1, NUM_LEVELS):
            ei, ej = ve_infos[i]["error_curve"], ve_infos[j]["error_curve"]
            min_len = min(len(ei), len(ej))
            if min_len > 0:
                ve_dists.append(float(np.mean(np.abs(ei[:min_len] - ej[:min_len]))))
            else:
                ve_dists.append(0.0)
    ve_dists = np.array(ve_dists)

    # ============================================================
    # FIGURE 4: Deep-dive by value error distance
    # ============================================================
    pair_sim_ve = pair_indices[np.argmin(ve_dists)]
    pair_diff_ve = pair_indices[np.argmax(ve_dists)]

    fig4, axes4 = plt.subplots(2, 7, figsize=(35, 10))
    fig4.suptitle("Deep-Dive: Most Similar vs Most Different by Value Error Profile",
                  fontsize=14, fontweight='bold')
    plot_deepdive_experience(axes4[0], pair_sim_ve, "Most Similar (value error)",
                             trajectories, ve_infos, mode_infos, wall_maps)
    plot_deepdive_experience(axes4[1], pair_diff_ve, "Most Different (value error)",
                             trajectories, ve_infos, mode_infos, wall_maps)
    plt.tight_layout()
    fig4.savefig(f"{PLOT_DIR}/plots_deepdive_value_error.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {PLOT_DIR}/plots_deepdive_value_error.png")
    plt.close(fig4)

    # ============================================================
    # FIGURE 5: Deep-dive by mode transition divergence
    # ============================================================
    pair_sim_mode = pair_indices[np.argmin(mode_div_dists)]
    pair_diff_mode = pair_indices[np.argmax(mode_div_dists)]

    fig5, axes5 = plt.subplots(2, 7, figsize=(35, 10))
    fig5.suptitle("Deep-Dive: Most Similar vs Most Different by Experience Mode Divergence",
                  fontsize=14, fontweight='bold')
    plot_deepdive_experience(axes5[0], pair_sim_mode, "Most Similar (experience)",
                             trajectories, ve_infos, mode_infos, wall_maps)
    plot_deepdive_experience(axes5[1], pair_diff_mode, "Most Different (experience)",
                             trajectories, ve_infos, mode_infos, wall_maps)
    plt.tight_layout()
    fig5.savefig(f"{PLOT_DIR}/plots_deepdive_mode_transition.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {PLOT_DIR}/plots_deepdive_mode_transition.png")
    plt.close(fig5)

    # ============================================================
    # FIGURE 6: Deep-dive by TD error distribution EMD
    # ============================================================
    pair_sim_td = pair_indices[np.argmin(td_emd_dists)]
    pair_diff_td = pair_indices[np.argmax(td_emd_dists)]

    fig6, axes6 = plt.subplots(2, 7, figsize=(35, 10))
    fig6.suptitle("Deep-Dive: Most Similar vs Most Different by TD Error Distribution (EMD)",
                  fontsize=14, fontweight='bold')
    plot_deepdive_td_error(axes6[0], pair_sim_td, "Most Similar (TD error)",
                           trajectories, wall_maps)
    plot_deepdive_td_error(axes6[1], pair_diff_td, "Most Different (TD error)",
                           trajectories, wall_maps)
    plt.tight_layout()
    fig6.savefig(f"{PLOT_DIR}/plots_deepdive_td_error.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {PLOT_DIR}/plots_deepdive_td_error.png")
    plt.close(fig6)

    print(f"\nDone! All figures saved to {PLOT_DIR}/")
    corr_pa = np.corrcoef(pos_dtw_dists, act_dtw_dists)[0, 1]
    corr_pr = np.corrcoef(pos_dtw_dists, reg_dtw_dists)[0, 1]
    corr_ar = np.corrcoef(act_dtw_dists, reg_dtw_dists)[0, 1]
    corr_pm = np.corrcoef(pos_dtw_dists, mode_div_dists)[0, 1]
    corr_am = np.corrcoef(act_dtw_dists, mode_div_dists)[0, 1]
    corr_rm = np.corrcoef(reg_dtw_dists, mode_div_dists)[0, 1]
    corr_ve_m = np.corrcoef(ve_dists, mode_div_dists)[0, 1]
    corr_pt = np.corrcoef(pos_dtw_dists, td_emd_dists)[0, 1]
    corr_at = np.corrcoef(act_dtw_dists, td_emd_dists)[0, 1]
    corr_rt = np.corrcoef(reg_dtw_dists, td_emd_dists)[0, 1]
    corr_mt = np.corrcoef(mode_div_dists, td_emd_dists)[0, 1]
    corr_vet = np.corrcoef(ve_dists, td_emd_dists)[0, 1]
    print(f"\nStats:")
    print(f"  Pos DTW:         mean={pos_dtw_dists.mean():.3f}, std={pos_dtw_dists.std():.3f}")
    print(f"  Action DTW:      mean={act_dtw_dists.mean():.3f}, std={act_dtw_dists.std():.3f}")
    print(f"  Regret DTW:      mean={reg_dtw_dists.mean():.3f}, std={reg_dtw_dists.std():.3f}")
    print(f"  Value Error L1:  mean={ve_dists.mean():.3f}, std={ve_dists.std():.3f}")
    print(f"  Mode Div (KL):   mean={mode_div_dists.mean():.3f}, std={mode_div_dists.std():.3f}")
    print(f"  Norm TD Error EMD: mean={td_emd_dists.mean():.3f}, std={td_emd_dists.std():.3f}")
    print(f"\n  Correlations:")
    print(f"    pos-action r={corr_pa:.3f}, pos-regret r={corr_pr:.3f}, action-regret r={corr_ar:.3f}")
    print(f"    pos-mode r={corr_pm:.3f}, action-mode r={corr_am:.3f}, regret-mode r={corr_rm:.3f}")
    print(f"    pos-td r={corr_pt:.3f}, action-td r={corr_at:.3f}, regret-td r={corr_rt:.3f}")
    print(f"    mode-td r={corr_mt:.3f}, value_error-td r={corr_vet:.3f}")
    print(f"    value_error-mode r={corr_ve_m:.3f}")


if __name__ == "__main__":
    main()
