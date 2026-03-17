"""
Measure policy non-stationarity: evaluate the same set of levels (buffer at 10k)
under checkpoints from 10k to 20k PPO updates, computing SFL scores.

Usage:
    python trajectory/score_nonstationarity.py

Outputs a .npz + plot showing how scores drift as the policy trains.
"""
import os
import sys
import subprocess
import shutil
import numpy as np

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import orthogonal, constant
import distrax
import orbax.checkpoint as ocp

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from jaxued.environments.maze import Maze
from jaxued.linen import ResetRNN
from vae.vae_level_utils import tokens_to_level

# ── Config ──────────────────────────────────────────────────────────────────
GCS_BUCKET = "ucl-ued-project-bucket"
GCS_PREFIX = "accel"
RUN_NAME = "accel_sfl"
SEED = 1
GCLOUD_BIN = shutil.which("gsutil") or "/cs/student/project_msc/2025/csml/rhautier/google-cloud-sdk/bin/gsutil"

BUFFER_GCS = f"gs://{GCS_BUCKET}/{GCS_PREFIX}/buffer_dumps/{RUN_NAME}/{SEED}/buffer_dump_10k.npz"
CKPT_GCS_BASE = f"gs://{GCS_BUCKET}/{GCS_PREFIX}/checkpoints/{RUN_NAME}/{SEED}/models"

LOCAL_DIR = "/tmp/nonstationarity_analysis"
BUFFER_LOCAL = os.path.join(LOCAL_DIR, "buffer_dump_10k.npz")
CKPT_LOCAL_BASE = os.path.join(LOCAL_DIR, "models")

# eval_freq=250, checkpoints saved every 2 eval steps = every 500 updates
# eval_step 0 = after 250 updates, step N = after (N+1)*250 updates
# step 38 = 9750 updates (closest before 10k buffer dump)
# step 78 = 19750 updates, step 80 = 20250 updates
EVAL_STEPS = list(range(38, 82, 2))  # 38, 40, 42, ..., 80

NUM_SFL_ROLLOUTS = 5
AGENT_VIEW_SIZE = 5
MAX_STEPS = 250
BATCH_SIZE = 64


# ── Model (matches maze_plr.py exactly) ────────────────────────────────────
class ActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, inputs, hidden):
        obs, dones = inputs

        img_embed = nn.Conv(16, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(obs.image)
        img_embed = img_embed.reshape(*img_embed.shape[:-3], -1)
        img_embed = nn.relu(img_embed)

        dir_embed = jax.nn.one_hot(obs.agent_dir, 4)
        dir_embed = nn.Dense(5, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0), name="scalar_embed")(dir_embed)

        embedding = jnp.append(img_embed, dir_embed, axis=-1)

        hidden, embedding = ResetRNN(nn.OptimizedLSTMCell(features=256))((embedding, dones), initial_carry=hidden)

        actor_mean = nn.Dense(32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="actor0")(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name="actor1")(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(32, kernel_init=orthogonal(2), bias_init=constant(0.0), name="critic0")(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic1")(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)

    @staticmethod
    def initialize_carry(batch_dims):
        return nn.OptimizedLSTMCell(features=256).initialize_carry(jax.random.PRNGKey(0), (*batch_dims, 256))


# ── Data loading ────────────────────────────────────────────────────────────
def download_gcs(gcs_path, local_path):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    subprocess.run([GCLOUD_BIN, "cp", gcs_path, local_path], check=True)


def download_gcs_dir(gcs_path, local_path):
    os.makedirs(local_path, exist_ok=True)
    subprocess.run([GCLOUD_BIN, "-m", "cp", "-r", gcs_path, local_path], check=True)


def load_buffer_levels(buffer_path):
    data = np.load(buffer_path, allow_pickle=True)
    tokens = data["tokens"]
    scores = data["scores"]
    size = int(data["size"])
    print(f"Buffer: {size} levels, tokens shape={tokens.shape}, "
          f"scores range=[{scores[:size].min():.3f}, {scores[:size].max():.3f}]")
    tokens_jax = jnp.array(tokens[:size])
    levels = jax.vmap(tokens_to_level)(tokens_jax)
    return levels, size, scores[:size]


# ── Evaluation ──────────────────────────────────────────────────────────────
def evaluate_rnn(rng, env, env_params, apply_fn, params, init_hstate, init_obs, init_env_state, max_episode_length):
    num_levels = jax.tree_util.tree_flatten(init_obs)[0][0].shape[0]

    def step(carry, _):
        rng, hstate, obs, state, done, mask, episode_length = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, done))
        hstate, pi, _ = apply_fn(params, x, hstate)
        action = pi.sample(seed=rng_action).squeeze(0)

        obs, next_state, reward, done, _ = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_levels), state, action, env_params)

        next_mask = mask & ~done
        episode_length += mask

        return (rng, hstate, obs, next_state, done, next_mask, episode_length), (state, reward)

    (_, _, _, _, _, _, episode_lengths), (states, rewards) = jax.lax.scan(
        step,
        (rng, init_hstate, init_obs, init_env_state,
         jnp.zeros(num_levels, dtype=bool),
         jnp.ones(num_levels, dtype=bool),
         jnp.zeros(num_levels, dtype=jnp.int32)),
        None,
        length=max_episode_length,
    )
    return states, rewards, episode_lengths


def compute_sfl_batch(rng, env, env_params, apply_fn, params, levels, n_levels, num_rollouts=5):
    def single_rollout(rng_roll):
        rng_r, rng_e = jax.random.split(rng_roll)
        init_obs, init_env_state = jax.vmap(env.reset_to_level, (0, 0, None))(
            jax.random.split(rng_r, n_levels), levels, env_params)
        init_hstate = ActorCritic.initialize_carry((n_levels,))
        _, rewards, _ = evaluate_rnn(
            rng_e, env, env_params, apply_fn, params,
            init_hstate, init_obs, init_env_state, MAX_STEPS)
        success = (rewards.sum(axis=0) > 0).astype(jnp.float32)
        return success

    rollout_rngs = jax.random.split(rng, num_rollouts)
    successes = jax.vmap(single_rollout)(rollout_rngs)  # (num_rollouts, n_levels)
    p = successes.mean(axis=0)
    return p * (1 - p), p


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)

    # 1. Download buffer
    if not os.path.exists(BUFFER_LOCAL):
        print("Downloading buffer dump...")
        download_gcs(BUFFER_GCS, BUFFER_LOCAL)
    else:
        print(f"Buffer already cached at {BUFFER_LOCAL}")

    # 2. Download checkpoints
    for step in EVAL_STEPS:
        step_dir = os.path.join(CKPT_LOCAL_BASE, str(step))
        if not os.path.exists(step_dir):
            print(f"Downloading checkpoint step {step}...")
            download_gcs_dir(f"{CKPT_GCS_BASE}/{step}", CKPT_LOCAL_BASE)
        else:
            print(f"Checkpoint step {step} already cached")

    # 3. Load buffer levels
    levels, n_levels, original_scores = load_buffer_levels(BUFFER_LOCAL)
    print(f"Loaded {n_levels} levels from buffer")

    # 4. Set up environment and network
    env = Maze(max_height=13, max_width=13, agent_view_size=AGENT_VIEW_SIZE, normalize_obs=True)
    env_params = env.default_params
    network = ActorCritic(env.action_space(env_params).n)

    # 5. Set up checkpoint manager
    checkpoint_manager = ocp.CheckpointManager(
        CKPT_LOCAL_BASE,
        item_handlers=ocp.StandardCheckpointHandler(),
    )

    # 6. Evaluate each checkpoint
    results = {}
    all_sfl_scores = []
    all_solve_rates = []

    @jax.jit
    def eval_batch(rng, params, level_batch):
        return compute_sfl_batch(rng, env, env_params, network.apply, params,
                                 level_batch, BATCH_SIZE, NUM_SFL_ROLLOUTS)

    for step in EVAL_STEPS:
        update_num = step * 250
        print(f"\nEvaluating checkpoint at step {step} (update {update_num})...")

        loaded = checkpoint_manager.restore(step)
        params = loaded['params']

        rng_eval = jax.random.PRNGKey(step + 1000)
        sfl_scores_all = []
        solve_rates_all = []

        for batch_start in range(0, n_levels, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, n_levels)
            batch_n = batch_end - batch_start
            level_batch = jax.tree_util.tree_map(lambda x: x[batch_start:batch_end], levels)

            rng_eval, rng_batch = jax.random.split(rng_eval)

            # Pad to BATCH_SIZE for consistent JIT shapes
            if batch_n < BATCH_SIZE:
                level_batch = jax.tree_util.tree_map(
                    lambda x: jnp.concatenate([x, jnp.repeat(x[:1], BATCH_SIZE - batch_n, axis=0)]),
                    level_batch)
                sfl, sr = eval_batch(rng_batch, params, level_batch)
                sfl = sfl[:batch_n]
                sr = sr[:batch_n]
            else:
                sfl, sr = eval_batch(rng_batch, params, level_batch)

            sfl_scores_all.append(np.array(sfl))
            solve_rates_all.append(np.array(sr))

        sfl_scores = np.concatenate(sfl_scores_all)
        solve_rates = np.concatenate(solve_rates_all)

        results[update_num] = {
            'sfl_mean': float(sfl_scores.mean()),
            'sfl_std': float(sfl_scores.std()),
            'solve_rate_mean': float(solve_rates.mean()),
            'solve_rate_std': float(solve_rates.std()),
        }
        all_sfl_scores.append(sfl_scores)
        all_solve_rates.append(solve_rates)

        print(f"  SFL: {sfl_scores.mean():.4f} +/- {sfl_scores.std():.4f}, "
              f"Solve rate: {solve_rates.mean():.3f} +/- {solve_rates.std():.3f}")

    # 7. Compute non-stationarity metrics
    all_sfl_scores = np.array(all_sfl_scores)   # (num_checkpoints, n_levels)
    all_solve_rates = np.array(all_solve_rates)
    update_nums = np.array([s * 250 for s in EVAL_STEPS])

    per_level_sfl_var = all_sfl_scores.var(axis=0)
    per_level_sr_var = all_solve_rates.var(axis=0)

    sfl_diffs = np.diff(all_sfl_scores, axis=0)
    sr_diffs = np.diff(all_solve_rates, axis=0)

    print("\n" + "=" * 60)
    print("NON-STATIONARITY ANALYSIS")
    print("=" * 60)
    print(f"Levels evaluated: {n_levels}")
    print(f"Checkpoints: {len(EVAL_STEPS)} (updates {update_nums[0]} to {update_nums[-1]})")
    print(f"\nPer-level SFL variance across checkpoints:")
    print(f"  Mean: {per_level_sfl_var.mean():.6f}")
    print(f"  Median: {np.median(per_level_sfl_var):.6f}")
    print(f"  Max: {per_level_sfl_var.max():.6f}")
    print(f"\nPer-level solve rate variance across checkpoints:")
    print(f"  Mean: {per_level_sr_var.mean():.6f}")
    print(f"  Median: {np.median(per_level_sr_var):.6f}")
    print(f"  Max: {per_level_sr_var.max():.6f}")
    print(f"\nConsecutive SFL change (per 500 updates):")
    print(f"  Mean |delta|: {np.abs(sfl_diffs).mean():.6f}")
    print(f"  Max |delta|: {np.abs(sfl_diffs).max():.6f}")
    print(f"\nConsecutive solve rate change (per 500 updates):")
    print(f"  Mean |delta|: {np.abs(sr_diffs).mean():.6f}")
    print(f"  Max |delta|: {np.abs(sr_diffs).max():.6f}")

    # Solvability flips (threshold: solve_rate > 0.5)
    solvable = all_solve_rates > 0.5
    flips = np.diff(solvable.astype(int), axis=0)
    flip_rate = (flips != 0).mean(axis=1)
    print(f"\nSolvability flip rate (per 500 updates):")
    print(f"  Mean: {flip_rate.mean():.4f} ({flip_rate.mean()*100:.1f}% of levels flip)")
    print(f"  Max: {flip_rate.max():.4f}")

    # 8. Save results
    output_dir = os.path.join(LOCAL_DIR, "results")
    os.makedirs(output_dir, exist_ok=True)

    np.savez(os.path.join(output_dir, "nonstationarity.npz"),
             update_nums=update_nums,
             sfl_scores=all_sfl_scores,
             solve_rates=all_solve_rates,
             original_scores=original_scores,
             per_level_sfl_var=per_level_sfl_var,
             per_level_sr_var=per_level_sr_var)

    # 9. Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Mean SFL over time
    ax = axes[0, 0]
    means = [results[u]['sfl_mean'] for u in update_nums]
    stds = [results[u]['sfl_std'] for u in update_nums]
    ax.plot(update_nums, means, 'b-o', markersize=3)
    ax.fill_between(update_nums, np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds), alpha=0.2)
    ax.set_xlabel('PPO Updates')
    ax.set_ylabel('SFL Score')
    ax.set_title('Mean SFL Score Over Training')
    ax.grid(True, alpha=0.3)

    # Mean solve rate over time
    ax = axes[0, 1]
    means_sr = [results[u]['solve_rate_mean'] for u in update_nums]
    stds_sr = [results[u]['solve_rate_std'] for u in update_nums]
    ax.plot(update_nums, means_sr, 'r-o', markersize=3)
    ax.fill_between(update_nums, np.array(means_sr) - np.array(stds_sr),
                    np.array(means_sr) + np.array(stds_sr), alpha=0.2)
    ax.set_xlabel('PPO Updates')
    ax.set_ylabel('Solve Rate')
    ax.set_title('Mean Solve Rate Over Training')
    ax.grid(True, alpha=0.3)

    # Per-level score variance histogram
    ax = axes[0, 2]
    ax.hist(per_level_sfl_var, bins=50, alpha=0.7, color='green')
    ax.set_xlabel('SFL Variance Across Checkpoints')
    ax.set_ylabel('Count')
    ax.set_title(f'Per-Level Score Variance (mean={per_level_sfl_var.mean():.5f})')
    ax.axvline(per_level_sfl_var.mean(), color='red', linestyle='--', label='Mean')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Solvability flip rate over time
    ax = axes[1, 0]
    ax.bar(update_nums[1:], flip_rate * 100, width=400, alpha=0.7, color='orange')
    ax.set_xlabel('PPO Updates')
    ax.set_ylabel('% Levels Flipping Solvability')
    ax.set_title('Solvability Flip Rate Per Checkpoint Interval')
    ax.grid(True, alpha=0.3)

    # Heatmap of solve rates for top-variance levels
    ax = axes[1, 1]
    n_show = min(50, n_levels)
    top_var_idx = np.argsort(per_level_sr_var)[-n_show:]
    im = ax.imshow(all_solve_rates[:, top_var_idx].T, aspect='auto', cmap='RdYlGn',
                   extent=[update_nums[0], update_nums[-1], 0, n_show])
    ax.set_xlabel('PPO Updates')
    ax.set_ylabel('Level Index (sorted by variance)')
    ax.set_title(f'Solve Rate Heatmap (Top {n_show} Variable Levels)')
    plt.colorbar(im, ax=ax)

    # Consecutive change magnitude
    ax = axes[1, 2]
    mean_abs_change = np.abs(sr_diffs).mean(axis=1)
    ax.plot(update_nums[1:], mean_abs_change, 'purple', marker='o', markersize=3)
    ax.set_xlabel('PPO Updates')
    ax.set_ylabel('Mean |delta Solve Rate|')
    ax.set_title('Mean Absolute Solve Rate Change Between Checkpoints')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "nonstationarity.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")
    print(f"Results saved to {output_dir}/nonstationarity.npz")


if __name__ == "__main__":
    main()
