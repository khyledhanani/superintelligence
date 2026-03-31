"""
Post-hoc evaluation of agent checkpoints on ACCEL-paper test levels.

Computes a comprehensive set of metrics per checkpoint:
  1. Mean return per test level
  2. Optimality gap (optimal value - agent value) per test level
  3. Solve rate per test level (over N rollouts)
  4. Learnability score per test level: p*(1-p) where p = solve rate
  5. LSTM hidden state embeddings (256D mean h-state + mean action = 257D)
  6. Episode length distribution per test level (over N rollouts)
  7. Shortest path ratio: agent_steps / BFS_shortest_path (path efficiency)
  8. IQM of solve rates across levels (rliable-style aggregation)

Supports:
  - Single checkpoint: --checkpoint_dir path/to/seed0
  - Batch across seeds: --checkpoint_dirs seed0 seed1 seed2
  - Multiple checkpoint steps: --checkpoint_steps 20 40 60 80 100 120
  - Override eval levels: --eval_levels SixteenRooms Labyrinth ...

Usage:
    # Latest checkpoint, 100 rollouts, all 13x13 levels
    python examples/eval_test_levels.py \\
        --checkpoint_dir checkpoints/accel_sfl_13x13/0 \\
        --num_attempts 100

    # Batch: 3 seeds, final checkpoint
    python examples/eval_test_levels.py --batch \\
        --checkpoint_dirs checkpoints/accel_sfl_13x13/0 \\
                          checkpoints/accel_sfl_13x13/1 \\
                          checkpoints/accel_sfl_13x13/2 \\
        --num_attempts 100 \\
        --output_dir results/test_level_eval/accel_sfl_13x13/

    # Learning curve: multiple steps for one seed
    python examples/eval_test_levels.py --batch \\
        --checkpoint_dirs checkpoints/accel_sfl_13x13/0 \\
        --checkpoint_steps 20 40 60 80 100 120 \\
        --num_attempts 100

    # Download GCS checkpoints first if needed:
    gcloud storage cp -r gs://bucket/accel/checkpoints/run_name/seed/ checkpoints/run_name/seed/
"""
import argparse
import csv
import json
import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import wandb
sys.path.insert(0, os.path.dirname(__file__))

from jaxued.environments import Maze
from jaxued.environments.maze import Level, make_level_generator
from jaxued.environments.maze.level import prefabs
from jaxued.environments.maze.env_solved import MazeSolved
from jaxued.level_sampler import LevelSampler
from jaxued.wrappers import AutoReplayWrapper

from maze_plr import ActorCritic, TrainState

# ── Level sets by grid size ──────────────────────────────────────────────
LEVELS_13x13 = [
    "SixteenRooms", "SixteenRooms2",
    "Labyrinth", "LabyrinthFlipped", "Labyrinth2",
    "StandardMaze", "StandardMaze2", "StandardMaze3",
    "SmallCorridor", "SimpleCrossing",
]
LEVELS_19x19 = LEVELS_13x13 + ["LargeCorridor", "FourRooms"]
LEVELS_21x21 = LEVELS_19x19 + ["PerfectMaze"]


def get_default_levels_for_grid(maze_height, maze_width):
    size = max(maze_height, maze_width)
    if size >= 21:
        return LEVELS_21x21
    elif size >= 19:
        return LEVELS_19x19
    return LEVELS_13x13


def filter_levels_for_grid(level_names, maze_h, maze_w):
    """Return only levels that fit the grid, with warnings."""
    valid = []
    for name in level_names:
        if name not in prefabs:
            print(f"  [WARN] Unknown prefab '{name}', skipping")
            continue
        rows = prefabs[name].strip().split('\n')
        lh, lw = len(rows), len(rows[0])
        if lh > maze_h or lw > maze_w:
            print(f"  [SKIP] {name} ({lw}x{lh}) too large for {maze_w}x{maze_h} grid")
            continue
        valid.append(name)
    return valid


# ── Agent loading ────────────────────────────────────────────────────────
def load_agent(checkpoint_dir, checkpoint_step=-1):
    """Load agent params and config. Returns (train_state, config, env, env_params) or None-tuple."""
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    maze_h = config.get("maze_height", 13)
    maze_w = config.get("maze_width", 13)

    base_env = Maze(max_height=maze_h, max_width=maze_w,
                    agent_view_size=config.get("agent_view_size", 5),
                    normalize_obs=True)
    env = AutoReplayWrapper(base_env)
    env_params = env.default_params
    sample_random_level = make_level_generator(maze_h, maze_w, config.get("n_walls", 25))

    level_sampler = LevelSampler(
        capacity=config["level_buffer_capacity"],
        replay_prob=config["replay_prob"],
        staleness_coeff=config["staleness_coeff"],
        minimum_fill_ratio=config["minimum_fill_ratio"],
        prioritization=config["prioritization"],
        prioritization_params={"temperature": config["temperature"],
                               "k": config.get("topk_k", 4)},
        duplicate_check=config.get("buffer_duplicate_check", True),
    )

    rng = jax.random.PRNGKey(0)
    obs, _ = env.reset_to_level(rng, sample_random_level(rng), env_params)
    obs_batch = jax.tree_util.tree_map(
        lambda x: jnp.repeat(jnp.repeat(x[None, ...], config["num_train_envs"], axis=0)[None, ...], 256, axis=0),
        obs,
    )
    init_x = (obs_batch, jnp.zeros((256, config["num_train_envs"])))
    network = ActorCritic(env.action_space(env_params).n)
    network_params = network.init(rng, init_x, ActorCritic.initialize_carry((config["num_train_envs"],)))

    import optax
    tx = optax.chain(
        optax.clip_by_global_norm(config.get("max_grad_norm", 0.5)),
        optax.adam(learning_rate=config.get("lr", 1e-4), eps=1e-5),
    )

    pholder_level = sample_random_level(jax.random.PRNGKey(0))
    sampler = level_sampler.initialize(pholder_level, {"max_return": -jnp.inf})
    pholder_level_batch = jax.tree_util.tree_map(
        lambda x: jnp.array([x]).repeat(config["num_train_envs"], axis=0), pholder_level
    )

    template_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
        sampler=sampler,
        update_state=0,
        es_state=None,
        es_state_full=None,
        cenie_gmm_params=None,
        pca_mean=jnp.zeros(1),
        pca_components=jnp.zeros((1, 1)),
        num_dr_updates=0,
        num_replay_updates=0,
        num_mutation_updates=0,
        dr_last_level_batch=pholder_level_batch,
        replay_last_level_batch=pholder_level_batch,
        replay_last_level_inds=jnp.zeros(config["num_train_envs"], dtype=jnp.int32),
        mutation_last_level_batch=pholder_level_batch,
    )

    models_dir = os.path.join(checkpoint_dir, "models")
    checkpoint_manager = ocp.CheckpointManager(
        models_dir, item_handlers=ocp.StandardCheckpointHandler()
    )
    available = sorted(checkpoint_manager.all_steps())
    if not available:
        print(f"[Agent] No checkpoints found in {models_dir}")
        return None, config, None, None

    if checkpoint_step == -1:
        step = checkpoint_manager.latest_step()
    elif checkpoint_step in available:
        step = checkpoint_step
    else:
        below = [s for s in available if s <= checkpoint_step]
        above = [s for s in available if s >= checkpoint_step]
        step = max(below) if below else min(above)
        print(f"[Agent] Step {checkpoint_step} not available, using closest: {step}")

    eval_freq = config.get("eval_freq", 250)
    updates_at_step = (step + 1) * eval_freq
    print(f"[Agent] Loading step {step} (~{updates_at_step} updates) from {models_dir}")

    restored = checkpoint_manager.restore(step)
    train_state = template_state.replace(params=restored["params"])
    return train_state, config, env, env_params


# ── Extended evaluate_rnn that also returns hidden states ────────────────
def evaluate_rnn_extended(rng, env, env_params, train_state, init_hstate,
                          init_obs, init_env_state, max_episode_length):
    """Like evaluate_rnn but also returns LSTM hidden states and actions per step.

    Returns:
        states: (T, N, ...) env states
        rewards: (T, N)
        episode_lengths: (N,)
        hstates: (T, N, 256) — LSTM h-vector at each step
        actions: (T, N)
    """
    num_levels = jax.tree_util.tree_flatten(init_obs)[0][0].shape[0]

    def step(carry, _):
        rng, hstate, obs, state, done, mask, episode_length = carry
        rng, rng_action, rng_step = jax.random.split(rng, 3)

        x = jax.tree_util.tree_map(lambda x: x[None, ...], (obs, done))
        hstate, pi, value = train_state.apply_fn(train_state.params, x, hstate)
        action = pi.sample(seed=rng_action).squeeze(0)

        obs, next_state, reward, done, _ = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(jax.random.split(rng_step, num_levels), state, action, env_params)

        next_mask = mask & ~done
        episode_length += mask

        # Extract h from (c, h) LSTM state — h is the output hidden state
        h_vec = hstate[1]  # (num_levels, 256)

        return (rng, hstate, obs, next_state, done, next_mask, episode_length), \
               (state, reward, h_vec, action, value.squeeze(0))

    (_, _, _, _, _, _, episode_lengths), (states, rewards, hstates, actions, values) = jax.lax.scan(
        step,
        (rng, init_hstate, init_obs, init_env_state,
         jnp.zeros(num_levels, dtype=bool),
         jnp.ones(num_levels, dtype=bool),
         jnp.zeros(num_levels, dtype=jnp.int32)),
        None,
        length=max_episode_length,
    )
    return states, rewards, episode_lengths, hstates, actions, values


# ── Embedding computation ────────────────────────────────────────────────
def compute_embeddings(hstates, actions, episode_lengths, max_steps):
    """Compute 257D embedding per level: mean LSTM h-state (256) + mean action (1).

    Masks to first episode only (steps < episode_length).
    """
    # mask: (T, N) — 1 for valid steps
    mask = jnp.arange(max_steps)[:, None] < episode_lengths[None, :]
    mask_f = mask.astype(jnp.float32)
    ep_len = jnp.maximum(mask_f.sum(axis=0), 1.0)  # (N,)

    mean_h = (hstates * mask_f[..., None]).sum(axis=0) / ep_len[..., None]  # (N, 256)
    mean_a = (actions.astype(jnp.float32) * mask_f).sum(axis=0) / ep_len    # (N,)
    return jnp.concatenate([mean_h, mean_a[:, None]], axis=-1)               # (N, 257)


# ── Shortest path computation via MazeSolved ─────────────────────────────
def compute_shortest_paths(levels, maze_h, maze_w, agent_view_size=5):
    """Compute BFS shortest path length for each level.

    Returns (N,) array of shortest path lengths (inf if unsolvable).
    """
    solved_env = MazeSolved(max_height=maze_h, max_width=maze_w,
                            agent_view_size=agent_view_size, normalize_obs=True)

    def get_min_steps(level):
        state = solved_env.init_state_from_level(level)
        return solved_env.min_steps_to_goal(state)

    min_steps = jax.vmap(get_min_steps)(levels)
    return np.asarray(min_steps)


# ── Optimality gap computation ───────────────────────────────────────────
def compute_optimal_values(levels, maze_h, maze_w, gamma, max_steps,
                           penalize_time=True, agent_view_size=5):
    """Compute V*(s_0) for each level via BFS shortest path.

    Returns (N,) array of optimal values.
    """
    solved_env = MazeSolved(max_height=maze_h, max_width=maze_w,
                            agent_view_size=agent_view_size, normalize_obs=True,
                            penalize_time=penalize_time)
    env_params = solved_env.default_params

    def get_optimal_value(level):
        state = solved_env.init_state_from_level(level)
        return solved_env.optimal_value(state, gamma, env_params)

    return np.asarray(jax.vmap(get_optimal_value)(levels))


# ── IQM (Interquartile Mean) ─────────────────────────────────────────────
def iqm(values):
    """Compute interquartile mean: mean of values between 25th and 75th percentile."""
    values = np.asarray(values)
    q25, q75 = np.percentile(values, [25, 75])
    mask = (values >= q25) & (values <= q75)
    return values[mask].mean() if mask.any() else values.mean()


# ── Main evaluation function ─────────────────────────────────────────────
def evaluate_checkpoint(train_state, config, env, env_params, level_names,
                        num_attempts, compute_embeddings_flag=True):
    """Run full evaluation suite on one checkpoint.

    Returns a dict with all metrics.
    """
    maze_h = config.get("maze_height", 13)
    maze_w = config.get("maze_width", 13)
    gamma = config.get("gamma", 0.995)
    max_steps = env_params.max_steps_in_episode

    # Load and pad levels
    levels = Level.load_prefabs(level_names)
    levels = levels.pad_to_shape(maze_w, maze_h)
    num_levels = len(level_names)

    eval_env = Maze(max_height=maze_h, max_width=maze_w,
                    agent_view_size=config.get("agent_view_size", 5),
                    normalize_obs=True)

    # ── Compute shortest paths & optimal values (once, independent of rollouts) ──
    shortest_paths = compute_shortest_paths(levels, maze_h, maze_w,
                                            config.get("agent_view_size", 5))
    optimal_values = compute_optimal_values(levels, maze_h, maze_w, gamma,
                                            max_steps,
                                            agent_view_size=config.get("agent_view_size", 5))

    # ── Run rollouts ──
    all_cum_rewards = []      # (num_attempts, num_levels)
    all_ep_lengths = []       # (num_attempts, num_levels)
    all_embeddings = []       # (num_attempts, num_levels, 257)
    all_values_t0 = []        # (num_attempts, num_levels) — V^pi(s_0)

    for attempt_idx in range(num_attempts):
        rng = jax.random.PRNGKey(attempt_idx + 2000)
        rng, rng_eval, rng_reset = jax.random.split(rng, 3)

        init_obs, init_env_state = jax.vmap(eval_env.reset_to_level, (0, 0, None))(
            jax.random.split(rng_reset, num_levels), levels, env_params
        )
        init_hstate = ActorCritic.initialize_carry((num_levels,))

        states, rewards, ep_lengths, hstates, actions, values = evaluate_rnn_extended(
            rng_eval, eval_env, env_params, train_state, init_hstate,
            init_obs, init_env_state, max_steps,
        )

        mask = np.arange(max_steps)[:, None] < np.asarray(ep_lengths)[None, :]
        cum_rewards = (np.asarray(rewards) * mask).sum(axis=0)
        all_cum_rewards.append(cum_rewards)
        all_ep_lengths.append(np.asarray(ep_lengths))
        all_values_t0.append(np.asarray(values[0]))  # V^pi at t=0

        if compute_embeddings_flag:
            emb = compute_embeddings(hstates, actions, ep_lengths, max_steps)
            all_embeddings.append(np.asarray(emb))

    # Stack: (num_attempts, num_levels)
    all_cum_rewards = np.stack(all_cum_rewards)
    all_ep_lengths = np.stack(all_ep_lengths)
    all_values_t0 = np.stack(all_values_t0)

    # ── Metric 1: Mean return per level ──
    mean_returns = all_cum_rewards.mean(axis=0)  # (num_levels,)

    # ── Metric 2: Optimality gap ──
    # V*(s_0) - mean V^pi(s_0)
    mean_agent_values = all_values_t0.mean(axis=0)
    optimality_gaps = optimal_values - mean_agent_values  # (num_levels,)

    # ── Metric 3: Solve rate per level ──
    solve_matrix = (all_cum_rewards > 0).astype(float)  # (num_attempts, num_levels)
    solve_rates = solve_matrix.mean(axis=0)              # (num_levels,)

    # ── Metric 4: Learnability ──
    learnability = solve_rates * (1.0 - solve_rates)  # (num_levels,)

    # ── Metric 5: Embeddings ──
    if compute_embeddings_flag and all_embeddings:
        embeddings = np.stack(all_embeddings)  # (num_attempts, num_levels, 257)
        mean_embedding = embeddings.mean(axis=0)  # (num_levels, 257)
    else:
        embeddings = None
        mean_embedding = None

    # ── Metric 6: Episode length distribution ──
    # For solved episodes only
    solved_mask = all_cum_rewards > 0
    mean_ep_lengths = np.where(solved_mask, all_ep_lengths, np.nan)
    # Per-level: mean and std of solved episode lengths
    with np.errstate(all='ignore'):
        ep_length_means = np.nanmean(mean_ep_lengths, axis=0)
        ep_length_stds = np.nanstd(mean_ep_lengths, axis=0)
        ep_length_medians = np.nanmedian(mean_ep_lengths, axis=0)

    # ── Metric 7: Shortest path ratio (path efficiency) ──
    # agent_steps / shortest_path (1.0 = optimal, higher = less efficient)
    # Only for solved episodes; unsolvable levels get NaN
    with np.errstate(all='ignore'):
        sp_broadcast = shortest_paths[None, :]  # (1, num_levels)
        path_ratios_all = np.where(solved_mask, all_ep_lengths / sp_broadcast, np.nan)
        path_ratio_means = np.nanmean(path_ratios_all, axis=0)  # (num_levels,)

    # ── Metric 8: IQM of solve rates ──
    iqm_solve_rate = iqm(solve_rates)

    return {
        # Per-level metrics (arrays of length num_levels)
        "level_names": level_names,
        "solve_rates": solve_rates,
        "mean_returns": mean_returns,
        "optimality_gaps": optimality_gaps,
        "optimal_values": optimal_values,
        "learnability": learnability,
        "shortest_paths": shortest_paths,
        "ep_length_means": ep_length_means,
        "ep_length_stds": ep_length_stds,
        "ep_length_medians": ep_length_medians,
        "path_ratio_means": path_ratio_means,
        # Aggregates
        "mean_solve_rate": float(solve_rates.mean()),
        "iqm_solve_rate": float(iqm_solve_rate),
        "mean_return_agg": float(mean_returns.mean()),
        "mean_optimality_gap": float(np.nanmean(optimality_gaps)),
        # Full matrices for downstream analysis (num_attempts x num_levels)
        "solve_matrix": solve_matrix,
        "cum_rewards": all_cum_rewards,
        "ep_lengths_all": all_ep_lengths,
        "values_t0_all": all_values_t0,
        # Embeddings (num_attempts x num_levels x 257)
        "embeddings": embeddings,
        "mean_embedding": mean_embedding,
    }


# ── Single checkpoint evaluation ─────────────────────────────────────────
def run_single_eval(checkpoint_dir, checkpoint_step, level_names, num_attempts,
                    output_dir=None, no_embeddings=False):
    t0 = time.time()

    train_state, config, env, env_params = load_agent(checkpoint_dir, checkpoint_step)
    if train_state is None:
        return None

    run_name = config.get("run_name", os.path.basename(checkpoint_dir))
    seed = config.get("seed", "?")
    maze_h = config.get("maze_height", 13)
    maze_w = config.get("maze_width", 13)

    if level_names is None:
        level_names = get_default_levels_for_grid(maze_h, maze_w)
    valid_levels = filter_levels_for_grid(level_names, maze_h, maze_w)
    if not valid_levels:
        print(f"  [ERROR] No valid levels for {maze_w}x{maze_h} grid")
        return None

    print(f"\n[Eval] {run_name}/seed{seed} | {len(valid_levels)} levels | "
          f"{num_attempts} attempts | embeddings={'off' if no_embeddings else 'on'}")

    results = evaluate_checkpoint(
        train_state, config, env, env_params, valid_levels, num_attempts,
        compute_embeddings_flag=not no_embeddings,
    )
    elapsed = time.time() - t0

    # ── Pretty-print results ──
    print(f"\n  {'Level':<20s} {'Solve%':>6s} {'Return':>7s} {'OptGap':>7s} "
          f"{'Learn':>6s} {'SPR':>5s} {'EpLen':>10s} {'BFS':>4s}")
    print(f"  {'-'*72}")
    for i, name in enumerate(valid_levels):
        sr = results["solve_rates"][i]
        ret = results["mean_returns"][i]
        og = results["optimality_gaps"][i]
        lr = results["learnability"][i]
        spr = results["path_ratio_means"][i]
        epm = results["ep_length_means"][i]
        eps = results["ep_length_stds"][i]
        bfs = results["shortest_paths"][i]
        spr_str = f"{spr:.2f}" if not np.isnan(spr) else "N/A"
        epl_str = f"{epm:.1f}±{eps:.1f}" if not np.isnan(epm) else "N/A"
        bfs_str = f"{bfs:.0f}" if not np.isinf(bfs) else "inf"
        print(f"  {name:<20s} {sr:>6.1%} {ret:>7.3f} {og:>7.3f} "
              f"{lr:>6.3f} {spr_str:>5s} {epl_str:>10s} {bfs_str:>4s}")

    print(f"  {'-'*72}")
    print(f"  {'MEAN':<20s} {results['mean_solve_rate']:>6.1%} "
          f"{results['mean_return_agg']:>7.3f} {results['mean_optimality_gap']:>7.3f}")
    print(f"  IQM solve rate: {results['iqm_solve_rate']:.1%}")
    print(f"  Elapsed: {elapsed:.1f}s")

    # ── Determine checkpoint step tag ──
    models_dir = os.path.join(checkpoint_dir, "models")
    cm = ocp.CheckpointManager(models_dir, item_handlers=ocp.StandardCheckpointHandler())
    actual_step = cm.latest_step() if checkpoint_step == -1 else checkpoint_step
    eval_freq = config.get("eval_freq", 250)
    updates = (actual_step + 1) * eval_freq

    # ── Save ──
    output = {
        "run_name": run_name,
        "seed": seed,
        "checkpoint_step": int(actual_step),
        "updates": int(updates),
        "maze_height": maze_h,
        "maze_width": maze_w,
        "num_attempts": num_attempts,
        "elapsed_s": elapsed,
        # Per-level arrays
        "level_names": valid_levels,
        "solve_rates": results["solve_rates"],
        "mean_returns": results["mean_returns"],
        "optimality_gaps": results["optimality_gaps"],
        "optimal_values": results["optimal_values"],
        "learnability": results["learnability"],
        "shortest_paths": results["shortest_paths"],
        "ep_length_means": results["ep_length_means"],
        "ep_length_stds": results["ep_length_stds"],
        "ep_length_medians": results["ep_length_medians"],
        "path_ratio_means": results["path_ratio_means"],
        # Aggregates
        "mean_solve_rate": results["mean_solve_rate"],
        "iqm_solve_rate": results["iqm_solve_rate"],
        "mean_return_agg": results["mean_return_agg"],
        "mean_optimality_gap": results["mean_optimality_gap"],
        # Full matrices
        "solve_matrix": results["solve_matrix"],
        "cum_rewards": results["cum_rewards"],
        "ep_lengths_all": results["ep_lengths_all"],
        "values_t0_all": results["values_t0_all"],
    }

    if results["embeddings"] is not None:
        output["embeddings"] = results["embeddings"]
        output["mean_embedding"] = results["mean_embedding"]

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fname = f"eval_{run_name}_s{seed}_step{actual_step}.npz"
        out_path = os.path.join(output_dir, fname)
        np.savez_compressed(out_path, **output)
        print(f"  Saved: {out_path}")

    # ── Wandb logging (if active) ──
    if wandb.run is not None:
        log_dict = {
            "eval/mean_solve_rate": results["mean_solve_rate"],
            "eval/iqm_solve_rate": results["iqm_solve_rate"],
            "eval/mean_return": results["mean_return_agg"],
            "eval/mean_optimality_gap": results["mean_optimality_gap"],
            "eval/run_name": run_name,
            "eval/seed": seed,
            "eval/checkpoint_step": int(actual_step),
        }
        for i, name in enumerate(valid_levels):
            log_dict[f"eval_solve_rate/{name}"] = float(results["solve_rates"][i])
            log_dict[f"eval_return/{name}"] = float(results["mean_returns"][i])
            log_dict[f"eval_opt_gap/{name}"] = float(results["optimality_gaps"][i])
            log_dict[f"eval_learnability/{name}"] = float(results["learnability"][i])
            spr = results["path_ratio_means"][i]
            if not np.isnan(spr):
                log_dict[f"eval_path_ratio/{name}"] = float(spr)
            epl = results["ep_length_means"][i]
            if not np.isnan(epl):
                log_dict[f"eval_ep_length/{name}"] = float(epl)

        # Log bar chart of solve rates as a wandb Table
        table = wandb.Table(columns=["level", "solve_rate", "return", "opt_gap",
                                      "learnability", "path_ratio", "ep_length", "bfs"])
        for i, name in enumerate(valid_levels):
            table.add_data(
                name,
                float(results["solve_rates"][i]),
                float(results["mean_returns"][i]),
                float(results["optimality_gaps"][i]),
                float(results["learnability"][i]),
                float(results["path_ratio_means"][i]) if not np.isnan(results["path_ratio_means"][i]) else None,
                float(results["ep_length_means"][i]) if not np.isnan(results["ep_length_means"][i]) else None,
                float(results["shortest_paths"][i]) if not np.isinf(results["shortest_paths"][i]) else None,
            )
        log_dict[f"eval_table/{run_name}_s{seed}"] = table
        wandb.log(log_dict)

    return output


# ── Batch evaluation ─────────────────────────────────────────────────────
def run_batch_eval(checkpoint_dirs, checkpoint_steps, level_names, num_attempts,
                   output_dir, no_embeddings=False, wandb_project=None,
                   wandb_group=None):
    all_results = []

    if wandb_project:
        wandb.init(
            project=wandb_project,
            group=wandb_group or "post_hoc_eval",
            config={
                "checkpoint_dirs": checkpoint_dirs,
                "checkpoint_steps": checkpoint_steps,
                "level_names": level_names,
                "num_attempts": num_attempts,
            },
            job_type="eval",
            tags=["post-hoc-eval"],
        )

    for ckpt_dir in checkpoint_dirs:
        steps_to_eval = checkpoint_steps if checkpoint_steps else [-1]
        for step in steps_to_eval:
            print(f"\n{'='*72}")
            result = run_single_eval(ckpt_dir, step, level_names, num_attempts,
                                     output_dir, no_embeddings=no_embeddings)
            if result is not None:
                all_results.append(result)

    if not all_results:
        print("\n[Batch] No results collected.")
        return all_results

    # ── Summary table ──
    all_level_names = []
    for r in all_results:
        for name in r["level_names"]:
            if name not in all_level_names:
                all_level_names.append(name)

    print(f"\n{'='*72}")
    print("SUMMARY — Solve Rates")
    print(f"{'='*72}")
    header = f"{'Run':<25s} {'Seed':>4s} {'Step':>6s} {'Mean':>6s} {'IQM':>6s}"
    for name in all_level_names:
        header += f" {name[:10]:>10s}"
    print(header)
    print("-" * len(header))

    for r in all_results:
        line = (f"{r['run_name']:<25s} {str(r['seed']):>4s} "
                f"{str(r['checkpoint_step']):>6s} "
                f"{r['mean_solve_rate']:>6.1%} {r['iqm_solve_rate']:>6.1%}")
        name_to_rate = dict(zip(r["level_names"], r["solve_rates"]))
        for name in all_level_names:
            rate = name_to_rate.get(name)
            line += f" {rate:>10.1%}" if rate is not None else f" {'N/A':>10s}"
        print(line)

    # ── Aggregate across seeds (if multiple seeds at same step) ──
    from collections import defaultdict
    by_run_step = defaultdict(list)
    for r in all_results:
        key = (r["run_name"], r["checkpoint_step"])
        by_run_step[key].append(r)

    if any(len(v) > 1 for v in by_run_step.values()):
        print(f"\n{'='*72}")
        print("AGGREGATE (mean ± std across seeds)")
        print(f"{'='*72}")
        header = f"{'Run':<25s} {'Step':>6s} {'#Seeds':>6s} {'Solve%':>12s} {'IQM':>12s} {'OptGap':>12s}"
        print(header)
        print("-" * len(header))

        for (run_name, step), results_list in sorted(by_run_step.items()):
            if len(results_list) < 2:
                continue
            srs = [r["mean_solve_rate"] for r in results_list]
            iqms = [r["iqm_solve_rate"] for r in results_list]
            ogs = [r["mean_optimality_gap"] for r in results_list]
            print(f"{run_name:<25s} {str(step):>6s} {len(results_list):>6d} "
                  f"{np.mean(srs):>5.1%}±{np.std(srs):>4.1%} "
                  f"{np.mean(iqms):>5.1%}±{np.std(iqms):>4.1%} "
                  f"{np.mean(ogs):>5.3f}±{np.std(ogs):>4.3f}")

    # ── Save CSV summary ──
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "eval_summary.csv")
        with open(csv_path, "w", newline="") as f:
            fieldnames = [
                "run_name", "seed", "checkpoint_step", "updates", "num_attempts",
                "mean_solve_rate", "iqm_solve_rate", "mean_return", "mean_optimality_gap",
            ]
            for name in all_level_names:
                fieldnames += [f"solve_rate/{name}", f"return/{name}",
                               f"opt_gap/{name}", f"learnability/{name}",
                               f"path_ratio/{name}", f"ep_len_mean/{name}"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in all_results:
                row = {
                    "run_name": r["run_name"],
                    "seed": r["seed"],
                    "checkpoint_step": r["checkpoint_step"],
                    "updates": r["updates"],
                    "num_attempts": r["num_attempts"],
                    "mean_solve_rate": f"{r['mean_solve_rate']:.4f}",
                    "iqm_solve_rate": f"{r['iqm_solve_rate']:.4f}",
                    "mean_return": f"{r['mean_return_agg']:.4f}",
                    "mean_optimality_gap": f"{r['mean_optimality_gap']:.4f}",
                }
                nm2sr = dict(zip(r["level_names"], r["solve_rates"]))
                nm2ret = dict(zip(r["level_names"], r["mean_returns"]))
                nm2og = dict(zip(r["level_names"], r["optimality_gaps"]))
                nm2lr = dict(zip(r["level_names"], r["learnability"]))
                nm2spr = dict(zip(r["level_names"], r["path_ratio_means"]))
                nm2epl = dict(zip(r["level_names"], r["ep_length_means"]))
                for name in all_level_names:
                    row[f"solve_rate/{name}"] = f"{nm2sr[name]:.4f}" if name in nm2sr else ""
                    row[f"return/{name}"] = f"{nm2ret[name]:.4f}" if name in nm2ret else ""
                    row[f"opt_gap/{name}"] = f"{nm2og[name]:.4f}" if name in nm2og else ""
                    row[f"learnability/{name}"] = f"{nm2lr[name]:.4f}" if name in nm2lr else ""
                    v = nm2spr.get(name)
                    row[f"path_ratio/{name}"] = f"{v:.4f}" if v is not None and not np.isnan(v) else ""
                    v = nm2epl.get(name)
                    row[f"ep_len_mean/{name}"] = f"{v:.2f}" if v is not None and not np.isnan(v) else ""
                writer.writerow(row)
        print(f"\n[Summary] CSV saved: {csv_path}")

    # ── Wandb: log aggregate summary table ──
    if wandb.run is not None:
        # Summary table: one row per (run, step) with mean±std across seeds
        from collections import defaultdict as _dd
        by_run_step = _dd(list)
        for r in all_results:
            by_run_step[(r["run_name"], r["checkpoint_step"])].append(r)

        agg_table = wandb.Table(columns=[
            "run_name", "step", "n_seeds",
            "solve_rate_mean", "solve_rate_std",
            "iqm_mean", "iqm_std",
            "opt_gap_mean", "opt_gap_std",
        ])
        for (rn, st), rlist in sorted(by_run_step.items()):
            srs = [r["mean_solve_rate"] for r in rlist]
            iqms = [r["iqm_solve_rate"] for r in rlist]
            ogs = [r["mean_optimality_gap"] for r in rlist]
            agg_table.add_data(
                rn, st, len(rlist),
                float(np.mean(srs)), float(np.std(srs)),
                float(np.mean(iqms)), float(np.std(iqms)),
                float(np.mean(ogs)), float(np.std(ogs)),
            )
        wandb.log({"eval_aggregate": agg_table})
        wandb.finish()

    return all_results


# ── CLI ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc evaluation of agent checkpoints on test levels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Metrics computed per test level:
  1. Solve rate (fraction of rollouts that solve the level)
  2. Mean return (cumulative reward averaged over rollouts)
  3. Optimality gap (V* - V^pi at s_0, via BFS shortest path)
  4. Learnability (p*(1-p) where p = solve rate)
  5. LSTM embeddings (257D: mean h-state + mean action, for behavioral analysis)
  6. Episode length distribution (mean, std, median over solved episodes)
  7. Shortest path ratio (agent_steps / BFS_steps, path efficiency)
  8. IQM of solve rates across levels (rliable-style robust aggregate)
        """,
    )
    parser.add_argument("--batch", action="store_true",
                        help="Batch mode: evaluate multiple checkpoints")

    # Single mode
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Checkpoint dir (has config.json + models/)")
    parser.add_argument("--checkpoint_step", type=int, default=-1,
                        help="Checkpoint step (-1 for latest)")

    # Batch mode
    parser.add_argument("--checkpoint_dirs", nargs="+", default=[],
                        help="List of checkpoint directories")
    parser.add_argument("--checkpoint_steps", nargs="+", type=int, default=None,
                        help="Steps to evaluate (default: latest only)")

    # Eval config
    parser.add_argument("--eval_levels", nargs="+", default=None,
                        help="Override prefab levels (default: auto by grid size)")
    parser.add_argument("--num_attempts", type=int, default=100,
                        help="Rollouts per level (ACCEL paper uses 100)")
    parser.add_argument("--no_embeddings", action="store_true",
                        help="Skip LSTM embedding computation (faster)")

    # Output
    parser.add_argument("--output_dir", type=str, default="results/test_level_eval/",
                        help="Directory to save results")

    # Wandb
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="Wandb project name (logs if set)")
    parser.add_argument("--wandb_group", type=str, default=None,
                        help="Wandb group name for this eval run")

    args = parser.parse_args()

    if args.batch or args.checkpoint_dirs:
        dirs = args.checkpoint_dirs or ([args.checkpoint_dir] if args.checkpoint_dir else [])
        if not dirs:
            parser.error("--checkpoint_dirs required in batch mode")
        run_batch_eval(dirs, args.checkpoint_steps, args.eval_levels,
                       args.num_attempts, args.output_dir,
                       no_embeddings=args.no_embeddings,
                       wandb_project=args.wandb_project,
                       wandb_group=args.wandb_group)
    else:
        if not args.checkpoint_dir:
            parser.error("--checkpoint_dir required")
        if args.wandb_project:
            wandb.init(
                project=args.wandb_project,
                group=args.wandb_group or "post_hoc_eval",
                job_type="eval",
                tags=["post-hoc-eval"],
            )
        run_single_eval(args.checkpoint_dir, args.checkpoint_step, args.eval_levels,
                        args.num_attempts, args.output_dir,
                        no_embeddings=args.no_embeddings)
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()