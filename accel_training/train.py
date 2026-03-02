"""
ACCEL Training Loop with Custom UED (VAE + MAP-Elites).

Replaces DCD's random level generator + PLR with:
  - VAE latent-space sampling / MAP-Elites mutation  (ued_interface.py)
  - jaxued LevelSampler                               (PLR replay buffer)

Training loop (3-branch ACCEL, adapted from jaxued/examples/maze_plr.py):
  Branch 0 — on_new:    generate candidates via VAE/MAP-Elites, evaluate regret,
                         insert into buffer, optionally train.
  Branch 1 — on_replay: sample levels from buffer, train agent (PPO update).
  Branch 2 — on_mutate: mutate latents of last replay batch, evaluate, insert,
                         optionally train. (ACCEL-specific step)

Usage:
    python accel_training/train.py
    python accel_training/train.py --config accel_training/config.yml
    python accel_training/train.py --log_dir runs/my_run/ --seed 0
"""

import argparse
import csv
import json
import os
import sys
import time
from enum import IntEnum

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import core, struct
from flax.training.train_state import TrainState as _BaseTrainState

# ---------------------------------------------------------------------------
# Path setup — allow running from project root or from accel_training/
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..')
sys.path.insert(0, _HERE)          # for local imports: ppo_utils, ued_interface, agent_loader, map_elites, regret_fitness
sys.path.insert(0, _ROOT)          # for jaxued, src/jaxued

import yaml

from jaxued.environments.maze import Maze
from jaxued.level_sampler import LevelSampler
from jaxued.utils import compute_max_returns, max_mc, positive_value_loss
from jaxued.wrappers import AutoReplayWrapper

from agent_loader import ActorCritic
from archive import Archive
from ued_interface import (
    load_vae,
    generate_candidates,
    mutate_latents,
    update_archive,
    build_eval_fn,
)
from ppo_utils import compute_gae, sample_trajectories_rnn, update_actor_critic_rnn

import wandb
from regret_fitness import rollout_agent_on_levels_with_positions, extract_behavior_signature
from accel_training.es_components import NSESStrategy, CMAESStrategy, SVCMAESStrategy


# ---------------------------------------------------------------------------
# TrainState (extends Flax TrainState with PLR sampler)
# ---------------------------------------------------------------------------

class UpdateState(IntEnum):
    NEW = 0
    REPLAY = 1


class TrainState(_BaseTrainState):
    sampler: core.FrozenDict = struct.field(pytree_node=True)


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def compute_score(score_fn, dones, values, max_returns, advantages):
    if score_fn == "MaxMC":
        return max_mc(dones, values, max_returns)
    elif score_fn == "pvl":
        return positive_value_loss(dones, advantages)
    else:
        raise ValueError(f"Unknown score_function: {score_fn!r}")


# ---------------------------------------------------------------------------
# Archive warm-up (Phase 3) — pre-populates PLR buffer before training step 0
# ---------------------------------------------------------------------------

def run_archive_warmup(rng, rng_np, train_state, level_sampler, eval_fn,
                       eval_env, env_params, network, config):
    """Pre-populate PLR buffer with valid, solvable levels before training.

    Applies solvability gate: only entries where valid=True and no NaN in
    regret or behavior_sig are inserted into the buffer. Unsolvable levels
    (valid=False) and NaN entries are silently skipped.

    Warm-up step budget does NOT count toward num_updates.
    """
    n_warmup = config.get("warmup_n", 256)
    latent_dim = 64
    num_envs = config["num_train_envs"]

    print(f"  [warmup] Sampling {n_warmup} latents for archive warm-up...")
    latents_np = rng_np.standard_normal((n_warmup, latent_dim)).astype(np.float32)

    rng, rng_eval, rng_rollout = jax.random.split(rng, 3)
    sequences, levels, regrets, max_returns, valid = eval_fn(
        rng_eval, train_state.params, jnp.array(latents_np)
    )

    # Separate rollout to get agent positions for behavior_sig
    _, _, _, agent_positions = rollout_agent_on_levels_with_positions(
        rng_rollout, eval_env, env_params,
        train_state.params, network, levels,
        num_steps=config["eval_rollout_steps"],
    )
    behavior_sigs = extract_behavior_signature(agent_positions, config["eval_rollout_steps"])

    # NaN guard
    nan_mask = (
        np.isnan(np.asarray(regrets))
        | np.any(np.isnan(np.asarray(behavior_sigs)), axis=-1)
    )
    if nan_mask.any():
        print(f"  [warmup] WARNING: {int(nan_mask.sum())} entries have NaN — will be skipped.")

    # SOLVABILITY GATE (locked CONTEXT.md decision — mandatory):
    # Filter invalid and NaN entries BEFORE insertion — never insert unsolvable levels.
    valid_np = np.asarray(valid) & ~nan_mask
    valid_indices = np.where(valid_np)[0]

    if len(valid_indices) == 0:
        print("  [warmup] No valid levels produced — buffer remains empty")
        return rng, rng_np, train_state

    # Select only valid entries
    latents_np = latents_np[valid_indices]
    regrets = jnp.array(np.asarray(regrets)[valid_indices])
    max_returns = jnp.array(np.asarray(max_returns)[valid_indices])
    behavior_sigs = jnp.array(np.asarray(behavior_sigs)[valid_indices])
    levels = jax.tree_util.tree_map(lambda x: x[valid_indices], levels)
    n_valid = len(valid_indices)
    print(f"  [warmup] {n_valid}/{n_warmup} levels valid (solvable, no NaN)")

    # Tile/slice to num_envs (n_valid may be < num_envs)
    tile_reps = (num_envs + n_valid - 1) // n_valid  # ceil division
    latents_jax_pad = jnp.array(np.tile(latents_np, (tile_reps, 1))[:num_envs])
    regrets_pad = jnp.tile(regrets, tile_reps)[:num_envs]
    max_returns_pad = jnp.tile(max_returns, tile_reps)[:num_envs]
    behavior_sigs_pad = jnp.tile(behavior_sigs, (tile_reps, 1))[:num_envs]
    levels_pad = jax.tree_util.tree_map(
        lambda x: jnp.tile(x, (tile_reps,) + (1,) * (x.ndim - 1))[:num_envs],
        levels,
    )

    # Build level_extra with behavior_sig — satisfies the assert at insert_batch site
    level_extra = {
        "max_return": max_returns_pad,
        "latent": latents_jax_pad,
        "behavior_sig": behavior_sigs_pad,
    }

    sampler, _ = level_sampler.insert_batch(train_state.sampler, levels_pad, regrets_pad, level_extra)
    train_state = train_state.replace(sampler=sampler)
    print(f"  [warmup] Done: {int(sampler['size'])} entries in PLR buffer")
    return rng, rng_np, train_state


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(config):
    rng = jax.random.PRNGKey(config["seed"])
    rng_np = np.random.default_rng(config["seed"])

    # --- Setup log directory ---
    log_dir = config["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    metrics_path = os.path.join(log_dir, "metrics.csv")

    # --- Environment ---
    maze_env = Maze(
        max_height=13, max_width=13,
        agent_view_size=config["agent_view_size"],
        normalize_obs=True,
    )
    env = AutoReplayWrapper(maze_env)
    env_params = env.default_params

    # --- VAE decoder ---
    print("Loading VAE checkpoint...")
    decoder_params = load_vae(config["vae_checkpoint"])

    # --- UED: evaluation function (JIT-compiled decode + regret) ---
    print("Building JIT-compiled eval function...")
    network = ActorCritic(action_dim=7)
    eval_fn, eval_env = build_eval_fn(
        decoder_params, network, env_params,
        latent_dim=64,
        decode_temperature=config["decode_temperature"],
        num_steps=config["eval_rollout_steps"],
        min_obstacles=config["min_obstacles"],
        min_distance=config["min_distance"],
    )

    # --- WandB initialization ---
    run = wandb.init(
        config=config,
        project=config.get("wandb_project", "es-accel"),
        group=config["run_name"],
        tags=[config.get("es_strategy", "cma_es").upper()],
    )
    wandb.define_metric("update")
    for key in ["regret", "novelty_score", "replay_buffer_size", "buffer_occupied",
                "valid_fraction", "mean_buffer_score"]:
        wandb.define_metric(key, step_metric="update")
    for key in ["sv_behavior_dist_pre", "sv_behavior_dist_post"]:
        wandb.define_metric(key, step_metric="update")
    mean_novelty = 0.0  # tracked per-step for NS-ES
    sv_behavior_dist_pre = 0.0   # tracked per-step for SV-CMA-ES
    sv_behavior_dist_post = 0.0  # tracked per-step for SV-CMA-ES

    # --- ES strategy (NS-ES or CMA-ES baseline) ---
    es_strategy_name = config.get("es_strategy", "cma_es")
    es_config = {
        "sigma_init": config.get("es_sigma_init", 0.5),
    }
    latent_dim_es = 64
    pop_size_es = config.get("es_pop_size", 16)
    if es_strategy_name == "ns_es":
        es_strategy = NSESStrategy(param_dim=latent_dim_es, pop_size=pop_size_es)
    elif es_strategy_name == "sv_cma_es":
        es_config["sv_n_particles"] = config.get("sv_n_particles", 2)
        es_strategy = SVCMAESStrategy(param_dim=latent_dim_es, pop_size=pop_size_es)
    else:
        es_strategy = CMAESStrategy(param_dim=latent_dim_es, pop_size=pop_size_es)
    rng, rng_es_init = jax.random.split(rng)
    es_state = es_strategy.init_state(rng_es_init, es_config)

    # --- MAP-Elites archive ---
    archive = Archive(latent_dim=64)

    # --- Level buffer (PLR) ---
    level_sampler = LevelSampler(
        capacity=config["level_buffer_capacity"],
        replay_prob=config["replay_prob"],
        staleness_coeff=config["staleness_coeff"],
        minimum_fill_ratio=config["minimum_fill_ratio"],
        prioritization=config["prioritization"],
        prioritization_params={
            "temperature": config["temperature"],
            "k": config["topk_k"],
        },
        duplicate_check=False,
    )

    # --- Agent + optimizer ---
    def linear_schedule(count):
        n_mb_per_update = config["num_minibatches"] * config["epoch_ppo"]
        frac = 1.0 - (count // n_mb_per_update) / config["num_updates"]
        return config["lr"] * frac

    print("Initializing agent (random weights)...")
    # Build a placeholder level to initialize the network and level sampler
    rng, rng_init, rng_level = jax.random.split(rng, 3)
    z_ph = jax.random.normal(rng_level, (1, 64))
    # Use a random CLUTTR sequence to build placeholder Level
    from vae_decoder import decode_latent_to_env, repair_cluttr_sequence
    from env_bridge import cluttr_sequence_to_level
    seq_ph = decode_latent_to_env(decoder_params, z_ph, rng_key=rng_level)
    seq_ph = repair_cluttr_sequence(seq_ph[0])
    pholder_level = cluttr_sequence_to_level(seq_ph, rng_level)

    # Initialize network params — obs must be (n_steps=256, num_envs, ...) for init
    obs_ph, _ = env.reset_to_level(rng_init, pholder_level, env_params)
    obs_ph_batch = jax.tree_util.tree_map(
        lambda x: jnp.repeat(
            jnp.repeat(x[None, ...], config["num_train_envs"], axis=0)[None, ...],
            256, axis=0,
        ),
        obs_ph,
    )
    init_x = (obs_ph_batch, jnp.zeros((256, config["num_train_envs"])))
    network_params = network.init(rng_init, init_x, ActorCritic.initialize_carry((config["num_train_envs"],)))

    tx = optax.chain(
        optax.clip_by_global_norm(config["max_grad_norm"]),
        optax.adam(learning_rate=linear_schedule, eps=1e-5),
    )

    sampler = level_sampler.initialize(
        pholder_level,
        {
            "max_return":   jnp.array(-jnp.inf),
            "latent":       jnp.zeros(64, dtype=jnp.float32),
            "behavior_sig": jnp.zeros(169, dtype=jnp.float32),
        },
    )

    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
        sampler=sampler,
    )

    # --- JIT-compiled train step on given levels ---
    num_envs = config["num_train_envs"]
    num_steps = config["num_steps"]
    score_fn = config["score_function"]
    exploratory = config["exploratory_grad_updates"]

    @jax.jit
    def _train_on_levels(rng, train_state, levels):
        """Run one PPO update on the given levels."""
        rng_reset, rng_train = jax.random.split(rng)
        init_obs, init_env_state = jax.vmap(env.reset_to_level, in_axes=(0, 0, None))(
            jax.random.split(rng_reset, num_envs), levels, env_params
        )
        carry, traj = sample_trajectories_rnn(
            rng_train, env, env_params, train_state,
            ActorCritic.initialize_carry((num_envs,)),
            init_obs, init_env_state, num_envs, num_steps,
        )
        _, train_state, _, _, _, last_value = carry
        obs, actions, rewards, dones, log_probs, values, _ = traj
        advantages, targets = compute_gae(
            config["gamma"], config["gae_lambda"], last_value, values, rewards, dones
        )
        max_returns = compute_max_returns(dones, rewards)
        scores = compute_score(score_fn, dones, values, max_returns, advantages)
        (_, train_state), losses = update_actor_critic_rnn(
            rng_train, train_state,
            ActorCritic.initialize_carry((num_envs,)),
            (obs, actions, dones, log_probs, values, targets, advantages),
            num_envs, num_steps,
            config["num_minibatches"], config["epoch_ppo"],
            config["clip_eps"], config["entropy_coeff"], config["critic_coeff"],
            update_grad=True,
        )
        return train_state, scores, max_returns, losses

    # --- Metrics logging ---
    csv_file = open(metrics_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "update", "mode", "archive_filled", "buffer_size",
        "mean_regret_candidates", "valid_fraction",
        "mean_buffer_score", "time_s",
    ])

    def _log(update, mode, archive_filled, buffer_size,
             mean_regret, valid_frac, mean_buf_score, elapsed):
        row = [update, mode, archive_filled, buffer_size,
               f"{mean_regret:.4f}", f"{valid_frac:.3f}",
               f"{mean_buf_score:.4f}", f"{elapsed:.2f}"]
        csv_writer.writerow(row)
        csv_file.flush()
        if update % 10 == 0:
            print(
                f"[{update:6d}] mode={mode:8s} | archive={archive_filled:3d}/48 "
                f"| buf={buffer_size:4d} | regret={mean_regret:.3f} "
                f"| valid={valid_frac:.2f} | buf_score={mean_buf_score:.3f} | {elapsed:.1f}s"
            )

    # --- Checkpoint saving ---
    def _save_checkpoint(update, train_state):
        ckpt_dir = os.path.join(log_dir, f"checkpoint_{update:07d}")
        os.makedirs(ckpt_dir, exist_ok=True)
        import pickle
        params_np = jax.tree_util.tree_map(np.asarray, train_state.params)
        with open(os.path.join(ckpt_dir, "agent_params.pkl"), "wb") as f:
            pickle.dump(params_np, f)
        # Save archive
        data = archive.get_arrays()
        if data is not None:
            np.save(os.path.join(ckpt_dir, "archive_envs.npy"), data["sequences"])
            np.save(os.path.join(ckpt_dir, "archive_latents.npy"), data["latents"])
            np.save(os.path.join(ckpt_dir, "archive_regrets.npy"), data["regrets"])
        print(f"  Saved checkpoint: {ckpt_dir}")

    # --- Archive warm-up (Phase 3) — runs BEFORE training step 0 ---
    if config.get("es_strategy", "cma_es") != "cma_es" or config.get("warmup_n", 0) > 0:
        print("Running archive warm-up...")
        rng, rng_np, train_state = run_archive_warmup(
            rng, rng_np, train_state, level_sampler, eval_fn,
            eval_env, env_params, network, config,
        )

    # --- Main training loop ---
    last_replay_latents = None  # numpy (num_envs, 64) — for ACCEL mutation
    update_state = UpdateState.NEW

    print(f"\nStarting ACCEL training: {config['num_updates']} updates")
    print(f"  Log dir: {log_dir}")
    print(f"  n_candidates={config['n_candidates']}, n_envs={num_envs}, n_steps={num_steps}")
    print("-" * 80)

    for update in range(config["num_updates"]):
        t_start = time.time()

        # --- Decide which branch to take ---
        rng, rng_decision = jax.random.split(rng)
        if update_state == UpdateState.REPLAY and config["use_accel"]:
            mode = "mutate"
        else:
            should_replay = bool(
                level_sampler.sample_replay_decision(train_state.sampler, rng_decision)
            )
            mode = "replay" if should_replay else "new"

        # =========================================================
        # Branch 0: NEW — generate candidates via VAE/MAP-Elites
        # =========================================================
        if mode in ("new", "mutate"):
            if mode == "new":
                latents_np = generate_candidates(
                    archive, rng_np, config["n_candidates"],
                    latent_dim=64,
                    mutation_sigma=config["mutation_sigma"],
                    random_fraction=config["random_fraction"],
                )
            else:  # mutate
                if last_replay_latents is not None:
                    latents_np = mutate_latents(
                        last_replay_latents, rng_np,
                        mutation_sigma=config["mutation_sigma"],
                    )
                else:
                    # Fallback: random if we haven't replayed yet
                    latents_np = generate_candidates(
                        archive, rng_np, config["n_candidates"],
                        latent_dim=64,
                        mutation_sigma=config["mutation_sigma"],
                        random_fraction=1.0,
                    )

            latents_jax = jnp.array(latents_np)
            rng, rng_eval = jax.random.split(rng)
            sequences, levels, regrets, max_returns, valid = eval_fn(
                rng_eval, train_state.params, latents_jax
            )

            sequences_np = np.asarray(sequences)
            regrets_np = np.asarray(regrets)
            valid_np = np.asarray(valid)

            # Update MAP-Elites archive
            update_archive(archive, latents_np, sequences_np, regrets_np, valid_np)

            # Resize latents/levels to num_envs for buffer insertion.
            # (pad by tiling if n_candidates < num_envs; LevelSampler needs fixed batch)
            n_cand = latents_np.shape[0]
            if n_cand < num_envs:
                repeat = (num_envs + n_cand - 1) // n_cand
                latents_jax_pad = jnp.tile(latents_jax, (repeat, 1))[:num_envs]
                levels_pad = jax.tree_util.tree_map(
                    lambda x: jnp.tile(x, (repeat, *([1] * (x.ndim - 1))))[:num_envs],
                    levels,
                )
                regrets_pad = jnp.tile(regrets, repeat)[:num_envs]
                max_returns_pad = jnp.tile(max_returns, repeat)[:num_envs]
            else:
                latents_jax_pad = latents_jax[:num_envs]
                levels_pad = jax.tree_util.tree_map(lambda x: x[:num_envs], levels)
                regrets_pad = regrets[:num_envs]
                max_returns_pad = max_returns[:num_envs]

            # Extract behavior signatures for buffer insertion (Phase 3 forward-contract)
            rng, rng_bsig = jax.random.split(rng)
            _, _, _, agent_positions = rollout_agent_on_levels_with_positions(
                rng_bsig, eval_env, env_params,
                train_state.params, network, levels_pad,
                num_steps=config["eval_rollout_steps"],
            )
            candidate_sigs = extract_behavior_signature(agent_positions, config["eval_rollout_steps"])

            # Insert into PLR buffer (latent stored in level_extra)
            level_extra = {"max_return": max_returns_pad, "latent": latents_jax_pad, "behavior_sig": candidate_sigs}
            assert "behavior_sig" in level_extra, (
                "All PLR buffer insertions must include 'behavior_sig'. "
                "Call extract_behavior_signature() on rollout positions first."
            )
            sampler, _ = level_sampler.insert_batch(
                train_state.sampler,
                levels_pad,
                regrets_pad,
                level_extra,
            )
            train_state = train_state.replace(sampler=sampler)

            # ES strategy tell() — update ES state and capture novelty/diversity for logging
            if es_strategy_name == "ns_es":
                buf_size_es = int(train_state.sampler["size"])
                all_buf_sigs = train_state.sampler["levels_extra"]["behavior_sig"]
                buf_valid_mask = jnp.arange(level_sampler.capacity) < buf_size_es
                es_state, mean_novelty = es_strategy.tell(
                    es_state, latents_jax_pad, regrets_pad,
                    candidate_sigs, all_buf_sigs, buf_valid_mask,
                    alpha=config.get("es_alpha", 0.8),
                    beta=config.get("es_beta", 0.2),
                    k=config.get("es_k_novelty", 5),
                )
            elif es_strategy_name == "sv_cma_es":
                # SV-CMA-ES two-pass eval.
                # First pass already done above (latents_jax_pad -> candidate_sigs, regrets_pad).
                # Step 4 of CONTEXT: nudge candidate latents by Stein repulsion for second eval pass.
                # Compute per-particle mean bsigs from first pass for Stein repulsion.
                n_sv = len(es_state["particles"])
                pop_sv = config.get("es_pop_size", 16)
                particle_bsigs_pre = jnp.stack([
                    jnp.mean(candidate_sigs[i * pop_sv:(i + 1) * pop_sv], axis=0)
                    for i in range(n_sv)
                ])  # (N, D_bsig)
                particle_means_pre = jnp.stack([
                    p["es_state"].mean for p in es_state["particles"]
                ])  # (N, param_dim)
                from accel_training.es_components.stein import compute_stein_repulsion
                sv_epsilon = config.get("sv_epsilon", 0.01)
                repulsion = compute_stein_repulsion(particle_means_pre, particle_bsigs_pre, sv_epsilon)

                # Build repelled latents for second eval pass: tile repulsion[i] to pop_sv candidates
                repulsion_tiled = jnp.concatenate([
                    jnp.tile(repulsion[i:i + 1], (pop_sv, 1))
                    for i in range(n_sv)
                ], axis=0)  # (N*pop_sv, param_dim)
                post_latents = latents_jax_pad[:n_sv * pop_sv] + repulsion_tiled

                # Second eval pass on repelled latents
                rng, rng_eval2, rng_bsig2 = jax.random.split(rng, 3)
                _, levels2, regrets2, max_returns2, _ = eval_fn(rng_eval2, train_state.params, post_latents)
                _, _, _, agent_pos2 = rollout_agent_on_levels_with_positions(
                    rng_bsig2, eval_env, env_params,
                    train_state.params, network, levels2,
                    num_steps=config["eval_rollout_steps"],
                )
                post_bsigs = extract_behavior_signature(agent_pos2, config["eval_rollout_steps"])

                # PLR buffer receives post-repulsion data (locked CONTEXT decision).
                # First-pass insert already happened above. Overwrite with post-repulsion batch.
                # Pad regrets2 to num_envs if n_sv*pop_sv < num_envs.
                n_post = n_sv * pop_sv
                if n_post < num_envs:
                    tile_reps2 = (num_envs + n_post - 1) // n_post
                    regrets2_pad = jnp.tile(regrets2, tile_reps2)[:num_envs]
                    post_latents_pad = jnp.tile(post_latents, (tile_reps2, 1))[:num_envs]
                    post_bsigs_pad = jnp.tile(post_bsigs, (tile_reps2, 1))[:num_envs]
                    levels2_pad = jax.tree_util.tree_map(
                        lambda x: jnp.tile(x, (tile_reps2, *([1] * (x.ndim - 1))))[:num_envs],
                        levels2,
                    )
                    max_returns2_pad = jnp.tile(max_returns2, tile_reps2)[:num_envs]
                else:
                    regrets2_pad = regrets2[:num_envs]
                    post_latents_pad = post_latents[:num_envs]
                    post_bsigs_pad = post_bsigs[:num_envs]
                    levels2_pad = jax.tree_util.tree_map(lambda x: x[:num_envs], levels2)
                    max_returns2_pad = max_returns2[:num_envs]

                level_extra2 = {
                    "max_return": max_returns2_pad,
                    "latent": post_latents_pad,
                    "behavior_sig": post_bsigs_pad,
                }
                assert "behavior_sig" in level_extra2
                sampler2, _ = level_sampler.insert_batch(
                    train_state.sampler, levels2_pad, regrets2_pad, level_extra2
                )
                train_state = train_state.replace(sampler=sampler2)

                # SV-CMA-ES tell() — strategy receives both eval passes, handles Stein means internally.
                es_state, sv_metrics = es_strategy.tell(
                    es_state,
                    latents_jax_pad,   # pre_cands (N*pop_sv, param_dim)
                    candidate_sigs,    # pre_bsigs (N*pop_sv, D_bsig)
                    post_latents,      # post_cands (N*pop_sv, param_dim)
                    regrets2,          # post_regrets (N*pop_sv,)
                    post_bsigs,        # post_bsigs (N*pop_sv, D_bsig)
                    sv_epsilon,
                )
                sv_behavior_dist_pre = sv_metrics["sv_behavior_dist_pre"]
                sv_behavior_dist_post = sv_metrics["sv_behavior_dist_post"]
            else:
                # CMA-ES baseline: negate composite for evosax (which minimizes)
                fitness_for_cma = -(
                    config.get("es_alpha", 0.8) * regrets_pad
                )
                es_state = es_strategy.tell(es_state, latents_jax_pad, fitness_for_cma)

            # Optionally train on new/mutated levels (exploratory grad updates)
            if exploratory:
                rng, rng_train = jax.random.split(rng)
                train_state, _, _, _ = _train_on_levels(rng_train, train_state, levels_pad)

            mean_regret = float(np.nanmean(np.where(valid_np, regrets_np, np.nan))) if valid_np.any() else 0.0
            valid_frac = float(valid_np.mean())
            update_state = UpdateState.NEW

        # =========================================================
        # Branch 1: REPLAY — sample from buffer, PPO update
        # =========================================================
        elif mode == "replay":
            rng, rng_levels, rng_train = jax.random.split(rng, 3)
            sampler, (level_inds, replay_levels) = level_sampler.sample_replay_levels(
                train_state.sampler, rng_levels, num_envs
            )
            train_state = train_state.replace(sampler=sampler)

            # Store latents for ACCEL mutation on next step
            last_replay_latents = np.asarray(
                level_sampler.get_levels_extra(train_state.sampler, level_inds)["latent"]
            )

            # PPO update
            train_state, new_scores, new_max_returns, losses = _train_on_levels(
                rng_train, train_state, replay_levels
            )

            # Update buffer scores with fresh regret
            prev_extra = level_sampler.get_levels_extra(train_state.sampler, level_inds)
            updated_max_returns = jnp.maximum(prev_extra["max_return"], new_max_returns)
            sampler = level_sampler.update_batch(
                train_state.sampler,
                level_inds,
                new_scores,
                {"max_return": updated_max_returns,
                 "latent": jnp.array(last_replay_latents),
                 "behavior_sig": prev_extra["behavior_sig"]},
            )
            train_state = train_state.replace(sampler=sampler)

            mean_regret = float(jnp.mean(new_scores))
            valid_frac = 1.0  # Replay levels are always valid
            update_state = UpdateState.REPLAY

        t_elapsed = time.time() - t_start
        buf_size = int(train_state.sampler["size"])
        all_scores = np.asarray(train_state.sampler["scores"])
        mask = np.arange(level_sampler.capacity) < buf_size
        mean_buf_score = float(all_scores[mask].mean()) if mask.any() else 0.0

        _log(update, mode, archive.num_filled, buf_size,
             mean_regret, valid_frac, mean_buf_score, t_elapsed)

        if (update + 1) % config.get("wandb_log_freq", 10) == 0:
            wandb.log({
                "update":                  update,
                "regret":                  mean_regret,
                "novelty_score":           mean_novelty,
                "replay_buffer_size":      buf_size,
                "buffer_occupied":         buf_size / level_sampler.capacity,
                "valid_fraction":          valid_frac,
                "mean_buffer_score":       mean_buf_score,
                "mode":                    mode,
                "sv_behavior_dist_pre":    sv_behavior_dist_pre,
                "sv_behavior_dist_post":   sv_behavior_dist_post,
            }, step=update)

        if (update + 1) % config["checkpoint_every"] == 0:
            _save_checkpoint(update + 1, train_state)

    # --- Final checkpoint ---
    _save_checkpoint(config["num_updates"], train_state)
    csv_file.close()
    print("\nTraining complete.")
    print(f"  Archive: {archive.num_filled}/48 cells filled")
    print(f"  Checkpoints + metrics in: {log_dir}")
    return train_state, archive


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="ACCEL training with VAE + MAP-Elites UED")
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(_HERE, "config.yml"),
        help="Path to YAML config file.",
    )
    parser.add_argument("--log_dir", type=str, default=None, help="Override log_dir from config.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed from config.")
    parser.add_argument("--num_updates", type=int, default=None, help="Override num_updates.")
    parser.add_argument(
        "--es_strategy", type=str, default=None,
        choices=["cma_es", "ns_es", "sv_cma_es"],
        help="ES strategy: cma_es | ns_es | sv_cma_es (overrides config).",
    )
    parser.add_argument(
        "--n_particles", type=int, default=None,
        help="Number of SV-CMA-ES particles (sv_n_particles in config).",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    if args.log_dir is not None:
        config["log_dir"] = args.log_dir
    if args.seed is not None:
        config["seed"] = args.seed
    if args.num_updates is not None:
        config["num_updates"] = args.num_updates
    if args.es_strategy is not None:
        config["es_strategy"] = args.es_strategy
    if args.n_particles is not None:
        config["sv_n_particles"] = args.n_particles

    # Resolve vae_checkpoint relative to project root
    if not os.path.isabs(config["vae_checkpoint"]):
        config["vae_checkpoint"] = os.path.join(_ROOT, config["vae_checkpoint"])

    train(config)


if __name__ == "__main__":
    main()
