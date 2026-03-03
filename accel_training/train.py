"""
Two-Mode ES Training Loop.

Replaces the old three-branch loop with a clean two-mode pipeline:
  - es_step:  ES ask() -> VAE decode -> eval -> behavior_sig -> PLR insert -> ES tell()
  - replay:   PLR buffer sample -> PPO update -> buffer score update

Bootstrap loop fills the PLR buffer to bootstrap_min levels via es_step before
any replay is allowed.

ES strategy routing via config["es_strategy"]:
  "cma_es"    -> CMAESStrategy    (vanilla CMA-ES baseline)
  "ns_es"     -> NSESStrategy     (novelty-seeking CMA-ES, Phase 3)
  "sv_cma_es" -> SVCMAESStrategy  (Stein Variational CMA-ES, Phase 4, primary contribution)

Usage:
    python accel_training/train.py
    python accel_training/train.py --config accel_training/config.yml
    python accel_training/train.py --es_strategy sv_cma_es --seed 0
"""

import argparse
import csv
import json
import os
import pickle
import sys
import time

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
sys.path.insert(0, _HERE)   # for local imports: ppo_utils, ued_interface, agent_loader, regret_fitness
sys.path.insert(0, _ROOT)   # for jaxued, src/jaxued

import yaml

from jaxued.environments.maze import Maze
from jaxued.environments import MazeRenderer
from jaxued.level_sampler import LevelSampler
from jaxued.utils import compute_max_returns, max_mc, positive_value_loss
from jaxued.wrappers import AutoReplayWrapper

from agent_loader import ActorCritic
from ued_interface import load_vae, build_eval_fn
from ppo_utils import compute_gae, sample_trajectories_rnn, update_actor_critic_rnn

import wandb
from regret_fitness import rollout_agent_on_levels_with_positions, extract_behavior_signature
from accel_training.es_components import NSESStrategy, CMAESStrategy, SVCMAESStrategy


# ---------------------------------------------------------------------------
# TrainState (extends Flax TrainState with PLR sampler)
# ---------------------------------------------------------------------------

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
# Main training function
# ---------------------------------------------------------------------------

def train(config):
    rng = jax.random.PRNGKey(config["seed"])
    rng_np = np.random.default_rng(config["seed"])

    # --- New config keys ---
    bootstrap_min = config.get("bootstrap_min", 50)
    replay_ratio = config.get("replay_ratio", config.get("replay_prob", 0.8))

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
    renderer = MazeRenderer(maze_env, tile_size=16)

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
        name=config["run_name"],
        group=config.get("wandb_group", "phase5-comparison"),
        tags=[config.get("es_strategy", "cma_es").upper()],
    )
    wandb.define_metric("update")
    for key in ["regret", "novelty_score", "replay_buffer_size", "buffer_occupied",
                "valid_fraction", "mean_buffer_score"]:
        wandb.define_metric(key, step_metric="update")
    for key in ["sv_behavior_dist_pre", "sv_behavior_dist_post"]:
        wandb.define_metric(key, step_metric="update")
    mean_novelty = 0.0           # tracked per-step for NS-ES
    sv_behavior_dist_pre = 0.0   # tracked per-step for SV-CMA-ES
    sv_behavior_dist_post = 0.0  # tracked per-step for SV-CMA-ES

    # --- ES strategy ---
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

    # --- Level buffer (PLR) ---
    level_sampler = LevelSampler(
        capacity=config["level_buffer_capacity"],
        replay_prob=replay_ratio,
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
        frac = 1.0 - (count // n_mb_per_update) / max(config["num_updates"], 1)
        return config["lr"] * frac

    print("Initializing agent (random weights)...")
    # Build a placeholder level to initialize the network and level sampler
    rng, rng_init, rng_level = jax.random.split(rng, 3)
    z_ph = jax.random.normal(rng_level, (1, 64))
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

    # --- Shared config values used in JIT-compiled helper ---
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
        "update", "mode", "buffer_size",
        "mean_regret_candidates", "valid_fraction",
        "mean_buffer_score", "time_s",
    ])

    def _log(update, mode, buffer_size, mean_regret, valid_frac, mean_buf_score, elapsed):
        row = [update, mode, buffer_size,
               f"{mean_regret:.4f}", f"{valid_frac:.3f}",
               f"{mean_buf_score:.4f}", f"{elapsed:.2f}"]
        csv_writer.writerow(row)
        csv_file.flush()
        if update % 10 == 0:
            print(
                f"[{update:6d}] mode={mode:8s} "
                f"| buf={buffer_size:4d} | regret={mean_regret:.3f} "
                f"| valid={valid_frac:.2f} | buf_score={mean_buf_score:.3f} | {elapsed:.1f}s"
            )

    # --- ES state extraction (strategy-aware) ---
    def _get_es_stats(es_state):
        """Extract mean/std/C/D from es_state as plain numpy dicts, safe to pickle."""
        if es_strategy_name == "sv_cma_es":
            particles = []
            means = []
            for p in es_state["particles"]:
                evo = p["es_state"]
                particles.append({
                    "mean":         np.asarray(evo.mean),
                    "std":          float(evo.std),
                    "C":            np.asarray(evo.C),
                    "D":            np.asarray(evo.D),
                    "generation":   int(evo.generation_counter),
                    "best_solution": np.asarray(evo.best_solution),
                })
                means.append(np.asarray(evo.mean))
            dist = float(np.linalg.norm(means[0] - means[1])) if len(means) >= 2 else 0.0
            return {"particles": particles, "particle_mean_distance": dist}
        else:  # cma_es or ns_es
            evo = es_state["es_state"]
            return {
                "mean":          np.asarray(evo.mean),
                "std":           float(evo.std),
                "C":             np.asarray(evo.C),
                "D":             np.asarray(evo.D),
                "generation":    int(evo.generation_counter),
                "best_solution": np.asarray(evo.best_solution),
            }

    # --- Checkpoint saving ---
    def _save_checkpoint(update, train_state, es_state, rng):
        ckpt_dir = os.path.join(log_dir, f"checkpoint_{update:07d}")
        os.makedirs(ckpt_dir, exist_ok=True)

        # 1. Agent params
        params_np = jax.tree_util.tree_map(np.asarray, train_state.params)
        with open(os.path.join(ckpt_dir, "agent_params.pkl"), "wb") as f:
            pickle.dump(params_np, f)

        # 2. ES state stats
        es_stats = _get_es_stats(es_state)
        with open(os.path.join(ckpt_dir, "es_state.pkl"), "wb") as f:
            pickle.dump(es_stats, f)

        # 3. Decode mean latent(s) and log as WandB image
        try:
            rng, rng_img = jax.random.split(rng)
            if es_strategy_name == "sv_cma_es":
                for i, p in enumerate(es_stats["particles"]):
                    mean = jnp.array(p["mean"]).reshape(1, 64)
                    padded = jnp.concatenate([mean, jnp.zeros((pop_size_es - 1, 64))])
                    _, levels, _, _, valid = eval_fn(rng_img, train_state.params, padded)
                    if bool(valid[0]):
                        level_i = jax.tree_util.tree_map(lambda x: x[0], levels)
                        img = np.asarray(renderer.render_level(level_i, env_params))
                        wandb.log({f"es/particle_{i}_maze": wandb.Image(img)}, step=update)
            else:
                mean = jnp.array(es_stats["mean"]).reshape(1, 64)
                padded = jnp.concatenate([mean, jnp.zeros((pop_size_es - 1, 64))])
                _, levels, _, _, valid = eval_fn(rng_img, train_state.params, padded)
                if bool(valid[0]):
                    level_i = jax.tree_util.tree_map(lambda x: x[0], levels)
                    img = np.asarray(renderer.render_level(level_i, env_params))
                    wandb.log({"es/mean_maze": wandb.Image(img)}, step=update)
        except Exception as e:
            print(f"  [es_state] image logging skipped: {e}")

        print(f"  Saved checkpoint: {ckpt_dir}")

    # -----------------------------------------------------------------------
    # ES step helper — shared by bootstrap loop and main loop
    # -----------------------------------------------------------------------
    def _run_es_step(rng, train_state, es_state):
        """One ES step: ask -> eval -> bsig -> solvability gate -> insert -> tell.

        Returns (rng, train_state, es_state, mean_regret, valid_frac,
                 mean_novelty, sv_dist_pre, sv_dist_post)
        """
        nonlocal mean_novelty, sv_behavior_dist_pre, sv_behavior_dist_post

        # ES ask
        rng, rng_ask = jax.random.split(rng)
        latents_jax, es_state = es_strategy.ask(es_state, rng_ask)

        # Eval
        rng, rng_eval = jax.random.split(rng)
        sequences, levels, regrets, max_returns, valid = eval_fn(
            rng_eval, train_state.params, latents_jax
        )

        regrets_np = np.asarray(regrets)
        valid_np_raw = np.asarray(valid)

        # Behavior signature extraction (first pass)
        rng, rng_bsig = jax.random.split(rng)
        _, _, _, agent_positions = rollout_agent_on_levels_with_positions(
            rng_bsig, eval_env, env_params,
            train_state.params, network, levels,
            num_steps=config["eval_rollout_steps"],
        )
        candidate_sigs = extract_behavior_signature(agent_positions, config["eval_rollout_steps"])

        # NaN guard
        nan_mask = (
            np.isnan(regrets_np)
            | np.any(np.isnan(np.asarray(candidate_sigs)), axis=-1)
        )

        # SOLVABILITY GATE: filter invalid and NaN entries before insertion
        valid_np = valid_np_raw & ~nan_mask
        valid_indices = np.where(valid_np)[0]

        n_cand = latents_jax.shape[0]
        valid_frac = float(valid_np.mean()) if n_cand > 0 else 0.0
        mean_regret = (
            float(np.nanmean(np.where(valid_np, regrets_np, np.nan)))
            if valid_np.any() else 0.0
        )

        if len(valid_indices) > 0:
            # Pad/truncate to num_envs for LevelSampler
            regrets_valid = jnp.array(regrets_np[valid_indices])
            max_returns_valid = jnp.array(np.asarray(max_returns)[valid_indices])
            candidate_sigs_valid = jnp.array(np.asarray(candidate_sigs)[valid_indices])
            latents_valid = latents_jax[valid_indices]
            levels_valid = jax.tree_util.tree_map(lambda x: x[valid_indices], levels)

            n_valid = len(valid_indices)
            tile_reps = (num_envs + n_valid - 1) // n_valid
            regrets_pad = jnp.tile(regrets_valid, tile_reps)[:num_envs]
            max_returns_pad = jnp.tile(max_returns_valid, tile_reps)[:num_envs]
            candidate_sigs_pad = jnp.tile(candidate_sigs_valid, (tile_reps, 1))[:num_envs]
            latents_pad = jnp.tile(latents_valid, (tile_reps, 1))[:num_envs]
            levels_pad = jax.tree_util.tree_map(
                lambda x: jnp.tile(x, (tile_reps,) + (1,) * (x.ndim - 1))[:num_envs],
                levels_valid,
            )

            level_extra = {
                "max_return": max_returns_pad,
                "latent": latents_pad,
                "behavior_sig": candidate_sigs_pad,
            }
            assert "behavior_sig" in level_extra, (
                "All PLR buffer insertions must include 'behavior_sig'. "
                "Call extract_behavior_signature() first."
            )
            new_sampler, _ = level_sampler.insert_batch(
                train_state.sampler, levels_pad, regrets_pad, level_extra
            )
            train_state = train_state.replace(sampler=new_sampler)
        else:
            # No valid levels — keep sampler unchanged; use full latents for tell()
            levels_pad = levels
            regrets_pad = regrets
            candidate_sigs_pad = candidate_sigs
            latents_pad = latents_jax

        # ES tell (strategy-specific)
        if es_strategy_name == "ns_es":
            buf_size_es = int(train_state.sampler["size"])
            all_buf_sigs = train_state.sampler["levels_extra"]["behavior_sig"]
            buf_valid_mask = jnp.arange(level_sampler.capacity) < buf_size_es
            es_state, mean_novelty = es_strategy.tell(
                es_state, latents_pad, regrets_pad,
                candidate_sigs_pad, all_buf_sigs, buf_valid_mask,
                alpha=config.get("es_alpha", 0.8),
                beta=config.get("es_beta", 0.2),
                k=config.get("es_k_novelty", 5),
            )

        elif es_strategy_name == "sv_cma_es":
            # SV-CMA-ES two-pass eval.
            # First pass already done above: latents_jax, candidate_sigs, regrets (pre-repulsion).
            # Step 4: nudge candidate latents by Stein repulsion for second eval pass.
            n_sv = len(es_state["particles"])
            pop_sv = config.get("es_pop_size", 16)

            particle_bsigs_pre = jnp.stack([
                jnp.mean(candidate_sigs_pad[i * pop_sv:(i + 1) * pop_sv], axis=0)
                for i in range(n_sv)
            ])  # (N, D_bsig)
            particle_means_pre = jnp.stack([
                p["es_state"].mean for p in es_state["particles"]
            ])  # (N, param_dim)

            from accel_training.es_components.stein import compute_stein_repulsion
            sv_epsilon = config.get("sv_epsilon", 0.01)
            repulsion = compute_stein_repulsion(particle_means_pre, particle_bsigs_pre, sv_epsilon)

            # Build repelled latents for second eval pass
            repulsion_tiled = jnp.concatenate([
                jnp.tile(repulsion[i:i + 1], (pop_sv, 1))
                for i in range(n_sv)
            ], axis=0)  # (N*pop_sv, param_dim)
            post_latents = latents_jax[:n_sv * pop_sv] + repulsion_tiled

            # Second eval pass on repelled latents
            rng, rng_eval2, rng_bsig2 = jax.random.split(rng, 3)
            _, levels2, regrets2, max_returns2, _ = eval_fn(rng_eval2, train_state.params, post_latents)
            _, _, _, agent_pos2 = rollout_agent_on_levels_with_positions(
                rng_bsig2, eval_env, env_params,
                train_state.params, network, levels2,
                num_steps=config["eval_rollout_steps"],
            )
            post_bsigs = extract_behavior_signature(agent_pos2, config["eval_rollout_steps"])

            # PLR buffer receives post-repulsion data
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

            # SV-CMA-ES tell()
            es_state, sv_metrics = es_strategy.tell(
                es_state,
                latents_pad,       # pre_cands
                candidate_sigs_pad, # pre_bsigs
                post_latents,      # post_cands
                regrets2,          # post_regrets
                post_bsigs,        # post_bsigs
                sv_epsilon,
            )
            sv_behavior_dist_pre = sv_metrics["sv_behavior_dist_pre"]
            sv_behavior_dist_post = sv_metrics["sv_behavior_dist_post"]

        else:
            # CMA-ES baseline: negate regret for evosax (minimizes)
            fitness_for_cma = -(config.get("es_alpha", 0.8) * regrets_pad)
            es_state = es_strategy.tell(es_state, latents_pad, fitness_for_cma)

        # Exploratory grad updates (optional — train on ES levels too)
        if exploratory and len(valid_indices) > 0:
            rng, rng_train = jax.random.split(rng)
            train_state, _, _, _ = _train_on_levels(rng_train, train_state, levels_pad)

        return (
            rng, train_state, es_state,
            mean_regret, valid_frac,
            mean_novelty, sv_behavior_dist_pre, sv_behavior_dist_post,
        )

    # -----------------------------------------------------------------------
    # Bootstrap loop — fill buffer to bootstrap_min via es_step
    # -----------------------------------------------------------------------
    print(f"\nBootstrapping PLR buffer to {bootstrap_min} levels...")
    boot_step = 0
    while int(train_state.sampler["size"]) < bootstrap_min:
        (
            rng, train_state, es_state,
            _, _, _, _, _,
        ) = _run_es_step(rng, train_state, es_state)
        boot_step += 1
        buf_now = int(train_state.sampler["size"])
        if boot_step % 5 == 0 or buf_now >= bootstrap_min:
            print(f"  [bootstrap step {boot_step}] buffer size: {buf_now}/{bootstrap_min}")
    print(f"  Bootstrap done in {boot_step} steps. Buffer: {int(train_state.sampler['size'])} levels")

    # -----------------------------------------------------------------------
    # Main training loop
    # -----------------------------------------------------------------------
    print(f"\nStarting ES training: {config['num_updates']} updates")
    print(f"  Log dir: {log_dir}")
    print(f"  ES strategy: {es_strategy_name}")
    print(f"  bootstrap_min={bootstrap_min}, replay_ratio={replay_ratio}")
    print(f"  n_envs={num_envs}, n_steps={num_steps}")
    print("-" * 80)

    for update in range(config["num_updates"]):
        t_start = time.time()

        # Decide mode: force es_step if buffer below bootstrap_min
        rng, rng_decision = jax.random.split(rng)
        buf_size_now = int(train_state.sampler["size"])

        if buf_size_now < bootstrap_min:
            mode = "es_step"
        else:
            should_replay = bool(
                level_sampler.sample_replay_decision(train_state.sampler, rng_decision)
            )
            mode = "replay" if should_replay else "es_step"

        # =======================================================
        # ES STEP branch
        # =======================================================
        if mode == "es_step":
            (
                rng, train_state, es_state,
                mean_regret, valid_frac,
                mean_novelty, sv_behavior_dist_pre, sv_behavior_dist_post,
            ) = _run_es_step(rng, train_state, es_state)

        # =======================================================
        # REPLAY branch
        # =======================================================
        else:
            rng, rng_levels, rng_train = jax.random.split(rng, 3)
            new_sampler, (level_inds, replay_levels) = level_sampler.sample_replay_levels(
                train_state.sampler, rng_levels, num_envs
            )
            train_state = train_state.replace(sampler=new_sampler)

            # PPO update on replayed levels
            train_state, new_scores, new_max_returns, losses = _train_on_levels(
                rng_train, train_state, replay_levels
            )

            # Update buffer scores with fresh regret
            prev_extra = level_sampler.get_levels_extra(train_state.sampler, level_inds)
            updated_max_returns = jnp.maximum(prev_extra["max_return"], new_max_returns)
            updated_sampler = level_sampler.update_batch(
                train_state.sampler,
                level_inds,
                new_scores,
                {
                    "max_return":   updated_max_returns,
                    "latent":       prev_extra["latent"],
                    "behavior_sig": prev_extra["behavior_sig"],
                },
            )
            train_state = train_state.replace(sampler=updated_sampler)

            mean_regret = float(jnp.mean(new_scores))
            valid_frac = 1.0  # Replay levels are always valid

        t_elapsed = time.time() - t_start
        buf_size = int(train_state.sampler["size"])
        all_scores = np.asarray(train_state.sampler["scores"])
        mask = np.arange(level_sampler.capacity) < buf_size
        mean_buf_score = float(all_scores[mask].mean()) if mask.any() else 0.0

        _log(update, mode, buf_size, mean_regret, valid_frac, mean_buf_score, t_elapsed)

        if (update + 1) % config.get("wandb_log_freq", 10) == 0:
            log_dict = {
                "update":                update,
                "regret":                mean_regret,
                "novelty_score":         mean_novelty,
                "replay_buffer_size":    buf_size,
                "buffer_occupied":       buf_size / level_sampler.capacity,
                "valid_fraction":        valid_frac,
                "mean_buffer_score":     mean_buf_score,
                "mode":                  mode,
                "sv_behavior_dist_pre":  sv_behavior_dist_pre,
                "sv_behavior_dist_post": sv_behavior_dist_post,
            }
            # ES state scalar metrics (only meaningful after an es_step)
            if mode == "es_step":
                if es_strategy_name in ("cma_es", "ns_es"):
                    evo = es_state["es_state"]
                    log_dict["es/std"]              = float(evo.std)
                    log_dict["es/mean_norm"]        = float(jnp.linalg.norm(evo.mean))
                    log_dict["es/cov_max_eigenval"] = float(evo.D.max())
                    log_dict["es/cov_min_eigenval"] = float(evo.D.min())
                    log_dict["es/cov_condition"]    = float(evo.D.max() / (evo.D.min() + 1e-8))
                elif es_strategy_name == "sv_cma_es":
                    for i, p in enumerate(es_state["particles"]):
                        evo = p["es_state"]
                        log_dict[f"es/p{i}_std"]       = float(evo.std)
                        log_dict[f"es/p{i}_mean_norm"] = float(jnp.linalg.norm(evo.mean))
                    means = jnp.stack([p["es_state"].mean for p in es_state["particles"]])
                    log_dict["es/particle_distance"] = float(jnp.linalg.norm(means[0] - means[1]))
            wandb.log(log_dict, step=update)

        if (update + 1) % config["checkpoint_every"] == 0:
            _save_checkpoint(update + 1, train_state, es_state, rng)

    # --- Final checkpoint ---
    _save_checkpoint(config["num_updates"], train_state, es_state, rng)
    csv_file.close()
    print("\nTraining complete.")
    print(f"  Buffer: {int(train_state.sampler['size'])} levels")
    print(f"  Checkpoints + metrics in: {log_dir}")
    return train_state


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Two-mode ES training: replay / es_step")
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(_HERE, "config.yml"),
        help="Path to YAML config file.",
    )
    parser.add_argument("--log_dir", type=str, default=None, help="Override log_dir from config.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed from config.")
    parser.add_argument("--num_updates", type=int, default=None, help="Override num_updates.")
    parser.add_argument("--run_name", type=str, default=None, help="Override run_name from config.")
    parser.add_argument(
        "--group", type=str, default=None,
        help="WandB group tag (overrides wandb_group in config).",
    )
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
    if args.run_name is not None:
        config["run_name"] = args.run_name
    if args.group is not None:
        config["wandb_group"] = args.group
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
