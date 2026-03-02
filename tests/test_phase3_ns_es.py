"""Phase 3 integration and unit tests for NS-ES components.

Tests:
  1. NSESStrategy ask/tell cycle (ALGO-01)
  2. NSESStrategy tell() uses composite fitness with actual novelty variation (ALGO-01)
  3. extract_behavior_signature returns (pop_size, 169) L1-normalized float32 (INTEG-02)
  4. Empty buffer guard: sample_replay_decision returns False when buffer empty (INTEG-01)
  5. run_archive_warmup() populates the PLR buffer (sampler['size'] > 0) (INTEG-02)
  6. End-to-end 3-update smoke test with ns_es strategy and warmup_n=4 (INTEG-03)

Run with:
    /cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python tests/test_phase3_ns_es.py
from project root (no pytest required).
"""

import os
os.environ["WANDB_MODE"] = "disabled"

import sys

# Path setup — allow imports from project root (mirror tests/test_es_components.py exactly)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, 'accel_training'))
sys.path.insert(0, _ROOT)

import jax
import jax.numpy as jnp


def test_nses_strategy_ask_tell():
    """Test NSESStrategy ask/tell produces correct shapes and types (ALGO-01).

    Verifies:
    - init_state returns dict with 'es_state' key
    - ask returns candidates of shape (pop_size, param_dim)
    - tell returns a tuple of length 2
    - tell returns (new_state_dict, mean_novelty_float)
    - 'es_state' is in the returned new_state dict
    - mean_novelty is a Python float
    - A second tell() call succeeds cleanly (state updates without error)
    """
    from accel_training.es_components.nses_strategy import NSESStrategy

    param_dim = 4
    pop_size = 3
    strategy = NSESStrategy(param_dim=param_dim, pop_size=pop_size)

    rng = jax.random.PRNGKey(0)
    rng, rng_init = jax.random.split(rng)
    state = strategy.init_state(rng_init, {"sigma_init": 0.5})
    assert "es_state" in state, "init_state must return dict with 'es_state' key"

    # Ask — verify candidate shapes
    rng, rng_ask = jax.random.split(rng)
    candidates, state = strategy.ask(state, rng_ask)
    assert candidates.shape == (pop_size, param_dim), (
        f"ask() must return shape ({pop_size}, {param_dim}), got {candidates.shape}"
    )

    # Tell — verify return type and structure
    regrets = jnp.ones(pop_size, dtype=jnp.float32)
    candidate_sigs = jnp.ones((pop_size, 4), dtype=jnp.float32)
    buffer_sigs = jnp.zeros((10, 4), dtype=jnp.float32)
    valid_mask = jnp.arange(10) < 3  # first 3 True

    result = strategy.tell(
        state, candidates, regrets,
        candidate_sigs, buffer_sigs, valid_mask,
        alpha=0.8, beta=0.2,
    )
    assert isinstance(result, tuple) and len(result) == 2, (
        f"tell() must return tuple of length 2, got {type(result)} len={len(result) if hasattr(result, '__len__') else 'N/A'}"
    )
    new_state, mean_novelty = result
    assert "es_state" in new_state, "tell() returned state must have 'es_state' key"
    assert isinstance(mean_novelty, float), (
        f"mean_novelty must be a Python float, got {type(mean_novelty)}"
    )

    # Second tell() — verify state updates cleanly
    rng, rng_ask2 = jax.random.split(rng)
    candidates2, new_state2 = strategy.ask(new_state, rng_ask2)
    result2 = strategy.tell(
        new_state2, candidates2, regrets,
        candidate_sigs, buffer_sigs, valid_mask,
        alpha=0.8, beta=0.2,
    )
    assert isinstance(result2, tuple) and len(result2) == 2, (
        "Second tell() must also return tuple of length 2"
    )

    print("  PASS test_nses_strategy_ask_tell")


def test_nses_tell_uses_composite_fitness():
    """Test that tell() novelty score varies with actual novelty (ALGO-01).

    Verifies:
    - Scenario A: candidates far from buffer -> high novelty
    - Scenario B: candidates identical to buffer -> low novelty
    - novelty_A > novelty_B + 0.01 (non-trivial margin)
    This proves novelty is actually computed and varies with actual distance.
    """
    from accel_training.es_components.nses_strategy import NSESStrategy

    param_dim = 4
    pop_size = 2
    strategy = NSESStrategy(param_dim=param_dim, pop_size=pop_size)

    rng = jax.random.PRNGKey(42)
    rng, rng_init = jax.random.split(rng)
    state = strategy.init_state(rng_init, {"sigma_init": 0.5})

    rng, rng_ask = jax.random.split(rng)
    candidates, state_asked = strategy.ask(state, rng_ask)

    regrets = jnp.ones(pop_size, dtype=jnp.float32)

    # Scenario A: candidates far from buffer (zeros vs ones -> high distance -> high novelty)
    buffer_sigs_A = jnp.zeros((5, param_dim), dtype=jnp.float32)
    candidate_sigs_A = jnp.ones((pop_size, param_dim), dtype=jnp.float32)
    valid_mask_A = jnp.ones(5, dtype=jnp.bool_)

    _, novelty_A = strategy.tell(
        state_asked, candidates, regrets,
        candidate_sigs_A, buffer_sigs_A, valid_mask_A,
        alpha=0.8, beta=0.2,
    )

    # Scenario B: candidates identical to buffer (ones vs ones -> zero distance -> low novelty)
    buffer_sigs_B = jnp.ones((5, param_dim), dtype=jnp.float32)
    candidate_sigs_B = jnp.ones((pop_size, param_dim), dtype=jnp.float32)
    valid_mask_B = jnp.ones(5, dtype=jnp.bool_)

    _, novelty_B = strategy.tell(
        state_asked, candidates, regrets,
        candidate_sigs_B, buffer_sigs_B, valid_mask_B,
        alpha=0.8, beta=0.2,
    )

    assert novelty_A > novelty_B + 0.01, (
        f"Novelty must be higher when candidates are far from buffer. "
        f"Got novelty_A={novelty_A:.4f}, novelty_B={novelty_B:.4f} "
        f"(expected novelty_A > novelty_B + 0.01)"
    )

    print("  PASS test_nses_tell_uses_composite_fitness")


def test_behavior_sig_extraction():
    """Test extract_behavior_signature returns (pop_size, 169) L1-normalized float32 (INTEG-02).

    Verifies:
    - output.shape == (1, 169)
    - L1-normalized: abs(sum(output[0]) - 1.0) < 1e-4
    - output.dtype == jnp.float32
    """
    from regret_fitness import extract_behavior_signature

    # (2 steps, 1 env, 2D coord [col, row])
    agent_positions = jnp.array([[[2, 3]], [[5, 7]]])
    assert agent_positions.shape == (2, 1, 2), (
        f"Test setup: expected (2, 1, 2), got {agent_positions.shape}"
    )

    output = extract_behavior_signature(agent_positions, num_steps=2)

    assert output.shape == (1, 169), (
        f"extract_behavior_signature must return shape (1, 169), got {output.shape}"
    )
    assert abs(float(jnp.sum(output[0])) - 1.0) < 1e-4, (
        f"Output must be L1-normalized (sum=1.0), got sum={float(jnp.sum(output[0])):.6f}"
    )
    assert output.dtype == jnp.float32, (
        f"Output dtype must be float32, got {output.dtype}"
    )

    print("  PASS test_behavior_sig_extraction")


def test_two_bucket_empty_buffer_guard():
    """Test that empty buffer falls back to frontier (sample_replay_decision returns False) (INTEG-01).

    Verifies:
    - LevelSampler with capacity=10, minimum_fill_ratio=0.5 returns False when empty
    - This proves the two-bucket empty-buffer guard: frontier sampling is the fallback
    """
    from jaxued.level_sampler import LevelSampler
    from jaxued.environments.maze.level import Level

    # Build a minimal placeholder level (matching train.py pattern, without VAE)
    wall_map = jnp.zeros((13, 13), dtype=jnp.bool_)
    goal_pos = jnp.array([2, 2], dtype=jnp.int32)   # [col, row]
    agent_pos = jnp.array([5, 5], dtype=jnp.int32)
    agent_dir = jnp.int32(0)
    width = jnp.int32(13)
    height = jnp.int32(13)
    pholder_level = Level(
        wall_map=wall_map,
        goal_pos=goal_pos,
        agent_pos=agent_pos,
        agent_dir=agent_dir,
        width=width,
        height=height,
    )

    level_sampler = LevelSampler(
        capacity=10,
        replay_prob=0.8,
        staleness_coeff=0.3,
        minimum_fill_ratio=0.5,
        prioritization="rank",
        prioritization_params={"temperature": 0.3, "k": 4},
        duplicate_check=False,
    )

    sampler = level_sampler.initialize(
        pholder_level,
        {
            "max_return": jnp.array(-jnp.inf),
            "latent": jnp.zeros(64),
            "behavior_sig": jnp.zeros(169),
        },
    )

    result = level_sampler.sample_replay_decision(sampler, jax.random.PRNGKey(0))
    assert bool(result) == False, (
        f"Empty buffer must fall back to frontier (sample_replay_decision must return False), "
        f"got {result}"
    )

    print("  PASS test_two_bucket_empty_buffer_guard")


def test_archive_warmup_populates_buffer():
    """Test that run_archive_warmup() inserts entries into sampler['size'] > 0 (INTEG-02).

    Guards on VAE checkpoint presence (skips if not found).
    Directly calls run_archive_warmup() with warmup_n=4 and verifies buffer is populated.
    """
    if not os.path.exists("vae/model/checkpoint_420000.pkl"):
        print("  SKIP test_archive_warmup_populates_buffer (VAE checkpoint not found at vae/model/checkpoint_420000.pkl)")
        return

    import numpy as np
    import optax
    from flax import core, struct
    from flax.training.train_state import TrainState as _BaseTrainState
    from jaxued.environments.maze import Maze
    from jaxued.level_sampler import LevelSampler
    from jaxued.wrappers import AutoReplayWrapper
    from agent_loader import ActorCritic
    from ued_interface import load_vae, build_eval_fn
    from accel_training.train import run_archive_warmup, TrainState

    # Minimal config mirroring config.yml defaults for warmup test
    config = {
        "seed": 0,
        "num_train_envs": 4,
        "num_steps": 256,
        "agent_view_size": 5,
        "level_buffer_capacity": 32,
        "replay_prob": 0.8,
        "staleness_coeff": 0.3,
        "minimum_fill_ratio": 0.5,
        "prioritization": "rank",
        "temperature": 0.3,
        "topk_k": 4,
        "score_function": "MaxMC",
        "exploratory_grad_updates": False,
        "use_accel": True,
        "n_candidates": 16,
        "mutation_sigma": 0.5,
        "random_fraction": 0.3,
        "eval_rollout_steps": 128,
        "min_obstacles": 5,
        "min_distance": 3,
        "decode_temperature": 0.25,
        "num_updates": 3,
        "eval_freq": 500,
        "checkpoint_every": 2000,
        "vae_checkpoint": "vae/model/checkpoint_420000.pkl",
        "log_dir": "/tmp/test_phase3_warmup",
        "run_name": "test_warmup",
        "warmup_n": 4,
        "es_strategy": "ns_es",
        "es_alpha": 0.8,
        "es_beta": 0.2,
        "es_pop_size": 16,
        "es_sigma_init": 0.5,
        "es_k_novelty": 5,
        "wandb_project": "test",
        "wandb_log_freq": 1,
        "lr": 0.0001,
        "max_grad_norm": 0.5,
        "gamma": 0.995,
        "gae_lambda": 0.98,
        "clip_eps": 0.2,
        "entropy_coeff": 0.001,
        "critic_coeff": 0.5,
        "epoch_ppo": 5,
        "num_minibatches": 1,
    }

    rng = jax.random.PRNGKey(config["seed"])
    rng_np = np.random.default_rng(config["seed"])

    # Setup environment
    maze_env = Maze(
        max_height=13, max_width=13,
        agent_view_size=config["agent_view_size"],
        normalize_obs=True,
    )
    env = AutoReplayWrapper(maze_env)
    env_params = env.default_params

    # Load VAE and build eval function
    decoder_params = load_vae(config["vae_checkpoint"])
    network = ActorCritic(action_dim=7)
    eval_fn, eval_env = build_eval_fn(
        decoder_params, network, env_params,
        latent_dim=64,
        decode_temperature=config["decode_temperature"],
        num_steps=config["eval_rollout_steps"],
        min_obstacles=config["min_obstacles"],
        min_distance=config["min_distance"],
    )

    # Level sampler
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

    # Placeholder level for sampler initialization (from VAE decode)
    from vae_decoder import decode_latent_to_env, repair_cluttr_sequence
    from env_bridge import cluttr_sequence_to_level
    rng, rng_init, rng_level = jax.random.split(rng, 3)
    z_ph = jax.random.normal(rng_level, (1, 64))
    seq_ph = decode_latent_to_env(decoder_params, z_ph, rng_key=rng_level)
    seq_ph = repair_cluttr_sequence(seq_ph[0])
    pholder_level = cluttr_sequence_to_level(seq_ph, rng_level)

    # Initialize network params
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

    def linear_schedule(count):
        n_mb_per_update = config["num_minibatches"] * config["epoch_ppo"]
        frac = 1.0 - (count // n_mb_per_update) / config["num_updates"]
        return config["lr"] * frac

    tx = optax.chain(
        optax.clip_by_global_norm(config["max_grad_norm"]),
        optax.adam(learning_rate=linear_schedule, eps=1e-5),
    )

    sampler = level_sampler.initialize(
        pholder_level,
        {
            "max_return": jnp.array(-jnp.inf),
            "latent": jnp.zeros(64, dtype=jnp.float32),
            "behavior_sig": jnp.zeros(169, dtype=jnp.float32),
        },
    )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
        sampler=sampler,
    )

    # Call run_archive_warmup with warmup_n=4
    rng, rng_np, train_state_out = run_archive_warmup(
        rng, rng_np, train_state, level_sampler, eval_fn,
        eval_env, env_params, network, config,
    )

    buf_size = int(train_state_out.sampler["size"])
    assert buf_size > 0, (
        f"run_archive_warmup did not populate the buffer (sampler['size'] = {buf_size})"
    )
    assert buf_size <= 4, (
        f"warmup_n=4 should insert at most 4 entries, got {buf_size}"
    )

    print(f"  PASS test_archive_warmup_populates_buffer — buffer size: {buf_size}")


def test_end_to_end_3_updates():
    """Smoke test: train.py runs 3 updates with warmup_n=4 and ns_es strategy (INTEG-03).

    Verifies that the full NS-ES pipeline (warmup -> training loop -> ES routing) runs
    without crashing. warmup_n=4 exercises run_archive_warmup in the smoke test.
    """
    if not os.path.exists("vae/model/checkpoint_420000.pkl"):
        print("  SKIP test_end_to_end_3_updates (VAE checkpoint not found at vae/model/checkpoint_420000.pkl)")
        return

    import os as _os
    _os.makedirs("/tmp/test_phase3_smoke", exist_ok=True)

    from accel_training.train import train

    config = {
        # Agent / PPO
        "lr": 0.0001,
        "max_grad_norm": 0.5,
        "gamma": 0.995,
        "gae_lambda": 0.98,
        "clip_eps": 0.2,
        "entropy_coeff": 0.001,
        "critic_coeff": 0.5,
        "epoch_ppo": 5,
        "num_minibatches": 1,
        # Rollout
        "num_train_envs": 32,
        "num_steps": 256,
        "agent_view_size": 5,
        # Level buffer
        "level_buffer_capacity": 4000,
        "replay_prob": 0.8,
        "staleness_coeff": 0.3,
        "minimum_fill_ratio": 0.5,
        "prioritization": "rank",
        "temperature": 0.3,
        "topk_k": 4,
        "score_function": "MaxMC",
        "exploratory_grad_updates": False,
        # ACCEL / MAP-Elites
        "use_accel": True,
        "n_candidates": 16,
        "mutation_sigma": 0.5,
        "random_fraction": 0.3,
        "eval_rollout_steps": 128,
        "min_obstacles": 5,
        "min_distance": 3,
        "decode_temperature": 0.25,
        # Training loop — overrides for smoke test
        "num_updates": 3,
        "eval_freq": 500,
        "checkpoint_every": 2000,
        # Paths
        "vae_checkpoint": "vae/model/checkpoint_420000.pkl",
        "log_dir": "/tmp/test_phase3_smoke",
        "run_name": "test_smoke",
        # ES / NS-ES
        "warmup_n": 4,
        "es_strategy": "ns_es",
        "es_alpha": 0.8,
        "es_beta": 0.2,
        "es_pop_size": 16,
        "es_sigma_init": 0.5,
        "es_k_novelty": 5,
        # WandB
        "wandb_project": "test",
        "wandb_log_freq": 1,
        # Misc
        "seed": 0,
    }

    train_state, archive = train(config)

    assert train_state is not None, "train() must return a non-None train_state"
    assert "size" in train_state.sampler, "train_state.sampler must have 'size' key"

    print("  PASS test_end_to_end_3_updates")


if __name__ == "__main__":
    print("Running Phase 3 NS-ES tests...")
    test_nses_strategy_ask_tell()
    test_nses_tell_uses_composite_fitness()
    test_behavior_sig_extraction()
    test_two_bucket_empty_buffer_guard()
    test_archive_warmup_populates_buffer()
    test_end_to_end_3_updates()
    print("\nAll Phase 3 tests passed.")
