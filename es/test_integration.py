"""
Integration tests for the ES regret fitness pipeline.

Tests:
    1. env_bridge: CLUTTR sequence -> Maze Level (golden comparison with visualize_envs.py)
    2. flood_fill_solvable: known solvable/unsolvable cases
    3. agent_loader: checkpoint loading + contract test
    4. decoder/regret unit checks for new sampling + complexity gating
    5. Small end-to-end run (placeholder fitness, 2 gens) to verify wiring

Run:
    python es/test_integration.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import os
import sys
import traceback

# Add parent dir to path (same as other es/ modules)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ── Helpers ──────────────────────────────────────────────────────────────
def run_test(name, fn):
    """Run a single test, print PASS/FAIL."""
    try:
        fn()
        print(f"  PASS: {name}")
        return True
    except Exception as e:
        print(f"  FAIL: {name}")
        traceback.print_exc()
        return False


# ── Test 1: Environment Bridge ──────────────────────────────────────────
def test_env_bridge_golden():
    """Golden test: compare env_bridge.cluttr_sequence_to_level with visualize_envs.sequence_to_grid."""
    from env_bridge import cluttr_sequence_to_level
    from visualize_envs import sequence_to_grid

    # Create a known test sequence:
    # Obstacles at linear positions 1, 14, 27 (1-indexed)
    #   pos 1  -> row=0, col=0
    #   pos 14 -> row=1, col=0
    #   pos 27 -> row=2, col=0
    # Goal at position 85 -> row=6, col=6 (center)
    # Agent at position 169 -> row=12, col=12 (bottom-right corner)
    obstacles = np.zeros(50, dtype=np.int32)
    obstacles[0] = 1
    obstacles[1] = 14
    obstacles[2] = 27
    seq = np.concatenate([obstacles, np.array([85, 169], dtype=np.int32)])

    # env_bridge: JAX path
    level = cluttr_sequence_to_level(jnp.array(seq), jax.random.PRNGKey(0))

    # visualize_envs: numpy reference
    grid = sequence_to_grid(seq)

    # Compare wall positions
    for r in range(13):
        for c in range(13):
            is_wall_bridge = bool(level.wall_map[r, c])
            is_wall_viz = (grid[r, c] == 1)
            # Goal and agent positions are cleared in wall_map
            if (r, c) in [(6, 6), (12, 12)]:
                continue
            assert is_wall_bridge == is_wall_viz, (
                f"Wall mismatch at ({r},{c}): bridge={is_wall_bridge}, viz={is_wall_viz}"
            )

    # Check goal position: Level uses [x=col, y=row]
    assert int(level.goal_pos[0]) == 6, f"Goal col: expected 6, got {int(level.goal_pos[0])}"
    assert int(level.goal_pos[1]) == 6, f"Goal row: expected 6, got {int(level.goal_pos[1])}"

    # Check agent position
    assert int(level.agent_pos[0]) == 12, f"Agent col: expected 12, got {int(level.agent_pos[0])}"
    assert int(level.agent_pos[1]) == 12, f"Agent row: expected 12, got {int(level.agent_pos[1])}"

    # Walls at agent/goal positions should be cleared
    assert not bool(level.wall_map[6, 6]), "Wall at goal position should be cleared"
    assert not bool(level.wall_map[12, 12]), "Wall at agent position should be cleared"


def test_env_bridge_edge_cases():
    """Edge cases: boundary indices, obstacle on goal/agent, duplicate obstacles."""
    from env_bridge import cluttr_sequence_to_level

    # All zeros except goal and agent (no obstacles)
    obstacles = np.zeros(50, dtype=np.int32)
    seq = np.concatenate([obstacles, np.array([1, 169], dtype=np.int32)])
    level = cluttr_sequence_to_level(jnp.array(seq), jax.random.PRNGKey(0))

    # No walls should exist
    assert int(level.wall_map.sum()) == 0, f"Expected 0 walls, got {int(level.wall_map.sum())}"

    # Goal at pos 1 -> (row=0, col=0), Agent at pos 169 -> (row=12, col=12)
    assert int(level.goal_pos[0]) == 0 and int(level.goal_pos[1]) == 0
    assert int(level.agent_pos[0]) == 12 and int(level.agent_pos[1]) == 12

    # Duplicate obstacles: should produce same result as single
    obstacles2 = np.zeros(50, dtype=np.int32)
    obstacles2[0] = 50
    obstacles2[1] = 50  # duplicate
    obstacles2[2] = 50  # duplicate
    seq2 = np.concatenate([obstacles2, np.array([85, 169], dtype=np.int32)])
    level2 = cluttr_sequence_to_level(jnp.array(seq2), jax.random.PRNGKey(0))
    # Should have exactly 1 wall
    assert int(level2.wall_map.sum()) == 1, f"Duplicate obstacles: expected 1 wall, got {int(level2.wall_map.sum())}"


def test_env_bridge_vmap():
    """Test that cluttr_sequence_to_level works with vmap."""
    from env_bridge import cluttr_sequence_to_level

    batch_size = 4
    seqs = np.zeros((batch_size, 52), dtype=np.int32)
    for i in range(batch_size):
        seqs[i, 0] = i + 1  # different obstacle per env
        seqs[i, 50] = 85    # goal: center
        seqs[i, 51] = 169   # agent: bottom-right

    rng_keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
    levels = jax.vmap(cluttr_sequence_to_level)(jnp.array(seqs), rng_keys)

    assert levels.wall_map.shape == (batch_size, 13, 13)
    assert levels.goal_pos.shape == (batch_size, 2)
    assert levels.agent_pos.shape == (batch_size, 2)


# ── Test 2: Solvability ─────────────────────────────────────────────────
def test_solvability_empty_grid():
    """Empty 13x13 grid: agent at (0,0), goal at (12,12) -> solvable."""
    from env_bridge import flood_fill_solvable

    wall_map = jnp.zeros((13, 13), dtype=jnp.bool_)
    agent_pos = jnp.array([0, 0], dtype=jnp.uint32)  # [x=col, y=row]
    goal_pos = jnp.array([12, 12], dtype=jnp.uint32)

    assert bool(flood_fill_solvable(wall_map, agent_pos, goal_pos)), "Empty grid should be solvable"


def test_solvability_walled_off():
    """Wall bisecting grid vertically: agent on left, goal on right -> unsolvable."""
    from env_bridge import flood_fill_solvable

    wall_map = jnp.zeros((13, 13), dtype=jnp.bool_)
    # Vertical wall at col=6 (all rows)
    wall_map = wall_map.at[:, 6].set(True)

    agent_pos = jnp.array([0, 6], dtype=jnp.uint32)  # col=0, row=6
    goal_pos = jnp.array([12, 6], dtype=jnp.uint32)   # col=12, row=6

    assert not bool(flood_fill_solvable(wall_map, agent_pos, goal_pos)), (
        "Grid bisected by wall should be unsolvable"
    )


def test_solvability_adjacent():
    """Agent and goal adjacent, no walls -> solvable."""
    from env_bridge import flood_fill_solvable

    wall_map = jnp.zeros((13, 13), dtype=jnp.bool_)
    agent_pos = jnp.array([5, 5], dtype=jnp.uint32)
    goal_pos = jnp.array([6, 5], dtype=jnp.uint32)  # one step right

    assert bool(flood_fill_solvable(wall_map, agent_pos, goal_pos)), (
        "Adjacent agent/goal with no walls should be solvable"
    )


def test_solvability_fully_walled():
    """Fully walled grid (except agent/goal cells) -> unsolvable if not adjacent."""
    from env_bridge import flood_fill_solvable

    wall_map = jnp.ones((13, 13), dtype=jnp.bool_)
    # Clear agent and goal positions
    wall_map = wall_map.at[0, 0].set(False)  # agent at (col=0, row=0)
    wall_map = wall_map.at[12, 12].set(False)  # goal at (col=12, row=12)

    agent_pos = jnp.array([0, 0], dtype=jnp.uint32)
    goal_pos = jnp.array([12, 12], dtype=jnp.uint32)

    assert not bool(flood_fill_solvable(wall_map, agent_pos, goal_pos)), (
        "Fully walled grid should be unsolvable"
    )


def test_solvability_vmap():
    """Test solvability with vmap over a batch."""
    from env_bridge import flood_fill_solvable

    # Two envs: one solvable (empty), one unsolvable (bisected)
    walls = jnp.zeros((2, 13, 13), dtype=jnp.bool_)
    walls = walls.at[1, :, 6].set(True)  # second env: bisected

    agents = jnp.array([[0, 0], [0, 6]], dtype=jnp.uint32)
    goals = jnp.array([[12, 12], [12, 6]], dtype=jnp.uint32)

    results = jax.vmap(flood_fill_solvable)(walls, agents, goals)

    assert bool(results[0]), "First env (empty) should be solvable"
    assert not bool(results[1]), "Second env (bisected) should be unsolvable"


# ── Test 3: Agent Loading ────────────────────────────────────────────────
def test_agent_load_and_contract():
    """Load agent checkpoint and run contract test."""
    from agent_loader import load_agent, verify_agent_contract

    checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', 'agent_folder')
    agent_params, network = load_agent(checkpoint_dir, action_dim=7)

    # Contract test (this asserts internally)
    verify_agent_contract(agent_params, network)


def test_agent_forward_pass_batched():
    """Run agent forward pass on a batch to verify shapes."""
    from agent_loader import load_agent, ActorCritic
    from jaxued.environments.maze.env import Observation

    checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', 'agent_folder')
    agent_params, network = load_agent(checkpoint_dir, action_dim=7)

    batch_size = 4
    dummy_image = jnp.zeros((batch_size, 5, 5, 3), dtype=jnp.float32)
    dummy_dir = jnp.zeros(batch_size, dtype=jnp.uint8)
    dummy_obs = Observation(image=dummy_image, agent_dir=dummy_dir)
    dummy_done = jnp.zeros(batch_size, dtype=jnp.bool_)

    hstate = ActorCritic.initialize_carry((batch_size,))
    # Add leading batch dim [1, batch_size, ...]
    x = jax.tree_util.tree_map(lambda a: a[None, ...], (dummy_obs, dummy_done))
    hstate_out, pi, value = network.apply({'params': agent_params}, x, hstate)

    assert pi.logits.shape == (1, batch_size, 7), f"Expected (1, {batch_size}, 7), got {pi.logits.shape}"
    assert value.shape == (1, batch_size), f"Expected (1, {batch_size}), got {value.shape}"


# ── Test 4: Decoder + complexity checks ──────────────────────────────────
def test_decoder_backward_compatible_none_rng():
    """decode_latent_to_env(..., rng_key=None) should match legacy argmax path."""
    from vae_decoder import load_vae_params, extract_decoder_params, decode_latent_to_env

    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, '..', 'vae', 'model', 'checkpoint_420000.pkl')
    full_params = load_vae_params(checkpoint_path)
    decoder_params = extract_decoder_params(full_params)

    z = jnp.zeros((3, 64), dtype=jnp.float32)
    seq_legacy = decode_latent_to_env(decoder_params, z)
    seq_none_rng = decode_latent_to_env(decoder_params, z, rng_key=None, temperature=1.0)
    assert jnp.array_equal(seq_legacy, seq_none_rng), "rng_key=None path changed legacy decode behavior"


def test_decoder_sampling_executes():
    """Sampling path should run and return valid token ranges/shapes."""
    from vae_decoder import load_vae_params, extract_decoder_params, decode_latent_to_env

    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, '..', 'vae', 'model', 'checkpoint_420000.pkl')
    full_params = load_vae_params(checkpoint_path)
    decoder_params = extract_decoder_params(full_params)

    z = jnp.zeros((4, 64), dtype=jnp.float32)
    seq = decode_latent_to_env(
        decoder_params, z, rng_key=jax.random.PRNGKey(0), temperature=1.0
    )
    assert seq.shape == (4, 52), f"Unexpected sampled decode shape: {seq.shape}"
    assert jnp.all((seq >= 0) & (seq <= 169)), "Sampled tokens out of expected [0,169] range"


def test_complexity_mask_filters_trivial():
    """Complexity mask should reject trivial envs and keep sufficiently complex ones."""
    from regret_fitness import compute_complexity_mask

    # Trivial: no obstacles, adjacent goal/agent.
    trivial = np.zeros(52, dtype=np.int32)
    trivial[50] = 1
    trivial[51] = 2

    # Complex enough: >= 5 obstacles and large agent-goal distance.
    complex_seq = np.zeros(52, dtype=np.int32)
    complex_seq[:6] = np.array([10, 20, 30, 40, 50, 60], dtype=np.int32)
    complex_seq[50] = 169
    complex_seq[51] = 1

    seqs = jnp.array(np.stack([trivial, complex_seq], axis=0))
    mask = compute_complexity_mask(seqs, min_obstacles=5, min_distance=3, inner_dim=13)
    assert bool(mask[0]) is False, "Trivial sequence should fail complexity mask"
    assert bool(mask[1]) is True, "Complex sequence should pass complexity mask"


# ── Test 5: Small integration (placeholder, 2 gens) ─────────────────────
def test_small_placeholder_run():
    """Run 2 generations of placeholder fitness to verify the full loop."""
    import yaml

    # Load configs
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '..', 'vae', 'vae_train_config.yml')
    with open(config_path, 'r') as f:
        vae_config = yaml.safe_load(f)

    checkpoint_path = os.path.join(script_dir, '..', 'vae', 'model', 'checkpoint_420000.pkl')
    output_dir = os.path.join(script_dir, '_test_output')

    config = {
        'seed': 42,
        'pop_size': 4,
        'num_generations': 2,
        'sigma_init': 1.0,
        'fitness_mode': 'placeholder',
        'warm_start': False,
        'latent_dim': vae_config['latent_dim'],
        'inner_dim': 13,
        'checkpoint_path': checkpoint_path,
        'output_dir': output_dir,
        'log_freq': 1,
        'w_obstacles': 0.4,
        'w_distance': 0.4,
        'w_validity': 0.2,
    }

    from evolve_envs import run_evolution
    es_state, final_seqs = run_evolution(config)

    # Check outputs were saved
    assert os.path.exists(os.path.join(output_dir, 'evolved_envs.npy')), "evolved_envs.npy not saved"
    assert os.path.exists(os.path.join(output_dir, 'best_env.npy')), "best_env.npy not saved"
    assert os.path.exists(os.path.join(output_dir, 'fitness_history.npy')), "fitness_history.npy not saved"

    # Cleanup
    import shutil
    shutil.rmtree(output_dir, ignore_errors=True)


# ── Main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("ES Pipeline Integration Tests")
    print("=" * 60)

    results = []

    print("\n[1] Environment Bridge Tests")
    results.append(run_test("env_bridge golden comparison", test_env_bridge_golden))
    results.append(run_test("env_bridge edge cases", test_env_bridge_edge_cases))
    results.append(run_test("env_bridge vmap", test_env_bridge_vmap))

    print("\n[2] Solvability Tests")
    results.append(run_test("empty grid solvable", test_solvability_empty_grid))
    results.append(run_test("walled-off unsolvable", test_solvability_walled_off))
    results.append(run_test("adjacent solvable", test_solvability_adjacent))
    results.append(run_test("fully walled unsolvable", test_solvability_fully_walled))
    results.append(run_test("solvability vmap", test_solvability_vmap))

    print("\n[3] Agent Loading Tests")
    results.append(run_test("agent load + contract test", test_agent_load_and_contract))
    results.append(run_test("agent batched forward pass", test_agent_forward_pass_batched))

    print("\n[4] Decoder + Complexity Tests")
    results.append(run_test("decoder backward compatibility", test_decoder_backward_compatible_none_rng))
    results.append(run_test("decoder sampling path", test_decoder_sampling_executes))
    results.append(run_test("complexity mask filtering", test_complexity_mask_filters_trivial))

    print("\n[5] Small Integration Run (placeholder, 2 gens)")
    results.append(run_test("placeholder evolution loop", test_small_placeholder_run))

    # Summary
    passed = sum(results)
    total = len(results)
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    if passed == total:
        print("All tests PASSED!")
    else:
        print(f"{total - passed} test(s) FAILED")
        sys.exit(1)
