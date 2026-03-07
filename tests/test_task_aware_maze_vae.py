import jax
import jax.numpy as jnp
import numpy as np

from es.maze_ae import MazeTaskAwareVAE, compute_structural_targets
from es.plwm_scoring import task_aware_objective, structural_difficulty_surrogate
from vae.train_maze_ae import compute_loss


def test_structural_targets_solvable_vs_unsolvable():
    h = w = 13

    empty = np.zeros((h, w), dtype=bool)
    blocked = np.zeros((h, w), dtype=bool)
    blocked[:, 6] = True

    wall_map = jnp.asarray(np.stack([empty, blocked], axis=0))
    goal_pos = jnp.asarray([[12, 12], [12, 12]], dtype=jnp.uint32)
    agent_pos = jnp.asarray([[0, 0], [0, 0]], dtype=jnp.uint32)

    s = compute_structural_targets(wall_map, goal_pos, agent_pos)
    # s1 solvable
    assert float(s[0, 0]) == 1.0
    assert float(s[1, 0]) == 0.0


def test_compute_loss_curriculum_zero_when_no_confidence():
    key = jax.random.PRNGKey(0)
    b, h, w = 8, 13, 13
    grids = jnp.zeros((b, h, w, 3), dtype=jnp.float32)
    # place simple goal/agent
    grids = grids.at[:, 0, 0, 2].set(1.0)
    grids = grids.at[:, 12, 12, 1].set(1.0)

    static = compute_structural_targets(
        grids[:, :, :, 0] > 0.5,
        jnp.tile(jnp.array([[12, 12]], dtype=jnp.uint32), (b, 1)),
        jnp.tile(jnp.array([[0, 0]], dtype=jnp.uint32), (b, 1)),
    )

    model = MazeTaskAwareVAE(latent_dim=16, height=h, width=w)
    params = model.init(key, grids, z_rng=key, deterministic=False)["params"]

    batch = {
        "grids": grids,
        "static_targets": static,
        "p_ema": jnp.zeros((b,), dtype=jnp.float32),
        "success_obs_count": jnp.zeros((b,), dtype=jnp.float32),
    }
    cfg = {
        "latent_dim": 16,
        "height": h,
        "width": w,
        "wall_pos_weight": 7.0,
        "wall_bce_weight": 1.0,
        "wall_dice_weight": 1.5,
        "goal_ce_weight": 2.0,
        "agent_ce_weight": 2.0,
        "overlap_penalty_weight": 0.5,
        "beta": 0.05,
        "lambda_static": 1.0,
        "lambda_curriculum": 0.5,
        "lambda_metric": 0.2,
        "lambda_valid": 0.5,
        "dynamic_confidence_ref": 20.0,
        "metric_y_weights": [1.0] * 9,
        "train_stage": "stage2",
    }

    total, metrics = compute_loss(params, batch, key, cfg)
    assert jnp.isfinite(total)
    assert jnp.isfinite(metrics["curr"])
    assert abs(float(metrics["curr"])) < 1e-8


def test_task_aware_objective_prefers_in_band_candidate():
    # Same values except p_pred, so in-band candidate should win.
    score = task_aware_objective(
        p_pred=jnp.array([[0.5, 0.95]], dtype=jnp.float32),
        learnability_pred=jnp.array([[0.2, 0.2]], dtype=jnp.float32),
        invalid_prob=jnp.array([[0.1, 0.1]], dtype=jnp.float32),
        bfs_norm_pred=jnp.array([[0.3, 0.3]], dtype=jnp.float32),
        wall_density_pred=jnp.array([[0.2, 0.2]], dtype=jnp.float32),
        parent_bfs_norm=jnp.array([[0.29]], dtype=jnp.float32),
        parent_wall_density=jnp.array([[0.2]], dtype=jnp.float32),
        low=0.3,
        high=0.7,
    )
    assert float(score[0, 0]) > float(score[0, 1])


def test_structural_surrogate_runs_for_single_and_multi_candidate_shapes():
    wall = jnp.zeros((2, 13, 13), dtype=jnp.bool_)
    goal = jnp.array([[12, 12], [12, 12]], dtype=jnp.uint32)
    agent = jnp.array([[0, 0], [0, 0]], dtype=jnp.uint32)

    s = structural_difficulty_surrogate(
        wall,
        goal,
        agent,
        weight_bfs=1.0,
        weight_slack=1.0,
        weight_dead_ends=0.2,
        weight_walls=0.05,
        weight_branches=0.0,
        require_solvable=True,
    )
    assert s.shape == (2,)
