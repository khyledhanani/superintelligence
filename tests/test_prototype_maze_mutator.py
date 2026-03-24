from pathlib import Path
import sys

import jax
import numpy as np
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from es.prototype_maze_mutator import mutate_wall_maps_with_prototypes


def test_mutate_levels_with_prototypes_changes_center_and_preserves_agent_goal():
    library = {
        "patch_size": 5,
        "center_size": 3,
        "num_codes": 2,
        "context_ids": jnp.array([0], dtype=jnp.int32),
        "context_code_probs": jnp.array([[0.1, 0.9]], dtype=jnp.float32),
        "global_probs": jnp.array([0.5, 0.5], dtype=jnp.float32),
        "prototypes": jnp.array(
            [
                np.zeros((9,), dtype=np.float32),
                np.ones((9,), dtype=np.float32),
            ]
        ),
        "prototype_ids": jnp.array([0, 511], dtype=jnp.int32),
    }
    child_walls, info = mutate_wall_maps_with_prototypes(
        jax.random.PRNGKey(0),
        jnp.zeros((1, 5, 5), dtype=jnp.bool_),
        jnp.array([[4, 4]], dtype=jnp.uint32),
        jnp.array([[0, 0]], dtype=jnp.uint32),
        library,
        deterministic=True,
    )

    child = np.asarray(child_walls[0], dtype=bool)
    assert int(np.asarray(info["chosen_code"])[0]) == 1
    assert child[1:4, 1:4].sum() == 9
    assert not child[0, 0]
    assert not child[4, 4]
