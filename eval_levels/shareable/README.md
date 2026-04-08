# Shareable OOD Eval Mazes

A flat collection of out-of-distribution maze sets for evaluating agents
trained at 13x13. All files are NPZ-format and compatible with
`scripts/ood_evaluate.py evaluate`.

## Contents

| File                        | N   | Size  | Source / notes |
|-----------------------------|-----|-------|----------------|
| `random_13x13.npz`          | 100 | 13x13 | In-distribution (DR). Random levels, 25 walls. |
| `random_15x15.npz`          | 100 | 15x15 | Random levels, 25 walls. |
| `random_17x17.npz`          | 100 | 17x17 | Random levels, 25 walls. |
| `random_19x19.npz`          |  10 | 19x19 | Random levels, 25 walls. |
| `random_21x21.npz`          |  10 | 21x21 | Random levels, 25 walls. |
| `perfect_13x13.npz`         | 100 | 13x13 | Perfect mazes (mazelib RecursiveBacktracker). |
| `perfect_15x15.npz`         | 100 | 15x15 | Perfect mazes. |
| `perfect_17x17.npz`         | 100 | 17x17 | Perfect mazes. |
| `perfect_19x19.npz`         |  10 | 19x19 | Perfect mazes. |
| `perfect_21x21.npz`         |  10 | 21x21 | Perfect mazes. |
| `dense_15x15.npz`           |  10 | 15x15 | Random levels with ~40% walls (harder). |
| `dense_17x17.npz`           |  10 | 17x17 | Dense random levels. |
| `dense_19x19.npz`           |  10 | 19x19 | Dense random levels. |
| `dense_21x21.npz`           |  10 | 21x21 | Dense random levels. |
| `prefabs_21x21.npz`         |   8 | 21x21 | Friend's 8 hand-designed 21x21 prefabs (PerfectMaze21_1..4, Rooms21_1/2, Labyrinth21_1/2) from the `llm-injection-exp` branch. |
| `labyrinth17_custom.npz`    |   2 | 17x17 | Two new hand-designed 17x17 labyrinths: `Labyrinth17_1` (inward spiral, path length 126) and `Labyrinth17_2` (horizontal serpentine, path length 126). |
| `labyrinth17_custom.txt`    |  —  |  —    | ASCII rendering of the 2 custom labyrinths. |

## NPZ schema

Each `.npz` contains a **stacked** `jaxued.environments.maze.level.Level`:

| Field        | Shape             | Notes |
|--------------|-------------------|-------|
| `wall_map`   | `(N, H, W)` bool  | True = wall. |
| `agent_pos`  | `(N, 2)` uint32   | `(x, y)` starting position. |
| `agent_dir`  | `(N,)`   uint8    | 0=right, 1=down, 2=left, 3=up. |
| `goal_pos`   | `(N, 2)` uint32   | `(x, y)` goal. |
| `width`      | `(N,)`   int32    | Maze width (cols). |
| `height`     | `(N,)`   int32    | Maze height (rows). |
| `names`      | `(N,)` str (opt.) | Present for prefab sets (`prefabs_21x21`, `labyrinth17_custom`). |

## Loading example

```python
import numpy as np
import jax.numpy as jnp
from jaxued.environments.maze.level import Level

data = np.load("eval_levels/shareable/prefabs_21x21.npz", allow_pickle=True)
levels = Level(
    wall_map=jnp.array(data["wall_map"]),
    goal_pos=jnp.array(data["goal_pos"]),
    agent_pos=jnp.array(data["agent_pos"]),
    agent_dir=jnp.array(data["agent_dir"]),
    width=jnp.array(data["width"]),
    height=jnp.array(data["height"]),
)
names = [str(n) for n in data["names"]] if "names" in data.files else None
```
