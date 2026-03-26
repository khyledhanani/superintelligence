# Coding Conventions

**Analysis Date:** 2025-03-23

## Naming Patterns

**Files:**
- Lowercase with underscores: `maze_plr.py`, `level_sampler.py`, `autoreplay.py`
- Environment files: `env.py`, `env_editor.py`, `env_solved.py`
- Utility/helper files: `utils.py`, `util.py`
- No class-based file naming

**Functions:**
- Snake case for all functions: `compute_gae()`, `sample_trajectories_rnn()`, `initialize_carry()`
- Descriptive names indicating purpose: `accumulate_rollout_stats()`, `positive_value_loss()`
- Nested internal functions use lowercase: `compute_gae_at_timestep()`, `update_minibatch()`, `loss_fn()`
- Private/internal methods prefixed with underscore: `_insert_new()`, `_proportion_filled()`, `_get_next_idx()`

**Variables:**
- Snake case for all variables: `max_episode_length`, `clip_eps`, `loss_fn`
- Abbreviations used sparingly: `rng` (random number generator), `obs` (observation), `env` (environment)
- Type aliases and constants: `Prioritization`, `Sampler`, `UpdateState` (CapitalCase)
- Loop variables: `carry`, `x`, `input` (single letters acceptable in functional code)
- Batch dimensions: `num_envs`, `num_steps`, `batch_size`

**Classes:**
- PascalCase for all classes: `ResetRNN`, `LevelSampler`, `TrainState`, `Maze`, `ActorCritic`
- Enum subclasses: `UpdateState`, `Actions` (PascalCase)
- Dataclass names: `Level`, `EnvState`, `Observation`, `AutoReplayState`

**Type Hints:**
- Used consistently throughout: `def compute_gae(...) -> Tuple[chex.Array, chex.Array]:`
- Import types from `typing`: `Tuple`, `Optional`, `Callable`, `Union`, `Sequence`
- JAX/Chex types: `chex.Array`, `chex.PRNGKey`, `chex.ArrayTree`
- Custom types defined at module level: `Prioritization = Literal["rank", "topk"]`

## Code Style

**Formatting:**
- No formatter configured (no `.black`, `.prettier`, `.flake8` found)
- Consistent style observed in codebase despite lack of tool enforcement
- Line breaks: Functions broken across lines for readability
- Indentation: 4 spaces (Python standard)

**Linting:**
- No linting configuration detected (`.eslintrc*`, `pylint` config absent)
- Code relies on manual style adherence

## Import Organization

**Order:**
1. Standard library: `import json`, `import time`, `from typing import ...`
2. Third-party packages: `import jax`, `import jax.numpy as jnp`, `import numpy as np`
3. Third-party ML/RL libraries: `from flax import struct`, `import optax`, `import distrax`
4. Project imports: `from jaxued.environments import ...`, `from jaxued.level_sampler import ...`
5. Conditional imports: `sys.path.insert(0, ...)` for optional modules (VAE, CMA-ES)

**Path Aliases:**
- No import aliases configured (`PYTHONPATH` management only)
- Relative imports within package: `from .level import Level`, `from .env import Maze`
- Conditional sys.path injection for optional features (e.g., VAE/CMA-ES modules)

**Common Patterns:**
```python
import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Callable
import chex
from flax import struct
from jaxued.environments.underspecified_env import EnvState, Observation, Level
```

## Error Handling

**Patterns:**
- `assert` statements for preconditions: `assert all(len(row) == len(rows[0]) for row in rows), "All rows must have same length"`
- `raise Exception` for invalid configuration: `raise Exception(f"\"{prioritization}\" not a valid prioritization.")`
- `raise ValueError` for invalid user input: `raise ValueError(f"Unknown score function: {config['score_function']}")`
- Try-except for external system operations (GCS uploads): `try: ... except (ImportError, Exception) as e: ...`

**Guard clauses:**
- JAX lax.cond for conditional logic: `jax.lax.cond(replace_cond, _replace, lambda: sampler)`
- JAX lax.select for value selection: `jax.lax.select(done, x, y)`

**No custom exception classes detected** — uses built-in Exception, ValueError, ImportError

## Logging

**Framework:** `print()` for debug output, `wandb` for metrics logging

**Patterns:**
- Prefix logging with context: `print(f"[CMA-ES] Initialized es_state: ...")`, `print(f"[GCS] Uploaded ...")`
- WandB for training metrics: `wandb.log(log_dict)`, `wandb.init(config=config, project=project)`
- Structured logging dictionaries: Build dicts before logging (`log_dict = {...}; wandb.log(log_dict)`)
- No timestamp or level prefixes in print statements

**Examples from codebase:**
```python
print(f"[CMA-ES] VAE loaded from {config['vae_checkpoint_path']}")
print(f"[GCS] Config saved to {overall_save_dir}/config.json")
print(f"Logging update: {stats['update_count']}")
```

## Comments

**When to Comment:**
- Complex JAX operations: `# Capture agent position before the step (matches obs)`
- Non-obvious calculations: `# CMA-ES minimizes; negate scores so high-regret = low fitness`
- State transitions: `# === Below is used for logging ===`
- Warnings and special notes: `# CRITICAL:`, `# TODO:` not observed but raises used

**Docstring Format:**

Triple-quoted docstrings for all public functions and classes. Two formats observed:

**Simple docstring (one-liner + Args/Returns):**
```python
def sample_replay_level(self, sampler: Sampler, rng: chex.PRNGKey) -> Tuple[Sampler, Tuple[int, Level]]:
    """
    Samples a replay level from the buffer...

    Args:
        sampler (Sampler): The sampler object
        rng (chex.PRNGKey):

    Returns:
        Tuple[Sampler, Tuple[int, Level]]: The updated sampler...
    """
```

**Class docstring with Examples:**
```python
class LevelSampler:
    """
    The `LevelSampler` provides all of the functionality...

    Examples:
        >>>
        level_sampler = LevelSampler(4000)
        sampler = level_sampler.initialize(...)

    Args:
        capacity (int): The maximum number of levels...
    """
```

**Inline method docstrings:**
```python
def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
    """Check whether state is terminal."""
```

## Function Design

**Size:** Functions typically 10-80 lines; nested helper functions for JAX scan/vmap callbacks

**Parameters:**
- Use `rng: chex.PRNGKey` as first param for functions needing randomness
- Environment/config as single argument (not unpacked): `env: UnderspecifiedEnv`, `config: dict`
- `params: EnvParams` always passed explicitly (not stored globally)
- Keyword-only arguments for optional flags: `time_average` (uses `*` separator in signature)

**Return Values:**
- Explicit tuple returns with type hints: `Tuple[Sampler, Tuple[int, Level]]`
- Unpacking at call site: `sampler, (idx, level) = level_sampler.sample_replay_level(...)`
- Functions mutate through return (JAX functional style, no in-place modifications)

**JAX-specific patterns:**
- Use `jax.lax.scan()` for loops: `jax.lax.scan(scan_fn, init_carry, xs)`
- Use `jax.lax.cond()` for conditionals: `jax.lax.cond(condition, true_fn, false_fn)`
- Use `jax.tree_util.tree_map()` for recursive operations: `jax.tree_util.tree_map(lambda x: x + 1, tree)`

## Module Design

**Exports:**
- Explicit exports via `__all__` not observed
- Package `__init__.py` imports key classes: `from .autoreplay import AutoReplayWrapper`
- Location: `src/jaxued/wrappers/__init__.py`, `src/jaxued/environments/__init__.py`

**Barrel Files:**
Yes, used for re-exporting:
- `src/jaxued/environments/__init__.py`: `from .maze import Maze, MazeEditor, MazeSolved, MazeRenderer`
- `src/jaxued/wrappers/__init__.py`: `from .autoreplay import AutoReplayWrapper`

**Module Structure:**
- Core library: `src/jaxued/` (base classes, utils)
- Environments: `src/jaxued/environments/` (specific env implementations)
- Wrappers: `src/jaxued/wrappers/` (composable environment decorators)
- Examples/scripts: `examples/` (runnable training scripts)

**Dataclasses (Flax struct):**
Used throughout for state representation:
```python
@struct.dataclass
class Level:
    wall_map: chex.Array
    goal_pos: chex.Array
    agent_pos: chex.Array
    agent_dir: int
```

Benefits: JAX-compatible, pytree registration automatic, immutable-by-default

---

*Convention analysis: 2025-03-23*
