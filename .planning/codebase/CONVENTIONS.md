# Coding Conventions

**Analysis Date:** 2026-03-11

## Naming Patterns

**Files:**
- `snake_case.py` for module files: `maze/env.py`, `level_sampler.py`, `autoreplay.py`
- Class implementations with PascalCase: `ActorCritic`, `Level`, `Maze`, `TrainState`
- Wrapper classes follow `Wrapper` pattern: `AutoReplayWrapper`, `AutoResetWrapper`

**Functions:**
- `snake_case` for all function names: `compute_gae()`, `sample_trajectories_rnn()`, `evaluate_rnn()`, `update_actor_critic_rnn()`
- Helper functions use underscore prefix when internal: `_upload_to_gcs()`
- JAX scan/vmap callbacks use `snake_case`: `sample_step()`, `update_minibatch()`, `compute_gae_at_timestep()`

**Variables:**
- `snake_case` for variables and parameters: `num_envs`, `max_episode_length`, `train_state`, `init_hstate`
- Single letter names reserved for loop indices in JAX patterns: `x`, `y` for coordinates
- Prefix convention for state tracking: `next_obs`, `last_obs`, `init_obs`, `init_hstate`, `init_env_state`

**Types:**
- PascalCase for dataclass decorators: `@struct.dataclass`, `@nn.Module`
- Type annotations used throughout: `chex.Array`, `chex.PRNGKey`, `Tuple[...]`, `Optional[...]`
- Custom enums use PascalCase and IntEnum: `class UpdateState(IntEnum)`, `class Actions(IntEnum)`

**Constants:**
- UPPERCASE for module-level constants: `OBJECT_TO_INDEX`, `COLOR_TO_INDEX`, `COLORS`, `DIR_TO_VEC`
- Dictionary constants for mappings: `OBJECT_TO_INDEX = {...}`, `COLOR_TO_INDEX = {...}`

## Code Style

**Formatting:**
- No explicit formatter configured (no .prettierrc or eslint config found)
- Follows PEP 8 conventions by convention
- Line length appears to follow standard ~100-120 character limit (based on samples)
- 4-space indentation (Python standard)

**Linting:**
- No active linter configured (flake8 commented out in tox.ini)
- Tox used for testing across Python versions 3.9-3.13

## Import Organization

**Order:**
1. Standard library imports: `os`, `json`, `time`, `subprocess`, `sys`, `tempfile`
2. Third-party scientific/ML imports: `numpy`, `jax`, `jax.numpy`, `flax`, `chex`, `distrax`
3. Framework-specific: `orbax.checkpoint`, `wandb`, `gymnax`
4. Project imports: `from jaxued.environments...`, `from jaxued.linen...`, `from jaxued.utils...`
5. Conditional/dynamic imports placed after main setup: `sys.path.insert(0, ...)` followed by conditional `from vae_model import ...`

**Path Aliases:**
- Direct relative imports: `from jaxued.environments import Maze, MazeRenderer`
- Qualified imports for JAX operations: `jax.numpy as jnp`, `jax.lax`, `jax.random`, `jax.tree_util`
- No explicit path aliases detected; project uses `src/jaxued` structure

## Error Handling

**Patterns:**
- Assertions for invariant validation: `assert os.path.exists(script_path)`, `assert all(len(row) == len(rows[0]))`
- Exception raising for parsing errors: `raise Exception("Unexpected character.")`, `raise Exception(f'"{prioritization}" not a valid prioritization.')`
- Try-except for optional dependencies:
  ```python
  try:
      from google.cloud import storage
      # ...
  except (ImportError, Exception) as e:
      print(f"[GCS] Python client failed ({e}), falling back to gcloud CLI")
      # fallback implementation
  ```
- Subprocess error handling with `check=True`: `subprocess.run([...], check=True)`
- Silent failure suppression in tests: `except subprocess.TimeoutExpired: pass` for timeout scenarios

**JAX-specific error handling:**
- Uses JAX operations (`jax.lax.select`, conditionals) instead of Python control flow for array operations
- No try-except within JAX-traced functions; computations are assumed valid
- Validation occurs at initialization time, not during traced execution

## Logging

**Framework:** `print()` for CLI output, `wandb` for metrics/experiment tracking

**Patterns:**
- Status messages use descriptive prefixes: `[GCS]` for cloud operations, `[PyTorch]` style markers
- WandB logging via dictionaries: `{"log": {...}, "info": {...}}`
- No dedicated logging module (no `logging` imports found in main code)
- Console output for debugging: `print(f"[message] status")` style

## Comments

**When to Comment:**
- Inline comments for non-obvious JAX operations: `# NOTE: mark agent position as valid only if there is not a wall there AND agent has been displaced`
- Section markers using `# region` and `# endregion`: `# region PPO helper functions`, `# region checkpointing`
- Algorithm notes and references: Credit to source repositories in file headers

**JSDoc/TSDoc:**
- Python-style docstrings used throughout
- Triple-quoted docstrings for functions and classes
- Docstring format includes:
  ```python
  """Short description.

  Args:
      param_name (type): Description

  Returns:
      type: Description
  """
  ```

**Example from codebase:**
```python
def compute_gae(
    gamma: float,
    lambd: float,
    last_value: chex.Array,
    values: chex.Array,
    rewards: chex.Array,
    dones: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """This takes in arrays of shape (NUM_STEPS, NUM_ENVS) and returns the advantages and targets.

    Args:
        gamma (float):
        lambd (float):
        last_value (chex.Array):  Shape (NUM_ENVS)
        values (chex.Array): Shape (NUM_STEPS, NUM_ENVS)
        rewards (chex.Array): Shape (NUM_STEPS, NUM_ENVS)
        dones (chex.Array): Shape (NUM_STEPS, NUM_ENVS)

    Returns:
        Tuple[chex.Array, chex.Array]: advantages, targets; each of shape (NUM_STEPS, NUM_ENVS)
    """
```

## Function Design

**Size:**
- Helper functions range 5-30 lines (small, focused)
- Complex functions like `update_actor_critic_rnn()` 40-80 lines with nested inner functions
- Inner functions defined within outer functions for JAX `scan`/`vmap` operations

**Parameters:**
- Explicit typing for all parameters: `rng: chex.PRNGKey`, `env: UnderspecifiedEnv`, `train_state: TrainState`
- Long parameter lists acceptable when logically grouped: `(rng, env, env_params, train_state, init_hstate, init_obs, init_env_state, num_envs, max_episode_length)`
- Default parameters used sparingly: mostly in class constructors and dataclass fields

**Return Values:**
- Tuple unpacking for multiple returns: `(rng, train_state, hstate, last_obs, last_env_state, last_value), traj = ...`
- Structured returns using custom types: `TrainState`, dataclasses with `@struct.dataclass`
- Dictionary returns for logging: `{"log": {...}, "info": {...}}`

## Module Design

**Exports:**
- Dataclass definitions exported at module level: `@struct.dataclass class Level`, `@struct.dataclass class Observation`
- Functions defined at module level (no wrapper classes for free functions)
- JAX environment implementations as classes inheriting from `UnderspecifiedEnv`

**Barrel Files:**
- `__init__.py` files use explicit imports: `from .autoreplay import AutoReplayWrapper`, `from .autoreset import AutoResetWrapper`
- Not all exports re-exported; selective public API
- Example from `src/jaxued/wrappers/__init__.py`: explicit imports of wrapper classes

**Class Patterns:**
- JAX-compatible dataclasses use `@struct.dataclass` from Flax
- Environment classes inherit from `UnderspecifiedEnv`
- RNN-based agents use Flax `nn.Module` with `@nn.compact` decorator
- State management via frozen dataclasses: `@struct.dataclass class TrainState(BaseTrainState)`

---

*Convention analysis: 2026-03-11*
