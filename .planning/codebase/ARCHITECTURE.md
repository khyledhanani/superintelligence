# Architecture

**Analysis Date:** 2026-02-26

## Pattern Overview

**Overall:** Modular Unsupervised Environment Design (UED) library using JAX, implementing reinforcement learning algorithms with environment generation and curriculum learning.

**Key Characteristics:**
- JAX-based functional/immutable design with pytree structure for state management
- Clear separation between core library (`src/jaxued/`) and example implementations (`examples/`)
- Environment-as-interface pattern using abstract `UnderspecifiedEnv` class
- Pluggable level sampling and curriculum strategies (PLR, ACCEL, PAIRED, DR)
- Evolution strategy integration for environment search in latent spaces (VAE/Maze-AE)

## Layers

**Core Library (`src/jaxued/`):**
- Purpose: Reference UED implementations and environment abstractions
- Location: `src/jaxued/`
- Contains: Abstract environment interfaces, wrappers, utilities, neural network modules
- Depends on: JAX, Flax, Gymnax, Chex
- Used by: Examples, ES integration, custom implementations

**Environments (`src/jaxued/environments/`):**
- Purpose: Concrete environment implementations (Maze, Gymnax variants)
- Location: `src/jaxued/environments/`
- Contains: Maze environment (minigrid-style), Gymnax wrappers (Acrobot, Cartpole, Pendulum)
- Depends on: UnderspecifiedEnv interface, JAX
- Used by: Training examples, wrapper modules

**Wrappers (`src/jaxued/wrappers/`):**
- Purpose: Decorate environments with automatic reset/replay functionality
- Location: `src/jaxued/wrappers/`
- Contains: `AutoReplayWrapper`, `AutoResetWrapper`, `AutoResetFiniteWrapper`
- Depends on: UnderspecifiedEnv interface
- Used by: Training loops, environment initialization

**Training Examples (`examples/`):**
- Purpose: Single-file reference implementations of UED algorithms
- Location: `examples/`
- Contains: DR, PLR, PAIRED, ACCEL implementations for Maze and Craftax
- Depends on: jaxued core, training utilities, logging (Weights & Biases)
- Used by: Researchers for algorithm prototyping and modification

**Evolution Strategy Integration (`es/`):**
- Purpose: Environment evolution using VAE/Maze-AE in latent space with MAP-Elites
- Location: `es/`
- Contains: VAE/Maze-AE loaders, CMA-ES evolution, MAP-Elites archive, agent evaluation
- Depends on: jaxued core, regret fitness computation, VAE/AE models
- Used by: ACCEL mutation integration, environment search

## Data Flow

**Training Loop (PLR/ACCEL Example):**

1. Initialization phase:
   - Load agent checkpoint from `agent_folder/`
   - Initialize level sampler with capacity (4000 levels default)
   - Create Maze environment with parameters
   - Wrap environment with AutoReplayWrapper

2. Training loop per update:
   - Sample new or replay level from level sampler
   - Run environment interactions with batch of parallel episodes
   - Compute agent advantages (GAE) and policy/value gradients
   - Compute level scores (regret via MaxMC or positive advantage)
   - Insert/update scored levels in level buffer
   - Log metrics to Weights & Biases

3. State progression:
   - TrainState (agent network params, optimizer state, sampler dict)
   - Sampler dict (levels array, scores, timestamps, size, episode_count)
   - Synchronized checkpoint saving for agent and level replay data

**Evolution Strategy Flow (es/evolve_envs.py):**

1. Load VAE decoder parameters and optional agent
2. Initialize CMA-ES in latent space (mean, covariance)
3. Per generation:
   - Sample latents from CMA-ES distribution
   - Decode latents to environment sequences via VAE
   - Repair invalid CLUTTR sequences
   - Compute fitness (placeholder complexity or regret if agent available)
   - Insert best solutions into MAP-Elites archive (if enabled)
   - Update CMA-ES with fitness ranks

4. MAP-Elites archive:
   - Fixed-size grid of behavior cells (8×6 for obstacle×distance)
   - Each cell tracks latent, sequence, fitness, occupancy, last update
   - Elitist replacement: only update if new fitness > current

**State Management:**

- All state immutable using Flax `@struct.dataclass` decorators
- JAX tree utilities for recursive operations on complex nested state
- Pytrees enable efficient vmap/pmap operations across batch dimensions
- Level sampler uses PyTree-compatible array storage for batch operations

## Key Abstractions

**UnderspecifiedEnv (`src/jaxued/environments/underspecified_env.py`):**
- Purpose: Base interface for UED environments with level concept
- Examples: `Maze`, Gymnax wrappers
- Pattern: Abstract class with concrete implementations of `step()` and `reset_to_level()` that call abstract `step_env()`, `reset_env_to_level()`, `action_space()`
- Enables environment-agnostic algorithm implementations

**Level and EnvState:**
- Purpose: Represent environment configuration and execution state
- `Level`: Contains wall_map, goal_pos, agent_pos, agent_dir, width, height
- `EnvState`: Contains agent_pos, agent_dir, goal_pos, wall_map, maze_map, time, terminal
- Pattern: Immutable Flax dataclasses enabling JAX compilation and vmap batching

**LevelSampler (`src/jaxued/level_sampler.py`):**
- Purpose: Manage PLR/ACCEL level buffer with prioritization and staleness
- Key methods: `sample_replay_decision()`, `sample_replay_level()`, `insert()`, `level_weights()`
- Pattern: Stateless class operating on immutable sampler dict with rank or topk prioritization
- Supports staleness weighting and capacity-managed insertion

**TrainState (examples):**
- Purpose: Encapsulate all training loop state for checkpointing
- Pattern: Extends Flax BaseTrainState with agent network params, optimizer state, sampler, metadata
- Enables orbax checkpointing with selective save/restore

**MapElitesArchive (`es/map_elites_mutation_service.py`):**
- Purpose: Quality-diversity grid for environment evolution with behavior descriptors
- Pattern: Flax dataclass tracking latents, sequences, fitness, occupancy, last_update per cell
- Descriptors: Obstacle count × Manhattan distance (primary), BFS path × obstacles (recommended)

## Entry Points

**maze_plr.py (`examples/maze_plr.py`):**
- Location: `examples/maze_plr.py`
- Triggers: Direct execution with argparse configuration
- Responsibilities:
  - Initialize Maze environment and agent network
  - Run PPO+PLR/ACCEL training loop
  - Sample levels, collect trajectories, compute regret scores
  - Checkpoint models and track metrics to Weights & Biases

**maze_dr.py (`examples/maze_dr.py`):**
- Location: `examples/maze_dr.py`
- Triggers: Direct execution with argparse configuration
- Responsibilities:
  - Run PPO with Domain Randomization
  - Sample random levels per episode
  - Simpler baseline without level buffer

**evolve_envs.py (`es/evolve_envs.py`):**
- Location: `es/evolve_envs.py`
- Triggers: Direct execution with fitness mode (placeholder or regret)
- Responsibilities:
  - Optimize environments in VAE latent space using CMA-ES
  - Evaluate against agent via regret fitness
  - Output evolved environment latents and decoded sequences

**map_elites.py (`es/map_elites.py`):**
- Location: `es/map_elites.py`
- Triggers: Direct execution with behavior descriptor mode
- Responsibilities:
  - Run MAP-Elites archive construction in VAE latent space
  - Maintain quality-diversity grid binned by behavioral properties
  - Checkpoint and visualize archive contents

## Error Handling

**Strategy:** JAX lax conditionals for functional control flow; try-catch for I/O operations.

**Patterns:**
- Level validity checks: `is_well_formatted()` returns boolean for structural validation
- Configuration validation: Argparse with type checking; YAML schema validation for VAE configs
- Trajectory truncation: `done` flags control episode boundaries and loss masking
- Archive insertion guards: Occupancy checks and fitness comparisons for elitist replacement

## Cross-Cutting Concerns

**Logging:**
- Weights & Biases integration for training metrics (rewards, losses, level stats)
- Orbax checkpointing for agent models and level replay data
- Optional visualization with PIL/imageio for maze renderings

**Validation:**
- Level well-formedness checks (bounded positions, non-wall placement, distinct agent/goal)
- CLUTTR sequence repair after VAE decode (obstacle count clamping, position validation)
- Agent action space verification and masking

**Authentication:**
- No explicit auth; relies on environment variable flags (WANDB_MODE for offline testing)
- Checkpoint paths as configuration for agent/VAE loading

---

*Architecture analysis: 2026-02-26*
