# Architecture

**Analysis Date:** 2026-03-11

## Pattern Overview

**Overall:** Modular JAX-based Unsupervised Environment Design (UED) framework with pluggable training algorithms and environment implementations.

**Key Characteristics:**
- Functional programming paradigm using JAX primitives (jit, vmap, scan)
- Stateless level sampler using dictionaries (FrozenDict) instead of class-based state
- Environment abstraction (`UnderspecifiedEnv`) enabling plug-and-play training
- Single-file reference implementations of algorithms (DR, PLR, ACCEL, PAIRED)
- Centralized training loop with modular components for level generation, sampling, and agent updates
- Integration with Flax (neural networks), Optax (optimization), and Distrax (distributions)

## Layers

**Environment Layer:**
- Purpose: Define UPOMDP environment interface and specific implementations
- Location: `src/jaxued/environments/`
- Contains: Base class `UnderspecifiedEnv` (underspecified_env.py), Maze implementation (maze/), and environment wrappers
- Depends on: JAX, Flax, Chex
- Used by: Training layer for rollouts and level resets

**Level Management Layer:**
- Purpose: Buffer-based level sampling with prioritization and staleness weighting
- Location: `src/jaxued/level_sampler.py`
- Contains: `LevelSampler` class managing level storage, insertion, and prioritized replay
- Depends on: JAX, Chex types
- Used by: Training loop for replay decisions and level selection

**Training Layer:**
- Purpose: Orchestrate PPO agent learning with environment interaction
- Location: `examples/maze_plr.py` (PLR/ACCEL), `examples/maze_dr.py` (DR), `examples/maze_paired.py` (PAIRED)
- Contains: Main training loop, `TrainState`, PPO loss computation, trajectory sampling, GAE calculation
- Depends on: Environment layer, Level sampler, Optax, Distrax, Flax
- Used by: Top-level main() function

**Level Generation Layer:**
- Purpose: Generate new procedural levels through mutation or learned VAE decoding
- Location: `src/jaxued/environments/maze/level.py` (procedural), `vae/` (CNN-VAE, CluttrVAE)
- Contains: Level representation, mutation operators, prefab library, VAE models for decoding latents
- Depends on: JAX, Flax (for VAE)
- Used by: Training loop for new level generation

**Scoring/Novelty Layer:**
- Purpose: Compute curriculum scores (MaxMC, PVL, MNA, CENIE, SFL)
- Location: Main `maze_plr.py` (score functions), `vae/cenie_scorer.py` (CENIE GMM)
- Contains: Score computation functions, CENIE GMM manager, regret-based scoring
- Depends on: Environment layer outputs, custom loss functions
- Used by: Training loop for level prioritization

**VAE & CMA-ES Integration Layer:**
- Purpose: Optional learned level generation using VAE latents optimized by CMA-ES
- Location: `vae/` directory (vae_model.py, cnn_vae_model.py, cmaes_manager.py)
- Contains: VAE architectures (CluttrVAE, CNN-LSTM VAE), latent optimization, CMA-ES wrapper
- Depends on: Flax, JAX, CMA library
- Used by: Training loop when `use_cmaes=True`

**Checkpoint/Logging Layer:**
- Purpose: Save/restore agent and level buffer state, log metrics to WandB
- Location: Main `maze_plr.py` (checkpoint manager), WandB integration throughout
- Contains: Orbax checkpoint management, WandB run initialization and metric logging
- Depends on: Orbax, WandB
- Used by: Training loop for periodic saves and metric tracking

## Data Flow

**Standard Training Loop (PLR/ACCEL/DR):**

1. **Initialization**
   - Load config (yaml), initialize RNG, WandB run
   - Create environment (`Maze`), level sampler (`LevelSampler`), network params
   - Initialize `TrainState` with params, optimizer state, sampler dict, counters

2. **Per-Update Cycle**
   - **Level Sampling Decision**: Sample from `sampler` dict; replay_prob determines new vs replay
   - **Level Generation**: If new level, mutate using `make_level_mutator_minimax()` or sample VAE latent
   - **Environment Reset**: Reset environment to sampled level via `env.reset_to_level()`
   - **Trajectory Rollout**: `sample_trajectories_rnn()` collects experience: obs, actions, rewards, dones, log_probs, values
   - **Score Computation**: Compute regret-based score (MaxMC, PVL, MNA, CENIE) from trajectory
   - **Level Insertion**: Insert scored level into sampler buffer via `level_sampler.insert_batch()`
   - **Agent Update**: PPO update on collected trajectories; compute GAE, clip objectives, apply optimizer
   - **Logging**: Log metrics to WandB, update level sampler stats

3. **Checkpoint & Eval** (periodic)
   - Save `TrainState` and sampler dict via Orbax
   - Optional: Evaluate on prefab/VAE mazes, log solve rates

**CMA-ES Integration Flow** (when `use_cmaes=True`):

1. Latent optimization loop (CMA-ES) runs in parallel or interleaved
2. VAE decoder decodes sampled/optimized latents → level tokens
3. Tokens → `Level` structure via `vae_level_utils.decode_latent_to_levels()`
4. Generated levels fed into training loop same as mutated levels
5. CENIE scorer optionally uses multi-rollout evaluation with GMM tracking

**State Management:**

- `TrainState` (Flax BaseTrainState subclass) holds:
  - `params`: Network parameters (JAX pytree)
  - `opt_state`: Optimizer state from Optax
  - `sampler`: FrozenDict containing levels, scores, timestamps, episode counts
  - `update_state`: Tracks DR vs REPLAY update type (IntEnum)
  - `es_state`: Optional CMA-ES internal state (if using CMA-ES)
  - Counters: `num_dr_updates`, `num_replay_updates`, `num_mutation_updates`

- All state is pure JAX arrays (pytrees); training functions are fully functional with jit/vmap applied

## Key Abstractions

**UnderspecifiedEnv Interface:**
- Purpose: UPOMDP abstraction for any procedurally-parameterized environment
- Examples: `Maze` (src/jaxued/environments/maze/env.py), Craftax, Gymnax environments
- Pattern: Subclass and implement `step_env()`, `reset_env_to_level()`, `action_space()`
- Used for: Pluggable environment support across all training methods

**LevelSampler:**
- Purpose: Functional level buffer with prioritization (rank-based, top-k) and staleness
- Examples: `src/jaxued/level_sampler.py`
- Pattern: Stateless class; methods take/return `Sampler` dict (pytree of arrays)
- Methods: `initialize()`, `sample_replay_decision()`, `insert_batch()`, `level_weights()`, `staleness_weights()`
- Used for: PLR, ACCEL, and any algorithm needing level prioritization

**TrainState Subclassing:**
- Purpose: Extend Flax BaseTrainState with custom fields for UED-specific state
- Examples: `src/jaxued/level_sampler.py` (sampler dict), `examples/maze_plr.py` (es_state, cenie_gmm_params)
- Pattern: Pytree registration via `@struct` decorator; all fields JAX arrays or nested dicts
- Used for: Checkpointing and training loop bookkeeping

**Score Functions:**
- Purpose: Compute level difficulty/novelty from agent trajectories
- Examples: `max_mc()` (max return), `positive_value_loss()` (PVL), `mna_score()` (neg advantages), `cenie_scorer.py` (GMM-based)
- Pattern: Pure functions taking trajectory arrays (dones, values, rewards, advantages) → scalar scores
- Used for: Prioritizing levels in sampler buffer

**VAE Decoder:**
- Purpose: Convert latent vector to level representation
- Examples: `vae/vae_model.py` (CluttrVAE token-based), `vae/cnn_vae_model.py` (CNN-LSTM for maze grids)
- Pattern: Flax nn.Module; `decode()` method takes latent → logits → sampled tokens/images
- Used for: VAE-based level generation when CMA-ES optimizes latent space

## Entry Points

**maze_plr.py (PLR/ACCEL/RPLR):**
- Location: `examples/maze_plr.py`
- Triggers: `python examples/maze_plr.py [--use_accel] [--use_cmaes] [--use_dred] ...`
- Responsibilities:
  - Parse config via argparse and yaml
  - Initialize environment, sampler, networks, optimizer
  - Main training loop (updates, checkpoint, evaluation)
  - VAE/CMA-ES integration if enabled

**maze_dr.py (Domain Randomization):**
- Location: `examples/maze_dr.py`
- Triggers: `python examples/maze_dr.py [args]`
- Responsibilities: Simpler variant; always samples new random levels, no replay buffer

**maze_paired.py (PAIRED):**
- Location: `examples/maze_paired.py`
- Triggers: `python examples/maze_paired.py [args]`
- Responsibilities: Two-agent co-evolution (protagonist + antagonist level designer)

**Evaluation Scripts:**
- `examples/cross_evaluate.py`: Cross-evaluation across trained agents
- `examples/evaluate_buffer.py`: Analyze trained level buffers
- `scripts/evaluate_checkpoints.py`: Eval checkpoints on prefab/VAE mazes

## Error Handling

**Strategy:** Exception-based with early validation; minimal try-catch in hot loops (for JAX jit efficiency)

**Patterns:**
- Config validation at startup (assertions on required keys)
- JAX shape asserts via `chex.assert_shape()` in development; removed via jit in production
- WandB init wrapped to gracefully degrade if offline
- Orbax checkpoint manager handles missing directories
- Level generation safeguards (e.g., retry mutator if invalid maze)

## Cross-Cutting Concerns

**Logging:**
- WandB integration: `wandb.log()` for per-update metrics (agent loss, level sampler stats, solve rates)
- Custom metrics defined via `wandb.define_metric()`
- Training state logged via `train_state_to_log_dict()` helper to avoid copying large pytrees to CPU
- Generated levels and maze images logged periodically for visualization

**Validation:**
- Level structure verified at generation time (maze must have start, goal, valid walls)
- Agent trajectory validation: dones and values shapes match, returns finite
- Score computation guarded against NaN (e.g., regret scores clipped to bounds)

**Authentication/Secrets:**
- WandB API key via environment variable (standard practice)
- No hardcoded credentials in codebase

---

*Architecture analysis: 2026-03-11*
