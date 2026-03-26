# Architecture

**Analysis Date:** 2025-03-23

## Pattern Overview

**Overall:** Modular UED (Unsupervised Environment Design) framework with pluggable algorithm components and optional LLM-based level generation.

**Key Characteristics:**
- Single-file algorithm implementations (minimal abstraction) for interpretability and rapid prototyping
- JAX-based vectorized training with vmap/lax.scan for parallelism
- Pluggable level sampling, generation, and mutation strategies
- Optional integration of CMA-ES for search in VAE latent space
- Integrated metric computation for diversity analysis and LLM prompt injection

## Layers

**Algorithm Layer (Trainers):**
- Purpose: Implements UED methods (DR, PLR, ACCEL, PAIRED) with PPO policy optimization
- Location: `examples/maze_plr.py`, `examples/maze_dr.py`, `examples/maze_paired.py`
- Contains: Complete training loops, state management, update cycle orchestration
- Depends on: Environment interface, level sampler, policy networks, VAE (optional), CMA-ES (optional)
- Used by: Experiment scripts and launch shell scripts

**Environment Layer:**
- Purpose: Defines the UPOMDP interface and concrete environment implementations
- Location: `src/jaxued/environments/`
  - `underspecified_env.py`: Base interface (EnvState, Observation, Level, EnvParams)
  - `maze/env.py`: Minigrid-style 2D maze environment
  - `gymnax/`: Classic control environments (Acrobot, Cartpole, Pendulum)
- Contains: JAX-jitted step/reset, environment state, action spaces
- Depends on: JAX, Flax
- Used by: Trainers via vmap/scan

**Wrappers Layer:**
- Purpose: Augment environments with cross-cutting concerns
- Location: `src/jaxued/wrappers/`
  - `autoreplay.py`: Replay same level repeatedly (AutoReplayWrapper)
  - `autoreset.py`: Auto-reset after done (AutoResetWrapper)
- Contains: Functional wrappers that preserve JAX composability
- Depends on: Environment interface
- Used by: Training loop to manage episode resets

**Level Management Layer:**
- Purpose: Manages the level buffer with prioritized replay and scoring
- Location: `src/jaxued/level_sampler.py`
- Contains: LevelSampler class with replay decisions, buffer operations
- Depends on: JAX, Flax struct
- Used by: Trainers to decide between new/replay/mutation branches

**Policy Network Layer:**
- Purpose: Actor-Critic network for RL agent
- Location: `examples/maze_plr.py` (ActorCritic class, lines ~308-341)
- Contains: CNN image embedding, LSTM hidden state, policy/value heads
- Depends on: Flax Linen, Distrax for distributions
- Used by: Training loop for rollouts and optimization

**Latent Generation Layer (optional CMA-ES):**
- Purpose: Search VAE latent space for interesting maze level seeds
- Location: `vae/cmaes_manager.py`, `vae/vae_model.py`
- Contains: CMA-ES state management, VAE decoder, token-to-level conversion
- Depends on: evosax, Flax, JAX
- Used by: Training loop when `--use_cmaes` flag is set
- Flow: CMA-ES ask → decode VAE logits → tokens_to_level → Level

**Metrics & Analysis Layer:**
- Purpose: Compute diversity, difficulty, and novelty metrics for analysis
- Location: `metrics/` directory
  - `base.py`: DiversityAnalyzer abstract interface
  - `standalone/`: Single-level metrics (regret, learnability, CENIE)
  - `pairwise/`: Level-pair metrics (DTW, mode transition, TD-error)
  - `utils.py`: Shared trajectory analysis functions
- Contains: Metric computation functions and Analyzer classes for LLM prompt formatting
- Depends on: JAX, scipy (DTW), numpy
- Used by: Post-training analysis, optionally during training for gates

**LLM Integration Layer (optional):**
- Purpose: Generate levels via LLM prompts with metric-informed context
- Location: `llm/` directory
  - `prompt_builder.py`: Compose system/reference/instruction prompts
  - `maze_generator.py`: Call Claude to generate maze tokens
  - `decision_gate.py`: Gate functions for difficulty/diversity decisions
  - `agent_evaluator.py`: Evaluate agent on LLM-generated levels
- Contains: Prompt formatting, LLM calls, level validation
- Depends on: Anthropic API, metrics system
- Used by: Optional post-training pipeline (not integrated into main training loop)

## Data Flow

**Training Cycle (Main Loop):**

```
1. Initialization (main):
   - Create environment + AutoReplayWrapper
   - Initialize LevelSampler (empty buffer)
   - Initialize policy + optimizer (TrainState)
   - [Optional] Load VAE, initialize CMA-ES manager

2. train_and_eval_step (every eval_freq steps):
   a) Collect eval_freq train_steps:
      - Compute replay decision → select branch
      - Branch 0 (on_new_levels):
        * Generate new levels (random or CMA-ES)
        * Rollout trajectories on new levels
        * Score levels (MaxMC or PVL)
        * Insert into buffer + extract new_levels
        * [Optional] Run CMA-ES ask → tell
      - Branch 1 (on_replay_levels):
        * Sample from buffer (prioritized by score + staleness)
        * Rollout trajectories on replayed levels
        * Update scores in buffer
      - Branch 2 (on_mutate_levels):
        * Mutate last replay batch (edit 1-3 walls)
        * Rollout on mutated levels
        * Insert into buffer
      - All branches: PPO update (if exploratory_grad_updates)

   b) Evaluation:
      - Eval on fixed benchmark levels
      - Compute solve rates, returns, episode lengths

3. Logging:
   - WandB metrics aggregation
   - [Optional] Buffer PCA/diversity analysis
   - [Optional] Level visualization + GCS upload
```

**State Management:**

```
TrainState (main container):
  ├── params: Policy network weights
  ├── opt_state: Optimizer state
  ├── sampler: LevelSampler state (levels, scores, timestamps, size)
  ├── update_state: UpdateState enum (DR=0, REPLAY=1)
  ├── es_state: CMA-ES state (mean, std, covariance) — optional
  └── Logging:
      ├── num_dr_updates
      ├── num_replay_updates
      ├── num_mutation_updates
      └── *_last_level_batch (for visualization)
```

**Score Computation:**

```
Raw Trajectory Data (per env):
  observations, actions, rewards, dones, log_probs, values
         ↓
  compute_gae() → advantages + targets
         ↓
  compute_score():
    - MaxMC: max_return over episodes
    - PVL: positive value loss (higher on harder levels)
         ↓
  Scores (shape: num_envs) → buffer insertion
```

## Key Abstractions

**UnderspecifiedEnv (UPOMDP interface):**
- Purpose: Abstract environment with configurable level space
- Examples: `Maze`, `Acrobot`, `Cartpole`, `Craftax`
- Pattern: Subclass defines `step_env()`, `reset_env_to_level()`, `action_space()`
- Core methods: `step(rng, state, action, params)`, `reset_to_level(rng, level, params)`
- Key design: No implicit reset to new level; wrappers handle that

**Level (dataclass):**
- Purpose: Represents a single environment configuration
- Examples: `Maze.Level`, `Acrobot.Level`
- For Maze: `wall_map` (13×13 bool), `goal_pos`, `agent_pos`, `agent_dir`
- Method: `is_well_formatted()` validates structure
- Serializable: loads from prefabs (e.g., `Level.load_prefabs(['StandardMaze'])`)

**LevelSampler:**
- Purpose: Prioritized level buffer management
- Key methods:
  - `initialize()`: Create empty sampler dict
  - `sample_replay_decision()`: Bernoulli based on fill ratio + replay_prob
  - `sample_replay_levels()`: Rank-based or top-k sampling
  - `insert_batch()`: Add generated levels with scores
  - `update_batch()`: Update existing level scores
  - `level_weights()`: Compute priority weights (score + staleness)
- Design: Functional (returns new sampler dict, no mutation)

**VAE-CMA-ES Pipeline (optional):**
- Purpose: Seed level generation via learned latent space
- Flow: `z ~ CMA-ES` → `VAE.decode(z)` → `tokens_to_level()` → `Level`
- Key components:
  - `CluttrVAE`: Flax module, encodes tokens → latent, decodes latent → logits
  - `CMAESManager`: Wraps evosax CMA-ES with ask/tell interface
  - `vae_level_utils.py`: JAX-jittable token↔level conversion
- Design: CMA-ES minimize negative score; caller negates for maximization

**Metrics Analyzers (optional):**
- Purpose: Compute and format metric results for LLM prompts
- Base class: `DiversityAnalyzer` with `analyze()` → `AnalysisSection`
- Examples: `LearnabilityAnalyzer`, `CEINEAnalyzer`, `PosDTWAnalyzer`
- Used by: `prompt_builder.py` to inject metric context into generation prompts

## Entry Points

**Training Entry (main function):**
- Location: `examples/maze_plr.py` lines ~450+
- Triggers: `python examples/maze_plr.py [args]`
- Responsibilities:
  - Parse config + command-line args
  - Initialize environment, networks, sampler, CMA-ES (optional)
  - Create TrainState and checkpoint manager
  - Run main training loop with jit-compiled steps
  - Log to WandB
  - Save checkpoints every N steps

**Evaluation Entry (eval mode):**
- Location: `examples/maze_plr.py` (lines ~980+ eval_checkpoint function)
- Triggers: `python examples/maze_plr.py --mode eval --checkpoint_directory=... --checkpoint_to_eval=...`
- Responsibilities:
  - Load saved checkpoint from orbax
  - Evaluate policy on fixed benchmark levels
  - Save results to npz (states, cum_rewards, episode_lengths, levels)

**Post-Training Analysis:**
- Location: `examples/evaluate_buffer.py` (example workflow)
- Purpose: Analyze trained level buffer in VAE latent space
- Responsibilities: Buffer PCA, diversity metrics, visualization

## Error Handling

**Strategy:** Fail-fast with descriptive messages; allow optional components to skip gracefully.

**Patterns:**
- CMA-ES loading (lines ~478-510): Assert VAE paths provided if `--use_cmaes`
- Level validity: `is_well_formatted()` check; log invalid % at each step
- Configuration: argparse with type checking; yaml validation for VAE config
- Checkpoint recovery: orbax handles version mismatch; fallback to latest_step()
- GCS upload (optional): Try Google Cloud API, fall back to gcloud CLI

## Cross-Cutting Concerns

**Logging:** WandB integration for metrics, images, videos, tables
- Metrics defined at init (lines ~463-473)
- Per-step metrics accumulated in dict
- Logging via `wandb.log()` every eval_freq steps

**Validation:** Level structure checked with `is_well_formatted()` (Maze.Level method)
- Ensures no overlapping agent/goal/walls
- Validates dimensions (13×13 for maze)

**Random State:** JAX PRNG key management throughout
- Split at each step for deterministic reproducibility
- vmap over split keys for parallelism

**Checkpointing:** Orbax-based save/restore
- Policy params saved to models/[step]/ directory
- Config serialized as JSON alongside
- Eval mode can restore and continue

---

*Architecture analysis: 2025-03-23*
