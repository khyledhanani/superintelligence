# Codebase Structure

**Analysis Date:** 2026-03-11

## Directory Layout

```
superintelligence/
├── src/
│   └── jaxued/                          # Core UED library (pip install)
│       ├── environments/                # Environment implementations
│       │   ├── maze/                    # Maze environment
│       │   ├── gymnax/                  # Gymnax environment wrappers
│       │   ├── underspecified_env.py   # Base UPOMDP interface
│       │   └── __init__.py
│       ├── wrappers/                    # Environment wrappers
│       │   ├── autoreplay.py
│       │   ├── autoreset.py
│       │   └── __init__.py
│       ├── level_sampler.py            # PLR/ACCEL level buffer
│       ├── utils.py                    # Score functions (max_mc, pvl)
│       ├── linen.py                    # ResetRNN helper
│       └── __init__.py
├── examples/                            # Reference implementations (single-file)
│       ├── maze_plr.py                 # PLR/ACCEL/RPLR main training
│       ├── maze_dr.py                  # Domain Randomization baseline
│       ├── maze_paired.py              # PAIRED co-evolution
│       ├── cross_evaluate.py           # Cross-eval agent performance
│       ├── evaluate_buffer.py          # Level buffer analysis
│       ├── craftax/                    # Craftax environment implementations
│       │   ├── craftax_plr.py
│       │   ├── craftax_wrappers.py
│       │   └── mutators.py
│       └── gymnax/
│           └── gymnax_plr.py
├── vae/                                # CNN-VAE & CMA-ES integration
│       ├── vae_model.py               # CluttrVAE (token-based)
│       ├── cnn_vae_model.py           # CNN-LSTM VAE (image-based)
│       ├── cnn_vae_losses.py          # VAE loss functions
│       ├── cnn_vae_data.py            # VAE dataset handling
│       ├── vae_level_utils.py         # Latent → level decoding
│       ├── cmaes_manager.py           # CMA-ES optimizer wrapper
│       ├── cenie_scorer.py            # CENIE novelty scorer with GMM
│       ├── buffer_latent_analysis.py  # Analyze VAE latent space
│       ├── latent_perturbation_diagnostic.py
│       └── compare_accel_vs_cmaes.py
├── accel_training/                    # Phase 5 ES experiments (currently empty)
│       └── es_components/
├── scripts/                            # Utility and batch scripts
│       ├── run_phase5_sequential.sh   # Phase 5 sequential training
│       ├── smoke_test_fixes.sh        # Quick test of fixes
│       ├── launch_sfl_cenie.sh        # SFL/CENIE launcher
│       └── evaluate_checkpoints.py    # Eval script
├── tests/
│       └── test_examples_kinda.py     # Basic smoke tests
├── checkpoints/                        # Saved agent & level buffer states
│       ├── phase5-accel/              # Phase 5 ACCEL results
│       └── smoke_test_accel_maxmc/    # Test checkpoints
├── logs/                               # Training logs
│       └── phase5.log
├── docs/                               # Documentation
├── figures/                            # Visualization outputs
├── wandb/                              # WandB offline/online runs
├── es_legacy/                          # Legacy ES code (archived)
└── .planning/
    └── codebase/                       # GSD planning docs (generated)
```

## Directory Purposes

**src/jaxued/:**
- Purpose: Reusable core library for UED research and applications
- Contains: Environment abstractions, level sampling, utility functions
- Key files: `level_sampler.py`, `underspecified_env.py`
- Status: Pip-installable package; defines interfaces

**src/jaxued/environments/:**
- Purpose: Environment implementations conforming to UnderspecifiedEnv interface
- Contains: Maze (13×13 grid world), Gymnax wrappers (CartPole, Acrobot, Pendulum)
- Key files: `maze/env.py` (main Maze class), `underspecified_env.py` (base class)

**src/jaxued/environments/maze/:**
- Purpose: Complete Maze environment implementation
- Contains:
  - `env.py`: Main Maze class with step/reset
  - `level.py`: Level representation (wall_map, agent_pos, goal_pos) + mutation operators
  - `renderer.py`: Visualization (image rendering)
  - `util.py`: Maze utilities
  - `env_editor.py`: Interactive level editing
  - `env_solved.py`: Solve state tracking

**examples/:**
- Purpose: Single-file reference implementations of UED algorithms
- Contains: DR, PLR, ACCEL, PAIRED, and environment-specific variants (Craftax, Gymnax)
- Key files: `maze_plr.py` (1701 lines, main reference), `maze_dr.py` (661 lines), `maze_paired.py` (749 lines)
- Design: Monolithic; intentionally not split into modules for understanding/modification

**vae/:**
- Purpose: Learned level generation via VAE + CMA-ES optimization
- Contains: VAE models (token-based CluttrVAE, CNN-LSTM for grids), CMA-ES wrapper, CENIE scorer
- Key files:
  - `vae_model.py`: CluttrVAE (Highways + BiLSTM)
  - `cnn_vae_model.py`: CNN-LSTM encoder/decoder
  - `cmaes_manager.py`: CMA-ES population optimizer
  - `cenie_scorer.py`: Novelty via Gaussian Mixture Model
  - `vae_level_utils.py`: Latent-to-level conversion

**accel_training/:**
- Purpose: Phase 5 ES strategy experiments (CMA-ES, NS-ES, SV-CMA-ES comparison)
- Contains: Currently empty; es_components/ reserved for future strategy modules
- Status: Active during phase5 runs; populated by `run_phase5_sequential.sh`

**scripts/:**
- Purpose: Training orchestration, batch processing, evaluation
- Key files:
  - `run_phase5_sequential.sh`: Sequential 4-strategy 20k-step runs
  - `smoke_test_fixes.sh`: 500-update quick health check
  - `evaluate_checkpoints.py`: Eval agents on prefab/VAE mazes
  - `launch_*.sh`: WandB launch configs for specific experiments

**checkpoints/:**
- Purpose: Orbax checkpoint storage (agent params, level sampler state)
- Structure: `{strategy_name}/{seed}/models/{update_step}/default/ocdbt.process_0/d/`
- Format: Jax pytree serialized via Orbax OCD database

**logs/:**
- Purpose: Text logs from training runs (stdout/stderr redirects)
- Key files: `phase5.log` (current phase 5 training)

**tests/:**
- Purpose: Basic smoke tests for examples
- Contains: `test_examples_kinda.py` (minimal functional checks)

## Key File Locations

**Entry Points:**
- `examples/maze_plr.py`: Main PLR/ACCEL/RPLR training (line 469: `def main()`)
- `examples/maze_dr.py`: DR baseline training
- `examples/maze_paired.py`: PAIRED co-evolution
- `scripts/run_phase5_sequential.sh`: Phase 5 orchestration script

**Configuration:**
- Config passed via command-line args in `maze_plr.py` (argparse)
- Converted to YAML for WandB logging
- Example: `python examples/maze_plr.py --use_accel --run_name accel_test --num_updates 20000`

**Core Logic - Environment:**
- `src/jaxued/environments/maze/env.py`: Maze stepping and reset logic
- `src/jaxued/environments/maze/level.py`: Level structure, mutation operators

**Core Logic - Training:**
- `examples/maze_plr.py` (lines 848-1240): `train_step()` function (PPO + level management)
- `examples/maze_plr.py` (lines 1241+): `train_and_eval_step()` wrapper

**Core Logic - Level Sampling:**
- `src/jaxued/level_sampler.py`: `LevelSampler` class with all buffer operations
- `examples/maze_plr.py` (lines 407-440): `train_state_to_log_dict()` for sampler stats

**Core Logic - Scoring:**
- `src/jaxued/utils.py`: `max_mc()`, `positive_value_loss()` score functions
- `examples/maze_plr.py` (lines 441-467): `compute_score()` dispatcher (MaxMC, PVL, MNA, CENIE, SFL)
- `vae/cenie_scorer.py`: CENIE GMM-based novelty

**Testing:**
- `tests/test_examples_kinda.py`: Smoke tests for maze_plr, maze_dr, etc.

## Naming Conventions

**Files:**
- Single-file examples use descriptive names: `maze_plr.py`, `maze_dr.py`
- Utility modules are action/noun based: `level_sampler.py`, `cmaes_manager.py`
- Score functions suffixed: `cenie_scorer.py`, `cnn_vae_losses.py`
- Internal modules lowercase with underscores: `vae_level_utils.py`, `cnn_vae_model.py`

**Directories:**
- Core library: lowercase, no underscores: `src/jaxued/`, `environments/`
- Feature domains: thematic: `vae/`, `craftax/`, `gymnax/`
- Outputs: operational names: `checkpoints/`, `logs/`, `figures/`

**Classes:**
- PascalCase: `Maze`, `TrainState`, `LevelSampler`, `UnderspecifiedEnv`, `CluttrVAE`
- Dataclasses (Flax struct): `EnvState`, `Observation`, `Level`, `EnvParams`

**Functions:**
- snake_case: `sample_trajectories_rnn()`, `compute_gae()`, `train_state_to_log_dict()`
- Score functions suffixed with domain: `max_mc()`, `positive_value_loss()`, `mna_score()`
- Helpers with underscore prefix: `_insert_new()`, `_proportion_filled()` (in LevelSampler)

**Variables:**
- Loop counters: `i`, `step`, `update`
- State containers: `state`, `carry`, `info`
- Random keys: `rng`, `rng_step`, `rng_action`
- JAX pytree maps: prefixed `tree_`: `tree_map`, `tree_flatten`

**Types:**
- Pytree arrays: `chex.Array`, `chex.ArrayTree`, `jnp.ndarray`
- State dicts: `Sampler` (TypedDict), `TrainState` (Flax struct)
- Enums: `UpdateState` (IntEnum: DR=0, REPLAY=1)

## Where to Add New Code

**New Training Algorithm (e.g., NovelD):**
- Create new file: `examples/maze_novelty_driven.py`
- Copy structure from `maze_plr.py` (imports, TrainState, main loop)
- Modify `train_step()` with custom level sampling/scoring logic
- Register new score function in `compute_score()` dispatcher if needed

**New Score Function:**
- If simple regret-based: add to `src/jaxued/utils.py`
- If complex (e.g., GMM-based): create module in `vae/` (e.g., `vae/my_scorer.py`)
- Import in `examples/maze_plr.py` and update `compute_score()` dispatcher

**New Environment:**
- Subclass `UnderspecifiedEnv` in new file: `src/jaxued/environments/my_env/env.py`
- Implement `step_env()`, `reset_env_to_level()`, `action_space()`
- Define `EnvState`, `Observation`, `Level` dataclasses
- Create wrapper example in `examples/my_env_plr.py` (copy `maze_plr.py`, swap environment)

**New VAE Architecture:**
- Add Flax module to `vae/` (e.g., `vae/transformer_vae.py`)
- Implement `encode()`, `decode()`, and `__call__()` methods
- Update `vae_level_utils.py` to handle new latent format if needed
- Register in `cmaes_manager.py` or optimizer wrapper

**Utilities/Helpers:**
- Shared functions: `src/jaxued/utils.py`
- Algorithm-specific helpers: inline in single-file examples (intentional for clarity)
- Data processing: `vae/cnn_vae_data.py` pattern (isolated module)

## Special Directories

**checkpoints/:**
- Purpose: Orbax checkpoint storage (agent params + level buffer)
- Generated: Yes (written by Orbax during training)
- Committed: No (gitignored; large file size)
- Structure: Deep nested directories per strategy/seed/step
- Access: Orbax `CheckpointManager` restores via `restore_from_latest_step()`

**logs/:**
- Purpose: Training stdout/stderr logs
- Generated: Yes (redirected from training scripts)
- Committed: No (gitignored; text logs)
- Access: Tail during training (`tail -f logs/phase5.log`)

**wandb/:**
- Purpose: WandB offline/online run metadata
- Generated: Yes (by WandB SDK)
- Committed: No (gitignored)
- Access: WandB UI for online runs; local sync for offline runs

**figures/:**
- Purpose: Generated visualizations (GIFs, PNGs of solved mazes)
- Generated: Yes (from maze_plr.py logging)
- Committed: Selectively (some example GIFs in repo for README)
- Format: PNG, GIF, MP4 via Pillow/imageio/moviepy

**es_legacy/:**
- Purpose: Archive of legacy ES implementations
- Generated: No (historical code)
- Committed: Yes (but unused in active training)
- Status: Do not modify; kept for reference only

---

*Structure analysis: 2026-03-11*
