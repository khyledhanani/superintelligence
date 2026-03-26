# Codebase Structure

**Analysis Date:** 2025-03-23

## Directory Layout

```
superintelligence/
├── src/jaxued/                    # Core JAX UED library (pip-installable)
│   ├── environments/              # UPOMDP implementations
│   │   ├── underspecified_env.py   # Base interface (Level, EnvState, Observation, EnvParams)
│   │   ├── maze/                  # Minigrid-style maze environment
│   │   │   ├── env.py             # Maze class, step/reset logic
│   │   │   ├── level.py           # Level dataclass, prefabs (StandardMaze, Labyrinth, etc.)
│   │   │   ├── env_editor.py      # Interactive level editor
│   │   │   ├── renderer.py        # Visualization (grid → RGB image)
│   │   │   ├── util.py            # Helper functions
│   │   │   └── __init__.py
│   │   ├── gymnax/                # Classic control environments
│   │   │   ├── acrobot.py         # Acrobot with configurable parameters
│   │   │   ├── cartpole.py        # CartPole variant
│   │   │   └── pendulum.py        # Pendulum variant
│   │   └── __init__.py
│   ├── wrappers/                  # Environment composition
│   │   ├── autoreplay.py          # AutoReplayWrapper (replay same level on reset)
│   │   ├── autoreset.py           # AutoResetWrapper (auto-reset after done)
│   │   └── __init__.py
│   ├── level_sampler.py           # LevelSampler class (prioritized buffer)
│   ├── linen.py                   # ResetRNN helper for LSTM
│   ├── utils.py                   # Utilities (compute_max_returns, max_mc, positive_value_loss)
│   └── __init__.py
│
├── examples/                       # Single-file algorithm implementations
│   ├── maze_plr.py                # Main training script (PLR/RPLR/ACCEL + CMA-ES + metrics)
│   ├── maze_dr.py                 # Domain Randomization (DR) baseline
│   ├── maze_paired.py             # PAIRED (teacher-learner co-evolution)
│   ├── cross_evaluate.py           # Evaluate one run's policy on multiple level sources
│   ├── evaluate_buffer.py          # Post-training buffer analysis (PCA, diversity)
│   ├── craftax/                    # Craftax environment support
│   │   ├── craftax_plr.py         # Craftax trainer
│   │   ├── craftax_wrappers.py    # Craftax-specific wrappers
│   │   ├── mutators.py            # Craftax level mutations
│   │   └── __init__.py
│   └── gymnax/                     # Gymnax environment support
│       └── gymnax_plr.py          # Gymnax trainer (Acrobot, Cartpole, Pendulum)
│
├── vae/                            # VAE + CMA-ES for level generation
│   ├── vae_model.py                # CluttrVAE (Flax model)
│   ├── cmaes_manager.py            # CMAESManager (evosax wrapper)
│   ├── vae_level_utils.py          # JAX functions for token↔level conversion
│   ├── buffer_latent_analysis.py   # Analyze VAE latent space of buffer
│   └── compare_accel_vs_cmaes.py  # Comparison script for plotting results
│
├── metrics/                        # Pluggable metric computation & analysis
│   ├── base.py                     # DiversityAnalyzer ABC, AnalysisSection
│   ├── dtw.py                      # DTW utilities (scipy-based)
│   ├── utils.py                    # Shared: episode reconstruction, trajectory processing
│   ├── standalone/                 # Single-level metrics
│   │   ├── cenie.py                # CENIE novelty metric
│   │   ├── learnability.py         # SFL learnability metric
│   │   ├── per_step_regret.py      # Per-step regret
│   │   ├── per_step_entropy.py     # Policy entropy per step
│   │   ├── per_step_action.py      # Action frequency analysis
│   │   ├── regret.py               # Episode regret (max return - actual)
│   │   ├── value_error.py          # Value function MSE
│   │   └── __init__.py
│   ├── pairwise/                   # Level-pair diversity metrics
│   │   ├── action_dtw_binary.py    # DTW on action sequences
│   │   ├── pos_dtw.py              # DTW on agent trajectories
│   │   ├── regret_dtw.py           # DTW on regret curves
│   │   ├── td_error_distribution.py# TD error histogram similarity
│   │   ├── mode_transition.py      # Mode-switching behavior
│   │   └── __init__.py
│   ├── scripts/                    # Standalone analysis scripts
│   │   └── plot_metrics_demo.py    # Example metric computation & plotting
│   └── __init__.py
│
├── llm/                            # LLM-based maze generation (optional)
│   ├── prompt_builder.py           # Compose system/reference/instruction prompts
│   ├── maze_generator.py           # Call Claude API to generate levels
│   ├── decision_gate.py            # Difficulty/diversity gates for selective generation
│   ├── agent_evaluator.py          # Evaluate agent on generated levels
│   ├── test_generator.py           # Unit tests for generation
│   └── __init__.py
│
├── accel_training/                 # Legacy ES components (deprecated)
│   └── es_components/              # CMA-ES/ES implementation details
│
├── tests/                          # Test suite
│   ├── test_examples_kinda.py      # Integration tests for example scripts
│   └── (other tests)
│
├── scripts/                        # Utility scripts
│   ├── compare_phase4_results.py   # WandB comparison table generator
│   └── (other utilities)
│
├── examples/                       # Launch scripts (shell)
│   ├── launch_50k_accel_baseline.sh
│   ├── launch_50k_cmaes_pruned.sh
│   ├── launch_50k_pca_refit.sh
│   └── (other launch scripts)
│
├── google_cloud_tpu/               # TPU deployment utilities
│   └── (TPU sync/deploy helpers)
│
├── docs/                           # Markdown documentation
│   └── (API docs, tutorials)
│
├── figures/                        # Demo visualizations
│   └── (GIF animations of solved mazes)
│
├── logs/                           # Training logs (created at runtime)
│   └── (checkpoint & log directories per run)
│
├── wandb/                          # Local WandB cache (created at runtime)
│   └── (offline sync data)
│
├── nlp/                            # NLP-related experiments (archived)
│   └── (legacy code)
│
├── es_legacy/                      # Legacy ES implementations (archived)
│   └── (deprecated evolution strategies)
│
├── .planning/                      # GSD planning documents (generated)
│   └── codebase/
│       ├── ARCHITECTURE.md
│       ├── STRUCTURE.md
│       ├── CONVENTIONS.md
│       ├── TESTING.md
│       ├── STACK.md
│       └── INTEGRATIONS.md
│
├── pyproject.toml                  # Package config (jaxued installable)
├── mkdocs.yml                      # Documentation config
├── tox.ini                         # Test runner config
├── .gitignore                      # Git exclusions (logs/, wandb/, *.npz, etc.)
├── README.md                       # Project overview
└── run_tests.sh                    # Test runner script
```

## Directory Purposes

**`src/jaxued/`** — Core library
- Purpose: Reusable UED components, distributed via pip
- Status: Actively maintained
- Key design: Functional, minimal state, JAX-first

**`examples/`** — Algorithm implementations
- Purpose: Single-file, readable implementations of UED methods (DR, PLR, ACCEL, PAIRED)
- Status: Primary training entry points
- Key: Intentionally not split into modules for pedagogical clarity
- Configuration: Command-line args (argparse) + optional YAML for VAE config

**`vae/`** — Latent space generation
- Purpose: Pre-trained VAE for maze level token sequences + CMA-ES search
- Status: Optional (only loaded if `--use_cmaes` flag set)
- Key: Pure JAX functions, fully vmappable/jittable

**`metrics/`** — Diversity & novelty analysis
- Purpose: Compute metrics on trajectories; format for LLM prompts
- Status: Extensible architecture for custom metrics
- Key: Metrics are pure functions; Analyzers format results for LLM

**`llm/`** — Language model integration
- Purpose: Generate levels via Claude API with metric-informed prompts
- Status: Optional post-training pipeline (not in main training loop)
- Key: Prompt builder is pluggable; metrics are injected as context

## Key File Locations

**Entry Points:**
- `examples/maze_plr.py`: Main training entry; run with `python examples/maze_plr.py --help` for all options
- `examples/maze_dr.py`: Domain Randomization (no level buffer)
- `examples/maze_paired.py`: PAIRED algorithm (teacher-learner)

**Configuration:**
- Command-line args: Defined in `examples/maze_plr.py` (lines ~1300+)
- VAE config: YAML file (path via `--vae_config_path`)
- Defaults: Hardcoded in argparse or config dict

**Core Logic:**
- Training loop: `examples/maze_plr.py` lines ~450-1100 (main function + train_and_eval_step)
- Environment base: `src/jaxued/environments/underspecified_env.py` (UnderspecifiedEnv interface)
- Maze implementation: `src/jaxued/environments/maze/env.py` (Maze class, step/reset)
- Level buffer: `src/jaxued/level_sampler.py` (LevelSampler class)
- VAE decoding: `vae/vae_model.py` (CluttrVAE) + `vae/vae_level_utils.py` (token↔level)
- CMA-ES: `vae/cmaes_manager.py` (CMAESManager wrapper)

**Testing:**
- Integration tests: `tests/test_examples_kinda.py`
- Run with: `bash run_tests.sh` or `python -m pytest tests/`

**Utilities:**
- Metrics computation: `metrics/base.py` (DiversityAnalyzer ABC)
- Specific metrics: `metrics/standalone/` (single-level), `metrics/pairwise/` (pair-level)
- Analysis scripts: `metrics/scripts/plot_metrics_demo.py` (example usage)

## Naming Conventions

**Files:**
- Python module files: `snake_case.py` (e.g., `maze_plr.py`, `vae_model.py`)
- Test files: `test_*.py` or `*_test.py` (e.g., `test_examples_kinda.py`)
- Shell scripts: `launch_*.sh` or `*.sh` (e.g., `launch_50k_pca_refit.sh`)
- Config files: `config.yaml`, `vae_config.yaml`, `.prettierrc`, `.eslintrc` (various)

**Directories:**
- Core library: `src/jaxued/` (Python package structure)
- Examples: `examples/` (single-file scripts or subdir with __init__.py)
- Data/output: `logs/`, `wandb/`, `checkpoints/`, `results/` (created at runtime)
- Config: Top-level or passed as env vars (e.g., `--vae_checkpoint_path`)

**Classes:**
- Environment: PascalCase + `Env` suffix (e.g., `Maze`, `UnderspecifiedEnv`, `AutoReplayWrapper`)
- Dataclasses: PascalCase (e.g., `Level`, `EnvState`, `TrainState`, `MetricEntry`)
- Managers: PascalCase + `Manager` suffix (e.g., `CMAESManager`)
- Analyzers: PascalCase + `Analyzer` suffix (e.g., `LearnabilityAnalyzer`, `DiversityAnalyzer`)

**Functions:**
- Pure utilities: `snake_case` (e.g., `compute_gae`, `sample_trajectories_rnn`, `tokens_to_level`)
- Loss/metric computation: `*_loss`, `compute_*`, `measure_*` (e.g., `positive_value_loss`, `compute_max_returns`)
- Step functions: `*_step` (e.g., `train_step`, `train_and_eval_step`)

**Variables:**
- PRNG keys: `rng`, `rng_*` (e.g., `rng_action`, `rng_reset`, `rng_eval`)
- State: `state`, `carry`, `*_state` (e.g., `env_state`, `train_state`, `hstate` for hidden state)
- Arrays: `x`, `obs`, `reward`, `done`, etc. (type/size from docstring)
- Indices: `i`, `j`, `idx`, `level_inds` (context-specific)

## Where to Add New Code

**New Environment:**
1. Subclass `UnderspecifiedEnv` in `src/jaxued/environments/your_env/env.py`
2. Implement: `step_env()`, `reset_env_to_level()`, `action_space()`
3. Define dataclasses: `EnvState`, `Observation`, `Level`, `EnvParams` (optional, inherit defaults)
4. Update `examples/maze_plr.py` line ~573: change `env = Maze(...)` to `env = YourEnv(...)`

**New Metric:**
1. Metric computation (pure function): `metrics/standalone/your_metric.py`
2. Analyzer class (format for LLM): Subclass `DiversityAnalyzer` in same file
3. Register: Import and add to metric list in analysis pipeline (if used)
4. Tests: `tests/test_metrics_your_metric.py`

**New UED Algorithm:**
1. Copy `examples/maze_plr.py` to `examples/maze_your_algo.py`
2. Modify the `train_step()` function (lines ~645-916) to implement your branching logic
3. Adjust TrainState fields if needed (add `your_state` to line ~38-52)
4. Update CLI args for your hyperparameters (lines ~1300+)

**Utility Functions:**
- Shared trajectory processing: `metrics/utils.py`
- RL utilities (compute_gae, etc.): `src/jaxued/utils.py`
- Level utilities: `src/jaxued/environments/maze/util.py`

## Special Directories

**`checkpoints/[run_name]/[seed]/models/[step]/`:**
- Purpose: Saved policy checkpoints
- Generated: Yes (created during training)
- Committed: No (`.gitignore` excludes)
- Contents: Orbax checkpoint (params dict) + config.json

**`results/[run_name]/[seed]/`:**
- Purpose: Evaluation outputs (npz files with states, rewards, episodes, levels)
- Generated: Yes (created in eval mode)
- Committed: No
- Contents: `eval_*.npz` files (NumPy binary format)

**`logs/`:**
- Purpose: Training logs, stderr redirects, metric aggregations
- Generated: Yes (created at runtime)
- Committed: No (`.gitignore`)
- Contents: Structured logs per run (optional)

**`wandb/`:**
- Purpose: Local WandB sync cache
- Generated: Yes (WandB offline sync)
- Committed: No
- Contents: Binary checkpoint data for sync to cloud

**`vae/`:**
- Purpose: Pre-trained VAE checkpoint + config
- Generated: No (external source)
- Committed: No (checkpoint is binary, too large)
- Contents: `vae_checkpoint.pkl` (params), `vae_config.yaml` (dimensions)

**`.planning/codebase/`:**
- Purpose: GSD-generated codebase analysis documents
- Generated: Yes (by `/gsd:map-codebase`)
- Committed: Yes (reference docs for future runs)
- Contents: ARCHITECTURE.md, STRUCTURE.md, CONVENTIONS.md, etc.

---

*Structure analysis: 2025-03-23*
