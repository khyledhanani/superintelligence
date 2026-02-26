# Codebase Structure

**Generated:** 2026-02-26
**Focus:** Directory layout, key locations, naming conventions

---

## Top-Level Layout

```
superintelligence/
├── src/jaxued/              # Core library (installable package)
├── examples/                # Example training scripts
├── es/                      # Evolution Strategy / MAP-Elites integration (custom)
├── accel_training/          # Accelerated training pipeline
├── tests/                   # Test suite
├── scripts/                 # Utility scripts
├── docs/                    # MkDocs documentation
├── figures/                 # GIF visualizations of trained agents
├── agent_folder/            # Saved agent checkpoints (orbax)
├── nlp/                     # Old Python 3.6 virtualenv (legacy, unused)
├── vae/                     # VAE-related files (partially integrated)
├── pyproject.toml           # Package definition
├── environment.yml          # Conda environment spec
├── mkdocs.yml               # Documentation config
└── tox.ini                  # Test runner config
```

---

## Core Library: `src/jaxued/`

The installable `jaxued` package. All public API lives here.

```
src/jaxued/
├── __init__.py                          # Public exports
├── level_sampler.py                     # LevelSampler — PLR/ACCEL buffer logic
├── linen.py                             # Flax linen utilities
├── utils.py                             # Shared utility functions
├── environments/
│   ├── underspecified_env.py            # Base class: UnderspecifiedEnv, Level, EnvState, EnvParams
│   ├── maze/
│   │   ├── env.py                       # Maze environment
│   │   ├── env_editor.py                # Maze level editor
│   │   ├── env_solved.py                # Maze oracle/solved variant
│   │   ├── level.py                     # Maze Level dataclass + generation
│   │   ├── renderer.py                  # Maze rendering
│   │   └── util.py                      # Maze utilities
│   └── gymnax/
│       ├── acrobot.py                   # Gymnax Acrobot wrapper
│       ├── cartpole.py                  # Gymnax CartPole wrapper
│       └── pendulum.py                  # Gymnax Pendulum wrapper
└── wrappers/
    ├── autoreplay.py                    # AutoReplayWrapper — auto-replays levels
    └── autoreset.py                     # AutoResetWrapper — auto-resets on episode end
```

---

## Examples: `examples/`

Runnable training scripts. Each script is self-contained.

```
examples/
├── maze_dr.py               # Domain Randomization on maze
├── maze_plr.py              # Prioritized Level Replay (PLR) on maze
├── maze_paired.py           # PAIRED algorithm on maze
├── craftax/
│   ├── craftax_plr.py       # PLR on CraftAx environment
│   ├── craftax_wrappers.py  # CraftAx UED wrappers
│   └── mutators.py          # Level mutation functions
└── gymnax/
    └── gymnax_plr.py        # PLR on gymnax environments
```

---

## ES Integration: `es/`

Custom MAP-Elites + Evolution Strategy pipeline for generating curricula. This is the active research extension.

```
es/
├── evolve_envs.py                    # Main ES evolution loop
├── map_elites.py                     # MAP-Elites archive implementation
├── map_elites_mutation_service.py    # Mutation operators for MAP-Elites
├── env_bridge.py                     # Bridge between ES and jaxued environments
├── agent_loader.py                   # Load trained agent checkpoints
├── cluttr_encoder.py                 # Encoder for environment representation
├── maze_ae.py                        # Maze autoencoder
├── vae_decoder.py                    # VAE decoder for level generation
├── regret_fitness.py                 # Fitness function based on agent regret
├── metrics.py                        # Evaluation metrics
├── visualize_envs.py                 # Visualization utilities
├── test_integration.py               # Integration tests for ES pipeline
├── evolve_config.yml                 # ES hyperparameter config
└── _me_full/                         # Saved MAP-Elites archive (numpy arrays)
    ├── archive_descriptors.npy
    ├── archive_envs.npy
    ├── archive_fitness.npy
    ├── archive_latents.npy
    ├── archive_gallery.png
    └── archive_heatmap.png
```

---

## Accelerated Training: `accel_training/`

Standalone training pipeline with PPO and UED interface.

```
accel_training/
├── train.py              # Main training entry point
├── ppo_utils.py          # PPO algorithm utilities
├── ued_interface.py      # Interface to jaxued level sampling
├── config.yml            # Training hyperparameters
└── README.md             # Usage instructions
```

---

## Tests: `tests/`

```
tests/
└── test_examples_kinda.py    # Integration tests — runs example scripts via subprocess
```

---

## Key File Locations

| What | Where |
|------|-------|
| Public API | `src/jaxued/__init__.py` |
| Core abstractions | `src/jaxued/environments/underspecified_env.py` |
| Level buffer | `src/jaxued/level_sampler.py` |
| Maze environment | `src/jaxued/environments/maze/env.py` |
| ES main loop | `es/evolve_envs.py` |
| Training pipeline | `accel_training/train.py` |
| Package definition | `pyproject.toml` |
| Conda environment | `environment.yml` |

---

## Naming Conventions

- **Files:** `snake_case.py`
- **Classes:** `PascalCase` (e.g., `LevelSampler`, `UnderspecifiedEnv`, `AutoResetWrapper`)
- **Functions/methods:** `snake_case`
- **Constants/types:** `PascalCase` type aliases (e.g., `Prioritization = Literal[...]`)
- **Private modules:** Prefixed with `_` (e.g., `_me_full/`)
- **Config files:** `.yml` extension (not `.yaml`)
- **Test files:** `test_*.py` prefix

---

*Generated by gsd-codebase-mapper · 2026-02-26*
