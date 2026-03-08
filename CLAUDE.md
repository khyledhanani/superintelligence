# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**JaxUED** is a JAX-based Unsupervised Environment Design (UED) library with single-file reference implementations of DR, PLR, ACCEL, and PAIRED algorithms. The codebase extends these with evolutionary environment search using VAE latent spaces and MAP-Elites quality-diversity optimization.

The core research direction: train agents on a curriculum of procedurally-generated maze environments, then evolve new environments via mutation in the VAE latent space to discover ones that maximize agent regret (hard but learnable).

## Commands

### Installation
```bash
conda env create -f environment.yml && conda activate superintelligence
# or
pip install -e ".[examples]"
```

### Running UED Training
```bash
python examples/maze_dr.py                          # Domain Randomization baseline
python examples/maze_plr.py --exploratory_grad_updates  # PLR
python examples/maze_plr.py --use_accel             # ACCEL
python examples/maze_paired.py                      # PAIRED (adversarial)
```

### Evaluation
```bash
python examples/maze_plr.py --mode eval \
  --checkpoint_directory=./checkpoints/<run_name>/<seed> \
  --checkpoint_to_eval <update_step>
```

### MAP-Elites Evolution
```bash
cd es/
python map_elites.py --agent_checkpoint_dir <agent_folder> --num_iterations 2000 --batch_size 32 --output_subdir _run1
python evolve_envs.py --fitness_mode regret --agent_checkpoint_dir <agent_folder>/119
```

### VAE Training
```bash
cd vae/
python sample_envs.py        # Generate dataset
python train_vae.py          # Train beta-VAE on CLUTTR sequences
python train_maze_ae.py --config maze_ae_config.yml  # Task-aware AE
```

### Testing
```bash
pytest tests/ -v -s
tox  # Runs across Python 3.9–3.13
```

## Architecture

### Core Library (`src/jaxued/`)

The library is built on a `UnderspecifiedEnv` base class (`underspecified_env.py`) defining the UPOMDP interface: `reset_to_level(rng, level, params)` and `step(rng, state, action, params)`. Every environment and wrapper implements this.

**Key abstractions:**
- `LevelSampler` (`level_sampler.py`): Prioritized buffer for PLR/ACCEL curriculum. Stores levels with scores and timestamps; supports rank-based or top-k prioritization with staleness weighting.
- `Maze` (`environments/maze/`): Primary environment. 7×7 egocentric observation, 7 discrete actions, parametrized by `Level` (wall_map, agent_pos, goal_pos, agent_dir).
- `ResetRNN` / `ResidualBlock` (`linen.py`): Policy network architectures used across all training scripts.

### UED Training Scripts (`examples/`)

Each is a self-contained single-file implementation (CleanRL-style). They share a common pattern: vmap over seeds → pmap over devices → PPO update loop with level curriculum.

### Environment Evolution Pipeline (`es/`)

The evolution pipeline operates in VAE latent space:

```
Latent z (64-dim)
  → vae_decoder.py: decode to CLUTTR sequence (52 tokens)
  → env_bridge.py: CLUTTR → Maze Level (wall map + positions)
  → regret_fitness.py: agent rollout → MaxMC regret score
  → map_elites.py: insert into 2D archive (obstacles × distance bins)
```

**MAP-Elites archive**: 8×6 grid (48 cells) keyed by (num_obstacles, manhattan_distance). Each cell stores the highest-regret environment found. `map_elites_mutation_service.py` handles mutations (z_child = z_parent + σ·N(0,I)) and behavior binning.

**Fitness modes** (set in `evolve_config.yml`):
- `placeholder`: structural complexity only (no agent needed)
- `regret`: MaxMC regret requires a frozen ACCEL agent checkpoint

### VAE / Autoencoder (`vae/`, `es/maze_ae.py`)

Two model types:
1. **Beta-VAE** (`train_vae.py`): Encodes CLUTTR sequences (52 discrete tokens) to 64-dim latent. Used as the primary latent space for MAP-Elites.
2. **Task-aware Maze AE** (`maze_ae.py`, `train_maze_ae.py`): Adds 7 auxiliary regression heads (wall_count, dead_ends, path_length, etc.) + validity head. Trained with multi-task loss to align latent space with structural properties.

CLUTTR is the intermediate representation: a variable-length sequence of (entity_type, x, y) tokens describing maze contents.

### Analysis Scripts (`scripts/`, `analysis/`)

Scripts produce reports in `analysis/` subdirectories. Key outputs:
- Latent space geometry (PCA, cross-model Spearman correlations)
- Mutation locality comparison: AE vs original VAE
- Structured random dataset with labeled characteristics

## Key Configuration

- `es/evolve_config.yml`: MAP-Elites/CMA-ES hyperparameters (sigma, batch_size, behavior bins, fitness_mode)
- `vae/vae_train_config.yml`: Beta-VAE training (latent_dim=64, sequence_length=52)
- `vae/maze_ae_config.yml`: Maze AE training with auxiliary head weights
- `.env`: `OLLAMA_API_KEY` and other secrets (not committed)

## Important Notes

- **Checkpoints are large**: `checkpoints/` (893 MB), `vae/` (186 MB), `wandb/` (80 MB) — don't accidentally commit these.
- **JAX device handling**: Scripts use `jax.pmap` for multi-device training. Single-GPU runs work with `--num_train_envs` adjusted down.
- **Orbax version**: Requires `orbax-checkpoint==0.5.3` exactly — newer versions have breaking API changes.
- **CLUTTR format**: The 52-token CLUTTR sequence is the bridge between VAE latent space and Maze levels. See `env_bridge.py` for encoding/decoding logic.
- **Solvability**: All generated mazes must pass BFS check in `env_bridge.py` before being used as training levels.
