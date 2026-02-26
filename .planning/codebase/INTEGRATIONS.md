# External Integrations

**Analysis Date:** 2026-02-26

## APIs & External Services

**Experiment Tracking:**
- Weights & Biases (Wandb)
  - What it's used for: ML experiment tracking, metric visualization, image/video logging, model comparison
  - SDK: `wandb` package
  - Auth: `wandb.login()` (uses local credentials file, typically `~/.wandb`)
  - Integration points:
    - `examples/maze_dr.py` - Domain Randomization training
    - `examples/maze_plr.py` - Prioritized Level Replay and ACCEL training
    - `examples/maze_paired.py` - PAIRED training
    - `examples/craftax/craftax_plr.py` - Craftax environment training
    - `examples/gymnax/gymnax_plr.py` - Gymnax environment training
  - Metrics logged:
    - Training progress: `num_updates`, `num_env_steps`
    - Performance: `solve_rate/*`, `return/*`, `eval_ep_lengths/*`
    - Level sampler stats: `level_sampler/*`
    - Agent metrics: `agent/*`
    - Images: highest scoring levels, highest weighted levels
    - Videos: training animations at 4 fps

## Data Storage

**Checkpointing:**
- Local filesystem via Orbax
  - Client: `orbax.checkpoint` (version 0.5.3)
  - Storage path: `./checkpoints/<run_name>/<seed>/models/<update_step>`
  - Format: JAX-native checkpoint format (compatible with JAX arrays and Flax models)
  - Used in: All example scripts for saving trained agents and level samplers

**Results Storage:**
- Local filesystem
  - Storage path: `./results/` (created during evaluation)
  - Format: `.npz` (NumPy compressed arrays)
  - Keys in results files: `states`, `cum_rewards`, `episode_lengths`, `levels`
  - Used by: Evaluation mode of example scripts

**Configuration Files:**
- YAML files for VAE/ML model configuration
  - Located: `vae/vae_train_config.yml`
  - Read by: `es/vae_decoder.py` for CluttrVAE decoder configuration
  - Content: Model hyperparameters (seq_len, latent dimensions, etc.)

**Binary Model Files:**
- Pickle-based checkpoints (inferred from JAX/Flax training patterns)
  - VAE decoder parameters loaded in `es/vae_decoder.py` using `pickle`
  - Encoder parameters loaded in `es/cluttr_encoder.py`
  - Maze autoencoder parameters in `es/maze_ae.py`

## File Storage

**Local Filesystem Only:**
- No cloud storage integration detected
- All outputs written locally:
  - Model checkpoints: `./checkpoints/`
  - Evaluation results: `./results/`
  - Evolved environments: `es/evolved/`
  - Generated images/visualizations: `es/*.png`

## Caching

**None Detected:**
- No Redis, Memcached, or other caching service
- JAX/Flax use JIT compilation caching automatically

## Authentication & Identity

**Auth Provider:**
- Custom local authentication
  - Wandb: Uses local credentials file (automatic via `wandb.login()`)
  - No centralized auth system required
  - Each training run is tagged with project/group name via Wandb config

## Monitoring & Observability

**Error Tracking:**
- None detected (standard Python exceptions propagate)

**Logs:**
- Standard output via Python `print()` and logging
- Wandb integration for structured metrics logging
- No dedicated logging service (Sentry, DataDog, etc.)

## CI/CD & Deployment

**Hosting:**
- Local/academic environment (no cloud deployment detected)
- Designed for single-machine training with JAX CPU/GPU

**CI Pipeline:**
- Tox for local testing (`tox.ini`)
  - Runs pytest across Python versions
  - Command: `pytest -v -s --cov=src/jaxued`
  - No remote CI service integration detected

**Deployment:**
- Not applicable - research/training library
- Installation: `pip install jaxued` (from PyPI or local source)
- Development install: `pip install -e .` (editable mode)

## Environment Configuration

**Required env vars:**
- `WANDB_API_KEY` - Wandb authentication (read from `~/.wandb/` if not set)
- `JAX_PLATFORMS` - JAX device selection (cpu/gpu, optional, defaults to available)

**Optional env vars:**
- `PIP_NO_CACHE_DIR` - Set to "1" in `environment.yml` (already configured)

**Secrets location:**
- Wandb credentials: `~/.wandb/settings` (local user file)
- No `.env` file pattern used (credentials via system environment)

## Webhooks & Callbacks

**Incoming:**
- None detected

**Outgoing:**
- Wandb event streaming (implicit via `wandb.log()`)
  - One-way push of metrics to Wandb backend

## Model Management

**VAE Integration:**
- CLUTTR VAE decoder extraction in `es/vae_decoder.py`
  - Loads pre-trained checkpoint from `vae/` sibling directory
  - Decodes latent vectors to environment sequences
  - Used for curriculum generation via ACCEL/MAP-Elites

**Maze Autoencoder:**
- Maze-specific encoder/decoder in `es/maze_ae.py`
  - Encodes maze levels to latent space
  - Decodes latents back to maze representations
  - Used for mutation-based level generation

## Third-party Environment Integrations

**Craftax:**
- Installation: `git+https://github.com/MichaelTMatthews/Craftax.git@main`
- Modules used: `craftax.envs`, `craftax.renderer`, `craftax.world_gen`, `craftax.constants`
- Wrapper integration: `examples/craftax/craftax_wrappers.py` adapts Craftax to JaxUED interface

**Gymnax:**
- Built-in dependency in `pyproject.toml`
- Modules: `gymnax.environments` (Acrobot, CartPole, Pendulum)
- Custom wrappers in `src/jaxued/environments/gymnax/`

---

*Integration audit: 2026-02-26*
