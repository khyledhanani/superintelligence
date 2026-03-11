# Technology Stack

**Analysis Date:** 2026-03-11

## Languages

**Primary:**
- Python 3.9+ - All training, VAE, and utility code
- Bash - Execution scripts and automation

**Secondary:**
- YAML - Configuration files (wandb, mkdocs)

## Runtime

**Environment:**
- Python 3.9.25 (verified on system)
- JAX 0.6.2 - Core computational framework
- JAX GPU: CUDA 12 via JAX CUDA pip releases (required for sideswipe/prowl nodes; CUDA 11.7 on blaze is incompatible with JAX 0.5.3+)

**Package Manager:**
- pip - Primary installation manager
- No lockfile detected; versions pinned in wandb run artifacts and `vae/requirements.txt`

## Frameworks

**Core ML/RL:**
- Flax 0.10.7 - Neural network definitions and training state management
  - Used for: Actor-critic networks, LSTM cells, RNN policies in `examples/maze_plr.py` and `src/jaxued/linen.py`
- JAX 0.6.2 - Differentiable programming and JIT compilation
  - Core to: vmap, scan, lax operations for vectorized rollouts
- Optax 0.2.7 - Gradient-based optimization
  - Used for: Adam optimizer in PPO training loop

**Environment & RL Algorithms:**
- Gymnax 0.0.9 - JAX-based vectorized environments (Cartpole, Acrobot, Pendulum)
  - Located: `examples/gymnax/gymnax_plr.py`
- Distrax 0.1.5 - Probabilistic distributions for policy sampling
  - Used in: Actor network's categorical action sampling (`examples/maze_plr.py` line 330)
- JaxUED 0.0.2 - Custom UED library (local package in `src/jaxued/`)
  - Provides: Maze environment, level sampler, utilities
  - Core classes: `UnderspecifiedEnv`, `LevelSampler`, `Maze`
- Chex 0.1.90 - Type checking and assertions for JAX code
  - Used for: `chex.ArrayTree` type hints throughout

**Evolution Strategies:**
- evosax (latest API via `evosax.algorithms`) - CMA-ES and other ES algorithms
  - Wrapper: `vae/cmaes_manager.py` - Provides init/ask/tell interface
  - Supports both old (`evosax.CMA_ES`) and new API (`evosax.algorithms.CMA_ES`)

**VAE & Level Generation:**
- Flax (same as above) - VAE encoder/decoder networks
  - Models: `vae/vae_model.py` (CluttrVAE), `vae/cnn_vae_model.py` (CNN-VAE)
  - LSTM-based bidirectional encoder, highway layers, dense decoder

**Checkpointing:**
- Orbax 0.11.33 - Checkpoint management and restoration
  - Config: `orbax.checkpoint as ocp` in `examples/maze_plr.py` line 17
  - Used for: Saving/loading PPO and ES state

**Monitoring & Logging:**
- Weights & Biases (wandb) 0.24.2 - Experiment tracking
  - Init: `examples/maze_plr.py` line 487
  - Metrics: define_metric calls for custom step metrics
  - Integration: Full logging of training metrics, generated levels, buffers

**Testing & Documentation:**
- pytest - Test framework
  - Config: `tox.ini` - Runs pytest with coverage on Python 3.9-3.13
- tox - Test environment management
  - Config: `tox.ini` - Specifies test envs for multiple Python versions

**Data & Analysis:**
- NumPy 2.2.6 - Numerical arrays (used via jnp but also raw numpy in scoring)
- Pandas 2.3.3 - Data manipulation for analysis scripts
- SciPy 1.15.3 - Scientific computing (used by sklearn)
- scikit-learn 1.7.2 - ML utilities
  - GaussianMixture in `vae/cenie_scorer.py` line 80
  - PCA in `vae/buffer_latent_analysis.py` line 24
- Matplotlib 3.10.8 - Plotting and visualization
  - Non-interactive backend ("Agg") in VAE scripts for headless operation
  - Used in: `vae/latent_perturbation_diagnostic.py`, `vae/compare_accel_vs_cmaes.py`

**Media & File Handling:**
- Pillow 11.3.0 - Image handling
  - Used in: Maze rendering pipeline
- imageio 2.37.2 + imageio-ffmpeg 0.6.0 - Video/frame writing
  - Used in: Level visualization exports
- moviepy 2.2.1 - Video composition
  - Used in: Creating training visualization videos (in optional dependencies)

**Utilities:**
- PyYAML 6.0.3 - YAML parsing
  - Used for: Config loading in training scripts
- python-dotenv 1.2.1 - Environment variable loading
  - Pattern: Load from `.env` files (none detected in repo)
- requests 2.32.5 - HTTP client
  - Used by: wandb, Google Cloud integration

**Cloud & Storage:**
- google-cloud-storage 3.9.0 - GCS client library
  - Fallback: gcloud CLI via subprocess if Python client fails
  - Used in: `_upload_to_gcs()` and checkpoint management
- google-cloud-core 2.5.0 - Core Google Cloud APIs
- google-auth 2.48.0 - Authentication for GCP
- gcsfs (latest, no version pin in requirements) - Filesystem interface to GCS
  - Used for: URI-based checkpoint path handling (gs://bucket/...)

**Jupyter/Interactive:**
- Jupyter/JupyterLab 4.5.3 - Notebook environment
  - Analysis and debugging of VAE, buffer contents
  - Config: `notebook==7.5.3`

**Dependencies at Risk/Special Notes:**
- tensorflow-probability 0.25.0 - Used only if needed for advanced probability operations (minimal usage expected)
- tensorstore 0.1.78 - Advanced tensor storage (unused in primary code)
- rdkit 2025.9.4 - Cheminformatics library (unused in core training)
- sentry-sdk 2.52.0 - Error tracking (minimal usage, can be optional)

## Configuration

**Environment:**
- JAX configuration: No `XLA_FLAGS` should be set (JAX 0.6.2 auto-detects libdevice)
  - Memory note: `.planning/memory/MEMORY.md` documents GPU node allocation (sideswipe/prowl have CUDA 12)
- OpenBLAS threading: `OPENBLAS_NUM_THREADS=4` set in training scripts to prevent contention
- Python OPENBLAS/MKL: `MKL_NUM_THREADS=4` in launch scripts

**Training Config:**
- YAML-based: Training parameters loaded via PyYAML
  - Example: `config.yml` with CMA-ES restart threshold, VAE paths, GCS bucket/prefix
  - Passed via `wandb.init(config=config)` for W&B tracking
- Default paths: Checkpoints in `./checkpoints/<run_name>/<seed>/models/<update_step>`
- GCS paths: `gs://<bucket>/<prefix>/checkpoints/<run_name>/<seed>/` if `--gcs_bucket` provided

**Build:**
- setuptools 61.0+ - Python packaging
  - Config: `pyproject.toml` defines jaxued as editable install
  - Entry point: `[project.optional-dependencies]` for examples (distrax, optax, orbax-checkpoint, wandb, etc.)

## Platform Requirements

**Development:**
- Linux (RHEL 9.7, kernel 5.14.0)
- CUDA 12 (on sideswipe/prowl TPU nodes; CUDA 11.7 on blaze head node is unsupported for training)
- JAX requires: libdevice auto-detection or JAX CUDA plugin
- Login shell: csh; scripts use bash (`#!/bin/bash`)
- Git 3.1.46 (for version control)

**Production/Deployment:**
- TPU VM environment (Google Cloud TPU nodes: sideswipe, prowl)
- GCS bucket access required for checkpoint/buffer storage
- Wandb account for experiment tracking (optional but integrated)

---

*Stack analysis: 2026-03-11*
