# Technology Stack

**Analysis Date:** 2026-03-23

## Languages

**Primary:**
- Python 3.9+ - Core implementation language for all training, evaluation, and utilities

**Secondary:**
- Bash - Launch scripts and experiment orchestration (`examples/launch_*.sh`)
- YAML - Configuration files for experiments and VAE training

## Runtime

**Environment:**
- Python 3.9, 3.10, 3.11, 3.12, 3.13 supported (per `pyproject.toml`)
- Conda environment: `jax_env` (both GPU and TPU variants)
- TPU path: `/home/gmaralla/miniconda3/envs/jax_env/bin/python`
- GPU path: `/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python`

**Package Manager:**
- pip - Primary package management
- Conda - Environment management (`jax_env`)
- setuptools>=61.0 - Build backend (`pyproject.toml`)

## Frameworks

**Core ML/RL:**
- JAX 0.6.2 - Numerical computing and JAX transformations (vmap, scan, jit)
- JAX-lib 0.6.2 - JAX compiled binaries
- Flax 0.10.7 - Neural network library built on JAX
- Optax 0.2.7 - Optimizer library
- Distrax 0.1.5 - Probabilistic distributions for JAX
- Chex 0.1.90 - Testing and type utilities for JAX

**Environment/Simulation:**
- Gymnax 0.0.9 - JAX-based environment suite (vectorized Gym-like interface)
- Gymnasium 1.2.3 - Standard RL environment interface
- MiniGrid 3.0.0 - Grid world environment
- Maze Dataset 1.4.2 - Maze environment data
- Craftax - Custom game environment for UED
- Craftax Classic - Legacy version of Craftax

**Search/Optimization:**
- EvoSax - Evolutionary strategies library (CMA-ES implementation via `vae/cmaes_manager.py`)

**Checkpointing:**
- Orbax 0.11.33 - Checkpoint management with 0.11.33 pinned version in optional deps

**Testing/Development:**
- MkDocs 0.0.0 - Documentation generation
- Jupyter/JupyterLab 4.5.3 - Interactive notebooks
- IPython - Interactive shell
- Pytest - Test framework (via `run_tests.sh`)

## Key Dependencies

**Critical:**
- numpy 2.2.6 - Numerical array operations
- jax/jaxlib 0.6.2 - Core computation engine for all training
- flax 0.10.7 - Neural networks and training state management
- optax 0.2.7 - Gradient-based optimization (Adam, SGD)
- chex 0.1.90 - JAX utilities and testing assertions

**Infrastructure:**
- orbax-checkpoint 0.11.33 - State checkpointing and recovery
- wandb 0.24.2 - Experiment tracking and visualization
- google-cloud-storage 3.9.0 - GCS bucket access
- gcsfs - Google Cloud Storage filesystem interface
- google-auth 2.48.0 - Cloud authentication
- google-cloud-core 2.5.0 - Cloud SDK utilities

**Data Processing & Visualization:**
- pandas 2.3.3 - Data manipulation
- matplotlib 3.10.8 - Plotting
- seaborn 0.13.2 - Statistical visualization
- plotly 6.5.2 - Interactive plots
- PIL/Pillow 11.3.0 - Image processing
- imageio 2.37.2 - Image I/O
- moviepy 2.2.1 - Video processing
- scipy 1.15.3 - Scientific computing
- scikit-learn 1.7.2 - Machine learning utilities

**ML-specific:**
- tensorflow-probability 0.25.0 - Probabilistic ML (for distributions)
- RDKit 2025.9.4 - Molecular/chemical toolkit (conditional dependency)
- jaxtyping 0.3.7 - Type annotations for JAX arrays

**Utilities:**
- tqdm 4.67.3 - Progress bars
- pydantic 2.12.5 - Data validation
- PyYAML 6.0.3 - YAML parsing
- python-dotenv 1.2.1 - Environment variable loading
- click 8.3.1 - CLI framework
- requests 2.32.5 - HTTP client
- GitPython 3.1.46 - Git operations

## Configuration

**Environment:**
- Conda environment variables: `WANDB_DIR=/tmp/wandb`, `JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache`
- LD_LIBRARY_PATH set for CUDA 13.1 support on GPU nodes
- PYTHONPATH configured to include `vae/` and `llm/` directories

**Build:**
- `pyproject.toml` - Package metadata and dependencies
  - Core dependencies: numpy, jax, flax, chex, gymnax
  - Optional `[examples]` group: distrax, optax, orbax-checkpoint, orbax, wandb, pillow, imageio, moviepy
- No build configuration files (setup.py, setup.cfg) - pure pyproject.toml

**Training Configs:**
- `llm/config.yaml` - LLM maze generator configuration (Claude provider, metrics injection, diversity gates)
- `vae/vae_train_config.yml` - VAE training parameters (embed_dim=300, latent_dim=64, seq_len=52)

## Platform Requirements

**Development:**
- Python 3.9+
- JAX GPU/TPU support:
  - GPU: CUDA 13.1 with custom LD_LIBRARY_PATH configuration
  - TPU: Google Cloud TPU (single core used, no pmap)
- Multiple training machines: TPU node (cma-es-v4), GPU nodes (albacore, smew, canada)

**Production/Deployment:**
- Cloud: Google Cloud Platform (TPU VMs, storage)
- GCS bucket: `ucl-ued-project-bucket` for checkpoint/artifact storage
- WandB project: `JAXUED_50K`, `JAXUED_COMPARISON` for monitoring

**Experiment Artifacts:**
- Local checkpoints: `./checkpoints/<run_name>/<seed>/models/`
- Local results: `./results/`
- GCS paths: `gcs_artifacts/agent/`, `gcs_artifacts/buffer/`
- Logs directory: `./logs/` with per-seed log files

---

*Stack analysis: 2026-03-23*
