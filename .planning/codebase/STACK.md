# Technology Stack

**Analysis Date:** 2026-02-26

## Languages

**Primary:**
- Python 3.11 - Core language for the entire project
  - Supported versions: 3.9, 3.10, 3.11, 3.12, 3.13 (as specified in `pyproject.toml`)

## Runtime

**Environment:**
- Python 3.11 (specified in `environment.yml`)
- Conda (package and environment manager)

**Package Manager:**
- pip (installed via conda)
- Conda-forge channels
- Lockfile: `environment.yml` (conda) and `pyproject.toml` (pip/setuptools)

## Frameworks

**Core ML/RL:**
- JAX - Numerical computing and automatic differentiation
  - Installation: `jax[cpu]` via pip (GPU support available via CUDA)
  - Used for all vectorized computations and environment simulations
- Flax - Neural network library built on JAX
  - Core components: `flax.linen` (nn), `flax.struct` (dataclasses), `flax.core`
  - Used in `src/jaxued/linen.py` and example scripts for policy networks
- Chex - JAX array utilities and type checking
  - Used throughout for array assertions and pytree operations

**RL/Evolution Algorithms:**
- Optax - Optimization library for JAX
  - Used for training optimization (SGD, Adam, etc.)
- Distrax - Probabilistic distributions for JAX
  - Used for policy learning and exploration
- Orbax - Checkpointing library for JAX
  - Version: 0.5.3 (pinned in `environment.yml`)
  - Used for saving/loading model checkpoints
- Evosax - Evolutionary algorithms in JAX
  - Used for MAP-Elites and evolution-based curriculum generation

**Environment Libraries:**
- Gymnax - JAX-compatible gymnasium environments
  - Used in `examples/gymnax/` for Acrobot, Cartpole, Pendulum
- Craftax - Extended JAX version of Crafter
  - Installed from git: `git+https://github.com/MichaelTMatthews/Craftax.git@main`
  - Used in `examples/craftax/` for complex environment generation

**Monitoring & Logging:**
- Wandb (Weights & Biases) - ML experiment tracking
  - Used in all example scripts (`examples/maze_*.py`, `examples/craftax/`, `examples/gymnax/`)
  - Metrics logged: solve rates, returns, level statistics, animations, images
  - Authentication via `wandb.login()`

**Scientific Computing:**
- NumPy - Numerical arrays (used as fallback for some operations)
- SciPy - Scientific computing utilities
- Matplotlib - Plotting and visualization
- Pillow - Image processing
- ImageIO - Image/video I/O
- MoviePy - Video composition and processing
- YAML - Configuration file parsing (`pyyaml`)
- TQDM - Progress bars

**Video/Media:**
- FFmpeg - Video encoding/decoding (system dependency)

## Key Dependencies

**Critical Core:**
- jax - Enables GPU-accelerated environment simulation and training
- flax - Neural network definitions and training loops
- chex - Type safety and array utilities
- optax - Training optimization

**Infrastructure:**
- orbax-checkpoint==0.5.3 - Model persistence (pinned version)
- evosax - Evolutionary search for curriculum generation
- gymnax - Standard JAX environment interface
- distrax - Probabilistic distributions for exploration

**Visualization & Logging:**
- wandb - Experiment tracking and visualization
- matplotlib - 2D plotting
- imageio + moviepy - Animation generation for logging

## Configuration

**Environment:**
- Defined in `environment.yml` (conda specification)
- Defines Python 3.11, core scientific stack, JAX, Flax, and optional dependencies
- Variables: `PIP_NO_CACHE_DIR=1` (disable pip cache)

**Build:**
- `pyproject.toml` - Modern Python packaging
  - Build system: setuptools>=61.0
  - Project metadata: jaxued (name), 0.0.2 (version)
  - Core dependencies: numpy, jax, flax, chex, gymnax
  - Optional dependencies (examples): distrax, optax, orbax, wandb, pillow, imageio, moviepy

**Documentation:**
- `mkdocs.yml` - MkDocs site generation
  - Theme: Material
  - Plugins: mkdocstrings with Python handler
  - Markdown extensions: toc, admonition, pymdownx

**Testing:**
- `tox.ini` - Test automation
  - Environments: py39, py310, py311, py312, py313
  - Test runner: pytest with coverage
  - Coverage target: `src/jaxued`

## Platform Requirements

**Development:**
- Python 3.9+ (3.11 recommended for testing environment)
- Conda (for environment management)
- FFmpeg (system package for video processing)
- Git (for installing Craftax from source)
- pip (for package installation)

**Production/Training:**
- Same as development
- GPU support optional (JAX can run on CPU but training is significantly slower)
- 8+ GB RAM recommended (for parallel environment simulation)

---

*Stack analysis: 2026-02-26*
