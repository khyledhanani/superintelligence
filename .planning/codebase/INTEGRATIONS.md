# External Integrations

**Analysis Date:** 2026-03-23

## APIs & External Services

**LLM/AI Models:**
- Claude (Anthropic) - Primary LLM provider for maze generation
  - SDK/Client: Claude Code API (configured in `llm/config.yaml`)
  - Config location: `llm/config.yaml` (provider: "claude-code", model: "sonnet")
  - Alternative providers: Ollama, OpenRouter (via `openrouter.ai/api/v1`)
  - Implementation: `llm/maze_generator.py` - Prompts Claude to generate mazes with metric injection

**Evolutionary Strategies:**
- EvoSax - CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
  - SDK/Client: `evosax` package (supports both old and new API versions)
  - Implementation: `vae/cmaes_manager.py` - Wrapper managing CMA-ES state for VAE latent space search
  - Used in: `examples/maze_plr.py` with `--use_cmaes` flag for latent-space curriculum generation
  - Configuration: population_size = num_train_envs (32), latent_dim (64), sigma_init (1.0)

## Data Storage

**Databases:**
- Not applicable - No relational database usage

**File Storage:**
- Google Cloud Storage (GCS)
  - Bucket: `ucl-ued-project-bucket`
  - Paths: `accel/`, `gcs_artifacts/agent/`, `gcs_artifacts/buffer/`
  - Connection: GOOGLE_APPLICATION_CREDENTIALS (standard GCP auth)
  - Client: `google.cloud.storage` (with `gcsfs` fallback for filesystem operations)
  - Usage locations:
    - `examples/maze_plr.py` - Uploads checkpoints, configs, buffer dumps to GCS
    - `vae/compare_accel_vs_cmaes.py` - Downloads/uploads comparison plots
    - `llm/config.yaml` - References `gcs_artifacts/buffer/buffer_dump_final.npz` and agent dirs

**Local Filesystem:**
- Checkpoint storage: `./checkpoints/<run_name>/<seed>/models/` (Orbax CheckpointManager)
- Results storage: `./results/` - Evaluation outputs (npz format)
- Logs: `./logs/` - Per-seed training logs
- VAE artifacts: `vae/checkpoints/` (configured in `vae_train_config.yml`)
- Temp cache: `/tmp/jax_cache/` (JAX compilation cache)

**Caching:**
- JAX compilation cache: `/tmp/jax_cache/` (JAX_COMPILATION_CACHE_DIR)
- WandB cache: `/tmp/wandb/` (WANDB_DIR)

## Authentication & Identity

**Auth Provider:**
- Google Cloud (GCP) native
  - Implementation: `google.cloud.storage` and `gcsfs` clients
  - Credentials: GOOGLE_APPLICATION_CREDENTIALS environment variable (service account key)
  - Fallback: `gcloud` CLI if Python client unavailable (in `examples/maze_plr.py`)
  - Note: Credentials file should NOT be committed (listed in `.gitignore`)

## Monitoring & Observability

**Error Tracking:**
- Sentry SDK 2.52.0 (installed but integration not visible in sample code - passive monitoring)

**Logs:**
- WandB (Weights & Biases) - Primary experiment tracking
  - Project names: `JAXUED_50K`, `JAXUED_COMPARISON`, `JAXUED_50K_VAE_TRAIN`
  - Group organization: `accel-baseline`, `pca-cmaes-accel`, `cmaes-cnn-vae-accel`, etc.
  - Implementation: `wandb.init()`, `wandb.log()` in all training scripts
  - Metrics logged:
    - `num_updates`, `num_env_steps` - Training progress
    - `solve_rate/*` - Performance on test levels
    - `level_sampler/*` - Curriculum statistics
    - `agent/*` - Agent loss, value loss, entropy
    - `return/*`, `eval_ep_lengths/*` - Episode statistics
    - Custom images: level visualizations, highest-scoring/highest-weighted levels
    - Videos: training-time environment rollouts
  - Config: `llm/config.yaml` defines metrics to inject into LLM prompts (per_step_entropy, scalar_regret, action_sequence, etc.)

- Local file logging: `./logs/<run_name>_seed<N>.log` (redirected stdout/stderr)

## CI/CD & Deployment

**Hosting:**
- Google Cloud Platform (GCP)
  - TPU VM: `cma-es-v4` (zone: `us-central2-b`)
  - GPU nodes: albacore (4070 Ti), smew (3090 Ti), canada (3090 Ti) with CUDA 13.1
  - Head node: NFS-shared home across all machines

**CI Pipeline:**
- Not applicable - No automated CI detected (manual launch scripts only)

**Deployment Method:**
- Manual script execution via SSH to remote machines
  - Launch scripts: `examples/launch_*.sh`
  - Commands stored in memory: e.g., `~/google-cloud-sdk/bin/gcloud compute tpus tpu-vm ssh cma-es-v4`

## Environment Configuration

**Required env vars:**
- `GOOGLE_APPLICATION_CREDENTIALS` - Path to GCP service account JSON (for GCS access)
- `WANDB_DIR` - WandB cache directory (set to `/tmp/wandb` in launch scripts)
- `JAX_COMPILATION_CACHE_DIR` - JAX JIT cache (set to `/tmp/jax_cache` in launch scripts)
- `LD_LIBRARY_PATH` - CUDA library path for GPU nodes (must include conda env lib path)
- `WANDB_PROJECT` - Optional, overrides config project name

**Secrets location:**
- `.env` files (present but contents never read per security policy)
- GCP credentials: Service account JSON referenced by `GOOGLE_APPLICATION_CREDENTIALS`
- Conda environment provides pip packages with pinned versions

## Webhooks & Callbacks

**Incoming:**
- Not applicable - No webhook endpoints

**Outgoing:**
- WandB logging callbacks: `wandb.log()` called on every training step
- LLM API callbacks: `llm/maze_generator.py` calls Claude API to generate mazes (synchronous, non-webhook)

## Data Flow - GCS Integration

**Upload Path:**
1. `examples/maze_plr.py` trains agent with PLR curriculum
2. Orbax CheckpointManager saves weights to `./checkpoints/<run_name>/<seed>/`
3. `save_plr_buffer()` function exports level buffer as `.npy` (VAE token format)
4. `upload_to_gcs()` in `examples/maze_plr.py` copies to `gs://ucl-ued-project-bucket/accel/buffer_dumps/<run_name>/<seed>/`
5. WandB artifacts link checkpoint locations

**Download Path (for evaluation):**
1. `llm/maze_generator.py` loads reference buffer from `gcs_artifacts/buffer/buffer_dump_final.npz`
2. `llm/agent_evaluator.py` loads agent checkpoint from `gcs_artifacts/agent/cmaes_vae_beta2.0_seed0_198`
3. Local evaluation scores computed, then uploaded to GCS if requested

---

*Integration audit: 2026-03-23*
