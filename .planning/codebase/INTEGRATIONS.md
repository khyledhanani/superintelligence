# External Integrations

**Analysis Date:** 2026-03-11

## APIs & External Services

**Google Cloud Storage (GCS):**
- Service: Cloud Storage bucket for checkpoint and buffer persistence
- What it's used for: Saving PPO checkpoints, VAE weights, PLR buffers, generated level datasets
- SDK/Client: `google-cloud-storage==3.9.0` + `gcsfs` (no pinned version)
- Auth: GCP credentials (gcloud CLI or GOOGLE_APPLICATION_CREDENTIALS env var)
- Implementation:
  - Primary: `google.cloud.storage.Client()` in `examples/maze_plr.py` lines 348-356
  - Fallback: `gcloud storage cp` subprocess command if Python client fails
  - Paths: `gs://<gcs_bucket>/<gcs_prefix>/checkpoints/<run_name>/<seed>/...`
  - Bucket config: `--gcs_bucket` and `--gcs_prefix` command-line arguments
  - Example bucket: `"ucl-ued-project-bucket"` in launch scripts

**Weights & Biases (wandb):**
- Service: Experiment tracking, metric logging, hyperparameter management
- What it's used for: Recording training metrics, level visualizations, buffer contents, run metadata
- SDK/Client: `wandb==0.24.2`
- Auth: WANDB_API_KEY env var (optional if offline mode used)
- Implementation:
  - Init: `wandb.init(config=config, project=project, group=config["run_name"], tags=tags)` (line 487)
  - Config sync: `config = wandb.config` (line 488) - overrides local config with W&B settings
  - Custom metrics: `wandb.define_metric("num_updates")`, `wandb.define_metric("num_env_steps")`
  - Step metrics: `wandb.define_metric("solve_rate/*", step_metric="num_updates")` (line 492)
  - Logging: All training losses, rewards, level stats sent to `group=<run_name>`
  - Offline mode: `WANDB_DIR=/tmp/wandb` in launch scripts to avoid disk bloat
  - Config: `--project` flag sets wandb project (e.g., "JAXUED_COMPARISON", "JAXUED_VAE_COMPARISON")

## Data Storage

**Databases:**
- Not used. No SQL/NoSQL databases integrated.

**File Storage:**
- **Local filesystem only:**
  - Checkpoints: `./checkpoints/<run_name>/<seed>/models/<update_step>/`
  - Results: `./results/` directory
  - Logs: `./logs/` directory
  - Buffer dumps: VAE token format (`.npy`) + metadata (`.npz`) saved locally before GCS upload
- **GCS (optional cloud persistence):**
  - Fallback destination if `--gcs_bucket` flag is set
  - URI scheme: `gs://bucket/prefix/...`
  - Upload mechanism: Python client or gcloud CLI

**Caching:**
- Not used. No Redis or in-memory caches configured.

## Authentication & Identity

**Auth Provider:**
- Custom (none used for API authentication)
- GCP service account credentials for GCS access
  - Method: Implicit auth via gcloud CLI (GOOGLE_APPLICATION_CREDENTIALS env var if needed)
- wandb API key:
  - Method: Implicit auth via WANDB_API_KEY env var or ~/.wandb/settings

## Monitoring & Observability

**Error Tracking:**
- `sentry-sdk==2.52.0` - Installed but minimal/no usage in primary code
- Primarily relies on wandb metrics for monitoring

**Logs:**
- **File logs:** Training loop outputs to console (captured in `logs/phase5.log` via `tail -f`)
- **wandb metrics:** Real-time training curves, sample efficiency, level diversity
- **Manual inspection:** VAE diagnostics and latent analysis scripts generate plots saved locally or to GCS

## CI/CD & Deployment

**Hosting:**
- Google Cloud Platform (TPU VMs)
  - Nodes: sideswipe, prowl (both have CUDA 12 + JAX support)
  - Head node: blaze (CUDA 11.7, not suitable for JAX 0.6.2 training)

**CI Pipeline:**
- None detected. Manual experiment execution via bash launch scripts.
- Test framework: pytest + tox (defined in `tox.ini` for regression testing)
  - Run: `pytest -v -s --cov=src/jaxued`
  - Envs: Python 3.9-3.13

## Environment Configuration

**Required env vars:**
- `OPENBLAS_NUM_THREADS=4` - OpenBLAS threading (set in scripts to prevent contention)
- `MKL_NUM_THREADS=4` - MKL threading (set in launch scripts)
- `WANDB_DIR=/tmp/wandb` - Offline wandb dir to avoid local disk bloat
- `GOOGLE_APPLICATION_CREDENTIALS` - (optional) Path to GCP service account JSON for GCS access
- `WANDB_API_KEY` - (optional) wandb API key for syncing to cloud

**Optional env vars:**
- `JAX_PLATFORMS=gpu` - Force JAX to use GPU (auto-detected if available)
- None of these should be hardcoded; all passed at runtime or in launch scripts

**Secrets location:**
- `.env` files: None present in repo (safe practice)
- GCP credentials: External to repo (provided via gcloud CLI auth)
- wandb key: External to repo (provided via ~/.wandb or WANDB_API_KEY)

## Webhooks & Callbacks

**Incoming:**
- None detected.

**Outgoing:**
- wandb callbacks: Implicit via `wandb.log()` calls in training loop
  - Triggered every `eval_freq` PPO updates
  - Data sent: metrics dict with losses, returns, level stats

## External Environment Integration

**VAE Models:**
- Location: `vae/runs/<run_id>/checkpoints/checkpoint_*.pkl`
- Loading: `vae_model.py` defines model architecture; checkpoints loaded via Flax/Orbax
- Path config: `--vae_checkpoint_path` and `--vae_config_path` flags in training

**Level Dataset:**
- Source: maze-dataset 1.4.2 PyPI package
  - Used for: Prefab maze set for evaluation (11 standard mazes)
  - Integration: Embedded in JAX pipeline, not external API

**Special TPU Integration:**
- JAX XLA compiler targets TPU hardware directly
- No special webhooks; all computation on-device with JAX jit()
- Output: Checkpoints and metrics sent to GCS/wandb post-training

---

*Integration audit: 2026-03-11*
