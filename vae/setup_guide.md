# CLUTTR VAE — GCP TPU / Local GPU Setup Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  train_vae.py                        │
│  ┌───────────┐  ┌───────────┐  ┌──────────────────┐│
│  │ CluttrVAE  │  │ IOManager │  │ExperimentTracker ││
│  │ (model)    │  │ (gcp_io)  │  │(Vertex AI)       ││
│  └───────────┘  └─────┬─────┘  └────────┬─────────┘│
│                       │                  │           │
│         ┌─────────────┴──────────────────┘           │
│         │                                            │
│    platform: "local"        platform: "gcp"          │
│    ├─ Local filesystem      ├─ GCS bucket            │
│    └─ No-op tracker         ├─ Vertex AI Experiments │
│                             └─ TPU data parallelism  │
└─────────────────────────────────────────────────────┘
```

## Files

| File | Purpose |
|---|---|
| `train_vae.py` | Main training script (replaces your original) |
| `vae_train_config.yml` | Unified config — switch `platform` between `"local"` and `"gcp"` |
| `gcp_io.py` | Transparent I/O: local filesystem ↔ GCS bucket |
| `experiment_tracker.py` | Vertex AI Experiments wrapper (no-op when local) |
| `utils.py` | Your existing evaluation metrics (unchanged) |

## Quick Start

### Local (UCL GPUs) — Nothing changes
```bash
# In vae_train_config.yml, ensure:
#   platform: "local"
python train_vae.py
```

### GCP TPU
```bash
# 1. SSH into your TPU VM
gcloud compute tpus tpu-vm ssh my-tpu-vm --zone=us-central1-a

# 2. Install dependencies
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install flax optax tqdm pyyaml matplotlib google-cloud-storage google-cloud-aiplatform

# 3. Edit config
#   platform: "gcp"
#   gcp_project: "your-project-id"
#   gcp_bucket: "your-bucket-name"

# 4. Run
python train_vae.py
```

## Key Changes Explained

### 1. Multi-device data parallelism (TPU cores)

A TPU v2-8 has **8 cores**. The script uses JAX's `Mesh` + `NamedSharding` API:

```python
mesh = Mesh(np.array(jax.devices()), axis_names=("batch",))
data_sharding = NamedSharding(mesh, P("batch"))  # shard batch dim
```

Each training step:
- Samples a **global batch** of size `batch_size × n_devices`
- Shards it across devices along the batch dimension
- Each device computes gradients on its shard
- `jax.lax.pmean(grads, axis_name="batch")` averages gradients across all devices

**On a single GPU**, `n_devices=1`, so this is a no-op — same behavior as before.

### 2. Data loading from GCS

`IOManager` in `gcp_io.py` provides a unified API:

```python
io = IOManager(config)
data = io.load_npy("datasets/training_200k_samples.npy")
```

- When `platform: "local"`: reads from `{working_path}/{vae_folder}/datasets/...`
- When `platform: "gcp"`: reads from `gs://{gcp_bucket}/{gcp_bucket_prefix}/datasets/...`

Your existing bucket structure should match:
```
gs://your-bucket/
  vae/
    datasets/
      training_200k_samples.npy
      validation_samples.npy
    checkpoints_200k_samples/
      ...
```

### 3. Checkpoints & plots saved to GCS

Same `IOManager` handles saves:
```python
io.save_pickle({"params": state.params, "step": step}, "checkpoints/step_1000.pkl")
io.save_figure(fig, "plot.png")
```

### 4. Vertex AI Experiment Tracking

When `enable_vertex_tracking: true` and `platform: "gcp"`, the script logs to **Vertex AI Experiments**:

- **Hyperparameters** (logged once at start)
- **Time-series metrics** (loss curves, accuracy, every 100 steps)
- **Summary metrics** (final values at end)

View in GCP Console: **Vertex AI → Experiments → cluttr-vae-training**

You get:
- Comparison across runs (different hyperparameters)
- Loss curve visualization
- Metric tables

When running locally, a no-op tracker is used — zero overhead.

### 5. Batch size scaling

The config `batch_size` is now **per-device**. On a TPU v2-8 with `batch_size: 32`:
- Global batch = 32 × 8 = 256 samples per step
- You may want to scale learning rate accordingly (linear scaling rule)

## GCP Bucket Setup

```bash
# Create bucket (if not already done)
gsutil mb -p YOUR_PROJECT -l us-central1 gs://your-bucket-name

# Upload datasets
gsutil cp training_200k_samples.npy gs://your-bucket-name/vae/datasets/
gsutil cp validation_samples.npy gs://your-bucket-name/vae/datasets/
```

## TRC Grant: Multiple TPU VMs

With the TRC grant you typically get access to TPU VMs (e.g., v2-8 or v3-8). Each VM has 8 cores.

**Single VM (recommended to start):**
- The script handles all 8 cores automatically via `jax.devices()`

**Multiple VMs (multi-host):**
- For multi-host TPU pods (e.g., v3-32 = 4 VMs × 8 cores), you'd need:
  - JAX's multi-process setup with `jax.distributed.initialize()`
  - This is more complex; start with single-VM first
  - If needed later, add this before `setup_devices()`:
    ```python
    jax.distributed.initialize(
        coordinator_address="COORDINATOR_IP:PORT",
        num_processes=NUM_VMS,
        process_id=THIS_VM_ID,
    )
    ```

## Monitoring Runs

### Vertex AI Console
Navigate to: **GCP Console → Vertex AI → Experiments**

### From CLI
```bash
# List experiments
gcloud ai experiments list --region=us-central1

# List runs in an experiment
gcloud ai experiments runs list --experiment=cluttr-vae-training --region=us-central1
```

### Download checkpoints
```bash
gsutil cp gs://your-bucket/vae/checkpoints_200k_samples/checkpoint_50000.pkl .
```

## Troubleshooting

| Issue | Fix |
|---|---|
| `jax.devices()` shows CPU | JAX TPU not installed correctly. Reinstall: `pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html` |
| GCS permission denied | Ensure the TPU VM service account has Storage Admin role |
| Vertex AI not found | `pip install google-cloud-aiplatform` and ensure Vertex AI API is enabled in your project |
| OOM on TPU | Reduce `batch_size` in config |
| Slow data loading from GCS | Data is loaded once at startup into host RAM, so this is a one-time cost |