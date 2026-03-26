# Phase 1: Checkpoint - Research

**Researched:** 2026-03-11
**Domain:** Orbax checkpoint loading, GCS download, CNN-VAE decoder param extraction and verification
**Confidence:** HIGH — all library APIs verified by execution in the actual `jax_env` conda environment; all param tree structures confirmed from direct model inspection

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CKPT-01 | CNN-VAE Orbax checkpoint downloaded from GCS to `vae/checkpoints/cnn_vae/` | GCS download via `google-cloud-storage` SDK (must install) or `gcloud storage cp`; `gsutil` not present, GCS SDK not installed — install is the first action |
| CKPT-02 | Decoder params loaded via Orbax and verified against `CnnLstmDecoder.init()` param tree | `ocp.PyTreeCheckpointer().restore()` OR `ocp.CheckpointManager().restore(200000)` depending on checkpoint directory structure; extract `restored["params"]["decoder"]`; print tree to verify all 8 sub-keys present |
| CKPT-03 | `decode_fn(z_zeros)` runs without error and returns `(wall_logits, goal_logits, agent_logits)` each `(13, 13)` | `CnnLstmDecoder(latent_dim=64).apply({"params": decoder_params}, jnp.zeros((1,64)))` returns 3x `(1,13,13)`; squeeze to `(13,13)` for single-sample check |
</phase_requirements>

---

## Summary

Phase 1 is a verification-only phase: no production code is written, no adapter functions are implemented. The sole deliverable is a confirmed local checkpoint with verified decoder params. This front-loading is essential because every subsequent phase depends on knowing the exact checkpoint directory structure, the correct Orbax API variant to use, and that `decoder_params` maps correctly to `CnnLstmDecoder`.

The technical domain is narrow: GCS-to-local file transfer, Orbax checkpoint restoration, and Flax param tree inspection. All three have been verified in the actual `jax_env` environment (JAX 0.5.3, Flax 0.10.7, Orbax 0.10.3). The only unresolved unknown is the checkpoint directory structure, which determines whether `PyTreeCheckpointer` or `CheckpointManager` is the correct loader — this cannot be known until the files are downloaded and inspected.

The GCS download is the only external dependency and the only potential blocker. `gsutil` is not installed and `google-cloud-storage` is not in `jax_env`. Installing the Python SDK is a one-command fix. GCS authentication must be pre-configured (application default credentials or a service account key) — this is the only item that cannot be resolved programmatically and may require user action.

**Primary recommendation:** Install `google-cloud-storage` into `jax_env`, download the checkpoint, inspect the directory structure, pick the correct Orbax loader, extract `restored["params"]["decoder"]`, and run a single `decoder.apply()` call on `z_zeros`. If Orbax loading fails for any reason, fall back to pickle extraction as documented in STACK.md.

---

## Standard Stack

### Core (unchanged from existing pipeline)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| JAX | 0.5.3 | Numerical compute + JIT | Existing pipeline; verified on sideswipe/prowl with CUDA 12 |
| Flax linen | 0.10.7 | Neural network module (`CnnLstmDecoder`) | CNN-VAE model already defined; no changes |
| orbax-checkpoint | 0.10.3 | Load CNN-VAE checkpoint from local directory | Verified working; two API variants documented below |

### New for Phase 1

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `google-cloud-storage` | ~2.x (installable) | One-time GCS checkpoint download | Install into `jax_env` before Phase 1; NOT needed at training runtime |
| `gcloud storage cp` (CLI) | in `google_cloud_tpu/google-cloud-sdk/` | Alternative GCS download | Use if Python SDK auth fails; requires bootstrapping the SDK first |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `google-cloud-storage` Python SDK | `gcloud storage cp` CLI | CLI requires bootstrapping the bundled SDK; Python SDK is simpler if auth is configured |
| `ocp.PyTreeCheckpointer` | `ocp.CheckpointManager` | Depends on how checkpoint was saved; PyTreeCheckpointer works for flat directories, CheckpointManager for step-indexed directories; inspect first, choose second |
| Orbax loading | pickle extraction | Fallback only if Orbax fails; convert params to pickle with matching Python env |

**Installation (one-time, before Phase 1 starts):**
```bash
/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/pip install google-cloud-storage
```

pytest is NOT in `jax_env` — if the verification script uses pytest, it must be installed:
```bash
/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/pip install pytest
```

---

## Architecture Patterns

### Recommended Project Structure (Phase 1 additions)

```
vae/
├── checkpoints/
│   └── cnn_vae/           # NEW — downloaded from GCS run10/step200000
│       ├── 200000/        # if saved with CheckpointManager (step-indexed)
│       │   └── ...        # Orbax internal files
│       └── ...            # OR flat Orbax files if saved with PyTreeCheckpointer
├── cnn_vae_model.py       # UNCHANGED — CnnLstmDecoder defined here
└── ...
```

### Pattern 1: Determine Loader by Inspecting Directory Structure

**What:** Before writing any loading code, inspect `vae/checkpoints/cnn_vae/` to decide between `PyTreeCheckpointer` and `CheckpointManager`.

**When to use:** Always — the correct loader depends on how the checkpoint was saved.

**Example:**
```bash
# Source: STACK.md (verified against actual orbax 0.10.3)
ls vae/checkpoints/cnn_vae/
# If you see numbered subdirectory (e.g., 200000/): use CheckpointManager
# If you see orbax metadata files directly (ocdbt*, manifest, _METADATA): use PyTreeCheckpointer
```

### Pattern 2: PyTreeCheckpointer (flat directory)

**What:** Use when checkpoint is a flat directory (no step-indexed subdirs).

**Example:**
```python
# Source: STACK.md — verified working in orbax 0.10.3
import orbax.checkpoint as ocp

checkpointer = ocp.PyTreeCheckpointer()
restored = checkpointer.restore('vae/checkpoints/cnn_vae/')
# restored is a nested dict: {"params": {"encoder": ..., "mean_layer": ...,
#                                         "logvar_layer": ..., "decoder": ...}}
decoder_params = restored["params"]["decoder"]
```

### Pattern 3: CheckpointManager (step-indexed directory)

**What:** Use when checkpoint directory contains step subdirectories (most likely for run10, which was saved with standard Orbax training pattern).

**Example:**
```python
# Source: STACK.md — verified working in orbax 0.10.3
import orbax.checkpoint as ocp

mgr = ocp.CheckpointManager(
    'vae/checkpoints/cnn_vae/',
    options=ocp.CheckpointManagerOptions()
)
restored = mgr.restore(200000)
decoder_params = restored["params"]["decoder"]
```

### Pattern 4: Decoder-Only Extraction and Verification

**What:** Extract `decoder_params`, instantiate `CnnLstmDecoder`, run `decoder.apply()` on `z_zeros`.

**Example:**
```python
# Source: STACK.md + cnn_vae_model.py direct inspection
import jax.numpy as jnp
from vae.cnn_vae_model import CnnLstmDecoder
import jax

# Step 1: Print param tree to verify structure
print(jax.tree_util.tree_map(lambda x: x.shape, decoder_params))
# Expected keys: dec_lstm, dec_proj, dec_conv1, dec_conv2, dec_conv3,
#                wall_head, goal_head, agent_head

# Step 2: Run decode on z_zeros
decoder = CnnLstmDecoder(latent_dim=64)
z_zeros = jnp.zeros((1, 64))
wall_logits, goal_logits, agent_logits = decoder.apply(
    {"params": decoder_params}, z_zeros
)
# Expected shapes: each (1, 13, 13)
print(wall_logits.shape, goal_logits.shape, agent_logits.shape)
# Squeeze for CKPT-03 single-sample check:
wall_logits_1 = wall_logits[0]   # (13, 13)
```

### Anti-Patterns to Avoid

- **Using `pickle.load()` for CNN-VAE checkpoint:** CNN-VAE was saved with Orbax, not pickle. Pickle will raise or return garbage. Only fallback to pickle if you manually extract and re-save the params in a separate step.
- **Loading the full VAE params tree into any persistent variable:** Extract only `params["decoder"]`; discard encoder, mean_layer, logvar_layer. They are unused for inference.
- **Calling `orbax.restore()` inside a jitted function:** Orbax is a Python/host operation; it cannot be traced by JAX. Always restore at Python-level startup.
- **Skipping the directory structure inspection:** Using `CheckpointManager` on a PyTreeCheckpointer-format directory (or vice versa) produces cryptic errors. Inspect first.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Checkpoint deserialization | Custom msgpack/protobuf parser | `ocp.PyTreeCheckpointer().restore()` | Orbax handles nested pytrees, dtypes, and sharding transparently |
| GCS download | Manual HTTP requests or boto3 | `google-cloud-storage` SDK or `gcloud storage cp` | Auth, retries, partial downloads, and blob enumeration already handled |
| Param tree inspection | Custom tree walker | `jax.tree_util.tree_map(lambda x: x.shape, params)` | One-liner that shows all key shapes recursively |

---

## Common Pitfalls

### Pitfall 1: GCS Auth Not Configured

**What goes wrong:** `google.cloud.storage.Client()` raises `DefaultCredentialsError` because application default credentials are not set up on the machine.

**Why it happens:** GCS requires authentication. The compute nodes (sideswipe, prowl) may not have `gcloud auth application-default login` configured, and no service account key may be present.

**How to avoid:** Before installing the SDK, verify auth with `gcloud auth list` (if gcloud CLI is bootstrapped from `google_cloud_tpu/google-cloud-sdk/`). If auth is not configured, request the checkpoint file be transferred via scp/rsync from a machine with access, or use a service account key JSON file set via `GOOGLE_APPLICATION_CREDENTIALS` env var.

**Warning signs:**
- `google.auth.exceptions.DefaultCredentialsError` on `storage.Client()` instantiation
- `403 Forbidden` on bucket access

### Pitfall 2: Orbax Checkpoint Directory Structure Unknown

**What goes wrong:** Using `CheckpointManager.restore(200000)` on a flat PyTreeCheckpointer-format directory raises `FileNotFoundError` looking for `200000/` subdirectory. Using `PyTreeCheckpointer` on a step-indexed directory also fails.

**Why it happens:** The CNN-VAE was trained in a separate repo (`cnn-vae-maze`). The training script's checkpoint saving pattern determines the directory structure — not known until download.

**How to avoid:** Always `ls vae/checkpoints/cnn_vae/` after download. If you see an integer subdirectory (like `200000`), use `CheckpointManager`. If you see orbax metadata files directly (`_METADATA`, `ocdbt.process_0`, or similar), use `PyTreeCheckpointer`.

**Warning signs:**
- `FileNotFoundError: .../200000/` when trying `CheckpointManager`
- `KeyError: 'params'` when the checkpoint tree is not nested as expected

### Pitfall 3: Param Key Mismatch Between Checkpoint and CnnLstmDecoder

**What goes wrong:** The checkpoint param tree has keys that don't match what `CnnLstmDecoder.apply()` expects. This causes `KeyError` on apply, or silently uses wrong parameters if keys partially match.

**Why it happens:** The param tree naming is determined by Flax module name arguments. `CnnLstmVAE` instantiates `CnnLstmDecoder(name='decoder')`, so the checkpoint tree is `params/decoder/...`. Loading with wrong key path (e.g., passing full `restored["params"]` instead of `restored["params"]["decoder"]`) fails.

**How to avoid:** Print the full param tree structure after restore:
```python
print(jax.tree_util.tree_map(lambda x: x.shape, restored["params"]))
```
Expected top-level keys: `encoder`, `mean_layer`, `logvar_layer`, `decoder`.
Then pass only `restored["params"]["decoder"]` to `CnnLstmDecoder.apply({"params": ...}, z)`.

**Warning signs:**
- `KeyError: 'dec_lstm'` — passing wrong subtree to decoder.apply()
- Output is all zeros or NaN after apply — wrong params silently applied

### Pitfall 4: Installing google-cloud-storage on blaze (Head Node)

**What goes wrong:** If the download script is run on the `blaze` head node with CUDA 11.7, importing JAX afterward to verify the checkpoint may fail due to incompatible CUDA versions.

**Why it happens:** The MEMORY.md explicitly states: do NOT run training on blaze (CUDA 11.7 only). However, the download itself (no GPU needed) can run on blaze. The verification step (loading params + running `decoder.apply()`) should run on sideswipe or prowl.

**How to avoid:** Keep the download step and the verification step separate. Download can happen anywhere with internet access and GCS auth. Verification must run on sideswipe or prowl via a `ssh sideswipe "cd ... && python verify_checkpoint.py"` or a tmux session.

---

## Code Examples

Verified patterns from direct codebase inspection and prior execution in `jax_env`:

### Full Phase 1 Verification Script Pattern

```python
# Source: STACK.md (all patterns verified in orbax 0.10.3 + jax 0.5.3)
# Run on sideswipe or prowl, NOT on blaze

import sys
sys.path.insert(0, '/cs/student/project_msc/2025/csml/gmaralla/superintelligence')

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from vae.cnn_vae_model import CnnLstmDecoder

CKPT_DIR = 'vae/checkpoints/cnn_vae'

# --- Step 1: Inspect directory structure ---
import os
entries = os.listdir(CKPT_DIR)
print("Checkpoint dir contents:", entries)

has_step_dirs = any(e.isdigit() for e in entries)

# --- Step 2: Load checkpoint ---
if has_step_dirs:
    # CheckpointManager path: step-indexed directory (most likely for run10)
    mgr = ocp.CheckpointManager(CKPT_DIR, options=ocp.CheckpointManagerOptions())
    restored = mgr.restore(200000)
else:
    # PyTreeCheckpointer path: flat directory
    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(CKPT_DIR)

# --- Step 3: Extract decoder params (CKPT-02) ---
decoder_params = restored["params"]["decoder"]
print("Decoder param keys:", list(decoder_params.keys()))
# Expected: dec_lstm, dec_proj, dec_conv1, dec_conv2, dec_conv3,
#           wall_head, goal_head, agent_head
print("Shapes:", jax.tree_util.tree_map(lambda x: x.shape, decoder_params))

# --- Step 4: Run decode_fn(z_zeros) (CKPT-03) ---
decoder = CnnLstmDecoder(latent_dim=64)
z_zeros = jnp.zeros((1, 64))
wall_logits, goal_logits, agent_logits = decoder.apply(
    {"params": decoder_params}, z_zeros
)
print("Output shapes:", wall_logits.shape, goal_logits.shape, agent_logits.shape)
# Expected: (1, 13, 13) each

# --- Step 5: Verify CKPT-03 single-sample shapes ---
assert wall_logits.shape == (1, 13, 13), f"wall_logits shape: {wall_logits.shape}"
assert goal_logits.shape == (1, 13, 13), f"goal_logits shape: {goal_logits.shape}"
assert agent_logits.shape == (1, 13, 13), f"agent_logits shape: {agent_logits.shape}"

# Squeeze for the documented CKPT-03 requirement (13,13) single sample
wl = wall_logits[0]   # (13, 13)
gl = goal_logits[0]   # (13, 13)
al = agent_logits[0]  # (13, 13)
assert wl.shape == (13, 13)
print("PASS: decode_fn(z_zeros) → (wall_logits, goal_logits, agent_logits) each (13, 13)")
```

### GCS Download Pattern

```bash
# Source: STACK.md — one-time download before Phase 1

# Install SDK (if not installed)
/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/pip install google-cloud-storage

# Download step 200000 from GCS (adjust bucket/path to actual run10 location)
/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python -c "
from google.cloud import storage
import os

client = storage.Client()
bucket = client.bucket('cnn-vae-maze-checkpoints')
prefix = 'run10/200000/'
local_dir = 'vae/checkpoints/cnn_vae/'

os.makedirs(local_dir, exist_ok=True)
for blob in bucket.list_blobs(prefix=prefix):
    relative = blob.name[len(prefix):]
    local_path = os.path.join(local_dir, relative)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)
    print(f'Downloaded: {local_path}')
"
```

### Fallback: Pickle Extraction (if Orbax fails)

```python
# Source: PITFALLS.md recovery strategy
# Use only if ocp.PyTreeCheckpointer and ocp.CheckpointManager both fail

# In the environment where CNN-VAE was trained (matching Orbax version):
import orbax.checkpoint as ocp
import pickle

restored = ocp.PyTreeCheckpointer().restore('vae/checkpoints/cnn_vae/')
decoder_params = restored["params"]["decoder"]

import pickle
with open('vae/checkpoints/cnn_vae_decoder_params.pkl', 'wb') as f:
    pickle.dump(decoder_params, f)

# Then in jax_env:
with open('vae/checkpoints/cnn_vae_decoder_params.pkl', 'rb') as f:
    decoder_params = pickle.load(f)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `ocp.Checkpointer` (generic) | `ocp.PyTreeCheckpointer` or `ocp.CheckpointManager` | Orbax 0.4+ | PyTreeCheckpointer no longer requires a target pytree for dict-format checkpoints; simpler for cross-run loading |
| `gsutil cp` (standalone binary) | `gcloud storage cp` or `google-cloud-storage` SDK | gcloud SDK 400+ | gsutil is bundled in the Google Cloud SDK but requires bootstrap; Python SDK is simpler for scripted one-shot downloads |

**Deprecated/outdated:**
- `gsutil` standalone binary: Not present in this environment; Google Cloud SDK (`google_cloud_tpu/google-cloud-sdk/`) is in bootstrap state (only `bin/bootstrapping/` exists, not `bin/gsutil`).
- `ocp.StandardCheckpointer` without target: Requires matching pytree at load time; fails without it. Use `PyTreeCheckpointer` for targetless restore.

---

## Open Questions

1. **Actual checkpoint directory structure on GCS**
   - What we know: The checkpoint is at `gs://cnn-vae-maze-checkpoints/run10/` at step 200000; the CNN-VAE training used standard Orbax patterns.
   - What's unclear: Whether the GCS path is `run10/200000/` (step-indexed by CheckpointManager) or `run10/` containing a flat Orbax checkpoint.
   - Recommendation: Download first, then `ls vae/checkpoints/cnn_vae/`. STACK.md covers both variants with verified code. No blocking decision needed before download.

2. **GCS authentication on sideswipe/prowl**
   - What we know: `google-cloud-storage` is not installed; `gsutil` is not present; the bundled Google Cloud SDK is in bootstrap state.
   - What's unclear: Whether application default credentials are configured on the nodes, or whether a service account key is available.
   - Recommendation: Test auth by attempting `gcloud auth list` after bootstrapping the bundled SDK, or attempt the Python SDK download and handle `DefaultCredentialsError` by requesting manual auth setup. If auth cannot be configured, the checkpoint must be transferred via scp from a machine that has GCS access.

3. **Actual GCS bucket name and path**
   - What we know: From STACK.md research: `gs://cnn-vae-maze-checkpoints/run10/` with step 200000.
   - What's unclear: The exact bucket name is from prior research (not directly verified against the actual GCS project). This should be confirmed before writing the download script.
   - Recommendation: Confirm bucket name and path in Phase 1 Plan 1 before running download.

---

## Sources

### Primary (HIGH confidence — verified by execution in `jax_env`)
- `vae/cnn_vae_model.py` — `CnnLstmDecoder` architecture, `CnnLstmVAE` module naming contract (`name='decoder'` → `params/decoder/...`), `latent_dim=64` default
- `.planning/research/STACK.md` (2026-03-11) — Orbax 0.10.3 API patterns verified by running in `jax_env`; both `PyTreeCheckpointer` and `CheckpointManager` load patterns; param tree structure confirmed from `decoder.init()`
- `.planning/research/PITFALLS.md` (2026-03-11) — Pitfall 2 (param key mismatch), Pitfall 6 (Orbax API version mismatch) — all from direct code analysis
- `.planning/research/SUMMARY.md` (2026-03-11) — GCS download blocker confirmed; `gsutil` not present, `google-cloud-storage` not installed
- Environment verification: `orbax-checkpoint==0.10.3`, `jax==0.5.3`, `flax==0.10.7` confirmed installed in `jax_env`

### Secondary (MEDIUM confidence — codebase inspection, not executed)
- `.planning/STATE.md` — GCS auth blocker noted; `google-cloud-storage` must be installed before Phase 1
- `vae/cnn_vae_losses.py` — `apply_wall_mask()` exists and is importable; not needed for Phase 1 but confirms the module structure

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all library versions confirmed installed; API patterns verified by prior execution
- Architecture: HIGH — param tree structure derived directly from `cnn_vae_model.py` module naming convention and prior `decoder.init()` execution
- GCS download: MEDIUM — `google-cloud-storage` install is confirmed installable; GCS auth status on sideswipe/prowl is unverified until attempted

**Research date:** 2026-03-11
**Valid until:** 2026-04-10 (Orbax and JAX are stable; GCS auth situation may change)
