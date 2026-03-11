---
phase: 01-checkpoint
plan: 01
subsystem: infra
tags: [gcs, orbax, cnn-vae, jax, flax, checkpoint, verification]

# Dependency graph
requires: []
provides:
  - "CNN-VAE Orbax checkpoint at vae/checkpoints/cnn_vae/ (run10, step 200000)"
  - "decoder_params dict with all 8 CnnLstmDecoder sub-keys verified"
  - "Confirmed Orbax loader pattern: PyTreeCheckpointer on default/ subdir"
  - "Standalone verification script at scripts/verify_checkpoint.py"
affects: [02-adapter, 03-integration, 04-training]

# Tech tracking
tech-stack:
  added: [google-cloud-storage~3.9.0]
  patterns:
    - "Orbax StandardCheckpointHandler saves to default/ subdir — use PyTreeCheckpointer on absolute path to cnn_vae/default/"
    - "Checkpoint must use absolute path with orbax.checkpoint (relative paths fail for tensorstore)"

key-files:
  created:
    - scripts/verify_checkpoint.py
  modified:
    - .gitignore

key-decisions:
  - "Orbax loader: PyTreeCheckpointer on cnn_vae/default/ subdir (StandardCheckpointHandler format detected from presence of default/ dir and absence of digit-only step dirs)"
  - "GCS credentials: legacy_credentials adc.json with explicit project='open-endedness-personal' (no ADC file; GOOGLE_APPLICATION_CREDENTIALS workaround not needed — app code uses same env)"
  - "Download scope: run10/200000/ step contents placed directly into vae/checkpoints/cnn_vae/ (not step-indexed)"

patterns-established:
  - "Pattern 1: Always use absolute paths with ocp.PyTreeCheckpointer (relative paths raise ValueError in tensorstore layer)"
  - "Pattern 2: Inspect checkpoint dir for digit-only subdirs to choose loader; if 'default/' present without digit dirs, use PyTreeCheckpointer on cnn_vae/default/"
  - "Pattern 3: Extract only params['decoder'] for CnnLstmDecoder.apply() — discard encoder, mean_layer, logvar_layer"

requirements-completed: [CKPT-01, CKPT-02, CKPT-03]

# Metrics
duration: 5min
completed: 2026-03-11
---

# Phase 1 Plan 01: Checkpoint Verification Summary

**CNN-VAE checkpoint downloaded from GCS (run10/step200000) and verified: PyTreeCheckpointer loads decoder_params with all 8 CnnLstmDecoder sub-keys, decode_fn(z_zeros) returns (1,13,13) logits with no NaN**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-11T20:01:46Z
- **Completed:** 2026-03-11T20:06:08Z
- **Tasks:** 2
- **Files modified:** 2 (.gitignore, scripts/verify_checkpoint.py)

## Accomplishments

- Downloaded 11-file CNN-VAE checkpoint (~35MB) from gs://cnn-vae-maze-checkpoints/run10/200000/ to vae/checkpoints/cnn_vae/
- Confirmed Orbax checkpoint format: StandardCheckpointHandler with default/ subdir; PyTreeCheckpointer on absolute path to default/ subdir is the correct loader
- Verified all 8 decoder sub-keys present: dec_lstm, dec_proj, dec_conv1, dec_conv2, dec_conv3, wall_head, goal_head, agent_head
- Confirmed decode_fn(z_zeros): CnnLstmDecoder(latent_dim=64).apply({"params": decoder_params}, jnp.zeros((1,64))) returns (1,13,13) tensors with no NaN

## Task Commits

Each task was committed atomically:

1. **Task 1: Install GCS SDK and download CNN-VAE checkpoint** - `eda0a36` (chore)
2. **Task 2: Create verification script and confirm decoder/decode_fn** - `e428e14` (feat)

**Plan metadata:** (created below)

## Files Created/Modified

- `scripts/verify_checkpoint.py` - Standalone verification script for CKPT-01, CKPT-02, CKPT-03; auto-detects Orbax format; exits 0 on success
- `.gitignore` - Added explicit `vae/checkpoints/` entry (large binary checkpoint files)

## Decisions Made

- **Orbax loader discovery:** The checkpoint structure has `_CHECKPOINT_METADATA`, `default/`, `commit_success.txt` at top level with no digit-only subdirectories. This is StandardCheckpointHandler format. The correct loader is `ocp.PyTreeCheckpointer()` pointed at the `default/` subdirectory with an absolute path.
- **Absolute paths required:** Orbax's tensorstore layer raises `ValueError: Checkpoint path should be absolute` when a relative path is passed. All Orbax calls must use absolute paths.
- **GCS credentials:** The gcloud legacy credentials at `~/.config/gcloud/legacy_credentials/giacomo.maralla@gmail.com/adc.json` provide the oauth2 token. The project `open-endedness-personal` must be passed explicitly to `storage.Client()` since there is no ADC file and no `GCLOUD_PROJECT` environment variable set.
- **Download structure:** Placed GCS `run10/200000/` content directly into `vae/checkpoints/cnn_vae/` (not step-indexed). Subsequent phases should load via `PyTreeCheckpointer` on `cnn_vae/default/`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Orbax requires absolute path; relative path causes ValueError in tensorstore**
- **Found during:** Task 2 (verification script development)
- **Issue:** Plan instructed using relative paths for checkpoint dir (`vae/checkpoints/cnn_vae/`). Orbax 0.10.3 with OCDBT backend raises `ValueError: Checkpoint path should be absolute` for relative paths.
- **Fix:** Updated verification script to use absolute path `CKPT_DIR_ABS` for all `ocp.PyTreeCheckpointer().restore()` calls. Also added discovery logic for `default/` subdir format.
- **Files modified:** scripts/verify_checkpoint.py
- **Verification:** Script runs with exit code 0; CKPT-01, CKPT-02, CKPT-03 all PASS
- **Committed in:** e428e14 (Task 2 commit)

**2. [Rule 1 - Bug] Checkpoint format uses default/ subdir (not flat); plain PyTreeCheckpointer on root fails**
- **Found during:** Task 2 (verification script development)
- **Issue:** Plan expected PyTreeCheckpointer on `cnn_vae/` root to work. Actual format has files inside `default/` subdir (StandardCheckpointHandler). Root restore raises `FileNotFoundError: No structure could be identified`.
- **Fix:** Verification script auto-detects format: if `default/` present without digit-only dirs, points PyTreeCheckpointer at `cnn_vae/default/`.
- **Files modified:** scripts/verify_checkpoint.py
- **Verification:** Script successfully loads checkpoint and passes all checks
- **Committed in:** e428e14 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 — bugs in assumed behavior that required code adjustment)
**Impact on plan:** Both fixes were necessary to correctly load the checkpoint. No scope creep; the delivered script and checkpoint location fully satisfy the plan's success criteria.

## Issues Encountered

- GCS project not auto-detected: `storage.Client()` without `project=` raised `OSError: Project was not passed`. Fixed by reading `~/.config/gcloud/configurations/config_default` to determine project name `open-endedness-personal` and passing it explicitly in the download script. Verification script does not need GCS — it loads from the local checkpoint only.
- `ocp.PyTreeCheckpointer().restore('vae/checkpoints/cnn_vae')` (flat root) raised `FileNotFoundError: No structure could be identified` — because the actual data is in the `default/` subdir. Resolved by pointing at `cnn_vae/default/`.

## User Setup Required

None — no external service configuration required for subsequent phases. The checkpoint is fully local.

## Next Phase Readiness

- Checkpoint fully verified: `vae/checkpoints/cnn_vae/` contains the run10/step200000 checkpoint
- Correct Orbax loader confirmed: `ocp.PyTreeCheckpointer().restore('/abs/path/to/cnn_vae/default/')`
- `decoder_params = restored["params"]["decoder"]` extracts the 8-key decoder dict
- `CnnLstmDecoder(latent_dim=64).apply({"params": decoder_params}, z)` works correctly
- Phase 2 (adapter) can now write production code with confidence about checkpoint structure

## Self-Check: PASSED

- FOUND: scripts/verify_checkpoint.py
- FOUND: vae/checkpoints/cnn_vae/
- FOUND: 01-01-SUMMARY.md
- FOUND commit eda0a36 (Task 1: chore)
- FOUND commit e428e14 (Task 2: feat)

---
*Phase: 01-checkpoint*
*Completed: 2026-03-11*
