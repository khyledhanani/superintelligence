# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-11)

**Core value:** CMA-ES with CNN-VAE produces valid, solvable maze Levels and runs a complete 20k training experiment comparable to the ACCEL baseline.
**Current focus:** Phase 6 — Run PCA-space CMA-ES on TPU (30k steps, 5 seeds) and compare solve_rate against Phase 4 results

## Current Position

Phase: 6 of 6 (Run PCA-CMA-ES on TPU + Compare Phase 4)
Plan: 2 of 3 in current phase — CHECKPOINT (human-verify)
Status: Phase 6 Plan 02 CHECKPOINT — all 4 files synced to TPU VM cma-es-v4 (verified via wc -l); awaiting human to SSH, check comparison3 done, and launch pca_run tmux session
Last activity: 2026-03-13 — Task 1 complete (file sync verified); stopped at Task 2 human checkpoint

Progress: [##########] 100%

## Performance Metrics

**Velocity:**
- Total plans completed: 4
- Average duration: 3.75min
- Total execution time: 0.25 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-checkpoint | 1 | 5min | 5min |
| 02-grid-adapter | 2 | 8min | 4min |
| 03-integration | 2 | 25min | 12.5min |
| 05-pca-space-cma-es-search | 3 | 43min | 14.3min |
| 06-run-pca-tpu-compare | 1 | 2min | 2min |

**Recent Trend:**
- Last 5 plans: 02-02 (2min), 03-01 (2min), 03-02 (23min), 05-01 (3min), 06-01 (2min)
- Trend: 06-01 fast (pure file creation, no GPU infrastructure work needed)

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Setup]: CNN-VAE as default decoder (user wants CNN-VAE as primary path, not opt-in)
- [Setup]: Download checkpoint locally to `vae/checkpoints/cnn_vae/` (simpler than GCS runtime loading)
- [Setup]: Reuse existing launch scripts (minimize new infrastructure)
- [01-01]: Orbax loader is PyTreeCheckpointer on cnn_vae/default/ subdir (StandardCheckpointHandler format, not step-indexed)
- [01-01]: Absolute paths required for ocp.PyTreeCheckpointer (relative paths fail in tensorstore layer)
- [01-01]: GCS project is 'open-endedness-personal'; must pass explicitly to storage.Client()
- [02-01]: New file vae/cnn_vae_level_utils.py — NOT modifying vae_level_utils.py (INTG-03: CluttrVAE path preserved)
- [02-01]: decode_fn is a static argument in JIT — use jax.jit(decode_latent_to_levels_grid, static_argnums=(0,)) or functools.partial
- [02-01]: Deterministic argmax (not stochastic sampling) for goal/agent placement ensures CMA-ES fitness is reproducible
- [02-02]: JIT static_argnums=(0,) required for decode_fn Python callable — confirmed passing all GRID-01..09 on 1000-sample batch
- [03-01]: CNN-VAE is default when --use_cmaes set; --use_clutr_vae flag gates CluttrVAE fallback
- [03-01]: vae_cfg["latent_dim"] only in elif use_clutr_vae branch; ternary guards in lines 561/569 use short-circuit to avoid NameError
- [03-01]: PCA block guarded with config.get("use_clutr_vae") — CNN-VAE has no encoder in maze_plr.py context
- [03-02]: MazeSolved constructor uses max_height/max_width (not height/width)
- [03-02]: VALD-02 validated via 1000-step simulation (is_valid.mean()*100 on popsize=32 batches) — GPU smoke test deferred (sideswipe occupied by NAMM training)
- [03-02]: cmaes/valid_structure_pct = 100.0% over 1000 simulated DR steps; BFS solvability = 100% on 50 levels
- [03-02]: maze_plr.py --use_cmaes confirmed to load CNN-VAE and start training (5-step CPU run successful)
- [05-01]: cumulative norm threshold 0.85 for Stage 1 K selection (not fixed K) — adapts to any checkpoint
- [05-01]: full_matrices=False in np.linalg.svd essential when N >> 64 (avoids large U matrix)
- [05-01]: Closures in make_*_decode_fn capture jnp arrays (not numpy) for JIT/vmap compatibility
- [05-01]: Stage 1 uses zeros as mu_mean baseline by default (VAE prior mean) — dataset not needed at search time
- [05-01]: pc_stds whitening ensures unit-variance CMA-ES dimensions (sigma_init=0.5 becomes meaningful)
- [05-02]: Removed @jax.jit from train_and_eval_step; use explicit jax.jit() variable for re-jitting after Stage 2 transition
- [05-02]: Stage 2 guarded by _pca_stage == 1 to fire exactly once at or after pca_stage2_step
- [05-02]: tokens_np = np.array(tokens_jax) required before encode_mazes_to_mu (clutr_to_grid needs numpy)
- [05-02]: PCA WandB logging done outside jit in outer loop (Python-level state, not JAX arrays)
- [05-03]: Stage 1 K=55 (at upper boundary of [15, 55]) — cumulative 86.5% norm threshold adapts to checkpoint quality
- [05-03]: Phase B smoke test run on CPU (GPU nodes unreachable); 500 steps completed in ~31 min — all functional checks pass
- [05-03]: pca_stage2_step=99999 in smoke test keeps Stage 1 active for full 500 steps (no dataset download needed)
- [06-01]: PCA launch script runs only 5 seeds (1 condition) — Phase 4 already has cmaes-cnn-vae-accel and accel-baseline data
- [06-01]: compare_phase4_results.py accepts --entity as optional arg (WandB infers from login if omitted)
- [06-01]: Final value extraction via hist[metric].dropna().iloc[-1] — robust to sparse WandB logging
- [06-02]: gcloud binary at /cs/student/project_msc/2025/csml/gmaralla/home/google-cloud-sdk/bin/gcloud (not in PATH on head node)
- [06-02]: SCP 'Attempting to connect to worker 0...' is normal gcloud SSH tunnel init — verify with subsequent ssh command

### Roadmap Evolution

- Phase 5 added: PCA-Space CMA-ES Search
- Phase 6 added: Run PCA-space CMA-ES on TPU (30k steps, 5 seeds) and compare solve_rate against Phase 4 results

### Pending Todos

None.

### Blockers/Concerns

None — all Phase 1 blockers resolved:
- google-cloud-storage SDK installed into jax_env
- Checkpoint structure confirmed: PyTreeCheckpointer on cnn_vae/default/
- GCS auth working via legacy credentials

## Session Continuity

Last session: 2026-03-13
Stopped at: 06-02-PLAN.md Task 2 checkpoint (human-verify) — files synced to TPU VM, awaiting SSH to check comparison3 done and launch pca_run tmux session with launch_pca_comparison.sh
Resume file: None
