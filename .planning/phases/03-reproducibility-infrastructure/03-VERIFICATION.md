---
phase: 03-reproducibility-infrastructure
verified: 2026-03-24T12:00:00Z
status: passed
score: 9/9 must-haves verified
re_verification: false
human_verification:
  - test: "Run bash examples/launch_llm_injection.sh on a GPU node (albacore/smew/canada)"
    expected: "Training starts, WandB JAXUED_LLM project receives runs with group accel-llm, level cache files appear at results/accel-llm/llm_levels/<seed>/"
    why_human: "Requires GPU node, live LLM API access, and WandB credentials — cannot verify programmatically"
  - test: "Run python scripts/compare_llm_results.py after at least one completed run exists in JAXUED_LLM"
    expected: "Prints comparison table showing solve rate mean+std for accel-llm and accel-only groups"
    why_human: "Requires completed WandB runs in the JAXUED_LLM project to be present"
---

# Phase 3: Reproducibility Infrastructure Verification Report

**Phase Goal:** Accepted LLM levels are cached to disk with wall_map hashes logged to WandB, and comparison launch scripts exist for running ACCEL+LLM vs ACCEL-only control with matching seeds and conditions
**Verified:** 2026-03-24T12:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Every accepted LLM level is written to disk as .npy + JSON sidecar with wall_map SHA-256 hash, injection_step, gate_scores, and timestamp | VERIFIED | `llm/level_cache.py:36-88` — `save_accepted()` writes `step_{step:05d}_idx_{idx:03d}.npy` and matching `.json` sidecar with all required fields |
| 2 | WandB receives a Table with wall_map hashes for each injection event | VERIFIED | `llm/injector.py:535-540` — `wandb.Table(columns=["step","batch_index","wall_map_hash"])` added to `log_payload["llm/accepted_level_hashes"]` guarded by `if accepted_hashes:` |
| 3 | `--llm_inject_start_step` flag controls when injection begins, replacing `--llm_warmup_steps` | VERIFIED | `examples/maze_plr.py:1411` — `add_argument("--llm_inject_start_step", ...)` present; no `add_argument` for `--llm_warmup_steps`; `injector.py:247` uses `self.config.inject_start_step` |
| 4 | `--llm_inject_interval` and `--llm_inject_start_step` are independently configurable for ablation | VERIFIED | Both flags exist independently in argparse (`maze_plr.py:1409,1411`); both map to independent fields `injection_interval` and `inject_start_step` in `LLMInjectionConfig` |
| 5 | `launch_llm_injection.sh` runs ACCEL+LLM training with 3 seeds and ablation-ready injection flags | VERIFIED | `examples/launch_llm_injection.sh:26-38` — seeds 0/1/2, `--use_accel --use_llm`, ablation variables `INJECT_START`/`INJECT_INTERVAL`/`BATCH_SIZE` at script top |
| 6 | `launch_accel_only_control.sh` runs ACCEL-only control with matching seeds and identical non-injection hyperparameters | VERIFIED | `examples/launch_accel_only_control.sh:21-28` — seeds 0/1/2, `--use_accel` only (no `--use_llm`), identical COMMON flags (same num_updates, eval_freq) |
| 7 | Both launch scripts target WandB project JAXUED_LLM with groups accel-llm and accel-only | VERIFIED | LLM script: `--project JAXUED_LLM --run_name "accel-llm"` (line 17,37); control script: `--project JAXUED_LLM --run_name "accel-only"` (lines 12,27) |
| 8 | `compare_llm_results.py` queries JAXUED_LLM WandB runs and prints mean+std solve rate plus LLM-specific metrics | VERIFIED | `scripts/compare_llm_results.py:138-139` — `api = wandb.Api(); runs = api.runs(project_path)`; prints solve_rate, acceptance_rate, injected count, diversity score per group |
| 9 | An ablation can be run by changing only `--llm_inject_start_step` and `--llm_inject_interval` in the launch script | VERIFIED | `launch_llm_injection.sh:7-9` — `INJECT_START=5000` and `INJECT_INTERVAL=3000` extracted as shell variables; used in `$PYTHON` invocation lines 32-33 |

**Score:** 9/9 truths verified

### Required Artifacts

#### Plan 03-01 Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `llm/level_cache.py` | LevelCache class with `save_accepted()` and `compute_hash()` methods | VERIFIED | 104-line file; `class LevelCache` at line 23; `save_accepted()` at line 36; `compute_hash()` static method at line 91 |
| `llm/injector.py` | LevelCache integration and WandB hash table logging in `_do_injection()` | VERIFIED | Imports `LevelCache` (line 32); `level_cache` param in `__init__` (line 155); Step 4b cache block (lines 400-413); `wandb.Table` in log_payload (lines 535-540) |
| `llm/injection_config.py` | `inject_start_step` field replacing `warmup_steps`, with backward compat in `from_config_dict()` | VERIFIED | `inject_start_step: int = 5000` at line 31; `from_config_dict()` reads `config.get("llm_inject_start_step", config.get("llm_warmup_steps", 5000))` at line 86 |
| `examples/maze_plr.py` | `--llm_inject_start_step` argparse flag; `LevelCache` import and instantiation | VERIFIED | `add_argument("--llm_inject_start_step", ...)` line 1411; `from llm.level_cache import LevelCache` line 40; `LevelCache(llm_cache_dir)` line 1070; passed to `LLMInjectionManager` line 1078 |

#### Plan 03-02 Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `examples/launch_llm_injection.sh` | ACCEL+LLM injection launch script for 3 seeds | VERIFIED | 44-line executable script; ablation variables at top; seeds 0/1/2 loop; all required flags present |
| `examples/launch_accel_only_control.sh` | ACCEL-only control launch script with matching seeds | VERIFIED | 34-line executable script; no `--use_llm`; identical COMMON hyperparameters; seeds 0/1/2 loop |
| `scripts/compare_llm_results.py` | WandB comparison table for accel-llm vs accel-only groups | VERIFIED | 164-line script; valid Python syntax; `wandb.Api()` usage; groups by run_name; prints mean/std for all metrics |

### Key Link Verification

#### Plan 03-01 Key Links

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `llm/injector.py` | `llm/level_cache.py` | import + `level_cache.save_accepted()` call in `_do_injection()` | WIRED | `from llm.level_cache import LevelCache` (line 32); `self.level_cache.save_accepted(level, current_step, idx, gate_metrics_for_seed)` (line 410) |
| `llm/injector.py` | `wandb.Table` | hash table added to `log_payload` | WIRED | `wandb.Table(columns=["step","batch_index","wall_map_hash"], data=...)` (lines 536-539); assigned to `log_payload["llm/accepted_level_hashes"]` (line 540) |
| `llm/injection_config.py` | `examples/maze_plr.py` | `from_config_dict` reads `llm_inject_start_step` from argparse config dict | WIRED | `maze_plr.py:1411` registers `--llm_inject_start_step`; `LLMInjectionConfig.from_config_dict(config)` called at `maze_plr.py:1054`; `from_config_dict` reads `config.get("llm_inject_start_step", ...)` at line 86 |

#### Plan 03-02 Key Links

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `examples/launch_llm_injection.sh` | `examples/maze_plr.py` | `--use_llm --use_accel --llm_inject_start_step` flags | WIRED | Script calls `$PYTHON examples/maze_plr.py ... --llm_inject_start_step ${INJECT_START}` (lines 29-38) |
| `scripts/compare_llm_results.py` | WandB API | `wandb.Api().runs()` with JAXUED_LLM project filter | WIRED | `api = wandb.Api()` (line 138); `api.runs(project_path)` (line 139); project defaults to "JAXUED_LLM" (line 128) |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| EXPT-02 | 03-01-PLAN.md | Accepted levels cached to disk (.npy + metadata JSON) with wall_map hashes logged to WandB | SATISFIED | `llm/level_cache.py` — full implementation; `llm/injector.py:400-413` — caching + hash collection; `injector.py:535-540` — WandB Table log |
| EXPT-03 | 03-01-PLAN.md | Ablatable via `--llm_inject_start_step` and `--llm_inject_interval` parameters | SATISFIED | Both flags in `maze_plr.py`; both map to independent `LLMInjectionConfig` fields; ablation variables in `launch_llm_injection.sh` |

**Orphaned requirements check:** REQUIREMENTS.md assigns EXPT-01 ("Comparison launch scripts for ACCEL+LLM injection vs ACCEL-only control with matching seeds") to Phase 4 in the traceability table. Plan 03-02 does not claim EXPT-01 in its frontmatter. The Phase 3 launch scripts (`launch_llm_injection.sh`, `launch_accel_only_control.sh`) satisfy the text of EXPT-01, but REQUIREMENTS.md marks EXPT-01 as still pending (Phase 4) and it is not formally assigned to Phase 3. No orphaned requirements for Phase 3 — both plans declare EXPT-02 and EXPT-03 and both are satisfied.

**Note:** REQUIREMENTS.md traceability has a minor inconsistency — Phase 3 delivers the physical launch scripts that satisfy EXPT-01's text, yet EXPT-01 is assigned to Phase 4. This is a tracking discrepancy, not an implementation gap. EXPT-01 is fully implemented by Plan 03-02 but the requirements doc has not been updated.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `scripts/smoke_test_llm_gate.sh` | 46 | `--llm_warmup_steps 100` — stale flag removed from argparse | Warning | Smoke test would fail with `unrecognized arguments` error if run. Script is from Phase 2 (commit `6c4c8eb`) and was not updated when `--llm_inject_start_step` replaced `--llm_warmup_steps`. Does not block Phase 3 goal — smoke test is not a Phase 3 artifact. |

No blocker anti-patterns in any Phase 3 artifact. All Phase 3 files are clean.

### Human Verification Required

#### 1. LLM Injection Launch Script — Full End-to-End Run

**Test:** SSH to a GPU node (albacore, smew, or canada) and run `bash examples/launch_llm_injection.sh`
**Expected:** Training starts for seed 0, WandB shows a run in JAXUED_LLM project under group `accel-llm`, files appear at `results/accel-llm/llm_levels/0/` with `.npy` + `.json` naming pattern `step_XXXXX_idx_XXX.*`
**Why human:** Requires live GPU node, working LLM API credentials (claude-code), and WandB login

#### 2. WandB Hash Table Visibility

**Test:** After at least one injection event completes, open the WandB run in JAXUED_LLM and look for the `llm/accepted_level_hashes` metric
**Expected:** A Table view with columns `step`, `batch_index`, `wall_map_hash` — one row per accepted LLM level in that injection event
**Why human:** Requires a live WandB run with actual injection events

#### 3. Comparison Script Output

**Test:** After runs complete, run `python scripts/compare_llm_results.py`
**Expected:** Prints formatted table with accel-llm and accel-only groups, showing mean +/- std solve rate and LLM-specific metrics. Per-run detail table also printed.
**Why human:** Requires completed runs in JAXUED_LLM WandB project

### Gaps Summary

No gaps. All must-haves from both Plan 03-01 and Plan 03-02 frontmatter are fully implemented and wired.

**One warning (non-blocking):** `scripts/smoke_test_llm_gate.sh` (a Phase 2 artifact) uses the old `--llm_warmup_steps` flag which was removed by Phase 3. This would cause the smoke test to fail if re-run. This does not block Phase 3's goal, but should be fixed before Phase 4 execution if the smoke test is used for pre-flight validation.

---

_Verified: 2026-03-24T12:00:00Z_
_Verifier: Claude (gsd-verifier)_
