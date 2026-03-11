---
phase: 05-ablations-and-analysis
verified: 2026-03-03T18:30:00Z
status: gaps_found
score: 7/10 must-haves verified
re_verification: false
gaps:
  - truth: "Pre-launch validation passes: SV-CMA-ES runs 1-2k updates and buf_score rises clearly above the ~0.004 ceiling"
    status: failed
    reason: "No phase5-smoke run directory exists. The only runs/ entry is sv_cma_es_run1 (pre-refactor, old three-branch architecture). The smoke test has not been executed."
    artifacts:
      - path: "runs/phase5-smoke/"
        issue: "Directory does not exist — smoke test has not been run"
    missing:
      - "Execute the smoke test step in run_phase5_comparison.sh (SV-CMA-ES, 1000 updates) to validate the refactored architecture produces buf_score > 0.004"
  - truth: "All four experiments complete at 20k updates (ACCEL baseline, CMA-ES, NS-ES, SV-CMA-ES) — runs named and grouped in WandB"
    status: failed
    reason: "No phase5-comparison experiment run directories exist under runs/. The runs/ directory contains only the pre-refactor sv_cma_es_run1. Four experiments have not been executed."
    artifacts:
      - path: "runs/phase5-cma-es/"
        issue: "Directory does not exist — CMA-ES 20k run has not been executed"
      - path: "runs/phase5-ns-es/"
        issue: "Directory does not exist — NS-ES 20k run has not been executed"
      - path: "runs/phase5-sv-cma-es/"
        issue: "Directory does not exist — SV-CMA-ES 20k run has not been executed"
    missing:
      - "Execute scripts/run_phase5_comparison.sh to run all four 20k-update experiments"
      - "Rename/tag ACCEL baseline WandB run to group=phase5-comparison via WandB UI after maze_plr.py completes"
  - truth: "A Jupyter notebook produces two thesis-quality figures from WandB data"
    status: failed
    reason: "figures/phase5_comparison.pdf and figures/phase5_comparison.png do not exist. The notebook is a template that will only produce figures once the experiments (gaps above) have completed. No WandB data for the phase5-comparison group exists yet."
    artifacts:
      - path: "figures/phase5_comparison.pdf"
        issue: "File does not exist — notebook has not been executed with live WandB data"
      - path: "figures/phase5_comparison.png"
        issue: "File does not exist — notebook has not been executed with live WandB data"
    missing:
      - "After running experiments (gaps above), execute notebooks/phase5_comparison.ipynb to produce thesis figures"
human_verification:
  - test: "Verify buf_score rises above 0.004 during smoke test"
    expected: "After 1000 updates of the smoke run, WandB shows mean_buffer_score or buf_score trending clearly above the old 0.004 ceiling"
    why_human: "Requires running the actual JAX training pipeline and observing WandB metric curves — cannot be verified statically"
  - test: "Verify ACCEL baseline WandB run is correctly tagged with group=phase5-comparison"
    expected: "After running examples/maze_plr.py, the resulting WandB run has been manually renamed to accel-baseline and tagged to group phase5-comparison so the notebook api.runs() filter picks it up"
    why_human: "ACCEL baseline runs black-box; WandB group assignment requires manual UI action post-run — cannot be automated"
  - test: "Verify thesis figures are visually publication-quality"
    expected: "figures/phase5_comparison.pdf shows four smooth regret curves (ACCEL, CMA-ES, NS-ES, SV-CMA-ES) with clear labels, readable axis scales, and distinguishable colors"
    why_human: "Visual quality of output figures requires human inspection — cannot be verified programmatically"
---

# Phase 5: Refactor and Four-Way Comparison — Verification Report

**Phase Goal:** accel_training/train.py is rewritten as a clean two-mode pipeline (replay / es_step), the full codebase is audited for compatibility, and four comparison experiments produce thesis-quality comparison plots
**Verified:** 2026-03-03T18:30:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths (from ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| SC-1 | train.py rewritten with only two modes (replay / es_step); no MAP-Elites archive, no archive warm-up; ratio configurable | VERIFIED | accel_training/train.py: 665 lines, zero old-arch references (archive, run_archive_warmup, UpdateState, use_accel, n_candidates, mutation_sigma, random_fraction, warmup_n), bootstrap_min + replay_ratio in use (12 hits) |
| SC-2 | Full pipeline audit complete; all three ES strategies (cma_es, ns_es, sv_cma_es) run without error under new architecture | VERIFIED | Tests test_phase3_ns_es.py + test_phase4_sv_cma_es.py updated; import pattern verified; strategy routing confirmed (lines 151-157, 381-476); SUMMARY reports 6/6 PASS on each |
| SC-3 | Pre-launch validation passes: SV-CMA-ES runs 1-2k updates, buf_score rises above ~0.004 ceiling | FAILED | No runs/phase5-smoke/ directory exists. Smoke test has not been executed. Only existing run is legacy sv_cma_es_run1 (pre-refactor architecture). |
| SC-4 | All four experiments complete at 20k updates (ACCEL baseline, CMA-ES, NS-ES, SV-CMA-ES); named and grouped in WandB | FAILED | No runs/phase5-cma-es/, phase5-ns-es/, phase5-sv-cma-es/, or ACCEL baseline run directories exist. Experiments have not been executed. |
| SC-5 | Jupyter notebook produces two thesis-quality figures from WandB data | FAILED | figures/phase5_comparison.pdf and .png do not exist. Notebook is a valid template but has not been executed with live WandB data. |

**Score (ROADMAP Success Criteria):** 2/5 criteria fully verified

---

### Must-Have Truths (from PLAN frontmatter — 05-01 and 05-02)

#### Plan 05-01 Must-Haves

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| P1-T1 | train.py has exactly two modes: replay and es_step; no MAP-Elites archive, no archive warm-up, no mutate branch | VERIFIED | grep count of old refs = 0; main loop contains only `if mode == "es_step":` and `else:` (replay) branch (lines 532-570) |
| P1-T2 | All three ES strategies (cma_es, ns_es, sv_cma_es) run 3 updates without error under new architecture | VERIFIED | test_phase3_ns_es.py and test_phase4_sv_cma_es.py updated; both sets of smoke tests use `train_state = train(config)` signature with bootstrap_min=5; SUMMARY reports 6/6 PASS per set |
| P1-T3 | Bootstrap loop fills buffer to bootstrap_min levels via es_step before any replay occurs | VERIFIED | Bootstrap loop present lines 491-502; `while int(train_state.sampler["size"]) < bootstrap_min:` calls `_run_es_step()`; does NOT count toward num_updates |
| P1-T4 | WandB init uses name=run_name and group=wandb_group (not group=run_name) | VERIFIED | Lines 127-133: `name=config["run_name"]`, `group=config.get("wandb_group", "phase5-comparison")` — correct pattern |
| P1-T5 | config.yml has bootstrap_min and replay_ratio keys; use_accel/n_candidates/mutation_sigma/random_fraction/warmup_n removed | VERIFIED | config.yml lines 24 (replay_ratio), 38 (bootstrap_min), 69 (wandb_group); grep for removed keys = 0 hits; replay_prob key absent |
| P1-T6 | Existing test files updated to match new train() return signature (train_state only, not tuple) | VERIFIED | Both test files: `train_state = train(config)` on end-to-end tests; assert `not isinstance(result, tuple)`; no `train_state, archive = train(config)` pattern |

#### Plan 05-02 Must-Haves

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| P2-T1 | A single launcher script runs all four experiments sequentially with consistent seed and WandB group | VERIFIED | scripts/run_phase5_comparison.sh: SEED=42, GROUP=phase5-comparison, UPDATES=20000; four train.py invocations with --es_strategy; bash syntax check passes |
| P2-T2 | The Jupyter notebook pulls WandB data via API and produces two thesis-quality figures | PARTIAL | notebooks/phase5_comparison.ipynb: valid nbformat=4 JSON, 4 cells, wandb.Api() call with group filter; Figure 2 placeholder present. BUT figures/phase5_comparison.pdf/.png do not exist — notebook unexecuted |
| P2-T3 | All four experiment configs use seed=42, group=phase5-comparison, and 20k updates | VERIFIED | Launcher script: SEED=42, GROUP=phase5-comparison, UPDATES=20000; all three train.py invocations use `--seed $SEED --group $GROUP --num_updates $UPDATES` |
| P2-T4 | ACCEL baseline runs examples/maze_plr.py as-is (black-box, no modifications) | VERIFIED | Launcher script line 79: `$PYTHON examples/maze_plr.py` — no flags, no modifications; comment explicitly states black-box |

**Score (Plan must-haves):** 9/10 truths verified (P2-T2 partial due to unexecuted notebook)

**Combined score:** 7/10 must-haves verified (counting P2-T2 as partial, SC-3/SC-4/SC-5 as failed)

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `accel_training/train.py` | Two-mode training pipeline (replay / es_step) with bootstrap loop | VERIFIED | 665 lines; bootstrap_min/replay_ratio used 12 times; clean two-mode structure confirmed |
| `accel_training/config.yml` | Updated config with bootstrap_min, replay_ratio, removed ACCEL keys | VERIFIED | bootstrap_min: 50, replay_ratio: 0.8, wandb_group: phase5-comparison; zero old ACCEL keys |
| `tests/test_phase3_ns_es.py` | Updated Phase 3 tests for new train() signature | VERIFIED | test_bootstrap_populates_buffer uses new API; run_archive_warmup import absent; train_state = train(config) |
| `tests/test_phase4_sv_cma_es.py` | Updated Phase 4 tests for new train() signature | VERIFIED | train_state = train(config); no tuple unpacking; bootstrap_min/replay_ratio in config |
| `scripts/run_phase5_comparison.sh` | Sequential launcher: pre-launch smoke + four 20k-update experiments + ACCEL baseline | VERIFIED | All four strategies invoked; SEED=42, GROUP=phase5-comparison; bash syntax OK; executable |
| `notebooks/phase5_comparison.ipynb` | Jupyter notebook with WandB API data pull, smoothed regret curves, two figures | PARTIAL | Valid nbformat=4, 4 cells, wandb.Api() wired; but unexecuted — no output figures exist |
| `runs/phase5-smoke/` | Smoke test run output | MISSING | Directory does not exist |
| `runs/phase5-cma-es/` | CMA-ES 20k run output | MISSING | Directory does not exist |
| `runs/phase5-ns-es/` | NS-ES 20k run output | MISSING | Directory does not exist |
| `runs/phase5-sv-cma-es/` | SV-CMA-ES 20k run output | MISSING | Directory does not exist |
| `figures/phase5_comparison.pdf` | Thesis comparison figure (PDF) | MISSING | File does not exist |
| `figures/phase5_comparison.png` | Thesis comparison figure (PNG) | MISSING | File does not exist |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `accel_training/train.py` | `accel_training/es_components` | ES strategy routing (cma_es/ns_es/sv_cma_es) | VERIFIED | Lines 151-157 (init), 381-476 (_run_es_step): if/elif/else routes all three strategies |
| `accel_training/train.py` | `accel_training/config.yml` | config dict loading: bootstrap_min/replay_ratio | VERIFIED | Lines 91-92: `bootstrap_min = config.get("bootstrap_min", 50)`, `replay_ratio = config.get("replay_ratio", ...)` |
| `tests/test_phase3_ns_es.py` | `accel_training/train.py` | import train | VERIFIED | Line 251: `from accel_training.train import train` |
| `tests/test_phase4_sv_cma_es.py` | `accel_training/train.py` | import train | VERIFIED | Line 332: `from accel_training.train import train` |
| `scripts/run_phase5_comparison.sh` | `accel_training/train.py` | CLI invocation with --es_strategy | VERIFIED | Lines 24, 38, 50, 62: `$PYTHON accel_training/train.py ... --es_strategy {cma_es,ns_es,sv_cma_es}` |
| `scripts/run_phase5_comparison.sh` | `examples/maze_plr.py` | CLI invocation ACCEL baseline | VERIFIED | Line 79: `$PYTHON examples/maze_plr.py` (black-box) |
| `notebooks/phase5_comparison.ipynb` | WandB API | wandb.Api().runs() with group filter | VERIFIED | Cell 2: `api = wandb.Api()` then `runs = api.runs("es-accel", filters={"group": "phase5-comparison"})` |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| COMP-01 | 05-01, 05-02 | Regret curve comparison across methods (vanilla ACCEL vs NS-ES vs SV-CMA-ES) | PARTIAL | Infrastructure complete: train.py refactored, launcher script ready, notebook template wired. Experiments NOT yet run; no WandB data; no comparison figures produced. COMP-01 cannot be fully satisfied until all four 20k-update runs complete and notebook is executed. |

**COMP-01 traceability note:** REQUIREMENTS.md marks COMP-01 as `[x]` Complete (Phase 5, Complete). This is premature — COMP-01 requires actual comparison plots, which do not exist. The plan infrastructure is complete, but the requirement evidence (runnable comparison) is absent.

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `accel_training/train.py` | 182 | Comment `# Build a placeholder level to initialize the network and level sampler` | Info | Code comment, not a stub — the placeholder level is a legitimate initialization pattern, not incomplete code |
| `notebooks/phase5_comparison.ipynb` | Cell 4 | Markdown placeholder for Phase 6 ablation curves | Info | Intentional design per plan — placeholder cell is correct; Phase 6 will add content |

No blockers or warnings found in any modified file. The "placeholder" at line 182 of train.py is a code comment describing a legitimate technique (building a dummy level for network shape initialization), not an incomplete implementation.

---

## Human Verification Required

### 1. Smoke Test Validation

**Test:** Run `bash scripts/run_phase5_comparison.sh` and observe the Step 1 smoke test output in WandB.
**Expected:** After 1000 SV-CMA-ES updates, `mean_buffer_score` in WandB trends clearly above the previous ~0.004 ceiling, confirming the refactored architecture is generating quality curriculum levels.
**Why human:** Requires executing the JAX training pipeline on GPU and observing WandB metric curves — cannot be verified statically.

### 2. ACCEL Baseline WandB Group Assignment

**Test:** After `examples/maze_plr.py` completes (Step 5 of launcher), navigate to the WandB project and manually rename the run to `accel-baseline` and tag its group as `phase5-comparison`.
**Expected:** The run appears in `api.runs("es-accel", filters={"group": "phase5-comparison"})` with name `accel-baseline`.
**Why human:** examples/maze_plr.py runs black-box with its own internal WandB config; group/name assignment requires post-run manual UI action.

### 3. Thesis Figure Quality

**Test:** After executing all four experiments and running notebooks/phase5_comparison.ipynb, inspect `figures/phase5_comparison.pdf`.
**Expected:** The figure shows four smooth regret curves (ACCEL, CMA-ES, NS-ES, SV-CMA-ES) with clear axis labels ("Training Updates", "Mean Buffer Score"), a legend, grid lines, and visually distinguishable colors (#555555, #1f77b4, #ff7f0e, #2ca02c). Appropriate for thesis inclusion.
**Why human:** Visual publication quality requires human inspection — cannot be verified programmatically.

---

## Gaps Summary

Phase 5 is split into two distinct achievement categories:

**Category A (Architecture Refactor) — COMPLETE.** The train.py rewrite is substantive and correct: 665 lines with the clean two-mode pipeline, zero legacy archive/MAP-Elites references, all three ES strategies routed correctly, bootstrap loop functioning, WandB init fixed, train() return signature corrected. The config.yml update is correct and complete. Both test files are updated to the new API and verified passing. The launcher script is syntactically valid and structurally correct. The notebook is a valid nbformat=4 template with proper WandB API wiring. All seven code-level must-haves from the plan frontmatter pass.

**Category B (Experiments and Figures) — NOT YET STARTED.** The four 20k-update comparison runs have not been executed. No phase5-smoke, phase5-cma-es, phase5-ns-es, or phase5-sv-cma-es directories exist under runs/. No phase5_comparison.pdf or .png figures exist. This is the portion of COMP-01 that requires actual compute time (several hours). The infrastructure is in place; the executions have not happened.

The three ROADMAP success criteria that fail (SC-3, SC-4, SC-5) are all in Category B. They cannot be satisfied by code changes — they require running the experiments.

**Root cause for all three gaps:** The experiments have not been launched yet. The launcher script `scripts/run_phase5_comparison.sh` is ready; it needs to be executed in tmux/screen on a machine with JAX GPU access.

---

_Verified: 2026-03-03T18:30:00Z_
_Verifier: Claude (gsd-verifier)_
