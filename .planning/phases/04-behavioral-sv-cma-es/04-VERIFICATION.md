---
phase: 04-behavioral-sv-cma-es
verified: 2026-03-02T21:15:00Z
status: human_needed
score: 3/4 must-haves verified (plus 1 requiring human confirmation)
re_verification: false
human_verification:
  - test: "Run train.py with es_strategy=sv_cma_es for 3 updates and confirm WandB logs sv_behavior_dist_pre and sv_behavior_dist_post > 0"
    expected: "Both sv_behavior_dist_pre and sv_behavior_dist_post appear in WandB with nonzero float values at every wandb_log_freq step"
    why_human: "Static analysis confirms the logging path is wired; actual values depend on runtime JAX execution and WandB connectivity with WANDB_MODE active"
  - test: "Run sv_cma_es for ~100 steps and inspect sv_behavior_dist_pre/post in WandB — confirm values do not collapse to near-zero"
    expected: "Particle behavior distance remains meaningfully above 0 (not collapsing); ROADMAP SC3 requires this holds within first 500 steps"
    why_human: "Particle diversity is an empirical property of the Stein repulsion dynamics — cannot be verified by static code inspection alone"
  - test: "Verify --es_strategy CLI flag availability for maze_plr.py entrypoint"
    expected: "Either maze_plr.py exposes --es_strategy sv_cma_es, or the project README/config.yml documents the correct invocation so the ROADMAP SC1 CLI command can actually be run"
    why_human: "maze_plr.py does not expose --es_strategy or --n_particles. accel_training/train.py main() has --n_particles but not --es_strategy. ROADMAP SC1 CLI command as written does not work against either entrypoint as-is."
---

# Phase 4: Behavioral SV-CMA-ES Verification Report

**Phase Goal:** The primary thesis contribution is implemented and runs end-to-end: N independent CMA-ES particles apply Stein repulsion in behavior space to maintain diversity across the particle population.
**Verified:** 2026-03-02T21:15:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (from ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| SC1 | Running `--es_strategy sv_cma_es --n_particles N` completes a full training run | ? UNCERTAIN | `accel_training/train.py main()` has `--n_particles` but no `--es_strategy` CLI flag; `maze_plr.py` has neither flag. `es_strategy` must be set in `config.yml`. End-to-end pipeline works when invoked via Python API (smoke test PASSES) but the exact CLI command in ROADMAP SC1 does not work against either script as-is. |
| SC2 | Behavior-space repulsion is active: Stein gradient computed between particles using bsigs, candidates adjusted before tell() | VERIFIED | `train.py:516-592` implements the full two-pass eval: particle_bsigs_pre aggregated from first-pass, `compute_stein_repulsion()` called on particle means, `post_latents = latents_jax_pad + repulsion_tiled`, second eval pass on `post_latents`, then `es_strategy.tell(..., pre_cands, pre_bsigs, post_latents, regrets2, post_bsigs, sv_epsilon)` |
| SC3 | Particle diversity maintained: mean pairwise behavior distance does not collapse within 500 steps | ? HUMAN | Metric is logged (WandB `sv_behavior_dist_pre/post`) and Stein repulsion formula is mathematically correct, but whether diversity holds at runtime is an empirical question |
| SC4 | SV-CMA-ES produces a regret curve plottable alongside NS-ES and vanilla CMA-ES | VERIFIED | `wandb.log({"regret": mean_regret, ...}, step=update)` is identical across all three strategies at lines 657-668 in `train.py`. Same key, same step metric — WandB multi-run panel works out of the box. |

**Score:** 2 of 4 truths fully verified by static analysis; 2 require human confirmation (SC1 CLI gap, SC3 empirical behavior)

### Four Phase Must-Haves (from prompt)

| # | Must-Have | Status | Evidence |
|---|-----------|--------|----------|
| 1 | SVCMAESStrategy in accel_training/es_components/ with N=2 distinct-mean particles that step correctly | VERIFIED | `stein.py` + `svcmaes_strategy.py` exist; `init_state` uses distinct RNG keys per particle (`mean_i = jax.random.normal(rng_i, (param_dim,)) * sigma_init`); `ask()` concatenates N particle populations; `tell()` calls evosax tell per particle then applies `compute_stein_repulsion`. All 6 unit tests PASS. |
| 2 | train.py accepts `--es_strategy sv_cma_es` and runs 3 updates without error | PARTIAL | `train.py` reads `es_strategy` from config dict (`config.get("es_strategy", "cma_es")`) but `train.py main()` does NOT expose `--es_strategy` as a CLI flag. Only `--n_particles` is exposed. End-to-end via Python API (`test_end_to_end_3_updates_sv_cma_es`) PASSES. CLI path requires editing `config.yml`. |
| 3 | sv_behavior_dist_pre and sv_behavior_dist_post logged to WandB each update | VERIFIED | `train.py:229-233` initializes both to 0.0 and registers `wandb.define_metric()`; lines 593-594 update them from `sv_metrics`; lines 666-667 include both in `wandb.log()` at every `wandb_log_freq` step. |
| 4 | SV-CMA-ES run produces a regret curve plottable alongside NS-ES and vanilla CMA-ES | VERIFIED | `regret` key logged via `wandb.log({"regret": mean_regret}, step=update)` for all three strategies under identical metric names and step indexing. WandB multi-run overlay works. |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `accel_training/es_components/stein.py` | compute_stein_repulsion() and mean_pairwise_behavior_dist() pure functions | VERIFIED | 105 lines; both functions fully implemented with median heuristic bandwidth; N=1 short-circuit returns `jnp.zeros_like(means)` before any computation; correct SVGD formula with K@means in latent space |
| `accel_training/es_components/svcmaes_strategy.py` | SVCMAESStrategy class with init_state, ask, tell | VERIFIED | 242 lines; `__init__`, `init_state`, `ask`, `tell` all implemented; state dict matches spec `{"particles": [{"es_state":..., "es_params":...}]}`; lazy import of stein inside tell() |
| `accel_training/es_components/__init__.py` | SVCMAESStrategy export | VERIFIED | Line 39: `from .svcmaes_strategy import SVCMAESStrategy`; line 41: `__all__` includes `"SVCMAESStrategy"` |
| `accel_training/train.py` | sv_cma_es routing branch + --n_particles argparse + WandB metrics | VERIFIED (partial CLI) | sv_cma_es branch exists at lines 244-246 (routing) and 516-594 (tell block); `--n_particles` at line 702; `sv_behavior_dist` metrics at lines 229-233, 593-594, 666-667. Missing: `--es_strategy` CLI flag. |
| `tests/test_phase4_sv_cma_es.py` | 6 Phase 4 test functions | VERIFIED | All 6 test functions present and substantive; end-to-end smoke test guards on VAE checkpoint; SUMMARY reports all 6 PASS including end-to-end. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `svcmaes_strategy.py tell()` | `stein.py compute_stein_repulsion()` | `from .stein import compute_stein_repulsion, mean_pairwise_behavior_dist` | WIRED | Line 177 in `svcmaes_strategy.py`; lazy import inside tell() body |
| `svcmaes_strategy.py tell()` | `evosax EvoState.replace(mean=new_mean)` | `new_particles[i]["es_state"].replace(mean=new_mean_i)` | WIRED | Lines 221-223; post-tell Stein mean update applied correctly |
| `train.py ES routing block` | `SVCMAESStrategy.ask()` | `es_strategy_name == 'sv_cma_es'` branch | WIRED | Lines 244-246 instantiate SVCMAESStrategy; line 428 calls `es_strategy.ask(es_state, rng_ask)` shared across all strategies |
| `train.py sv_cma_es branch` | `es_strategy.tell()` | post-repulsion second eval pass, then tell() | WIRED | Lines 584-592 call `es_strategy.tell(es_state, latents_jax_pad, candidate_sigs, post_latents, regrets2, post_bsigs, sv_epsilon)` |
| `train.py WandB log` | `sv_behavior_dist_pre/post metrics` | `wandb.log()` in `wandb_log_freq` block | WIRED | Lines 666-667 include both metrics unconditionally in every wandb.log call |
| `tests/test_phase4_sv_cma_es.py` | `accel_training.es_components.svcmaes_strategy` | `from accel_training.es_components.svcmaes_strategy import SVCMAESStrategy` | WIRED | Line 42 (and per-function) |
| `tests/test_phase4_sv_cma_es.py` | `accel_training.es_components.stein` | `from accel_training.es_components.stein import compute_stein_repulsion` | WIRED | Line 208 in test_stein_repulsion_pushes_apart |
| `test_end_to_end_3_updates_sv_cma_es` | `accel_training.train` | `from accel_training.train import train` | WIRED | Line 331; calls `train(config)` with `"es_strategy": "sv_cma_es"` |

### Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| ALGO-02 | 04-01, 04-02, 04-03 | Behavioral SV-CMA-ES with N CMA-ES particles and Stein repulsion in behavior space | SATISFIED | `stein.py` implements SVGD repulsion; `svcmaes_strategy.py` wraps N evosax CMA_ES instances with Stein mean updates after tell(); `train.py` routes to SVCMAESStrategy with two-pass eval; 6 tests cover all ALGO-02 success criteria. REQUIREMENTS.md marks ALGO-02 Complete (Phase 4). |

No orphaned requirements: all three plans declare `ALGO-02` and REQUIREMENTS.md maps ALGO-02 to Phase 4.

### Anti-Patterns Found

No stub or placeholder anti-patterns detected in the four phase 4 files:

- `accel_training/es_components/stein.py`: 105 lines of substantive JAX math; no TODO/FIXME; no empty returns
- `accel_training/es_components/svcmaes_strategy.py`: 242 lines; all methods fully implemented
- `accel_training/es_components/__init__.py`: clean re-export; no stubs
- `tests/test_phase4_sv_cma_es.py`: 409 lines; all 6 test functions have substantive assertions

One structural note (not a blocker): `train.py main()` exposes `--n_particles` but not `--es_strategy`. Users must edit `config.yml` to switch between strategies. The smoke test (test 6) confirms the Python API path works. This is a usability gap, not a correctness gap.

### Human Verification Required

#### 1. CLI Invocation via maze_plr.py

**Test:** Run `python examples/maze_plr.py --use_es_mutation --es_strategy sv_cma_es --n_particles 2` from project root.
**Expected:** Script accepts the flags and runs a training iteration, OR a clear error message explains the required invocation (e.g., edit config.yml and run `python -m accel_training.train --n_particles 2` with `es_strategy: sv_cma_es` in config).
**Why human:** `maze_plr.py` does not have `--es_strategy` or `--n_particles`; `accel_training/train.py main()` has `--n_particles` but not `--es_strategy`. The ROADMAP SC1 CLI command as documented does not map to either script. The Python API (smoke test) works, but the CLI surface is incomplete vs. the ROADMAP spec.

#### 2. WandB Behavior Distance Metrics at Runtime

**Test:** Run `python -m accel_training.train --n_particles 2` with `config.yml` setting `es_strategy: sv_cma_es`, for at least 10 updates with `wandb_log_freq: 1`. Check WandB dashboard.
**Expected:** `sv_behavior_dist_pre` and `sv_behavior_dist_post` appear as numeric metrics > 0.0, updated each logged step. Values should be stable (not NaN, not inf).
**Why human:** Static analysis confirms the logging path is correctly wired; runtime value depends on actual JAX execution, float32 precision at full D=169 bsig dimension, and real WandB connection.

#### 3. Particle Diversity Not Collapsing (ROADMAP SC3)

**Test:** Run sv_cma_es for at least 100 steps. Plot `sv_behavior_dist_pre` over time in WandB.
**Expected:** `sv_behavior_dist_pre` remains meaningfully above 0.0 — does not collapse to near-zero within the first 500 steps. Some fluctuation is expected; sustained near-zero would indicate Stein repulsion is failing to maintain diversity.
**Why human:** This is an empirical property of the Stein repulsion dynamics at actual behavioral diversity levels. The mathematical formula is correct (verified), but whether the bandwidth h, epsilon=0.01, and D=169 bsigs produce sufficient repulsion at runtime requires observation.

### Gaps Summary

No hard gaps block the core algorithmic goal. The implementation is substantive and wired end-to-end.

One minor structural gap exists: `train.py main()` does not expose `--es_strategy` as a CLI flag, so the ROADMAP SC1 invocation (`maze_plr.py --es_strategy sv_cma_es`) cannot be executed as written. The workaround is to edit `config.yml` directly (`es_strategy: sv_cma_es`) and run train.py with only `--n_particles`. This is a documentation/CLI gap, not an algorithmic gap — all four core must-haves are structurally implemented.

Three items require human verification:
1. CLI usability for the documented ROADMAP SC1 invocation
2. Runtime confirmation of sv_behavior_dist metrics appearing in WandB with nonzero values
3. Empirical confirmation of ROADMAP SC3 (particle diversity not collapsing within 500 steps)

---

_Verified: 2026-03-02T21:15:00Z_
_Verifier: Claude (gsd-verifier)_
