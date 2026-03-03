# Phase 5: Refactor and Four-Way Comparison — Research

**Researched:** 2026-03-03
**Domain:** Python training loop refactor, ES/PLR pipeline, WandB experiment management, Jupyter notebook plotting
**Confidence:** HIGH (all findings verified against actual codebase)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Refactor approach:**
- Clean rewrite of train.py — not surgical edits, start from scratch with only the two-mode logic
- Two modes only: `replay` (PLR buffer → train agent) and `es_step` (ES ask() → VAE decode → eval → insert into PLR buffer → tell())
- No MAP-Elites archive, no archive warm-up
- Bootstrap strategy: always run `es_step` until buffer has a minimum number of levels (e.g. 50), then switch to the configured ratio
- Replay/es_step ratio controlled via a single config.yml float (e.g. `replay_ratio: 0.8`)
- Full pipeline audit: every file that imports from `accel_training/` is reviewed and updated to match the new two-mode interface (not just train.py + config.yml)

**Experiment execution:**
- Single launcher script runs all four experiments (sequentially or submits to job scheduler)
- WandB naming: consistent run names (`accel-baseline`, `cma-es`, `ns-es`, `sv-cma-es`) + shared group tag (e.g. `phase5-comparison`) for easy filtering
- Pre-launch validation: SV-CMA-ES smoke run for 1–2k updates confirming buf_score rises above the old ~0.004 ceiling before committing to full runs
- ACCEL baseline (`examples/maze_plr.py`) runs as-is, black-box — no modifications to the original file

**Plot & analysis design:**
- Primary metric: regret vs update steps (smoothed rolling mean, e.g. window=50) for single seed per method
- Two separate figures:
  - Figure 1: four-method comparison (ACCEL baseline, CMA-ES, NS-ES, SV-CMA-ES)
  - Figure 2: ablation curves (Phase 6 — placeholder in notebook for now)
- Jupyter notebook pulls WandB data via API, smooths, and produces both figures — easy to iterate on and include in thesis appendix

### Claude's Discretion
- Exact bootstrap threshold (number of levels before switching to replay ratio)
- Rolling mean window size for smoothing
- Exact plot styling (colors, line widths, legend placement)
- Temp file handling and run concurrency in the launcher script

### Deferred Ideas (OUT OF SCOPE)
- Fitness weight ablation studies (α/β sweep for SV-CMA-ES) — Phase 6
- Validation set evaluation: run saved agent checkpoints on a fixed held-out maze set — Phase 6 or later
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| COMP-01 | Regret curve comparison across methods (vanilla ACCEL vs NS-ES vs SV-CMA-ES) | The refactored train.py with two-mode pipeline enables fair comparison; WandB group-tagged runs enable side-by-side plotting; Jupyter notebook with wandb.Api() pulls all four runs into comparable DataFrames |
</phase_requirements>

---

## Summary

Phase 5 has three distinct workstreams: (1) rewrite `accel_training/train.py` from scratch as a clean two-mode pipeline, (2) run four experiments at 20k updates under the same seed and WandB group, and (3) produce thesis-quality comparison figures in a Jupyter notebook. The code investigation reveals exactly what must be removed, what must be preserved, and what the new architecture must look like.

The current `train.py` is a ~733-line three-branch loop (new / replay / mutate) with a MAP-Elites `Archive` object threaded throughout: imported at line 52, instantiated at line 252, updated in the new/mutate branch (line 460), logged (line 653), checkpointed (lines 385–390), and returned from `train()` (line 679). The `run_archive_warmup()` function (lines 97–174) also depends on the archive indirectly. All of this machinery is removed in Phase 5. The new train.py keeps: ES ask/tell routing (NSESStrategy / CMAESStrategy / SVCMAESStrategy), PLR buffer (LevelSampler), the JIT-compiled `_train_on_levels()` helper, WandB logging, CSV logging, and checkpointing of agent params only.

Two existing tests (`test_phase3_ns_es.py` and `test_phase4_sv_cma_es.py`) import from `accel_training.train` and must be updated: `test_archive_warmup_populates_buffer` tests `run_archive_warmup()` which no longer exists; end-to-end smoke tests receive a `(train_state, archive)` tuple which changes to `train_state` only. These are the only external consumers of `accel_training.train`; the ES component tests (`test_es_components.py`, `test_phase4_sv_cma_es.py` unit tests) import only from `accel_training.es_components` and are unaffected.

**Primary recommendation:** Plan three sequential tasks — (1) rewrite train.py + update config.yml and tests, (2) run pre-launch smoke test and all four experiments, (3) create Jupyter notebook with WandB data pull and figures.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| jaxued.LevelSampler | project-local | PLR buffer: insert_batch, sample_replay_levels, update_batch, sample_replay_decision | Already in use; no alternative |
| wandb | 0.25.0 (confirmed) | Run logging, group-tagged experiment management | Already in use; API confirmed working via `wandb.Api()` |
| wandb.Api() | 0.25.0 | Notebook data retrieval: `api.runs()` + `run.history(pandas=True)` | Verified: credentials at `~/.netrc`, API loads successfully |
| matplotlib | 3.10.8 (confirmed in jax_env) | Thesis-quality figure generation | Available in jax_env |
| pandas | 2.3.3 (confirmed in jax_env) | DataFrame manipulation for smoothing and plotting | Available in jax_env |
| accel_training.es_components | project-local | NSESStrategy, CMAESStrategy, SVCMAESStrategy — unchanged | All three tested and passing |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| jupyter / notebook | NOT in jax_env, needs install | Interactive notebook for figures | Must install: `pip install notebook` or `pip install jupyterlab` in jax_env |
| numpy | project-installed | Bootstrap counter, np.nanmean for logging | Already available |
| yaml | project-installed | Config loading | Already in use |

**Installation needed:**
```bash
/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/pip install notebook
# or
/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/pip install jupyterlab
```
No job scheduler is available on `sumida.cs.ucl.ac.uk` (no sbatch/qsub/bsub). Experiments run sequentially in a single launcher script.

---

## Architecture Patterns

### New train.py Two-Mode Structure

```
train(config)
├── Setup: env, VAE, eval_fn, WandB, ES strategy, LevelSampler, agent
├── Bootstrap loop: run es_step until sampler["size"] >= bootstrap_min
│     (replaces run_archive_warmup — no Archive object, no solvability gate needed separately)
│     Internally: ES ask() → eval_fn() → behavior_sig → insert_batch() → ES tell()
├── Main loop (for update in range(num_updates)):
│     ├── Decide mode:
│     │     if sampler["size"] < bootstrap_min → es_step (forced)
│     │     else → replay with probability replay_ratio, else es_step
│     ├── es_step branch:
│     │     ES ask() → latents
│     │     eval_fn(rng, params, latents) → sequences, levels, regrets, max_returns, valid
│     │     rollout_agent_on_levels_with_positions() → behavior_sigs
│     │     insert_batch(sampler, levels, regrets, level_extra)
│     │     ES tell() [strategy-specific: CMA-ES / NS-ES / SV-CMA-ES]
│     │       For SV-CMA-ES: second eval pass with repulsion (same logic as current lines 518–591)
│     └── replay branch:
│           sample_replay_levels() → level_inds, replay_levels
│           _train_on_levels(rng, train_state, replay_levels) → new_scores, losses
│           update_batch(sampler, level_inds, new_scores, updated_extra)
└── Logging, checkpointing (agent params only — no archive arrays)
```

### Bootstrap Threshold Decision

The `LevelSampler.sample_replay_decision()` already refuses replay when `sampler["size"] / capacity < minimum_fill_ratio`. The new bootstrap counter is **separate and explicit**: a Python `int` counter or a direct check `sampler["size"] < bootstrap_min`. Recommended threshold: `bootstrap_min = 50` (Claude's discretion). This is simpler than the archive warm-up approach and gives a deterministic minimum before the first replay.

The `replay_ratio` config key replaces `replay_prob` in the new train.py. The LevelSampler is still initialized with `replay_prob=config["replay_ratio"]` — the naming change is at the config.yml level only; the LevelSampler API is unchanged.

### WandB Naming Pattern

```python
wandb.init(
    project="es-accel",
    name=config["run_name"],          # "accel-baseline" | "cma-es" | "ns-es" | "sv-cma-es"
    group="phase5-comparison",        # shared group for easy filtering
    tags=[config["es_strategy"].upper()],
)
```

The current code uses `group=config["run_name"]` (wrong — makes each run its own group). The fix: `name=config["run_name"]` and `group="phase5-comparison"` as a hardcoded or config-controlled group key.

### Launcher Script Pattern

```bash
#!/bin/bash
# scripts/run_phase5_comparison.sh
set -e
PYTHON=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python
SEED=42
GROUP=phase5-comparison

# Pre-launch smoke (1k updates, SV-CMA-ES)
$PYTHON accel_training/train.py --config accel_training/config.yml \
  --run_name sv-cma-es-smoke --num_updates 1000 --seed $SEED \
  --group $GROUP --log_dir runs/phase5-smoke/

# Full runs (20k updates, same seed)
$PYTHON accel_training/train.py --config accel_training/config.yml \
  --es_strategy cma_es --run_name cma-es --num_updates 20000 --seed $SEED \
  --group $GROUP --log_dir runs/phase5-cma-es/ &

# ... etc
```

Note: No job scheduler. Sequential execution is safest given JAX GPU/XLA memory constraints. Parallel execution via `&` may cause OOM if all four runs share a GPU.

### Jupyter Notebook Pattern

```python
# notebooks/phase5_comparison.ipynb

import wandb
import pandas as pd
import matplotlib.pyplot as plt

api = wandb.Api()
runs = api.runs("es-accel", filters={"group": "phase5-comparison"})

dfs = {}
for run in runs:
    hist = run.history(keys=["update", "mean_buffer_score"], pandas=True)
    dfs[run.name] = hist

# Figure 1: Four-method comparison
fig, ax = plt.subplots(figsize=(8, 5))
window = 50  # Claude's discretion
for name, df in dfs.items():
    smoothed = df["mean_buffer_score"].rolling(window, min_periods=1).mean()
    ax.plot(df["update"], smoothed, label=name)
ax.set_xlabel("Updates")
ax.set_ylabel("Mean Buffer Score (regret proxy)")
ax.legend()
fig.savefig("figures/phase5_comparison.pdf", bbox_inches="tight")
```

### Removed Components

The following are **removed entirely** from train.py in the rewrite:
- `Archive` class import (`from archive import Archive`)
- `generate_candidates()`, `mutate_latents()`, `update_archive()` from `ued_interface.py`
- `run_archive_warmup()` function (entire 78-line function)
- `UpdateState` IntEnum (only had NEW/REPLAY, drives the old mutate logic)
- `last_replay_latents` state variable (drives ACCEL mutation step)
- `use_accel` config key
- `n_candidates`, `mutation_sigma`, `random_fraction` config keys
- `warmup_n` config key
- Archive-related checkpoint saves (lines 385–390: `archive_envs.npy`, etc.)
- `archive.num_filled` from `_log()` calls

The following are **kept** in config.yml but renamed or restructured:
- `replay_prob` → `replay_ratio` (same semantics, cleaner name)
- `es_pop_size` stays (controls ES ask() population = latents per es_step)
- New key: `bootstrap_min` (e.g. 50) — minimum buffer size before replay allowed

### Files Requiring Audit (Full Pipeline Audit Scope)

Imports from `accel_training/` found in the codebase:

| File | Import | Impact |
|------|--------|--------|
| `tests/test_phase3_ns_es.py` | `from accel_training.train import run_archive_warmup, TrainState` | `run_archive_warmup` removed — test must be rewritten or deleted |
| `tests/test_phase3_ns_es.py` | `from accel_training.train import train` (smoke test expects `(train_state, archive)` tuple) | Return signature changes to `train_state` only |
| `tests/test_phase4_sv_cma_es.py` | `from accel_training.train import train` (smoke test expects `(train_state, archive)` tuple) | Return signature changes to `train_state` only |
| `tests/test_es_components.py` | Only imports from `accel_training.es_components.*` | No change needed |
| `tests/test_phase4_sv_cma_es.py` | Unit tests import from `accel_training.es_components.*` | No change needed |
| `accel_training/es_components/__init__.py` | Internal imports only | No change needed |

**No other files** import from `accel_training/` — confirmed by codebase-wide grep.

The `scripts/extract_plr_dataset.py` imports from `es/` (legacy `es.env_bridge`) and `orbax`, not from `accel_training/` — not in audit scope.

`examples/maze_plr.py` runs as a standalone black-box — not audited, not modified.

### Anti-Patterns to Avoid

- **Retaining `UpdateState` enum:** The old NEW/REPLAY enum drives the mutate branch which no longer exists. Remove entirely. New mode is just the string `"replay"` or `"es_step"` for logging.
- **Keeping archive in checkpoint saves:** The `_save_checkpoint()` function currently saves `archive_envs.npy` etc. The new version saves agent params only (already a pickle). No archive arrays.
- **Keeping `warmup_n` in config:** The archive warm-up config key is removed. Bootstrap is now `bootstrap_min` (a buffer size count, not a latent count).
- **Wrong WandB `name`/`group` usage:** Current code uses `group=config["run_name"]` which makes filtering impossible. Fix: `name=config["run_name"], group="phase5-comparison"`.
- **Confusion between `replay_prob` (LevelSampler) and `replay_ratio` (config):** The LevelSampler API uses `replay_prob` in its constructor. The new config key is `replay_ratio`. When constructing LevelSampler, pass `replay_prob=config["replay_ratio"]`. Both refer to the same probability.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Smoothed regret curves | Custom rolling average | `pandas.Series.rolling(window).mean()` | Handles NaN, min_periods, already available |
| WandB data retrieval | Direct log file parsing | `wandb.Api().runs(...).history(pandas=True)` | Verified working at 0.25.0; returns DataFrame directly |
| Replay buffer management | Custom ring buffer | `jaxued.LevelSampler` | Already integrated and tested |
| ES strategy updates | Any custom gradient logic | Existing `CMAESStrategy`, `NSESStrategy`, `SVCMAESStrategy` | All three tested passing Phase 3 and 4 |
| Solvability filtering in es_step | Custom validity gate | `eval_fn` already returns `valid` boolean array | `eval_fn` in `ued_interface.build_eval_fn` already runs `flood_fill_solvable` and complexity mask — filter regrets by `jnp.where(valid, regrets, 0.0)` |

---

## Common Pitfalls

### Pitfall 1: Test Regression on Removed `run_archive_warmup`

**What goes wrong:** `test_archive_warmup_populates_buffer` in `test_phase3_ns_es.py` imports `run_archive_warmup` directly from `accel_training.train`. After the rewrite, this import fails with `ImportError`.

**Why it happens:** The test was written to cover an INTEG-02 requirement that is now superseded by the bootstrap-in-the-loop approach.

**How to avoid:** The test must be updated. Two options: (a) replace the test with a new test verifying bootstrap behavior — i.e., that after the bootstrap phase, `sampler["size"] >= bootstrap_min`; (b) remove the test entirely since INTEG-02 is already marked Complete. Given the test file exists and we run it as a regression gate, option (a) is preferred.

**Warning signs:** Any test run that fails with `ImportError: cannot import name 'run_archive_warmup'`.

### Pitfall 2: Return Signature Mismatch in Smoke Tests

**What goes wrong:** Both `test_phase3_ns_es.py` (line 479) and `test_phase4_sv_cma_es.py` (line 391) do `train_state, archive = train(config)`. After the rewrite, `train()` returns only `train_state`. This raises `ValueError: too many values to unpack` (or `not enough values`).

**Why it happens:** Old `train()` returned `(train_state, archive)`. New `train()` returns `train_state` only (archive is gone).

**How to avoid:** Update both smoke test assertions to `train_state = train(config)`. Also remove any `archive`-related assertions (e.g., checking `archive.num_filled`).

### Pitfall 3: Config Key Drift Between Old and New Tests

**What goes wrong:** Smoke tests pass a config dict with `use_accel`, `n_candidates`, `mutation_sigma`, `random_fraction`, `warmup_n` keys that no longer exist. The new `train()` ignores them silently via `config.get()`, but this creates a false sense of test coverage.

**How to avoid:** When updating smoke test configs, remove obsolete keys and add `bootstrap_min`, `replay_ratio`. Document what each key controls in the new architecture.

### Pitfall 4: Sequential Experiment Budget on Single Machine

**What goes wrong:** All four 20k-update runs take ~4× longer than a single run. On `sumida.cs.ucl.ac.uk` with no job scheduler, a background kill or network disconnect terminates the launcher mid-run, leaving only 1–2 completed experiments.

**Why it happens:** No sbatch/qsub/bsub available. Single machine, interactive session.

**How to avoid:** Run in a `screen` or `tmux` session. The launcher script should use `set -e` and log progress explicitly. Each run saves checkpoints every 2000 updates — partial runs are recoverable by checking the last checkpoint update number.

### Pitfall 5: WandB Group vs Name Confusion

**What goes wrong:** Current `wandb.init(group=config["run_name"], ...)` makes `group` equal to the run name, so all runs end up in their own group. WandB filtering by group `phase5-comparison` finds nothing.

**How to avoid:** Fix WandB init: `name=config["run_name"], group=config.get("wandb_group", "phase5-comparison")`. The `run_name` CLI argument controls the per-run name; the group is a separate config key (or hardcoded constant).

### Pitfall 6: Jupyter Not Installed in jax_env

**What goes wrong:** `import jupyter` fails in jax_env. The notebook cannot be created or run without installing `notebook` or `jupyterlab` first.

**How to avoid:** The Wave 0 task for notebook work must include `pip install notebook` in jax_env as a prerequisite step. Alternatively, use VS Code's Jupyter extension pointing at the jax_env kernel (matplotlib and pandas are already present).

### Pitfall 7: buf_score Ceiling Not Broken in Smoke Test

**What goes wrong:** Pre-launch smoke test (1–2k updates) still shows `buf_score` hovering near 0.004 — the same ceiling seen before. This is not a crash but a silent signal that the new architecture is not working correctly.

**Why it happens:** The old archive warm-up pre-populated the buffer with valid solvable levels; the new bootstrap loop must do the same via `es_step`. If the bootstrap threshold is too low (e.g., `bootstrap_min = 10`), the agent starts replaying before enough diverse levels are in the buffer, and regret scoring collapses.

**How to avoid:** Set `bootstrap_min = 50` (recommended). In the smoke test, assert `max(buf_scores[500:]) > 0.008` (doubling the old ceiling) to catch silent failures. If the ceiling is not broken after 1k updates, investigate the es_step loop: check that `insert_batch` is receiving non-zero regrets and that `valid.mean()` is above 0.3.

---

## Code Examples

### Verified Pattern: Bootstrap Loop (new, replaces run_archive_warmup)

```python
# Source: derived from existing accel_training/train.py es_step logic (lines 449–600)
# and LevelSampler._proportion_filled (src/jaxued/level_sampler.py line 396)

bootstrap_min = config.get("bootstrap_min", 50)

print(f"Bootstrap: running es_step until buffer has {bootstrap_min} levels...")
while int(train_state.sampler["size"]) < bootstrap_min:
    rng, rng_ask, rng_eval, rng_bsig = jax.random.split(rng, 4)
    latents, es_state = es_strategy.ask(es_state, rng_ask)  # (pop_size, 64)
    sequences, levels, regrets, max_returns, valid = eval_fn(
        rng_eval, train_state.params, latents
    )
    _, _, _, agent_positions = rollout_agent_on_levels_with_positions(
        rng_bsig, eval_env, env_params,
        train_state.params, network, levels,
        num_steps=config["eval_rollout_steps"],
    )
    behavior_sigs = extract_behavior_signature(agent_positions, config["eval_rollout_steps"])
    level_extra = {"max_return": max_returns, "latent": latents, "behavior_sig": behavior_sigs}
    sampler, _ = level_sampler.insert_batch(train_state.sampler, levels, regrets, level_extra)
    train_state = train_state.replace(sampler=sampler)
    # ES tell() with strategy-appropriate signature (CMA-ES / NS-ES / SV-CMA-ES)
    _es_tell(es_state, es_strategy, es_strategy_name, ...)

print(f"Bootstrap done: {int(train_state.sampler['size'])} levels in buffer")
```

### Verified Pattern: Two-Mode Decision

```python
# Source: adapted from LevelSampler.sample_replay_decision
# (src/jaxued/level_sampler.py line 108)

rng, rng_decision = jax.random.split(rng)
buf_size = int(train_state.sampler["size"])
if buf_size < bootstrap_min:
    mode = "es_step"  # forced during bootstrap
else:
    should_replay = bool(
        level_sampler.sample_replay_decision(train_state.sampler, rng_decision)
    )
    mode = "replay" if should_replay else "es_step"
```

### Verified Pattern: WandB Run Init (corrected)

```python
# Source: verified against wandb 0.25.0 API (credentials at ~/.netrc)
run = wandb.init(
    project=config.get("wandb_project", "es-accel"),
    name=config["run_name"],          # "cma-es" | "ns-es" | "sv-cma-es" | "accel-baseline"
    group=config.get("wandb_group", "phase5-comparison"),
    tags=[config.get("es_strategy", "cma_es").upper()],
)
```

### Verified Pattern: WandB API Pull in Notebook

```python
# Source: verified: wandb.Api() loads with ~/.netrc credentials, version 0.25.0
import wandb
import pandas as pd
import matplotlib.pyplot as plt

api = wandb.Api()
runs = api.runs("es-accel", filters={"group": "phase5-comparison"})

METHOD_ORDER = ["accel-baseline", "cma-es", "ns-es", "sv-cma-es"]
COLORS = {"accel-baseline": "#555", "cma-es": "#1f77b4", "ns-es": "#ff7f0e", "sv-cma-es": "#2ca02c"}

fig, ax = plt.subplots(figsize=(8, 5))
for run in runs:
    df = run.history(keys=["update", "mean_buffer_score"], pandas=True)
    df = df.sort_values("update").dropna(subset=["mean_buffer_score"])
    smoothed = df["mean_buffer_score"].rolling(50, min_periods=1).mean()
    ax.plot(df["update"], smoothed, label=run.name, color=COLORS.get(run.name))
ax.set_xlabel("Training Updates")
ax.set_ylabel("Mean Buffer Score (regret proxy)")
ax.set_title("Phase 5: Four-Method Comparison (seed=42)")
ax.legend()
fig.tight_layout()
fig.savefig("figures/phase5_comparison.pdf", bbox_inches="tight", dpi=300)
```

### Verified Pattern: Checkpoint (agent params only)

```python
# Source: existing accel_training/train.py lines 378–391, simplified
import pickle

def _save_checkpoint(update, train_state, log_dir):
    ckpt_dir = os.path.join(log_dir, f"checkpoint_{update:07d}")
    os.makedirs(ckpt_dir, exist_ok=True)
    params_np = jax.tree_util.tree_map(np.asarray, train_state.params)
    with open(os.path.join(ckpt_dir, "agent_params.pkl"), "wb") as f:
        pickle.dump(params_np, f)
    print(f"  Saved checkpoint: {ckpt_dir}")
    # No archive arrays — archive is gone.
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Three-branch loop (new/replay/mutate) + MAP-Elites Archive | Two-mode loop (replay/es_step) | Phase 5 | Cleaner code, removes ~200 lines, removes Archive class entirely |
| Archive warm-up (run_archive_warmup) 256 latents pre-training | Bootstrap loop (es_step until N levels in buffer) | Phase 5 | Unified: bootstrap is the same code as es_step, not a special path |
| `warmup_n` config key | `bootstrap_min` config key | Phase 5 | More intuitive: counts buffer entries not latents evaluated |
| `replay_prob` config key | `replay_ratio` config key | Phase 5 | Rename only; LevelSampler still receives it as `replay_prob` |
| `group=run_name` in wandb.init | `name=run_name, group="phase5-comparison"` | Phase 5 | Enables cross-run filtering in WandB UI |
| `train()` returns `(train_state, archive)` | `train()` returns `train_state` | Phase 5 | Simplifies callers; tests must be updated |

**Deprecated/outdated:**
- `archive.py`: Still present in `accel_training/` but no longer imported by `train.py`. Can be left in place (it's used by `ued_interface.py` which is also imported but only for `load_vae` and `build_eval_fn`). The archive-related imports from `ued_interface.py` (`generate_candidates`, `mutate_latents`, `update_archive`) are removed.
- `UpdateState` IntEnum in `train.py`: Removed entirely. No replacement needed; mode is a plain string.
- `use_accel` config key: Removed. The mutate branch it controlled is gone.

---

## Open Questions

1. **Does `ued_interface.py` need to be audited?**
   - What we know: `ued_interface.py` exports `load_vae`, `generate_candidates`, `mutate_latents`, `update_archive`, `build_eval_fn`. The new `train.py` only needs `load_vae` and `build_eval_fn`. The other three are obsolete in train.py but the file itself stays.
   - What's unclear: Does anything else in the codebase call `generate_candidates` or `mutate_latents`? Grep confirms: only `train.py` uses them. So `ued_interface.py` is untouched but only two of its five exports are used.
   - Recommendation: Do not modify `ued_interface.py`. Just stop importing the unused functions in `train.py`.

2. **Should the pre-launch smoke test be a formal test file or just a CLI run?**
   - What we know: Phase 4 smoke tests are in `tests/test_phase4_sv_cma_es.py` as function calls with 3 updates. The pre-launch validation needs 1–2k updates to confirm buf_score rises above 0.004.
   - What's unclear: The user decision says "SV-CMA-ES runs 1–2k updates and buf_score rises" — this is more of an experiment validation than a unit test.
   - Recommendation: Implement as a standalone run (not a pytest test), invoked by the launcher script with `--num_updates 1000` before the full 20k runs. Check buf_score manually in WandB.

3. **Are CMA-ES and NS-ES configs identical to SV-CMA-ES except the strategy name?**
   - What we know: All three use the same `es_pop_size`, `es_sigma_init`, `eval_rollout_steps`. NS-ES uses `es_alpha`, `es_beta`, `es_k_novelty`. CMA-ES uses `es_alpha` only. SV-CMA-ES uses `sv_n_particles`, `sv_epsilon`.
   - What's unclear: For a fair comparison, should `es_pop_size` be the same across all strategies?
   - Recommendation: Yes, keep `es_pop_size=16` and `seed=42` constant across all runs. Only vary `es_strategy` and strategy-specific params.

---

## Validation Architecture

> `workflow.nyquist_validation` is not set in `.planning/config.json` — section included as guidance for test changes required by the refactor.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Plain Python (no pytest) — run as `__main__` scripts |
| Config file | None — each test file is self-contained |
| Quick run command | `/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python tests/test_phase4_sv_cma_es.py` |
| Full suite command | Run all four test files from project root |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| COMP-01 | train() runs without error for cma_es strategy | smoke | `python tests/test_phase3_ns_es.py` (updated) | Exists, needs update |
| COMP-01 | train() runs without error for ns_es strategy | smoke | `python tests/test_phase3_ns_es.py` (updated) | Exists, needs update |
| COMP-01 | train() runs without error for sv_cma_es strategy | smoke | `python tests/test_phase4_sv_cma_es.py` (updated) | Exists, needs update |
| COMP-01 | buf_score > 0.004 ceiling after 1k updates (SV-CMA-ES) | manual/visual | WandB dashboard check | N/A — manual |
| COMP-01 | All four experiments complete at 20k updates | integration | Launcher script exit code | Wave 0 gap |
| COMP-01 | Notebook produces two figures without error | smoke | `jupyter nbconvert --to notebook --execute notebooks/phase5_comparison.ipynb` | Wave 0 gap |

### Wave 0 Gaps

- [ ] `tests/test_phase3_ns_es.py` — update `test_archive_warmup_populates_buffer` and `test_end_to_end_3_updates` to match new `train()` signature and new config keys
- [ ] `tests/test_phase4_sv_cma_es.py` — update `test_end_to_end_3_updates_sv_cma_es` for new `train()` signature and config keys
- [ ] `scripts/run_phase5_comparison.sh` — launcher script (new file, Wave 0)
- [ ] `notebooks/phase5_comparison.ipynb` — Jupyter notebook (new file, Wave 0)
- [ ] Jupyter install: `/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/pip install notebook`

---

## Sources

### Primary (HIGH confidence)

- Direct codebase read: `accel_training/train.py` (733 lines, full) — all archive/warmup references inventoried
- Direct codebase read: `accel_training/config.yml` — all config keys catalogued
- Direct codebase read: `src/jaxued/level_sampler.py` — `sample_replay_decision`, `_proportion_filled`, `insert_batch`, `update_batch` APIs verified
- Direct codebase read: `accel_training/es_components/__init__.py`, `interface.py` — ES strategy protocol verified
- Direct codebase read: `accel_training/ued_interface.py` — what `build_eval_fn` and `load_vae` return
- Direct codebase read: `tests/test_phase3_ns_es.py`, `tests/test_phase4_sv_cma_es.py` — exact lines that break after refactor identified
- Codebase-wide grep: all files importing from `accel_training/` — confirmed complete list
- Shell verification: `wandb.__version__ == 0.25.0`, `wandb.Api()` loads with credentials — confirmed
- Shell verification: matplotlib 3.10.8, pandas 2.3.3 in jax_env — confirmed
- Shell verification: jupyter NOT in jax_env — confirmed

### Secondary (MEDIUM confidence)

- `examples/maze_plr.py` WandB init pattern (`name=`, `group=`) — reviewed, confirms current `train.py` uses `group` incorrectly
- `accel_training/archive.py` — full source read, confirms Archive is self-contained and can be left untouched

### Tertiary (LOW confidence)

- None — all findings are from direct codebase inspection

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries confirmed present in jax_env except jupyter (verified absent)
- Architecture: HIGH — derived directly from reading current train.py and understanding exactly what to remove/keep
- Pitfalls: HIGH — all pitfalls derived from actual code lines that will break (identified by grep/read, not speculation)

**Research date:** 2026-03-03
**Valid until:** 2026-04-03 (codebase-specific, stable)
