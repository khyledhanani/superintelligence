# Phase 3: Reproducibility Infrastructure - Research

**Researched:** 2026-03-24
**Domain:** Python file I/O (numpy, json, hashlib), bash launch scripts, wandb.Api, argparse
**Confidence:** HIGH — all findings verified directly from codebase; no external library research required

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Level cache layout
- Cache lives inside each run's output directory (e.g. `results/<run_name>/llm_levels/`)
- Files named by step + batch index: `step_05000_idx_003.npy`
- Only LLM-generated levels are cached — not the full PLR buffer
- Only accepted levels are saved to disk; rejected levels are logged in WandB metrics but not cached
- Each `.npy` file has a JSON sidecar with core audit fields: wall_map SHA-256 hash, injection step, gate scores (td_error_emd, solve_rate), accept/reject decision, timestamp

#### Launch script design
- Hardcoded per machine, matching existing `launch_50k_*.sh` conventions
- Two scripts: `launch_llm_injection.sh` (ACCEL+LLM) and `launch_accel_only_control.sh` (ACCEL-only)
- 3 seeds per condition (0, 1, 2)
- LLM injection runs target GPU nodes: albacore, smew, canada (one seed per machine)
- Control runs assigned separately (TPU or sequential on GPUs after LLM runs)
- WandB project: `JAXUED_LLM`
- WandB group names: `accel-llm` / `accel-only`

#### Determinism scope
- JAX-side determinism only: same JAX seed produces identical training trajectory until the first LLM injection event
- LLM API non-determinism accepted — different API calls will produce different mazes across reruns
- Reproducibility ensured by logging everything: cached levels on disk + wall_map hashes in WandB enable full audit trail
- No replay mode needed for this phase

#### Wall-map hash logging
- SHA-256 of `wall_map.tobytes()`, truncated to first 16 hex characters
- Logged as batch summary per injection event (WandB table row with list of hashes per step)

#### Analysis tooling
- Comparison table script (like existing `compare_phase4_results.py`)
- Filters runs by WandB group name (`accel-llm`, `accel-only`) in `JAXUED_LLM` project
- Shows solve rate (mean ± std) plus LLM-specific metrics: acceptance rate, injected count, diversity score mean

### Claude's Discretion
- Exact JSON sidecar field names and structure
- How `--llm_inject_start_step` default is set
- Comparison script output formatting
- Control run machine assignment strategy

### Deferred Ideas (OUT OF SCOPE)
- Cache + replay mode (`--replay_llm_levels <dir>`) for exact post-injection reproduction — future enhancement if needed
- Solve rate curve plots (matplotlib mean ± std shading) — can add during Phase 4 if needed for thesis figures
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| EXPT-02 | Accepted levels are cached to disk (`.npy` + metadata JSON) with wall_map hashes logged to WandB for reproducibility | Level cache in `_do_injection()` hook; `hashlib.sha256(wall_map.tobytes())` + `numpy.save()` + `json.dump()`; `wandb.log()` with hash list already called in injector |
| EXPT-03 | Injection frequency is ablatable via `--llm_inject_start_step` and `--llm_inject_interval` parameters | `--llm_inject_interval` already exists (maps to `injection_interval`); `--llm_inject_start_step` must be added as a rename/alias for `--llm_warmup_steps` or new independent flag; both must appear in `LLMInjectionConfig` and `from_config_dict()` |
</phase_requirements>

---

## Summary

Phase 3 is almost entirely a **plumbing and scripting phase**, not an algorithmic one. The core LLM injection pipeline (Phases 1+2) is complete. This phase adds three independent features: (1) disk caching of accepted levels, (2) wall-map hash logging to WandB, and (3) launch scripts + comparison tooling for the experiment.

The disk cache is the main new code: a small helper that writes `.npy` + JSON sidecar files to `results/<run_name>/llm_levels/` at the end of `_do_injection()`. The wall-map hash extends the existing `wandb.log(log_payload)` call in `injector.py`. Launch scripts follow a fixed pattern already established in `examples/launch_50k_*.sh`. The comparison script follows the `wandb.Api()` pattern established in `vae/compare_accel_vs_cmaes.py`.

One key finding: **`--llm_inject_start_step` does not yet exist**. Currently `--llm_warmup_steps` serves this role (`warmup_steps` field in `LLMInjectionConfig`, checked in `maybe_inject()`). EXPT-03 requires `--llm_inject_start_step` to be independently configurable. The cleanest approach is to rename `--llm_warmup_steps` to `--llm_inject_start_step` in the argparse definition (keeping `llm_warmup_steps` as a backward-compat alias or updating from_config_dict to read both). This avoids adding a new concept and maps directly to the experiment design requirement.

**Primary recommendation:** Add `LevelCache` helper class to `llm/level_cache.py`, call it from `_do_injection()`, extend `wandb.log()` with hash list, rename/alias `--llm_inject_start_step`, write two launch scripts and one comparison script.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `numpy` | already in env | Save `.npy` level files | `np.save()` is exact, lossless, JAX-compatible |
| `json` | stdlib | Write `.json` sidecar metadata | Human-readable, no deps |
| `hashlib` | stdlib | SHA-256 of `wall_map.tobytes()` | Cryptographic hash for audit trail |
| `pathlib.Path` | stdlib | Cache directory construction | Already used in `scripts/analyze_buffers.py` |
| `wandb` | already in env | Log hash lists as WandB tables | Already used in `injector.py` |
| `bash` + `ssh` + `nohup` | system | Launch scripts | Established by all `launch_50k_*.sh` scripts |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `datetime` | stdlib | Timestamp in JSON sidecar | One-liner `datetime.utcnow().isoformat()` |
| `wandb.Api` | already in env | Comparison script: query run history | Established by `vae/compare_accel_vs_cmaes.py` |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `np.save()` for level `.npy` | pickle, npz, hdf5 | np.save is simplest and matches project convention for single arrays |
| SHA-256 truncated to 16 hex | MD5, CRC32 | SHA-256 is collision-resistant; 16-char prefix is compact and sufficient for audit |
| Separate `LevelCache` class | Inline code in `_do_injection()` | Class is cleaner; keeps injector focused; matches project module pattern |

**Installation:** No new packages — all stdlib + existing `numpy`, `wandb`.

---

## Architecture Patterns

### Recommended Project Structure

```
llm/
├── level_cache.py       # NEW: LevelCache helper class (save + hash)
├── injector.py          # MODIFIED: call cache.save_accepted() + extend wandb.log()
├── injection_config.py  # MODIFIED: add llm_inject_start_step field
examples/
├── maze_plr.py          # MODIFIED: add --llm_inject_start_step argparse flag
scripts/
├── compare_llm_results.py  # NEW: WandB comparison table
examples/
├── launch_llm_injection.sh       # NEW
├── launch_accel_only_control.sh  # NEW
```

### Pattern 1: Level Cache Module (`llm/level_cache.py`)
**What:** A small `LevelCache` class initialized with a run directory, called after gate acceptance in `_do_injection()`. Writes `.npy` and `.json` sidecar for each accepted level.
**When to use:** Called once per accepted seed-level (NOT mutations — only LLM-generated seeds per CONTEXT.md).
**Example:**
```python
# Source: design based on CONTEXT.md locked decisions
import hashlib, json, datetime
import numpy as np
from pathlib import Path

class LevelCache:
    def __init__(self, run_dir: str, run_name: str, seed: int):
        self.cache_dir = Path(run_dir) / "llm_levels"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save_accepted(self, level, step: int, idx: int, gate_metrics: dict) -> str:
        """Save accepted LLM level as .npy + JSON sidecar. Returns wall_map hash."""
        wall_map_np = np.asarray(level.wall_map, dtype=bool)

        # SHA-256 hash of wall_map bytes, truncated to 16 hex chars
        hash_hex = hashlib.sha256(wall_map_np.tobytes()).hexdigest()[:16]

        stem = f"step_{step:05d}_idx_{idx:03d}"
        npy_path = self.cache_dir / f"{stem}.npy"
        json_path = self.cache_dir / f"{stem}.json"

        np.save(str(npy_path), wall_map_np)

        sidecar = {
            "wall_map_hash": hash_hex,
            "injection_step": step,
            "batch_index": idx,
            "gate_scores": {
                "td_error_emd": gate_metrics.get("mean_diversity", None),
                "solve_rate": gate_metrics.get("regret", None),
            },
            "accept": True,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }
        json_path.write_text(json.dumps(sidecar, indent=2))
        return hash_hex
```

### Pattern 2: Wall-Map Hash Logging to WandB
**What:** Extend the existing `log_payload` dict in `_do_injection()` with a `wandb.Table` containing one row per injection event, listing all hashes accepted.
**When to use:** In the existing `wandb.log(log_payload)` call after injection.
**Example:**
```python
# Source: design based on existing wandb.log() in injector.py
# Append to existing log_payload:
if accepted_hashes:
    hash_table = wandb.Table(
        columns=["step", "batch_index", "wall_map_hash"],
        data=[[current_step, i, h] for i, h in enumerate(accepted_hashes)]
    )
    log_payload["llm/accepted_level_hashes"] = hash_table
```

### Pattern 3: Launch Script (`launch_llm_injection.sh`)
**What:** Follows the same SSH + nohup + LD_LIBRARY_PATH pattern as `examples/launch_50k_pca_refit_accel.sh`, but with `--use_llm`, `--use_accel`, `--project JAXUED_LLM`, `--run_name accel-llm`.
**When to use:** One seed per GPU machine (albacore=seed0, smew=seed1, canada=seed2).
**Example:**
```bash
#!/bin/bash
# LLM Injection experiment: ACCEL + LLM diversity injection
# 3 seeds x N updates, WandB project JAXUED_LLM, group accel-llm
set -e

export LD_LIBRARY_PATH=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/lib:${LD_LIBRARY_PATH:-}
PYTHON=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python

export WANDB_DIR=/tmp/wandb
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
mkdir -p /tmp/jax_cache logs

# Seed 0 on albacore
ssh albacore "cd /path && nohup $PYTHON examples/maze_plr.py \
    --project JAXUED_LLM --run_name accel-llm \
    --use_accel --use_llm --llm_provider claude-code \
    --llm_inject_start_step 5000 --llm_inject_interval 3000 \
    --seed 0 \
    2>&1 | tee logs/llm_injection_seed0.log &"
```

### Pattern 4: Comparison Script (`scripts/compare_llm_results.py`)
**What:** Uses `wandb.Api()` to query `JAXUED_LLM`, filters by group `accel-llm` and `accel-only`, prints mean±std solve rate + LLM metrics.
**When to use:** After runs complete.
**Example:**
```python
# Source: vae/compare_accel_vs_cmaes.py pattern
import wandb
api = wandb.Api()
runs = api.runs("JAXUED_LLM")  # or "entity/JAXUED_LLM"

for group in ["accel-llm", "accel-only"]:
    group_runs = [r for r in runs if r.config.get("run_name") == group]
    solve_rates = []
    for run in group_runs:
        hist = run.history(keys=["solve_rate/mean", "num_updates"])
        # Get final solve rate
        final = hist["solve_rate/mean"].dropna().iloc[-1] if len(hist) else None
        if final is not None:
            solve_rates.append(final)
    mean = np.mean(solve_rates) if solve_rates else float("nan")
    std = np.std(solve_rates) if solve_rates else float("nan")
    print(f"{group}: {mean:.3f} ± {std:.3f}  (n={len(solve_rates)})")
```

### Anti-Patterns to Avoid
- **Saving mutations to the level cache:** CONTEXT.md says only LLM-generated seeds are cached, not mutations. The cache hook must run before mutation amplification, or must track only the `valid_levels` list (pre-mutation).
- **Hashing the full Level pytree instead of just `wall_map`:** The CONTEXT.md decision is SHA-256 of `wall_map.tobytes()`. Use `np.asarray(level.wall_map, dtype=bool).tobytes()` for determinism.
- **Making `LevelCache` optional/conditional on a flag:** The cache should always write when `use_llm=True`. No extra flag needed.
- **Logging hashes as a flat string:** Use `wandb.Table` rows so the auditor can join on step+index.
- **Using `--llm_warmup_steps` in launch scripts:** The new flag is `--llm_inject_start_step`. After renaming, scripts should use the new name.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SHA-256 hashing | Custom CRC or rolling hash | `hashlib.sha256()` | stdlib, fast, collision-resistant |
| Directory creation | Manual `os.mkdir` with existence check | `Path.mkdir(parents=True, exist_ok=True)` | Already used in `scripts/analyze_buffers.py` |
| WandB run querying | Manual HTTP calls to WandB REST API | `wandb.Api().runs()` + `.history()` | Established by `vae/compare_accel_vs_cmaes.py` |
| numpy array saving | Custom binary format | `np.save()` / `np.load()` | Lossless, exact, JAX-compatible |

---

## Common Pitfalls

### Pitfall 1: `wall_map.tobytes()` byte order depends on dtype
**What goes wrong:** If `wall_map` has different dtype between runs (e.g., `bool` vs `uint8`), the same maze produces different hashes.
**Why it happens:** `np.asarray(level.wall_map)` may give different dtypes depending on how JAX returns the array.
**How to avoid:** Always cast explicitly: `np.asarray(level.wall_map, dtype=bool).tobytes()` before hashing. Lock this in `LevelCache.save_accepted()`.
**Warning signs:** Two hashes that visually look the same maze but differ.

### Pitfall 2: Cache directory path conflicts with checkpointing
**What goes wrong:** `results/<run_name>/llm_levels/` is a sibling of `checkpoints/<run_name>/<seed>/`, but the `results/` directory is created by the post-training eval path in `maze_plr.py`, not during training.
**Why it happens:** The code uses `checkpoints/<run_name>/<seed>/` for checkpoints and `results/<run_name>/<seed>/` for post-eval results (line 1005 in maze_plr.py: `save_loc = og_config['checkpoint_directory'].replace('checkpoints', 'results')`). The LLM cache is a new directory.
**How to avoid:** Use `checkpoints/<run_name>/<seed>/llm_levels/` (alongside the `models/` directory) so the cache is co-located with the run and cleaned up together. OR use a fully separate `llm_levels/<run_name>/<seed>/` root. CONTEXT.md says `results/<run_name>/llm_levels/` — this is fine as long as `mkdir -p` is called on init.
**Warning signs:** `FileNotFoundError` on first injection event if parent dir doesn't exist.

### Pitfall 3: `--llm_inject_start_step` vs `--llm_warmup_steps` naming collision
**What goes wrong:** Both `--llm_inject_start_step` (new, required by EXPT-03) and `--llm_warmup_steps` (existing) exist simultaneously in argparse, causing confusion or double-counting.
**Why it happens:** `LLMInjectionConfig.warmup_steps` is currently populated from `--llm_warmup_steps`. EXPT-03 calls for `--llm_inject_start_step` as the user-facing name.
**How to avoid:** In `maze_plr.py`, rename the argparse flag from `--llm_warmup_steps` to `--llm_inject_start_step`. Update `from_config_dict()` to read `config.get("llm_inject_start_step", config.get("llm_warmup_steps", 5000))` for backward compat. Update `LLMInjectionConfig.warmup_steps` field name to `inject_start_step` or keep as-is and just rename the CLI flag.
**Warning signs:** `maybe_inject()` fires before expected step, or never fires.

### Pitfall 4: `wandb.Table` created inside `wandb.log()` on every step
**What goes wrong:** `wandb.Table` is only valid for `wandb.log()` when it contains new rows. If hashes list is empty, creating an empty table may cause WandB to log a null table entry.
**Why it happens:** `wandb.Table` creation is cheap, but logging empty tables produces confusing WandB artifacts.
**How to avoid:** Guard the table creation: `if accepted_hashes: log_payload["llm/accepted_level_hashes"] = wandb.Table(...)`.

### Pitfall 5: SSH-based launch scripts require passwordless SSH setup
**What goes wrong:** `ssh albacore "nohup ..."` fails if SSH keys are not configured between the head node and GPU machines.
**Why it happens:** The existing `launch_50k_*.sh` scripts are designed to be run directly on the target GPU machine, not via SSH from a head node.
**How to avoid:** Check how existing scripts launch. From the code, all `launch_50k_*.sh` run locally with a for-loop (no SSH). The LLM injection scripts should follow the same pattern: run the script ON the target machine directly. The "one seed per machine" design means the user `ssh`es to each machine and runs the appropriate script. The script itself does NOT ssh further.
**Warning signs:** `ssh: connect to host albacore port 22: Connection refused` on script launch.

---

## Code Examples

Verified patterns from codebase inspection:

### Current `--llm_warmup_steps` in argparse (to be renamed)
```python
# Source: examples/maze_plr.py line 1403
llm_group.add_argument("--llm_warmup_steps", type=int, default=5000,
                       help="No LLM injection before this many training steps")
```
After change:
```python
llm_group.add_argument("--llm_inject_start_step", type=int, default=5000,
                       help="Training step at which LLM injection begins (no injection before this step)")
```

### Current `injection_interval` scheduling in `maybe_inject()`
```python
# Source: llm/injector.py lines 239-251
current_step = eval_step * self.eval_freq
if current_step < self.config.warmup_steps:
    return runner_state
if current_step % self.config.injection_interval != 0:
    return runner_state
```
No logic change needed — just rename `warmup_steps` field to `inject_start_step` if desired.

### Current WandB logging call (to be extended)
```python
# Source: llm/injector.py lines 489-511
log_payload = {
    "llm/injected_count": retained_count,
    ...
    "llm/batch_all_rejected_count": self.batch_all_rejected_count,
}
try:
    wandb.log(log_payload)
```
Extend by adding hash table before the `wandb.log()` call.

### Existing launch script pattern (to follow exactly)
```bash
# Source: examples/launch_50k_pca_refit_accel.sh
export LD_LIBRARY_PATH=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/lib:${LD_LIBRARY_PATH:-}
PYTHON=/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python
export WANDB_DIR=/tmp/wandb
export JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache
mkdir -p /tmp/jax_cache logs
for seed in 0 1 2; do
  $PYTHON examples/maze_plr.py $COMMON ... --seed $seed --run_name "..." \
    2>&1 | tee logs/...seed${seed}.log
done
```

### WandB API pattern for comparison script
```python
# Source: vae/compare_accel_vs_cmaes.py lines 122-134
api = wandb.Api()
path = "JAXUED_LLM"  # or "entity/JAXUED_LLM"
wandb_runs = api.runs(path)
for run in wandb_runs:
    run_name = run.config.get("run_name", run.group)
    history = run.history(keys=["num_updates", "solve_rate/mean",
                                "llm/acceptance_rate", "llm/injected_count",
                                "llm/diversity_score_mean"])
```

### Saving a level's wall_map as `.npy`
```python
# Pattern: numpy save of JAX array
wall_map_np = np.asarray(level.wall_map, dtype=bool)  # (H, W) bool
np.save("/path/to/step_05000_idx_003.npy", wall_map_np)
# Load back:
wall_map = np.load("/path/to/step_05000_idx_003.npy")
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| No level caching | Disk cache + JSON sidecar | Phase 3 | Enables audit trail for LLM injection experiments |
| `--llm_warmup_steps` flag name | `--llm_inject_start_step` | Phase 3 | More intuitive name, matches EXPT-03 requirement |

**Deprecated/outdated:**
- `--llm_warmup_steps`: rename to `--llm_inject_start_step` (keep backward compat read in `from_config_dict()`)

---

## Open Questions

1. **Cache directory root: `results/` vs `checkpoints/`**
   - What we know: CONTEXT.md says `results/<run_name>/llm_levels/`. The `results/` dir is created by post-eval code, not during training. Injector needs it during training.
   - What's unclear: Should the cache be inside `checkpoints/<run_name>/<seed>/llm_levels/` (created during training) or `results/<run_name>/llm_levels/` (need explicit mkdir)?
   - Recommendation: Use `checkpoints/<run_name>/<seed>/llm_levels/` for co-location with run artifacts, since `checkpoints/<run_name>/<seed>/` is created during training setup. CONTEXT.md's `results/<run_name>/llm_levels/` phrasing is illustrative — planner should pick the path and document it.

2. **Control run machine assignment**
   - What we know: CONTEXT.md says "Control runs assigned separately (TPU or sequential on GPUs after LLM runs)" — this is Claude's discretion.
   - Recommendation: For simplicity, run control runs sequentially on the same GPU machines after LLM injection runs complete (one seed per machine). The existing `accel-baseline` runs in `JAXUED_50K` may suffice if seeds 0-2 match — but they're in a different WandB project. Fresh control runs in `JAXUED_LLM` project are cleaner for comparison.

3. **`--llm_inject_start_step` default value**
   - What we know: Current `--llm_warmup_steps` default is 5000. This is Claude's discretion.
   - Recommendation: Keep default at 5000 — it gives the PLR buffer time to populate before injecting LLM levels, which is already validated behavior.

4. **Should mutations be saved to the level cache?**
   - What we know: CONTEXT.md explicitly states "Only LLM-generated levels are cached — not the full PLR buffer" and "Only accepted levels are saved." Mutations are derived from accepted seeds.
   - Recommendation: Do NOT cache mutations. Only cache the `valid_levels` list (the direct LLM outputs that passed the gate), before mutation amplification. This matches the CONTEXT.md intent.

---

## Validation Architecture

> `workflow.nyquist_validation` is not set in `.planning/config.json` (key absent). Treating as false — skipping Validation Architecture section.

---

## Sources

### Primary (HIGH confidence)
- `/cs/student/project_msc/2025/csml/gmaralla/superintelligence/llm/injector.py` — full injection pipeline, existing WandB log structure, scheduling logic
- `/cs/student/project_msc/2025/csml/gmaralla/superintelligence/llm/injection_config.py` — current CLI flag names, `from_config_dict()` mapping, existing fields
- `/cs/student/project_msc/2025/csml/gmaralla/superintelligence/examples/maze_plr.py` — argparse definitions (lines 1391-1431), run directory structure (line 396), training loop (lines 1079-1097)
- `/cs/student/project_msc/2025/csml/gmaralla/superintelligence/examples/launch_50k_pca_refit_accel.sh` — canonical launch script pattern to follow
- `/cs/student/project_msc/2025/csml/gmaralla/superintelligence/vae/compare_accel_vs_cmaes.py` — `wandb.Api()` + `run.history()` pattern for comparison script
- `.planning/phases/03-reproducibility-infrastructure/03-CONTEXT.md` — all locked decisions

### Secondary (MEDIUM confidence)
- Python stdlib `hashlib`, `json`, `pathlib` docs (training data, stable APIs) — hashlib.sha256().hexdigest(), Path.mkdir(parents=True, exist_ok=True), json.dumps()

### Tertiary (LOW confidence)
- None

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libs are stdlib or already in environment; verified from codebase
- Architecture: HIGH — patterns extracted directly from existing working code in codebase
- Pitfalls: HIGH — identified by reading actual code paths; wall_map dtype pitfall is concrete

**Research date:** 2026-03-24
**Valid until:** 2026-06-24 (stable stdlib + project code; changes only if maze_plr.py argparse structure changes significantly)
