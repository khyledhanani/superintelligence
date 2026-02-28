---
phase: 01-foundation
plan: 02
subsystem: es
tags: [behavior-signature, jax, jit, diversity, ns-es]
dependency_graph:
  requires: [01-01]
  provides: [extract_behavior_signature, rollout_agent_on_levels_with_positions]
  affects: [phases/02, phases/03, phases/04]
tech_stack:
  added: []
  patterns: [one-hot-sum histogram, jax.nn.one_hot for JIT-safe scatter]
key_files:
  created: [.planning/DECISIONS.md]
  modified: [es/regret_fitness.py]
decisions:
  - DECISION-01: Behavior signature v1 is a 169-element L1-normalized visit-count histogram over the 13x13 CLUTTR maze grid using one-hot-sum (JIT-safe)
metrics:
  duration: 3 min
  completed: 2026-02-28
  tasks_completed: 2
  files_changed: 2
---

# Phase 01 Plan 02: Behavior Signature Implementation Summary

**One-liner:** JIT-compatible 169-cell L1-normalized visit-count histogram (one-hot-sum pattern) as core NS-ES diversity primitive, with DECISIONS.md design log.

## What Was Built

### Functions Added to es/regret_fitness.py

**1. `rollout_agent_on_levels_with_positions`**
- Thin wrapper around `rollout_agent_on_levels` that modifies `step_fn` to also emit `next_state.agent_pos` from the JAX `lax.scan`
- Returns: `rewards (num_steps, pop_size)`, `values (num_steps, pop_size)`, `dones (num_steps, pop_size)`, `agent_positions (num_steps, pop_size, 2)`
- Only emits `agent_pos` (not full `EnvState`) to avoid OOM from `maze_map` which is `(pop_size, padded_H, padded_W, 3)`
- Backward-compatible: existing callers of `rollout_agent_on_levels` are completely unaffected

**2. `extract_behavior_signature`**
- Signature: `extract_behavior_signature(agent_positions, num_steps, grid_h=13, grid_w=13)`
- Returns: `(pop_size, 169)` float32 array, L1-normalized
- Uses the JIT-safe one-hot-sum pattern: `jax.nn.one_hot(cell_idx, num_classes=169).sum(axis=0)`
- Coordinate convention: `agent_positions[..., 0]=col`, `agent_positions[..., 1]=row`
- Marked with `# TODO: EXPERIMENTAL v1` comment referencing `.planning/DECISIONS.md`

### Files Created

**`.planning/DECISIONS.md`**
- DECISION-01 documents v1 behavior signature design rationale
- Status: EXPERIMENTAL — subject to revision after Phase 3 NS-ES validation
- Includes known limitations (bag-of-positions, no temporal ordering) and revisit criteria

## Verification Results

### JIT Compatibility Test
```
Eager call: shape=(2, 169), dtype=float32 -- PASS
jit.lower().compile(): PASS
```
`jax.jit(extract_behavior_signature).lower(dummy_positions, 4).compile()` completed without error.

### Visual Distinctness Test
```
Sparse maze signature (top 1 nonzero cell):
  cell   0 (row=0, col=0): 1.0000

Dense maze signature (top 10 nonzero cells):
  cell  48 (row=3, col=9): 0.0273
  cell  19 (row=1, col=6): 0.0234
  ...

L1 distance between signatures: 2.0000 (expected > 0.5)
Visual distinctness: PASS
```
L1 distance of **2.0000** far exceeds the 0.1 threshold — signatures are qualitatively distinct.

### All 8 Verification Checks
1. `extract_behavior_signature` exists in module — PASS
2. `rollout_agent_on_levels_with_positions` exists in module — PASS
3. Original `rollout_agent_on_levels` signature unchanged — PASS
4. `jit.lower().compile()` passes — PASS
5. Output shape `(pop_size, 169)` float32 — PASS
6. L1 distance sparse vs dense = 2.0000 (> 0.1) — PASS
7. `.planning/DECISIONS.md` exists with DECISION-01 — PASS
8. EXPERIMENTAL v1 TODO comment with DECISIONS.md reference present — PASS

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| Task 1 | `6e789ab` | feat(01-02): add rollout_agent_on_levels_with_positions and extract_behavior_signature |
| Task 2 | `bcc7c6e` | feat(01-02): add DECISIONS.md with DECISION-01 behavior signature design log |

## Deviations from Plan

None — plan executed exactly as written.

Note: Plan instructed using `python` but JAX is installed in the `jax_env` conda environment at `/cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python3`. All verification used this interpreter. This is consistent with the project environment setup discovered in 01-01.

## Self-Check: PASSED

- FOUND: es/regret_fitness.py
- FOUND: .planning/DECISIONS.md
- FOUND: .planning/phases/01-foundation/01-02-SUMMARY.md
- FOUND commit: 6e789ab (Task 1)
- FOUND commit: bcc7c6e (Task 2)
