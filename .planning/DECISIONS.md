---
# Key Architectural Decisions

This document tracks key architectural decisions for the ES-ACCEL integration project.
Collaborators: refer to this document before modifying any EXPERIMENTAL-marked code.

---

## DECISION-01: Behavior Signature v1 Design

**Date:** 2026-02-28
**Status:** EXPERIMENTAL -- subject to revision after NS-ES validation (Phase 3)
**Requirement:** FOUND-02
**Code location:** es/regret_fitness.py::extract_behavior_signature()
**Marked with:** # TODO: EXPERIMENTAL v1

### Decision

v1 behavior signature is a 169-element (13x13) L1-normalized visit-count histogram
over the CLUTTR maze grid. Each element represents the fraction of rollout steps the
agent spent in that grid cell.

### Rationale

- Visit counts encode "where did the agent spend time?" -- captures maze traversal
  pattern without committing to trajectory ordering or action semantics
- 13x13 = 169 cells matches the CLUTTR maze resolution (inner_dim=13) -- no lossy
  binning in v1; full spatial resolution preserved
- L1 normalization makes signatures comparable across rollout lengths
- One-hot-sum pattern (jax.nn.one_hot(...).sum(axis=0)) is JIT-safe and vmap-safe
  without scatter complications
- Multi-episode accumulation (accumulate ALL steps across episode boundaries) gives
  richer coverage signal than single-episode; AutoReplayWrapper replays same level

### Known Limitations of v1

- Ignores action sequence (circling vs. straight path look the same if cells match)
- Does not capture temporal ordering of visits (bag-of-positions, not trajectory)
- May conflate topologically different mazes that happen to produce similar paths
- If the agent rarely reaches far cells, many bins near-zero carry no signal

### Planned Revisit Criteria

- After Phase 3 NS-ES validation: check if regret curves improve vs. vanilla CMA-ES
- If novelty reward hacks (high novelty, flat regret): redesign signature
- If mode collapse persists despite NS-ES: consider action-sequence or temporal features
- Ablation in Phase 5: compare 13x13 vs 7x7 resolution if k-NN is slow in Phase 2

### Implementation Reference

From RESEARCH.md recommendation:
  Grid resolution: 13x13 = 169 cells (full resolution)
  Normalization: L1 (divide by total steps per row)
  Episode boundary: Ignore (accumulate all steps)
  Temporal order: Ignored (histogram = bag of visits)

---
*Document created: 2026-02-28*
*Next review: After Phase 3 NS-ES training runs*
---
