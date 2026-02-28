# Phase 1: Foundation - Context

**Gathered:** 2026-02-28
**Status:** Ready for planning
**Source:** discuss-phase

<domain>
## Phase Boundary

Phase 1 delivers two independent prerequisites that all subsequent ES work depends on:

1. **Agent verified against DCD** (FOUND-01) — Confirm our PPO/ACCEL agent matches the Facebook Research DCD repo implementation for PPO update and regret computation. Level sampling will intentionally differ (we use ES instead of random mutation). Document all differences.

2. **Behavior signature extraction** (FOUND-02) — Build a JAX-JIT-compatible function to extract a behavior signature from any agent rollout on any maze level. Must be modular enough to swap the implementation later.

This phase does NOT build any ES algorithm, modify the replay buffer, or change training. It only verifies what exists and adds one new function.

</domain>

<decisions>
## Implementation Decisions

### Agent Verification (FOUND-01)

- **Approach:** Code comparison first, then smoke test — do NOT run DCD repo to compare training curves in Phase 1. Build our side first; full comparison happens after ES is working.
- **Scope:** Verify PPO update logic and regret computation (MaxMC). Level sampling will intentionally differ (no random env generation in our version) — document this as intentional.
- **Smoke test:** Run `maze_plr.py` with ACCEL for ~10k steps. Passing bar: no crash, regret > 0 and changing, solve rate between 0-1, WandB logs successfully.
- **Output:** Write all differences (code + smoke test results) to `.planning/phases/01-foundation/AGENT_VERIFICATION.md`. Classify differences as a flat list — no tiers needed. Document everything found.

### Behavior Signature (FOUND-02)

- **What it captures:** DEFER TO RESEARCH. The specific implementation (visit-count histogram, action sequences, stats) should be whatever the researcher recommends for this JAX/maze setup. Flag this as a critical open question.
- **Grid resolution:** DEFER TO RESEARCH.
- **Normalization:** DEFER TO RESEARCH.
- **Key constraint:** Must be JAX-JIT-compatible. All ops via `jax.lax.*`, fixed-size arrays.
- **⚠ CRITICAL FLAG:** The behavior signature is the most important design decision in the entire project. The wrong choice leads to mode collapse or novelty reward hacking. This is explicitly NOT finalized in Phase 1 — Phase 1 builds a v1 implementation that works, and the design is revisited after NS-ES validates the approach.

### Extractor Location and Interface

- **Location:** Extend `es/regret_fitness.py` (not a new file). Behavior extraction sits alongside regret computation since they share the rollout.
- **Execution:** Run as a **separate pass** from regret computation for simplicity during development. Efficiency optimization deferred.
- **Interface:** Claude's discretion — make it modular/swappable. Research will determine exact signature.
- **Marking:** Add `# TODO: EXPERIMENTAL v1 — behavior signature design is NOT final. See .planning/DECISIONS.md` comment in code AND document in `.planning/DECISIONS.md`.
- **Documentation:** Docstring with usage example (no standalone demo script needed).

### DECISIONS.md

- Create `.planning/DECISIONS.md` — a living document tracking key architectural decisions for collaborators.
- First entry: behavior signature design rationale, what v1 implements, and why this needs revisiting.
- This is the "big summary file" the user wants to reference and share with collaborators.

</decisions>

<specifics>
## Specific Ideas

- DCD repo: https://github.com/facebookresearch/dcd
- AGENT_VERIFICATION.md location: `.planning/phases/01-foundation/AGENT_VERIFICATION.md`
- Smoke test duration: ~10k steps
- ES level sampling intentionally differs from DCD — document as intentional, not a bug
- Behavior signature is **not final** — v1 is a starting point to be replaced once NS-ES validates

</specifics>

<deferred>
## Deferred Ideas

- Full training curve comparison vs DCD (deferred to after ES is working)
- Behavior signature redesign based on NS-ES results (Phase 3/5)
- Normalization choices for behavior signatures (left to researcher)
- Grid resolution for histogram (left to researcher)
- Standalone demo script for extractor (out of scope Phase 1)

</deferred>

---

*Phase: 01-foundation*
*Context gathered: 2026-02-28 via discuss-phase*
