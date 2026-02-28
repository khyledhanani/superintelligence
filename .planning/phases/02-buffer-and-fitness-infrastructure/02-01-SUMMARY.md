---
phase: 02-buffer-and-fitness-infrastructure
plan: 01
subsystem: infra
tags: [jax, evosax, cma-es, protocol, es-strategy, accel]

# Dependency graph
requires:
  - phase: 01-foundation
    provides: behavior_signature and regret_fitness foundation for ES wiring
provides:
  - ESStrategy typing.Protocol with init_state/ask/tell interface
  - CMAESStrategy thin wrapper around evosax CMA_ES satisfying ESStrategy
  - accel_training/es_components package importable from project root
affects:
  - 02-buffer (replay buffer will call ESStrategy.ask/tell)
  - 03-ns-es (NS-ES strategy will implement ESStrategy Protocol)
  - 04-sv-cmaes (SV-CMA-ES strategy will implement ESStrategy Protocol)

# Tech tracking
tech-stack:
  added: [evosax CMA_ES wrapper, typing.Protocol structural subtyping]
  patterns: [ask/tell/init_state Protocol contract, state dict absorbing evosax params]

key-files:
  created:
    - accel_training/es_components/interface.py
    - accel_training/es_components/cmaes_strategy.py
    - accel_training/es_components/__init__.py
  modified: []

key-decisions:
  - "evosax CMA_ES.init() requires 3 args (key, mean, params) not 2 as initially thought — mean sets distribution center, consistent with evolve_envs.py usage"
  - "CMAESStrategy stores es_params inside state dict so callers never touch evosax internals — clean Protocol surface"
  - "No @runtime_checkable on ESStrategy — structural type-check only, no isinstance overhead"
  - "dummy_key=PRNGKey(0) in tell() is correct — evosax uses key only for tie-breaking, not stochastic updates in CMA-ES tell"

patterns-established:
  - "ESStrategy Protocol pattern: all future ES strategies implement init_state/ask/tell and absorb params into state dict"
  - "CMAESStrategy delegation: Protocol wraps evosax API signature differences transparently"

requirements-completed: [INFRA-01]

# Metrics
duration: 3min
completed: 2026-02-28
---

# Phase 2 Plan 1: ES Strategy Interface Summary

**typing.Protocol ESStrategy with init_state/ask/tell contract and CMAESStrategy wrapping evosax CMA_ES with corrected 3-arg init API**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-28T18:51:05Z
- **Completed:** 2026-02-28T18:54:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- ESStrategy Protocol defined with init_state(rng, config), ask(state, rng), tell(state, candidates, fitness) — the common interface all future ES strategies will implement
- CMAESStrategy wraps evosax CMA_ES transparently: ask/tell cycle verified producing candidates of shape (pop_size, param_dim) with no behavioral change to the underlying algorithm
- Package __init__.py exports both ESStrategy and CMAESStrategy; importable as `from accel_training.es_components import ESStrategy, CMAESStrategy`

## Task Commits

Each task was committed atomically:

1. **Task 1: Define ESStrategy Protocol in interface.py** - `f59c0d5` (feat)
2. **Task 2: Implement CMAESStrategy wrapper and package init** - `cdfda1c` (feat)

## Files Created/Modified

- `accel_training/es_components/interface.py` - typing.Protocol ESStrategy with init_state, ask, tell
- `accel_training/es_components/cmaes_strategy.py` - CMAESStrategy thin wrapper around evosax CMA_ES
- `accel_training/es_components/__init__.py` - Package init exporting ESStrategy and CMAESStrategy

## Decisions Made

- Used 3-arg `es.init(rng, mean, params)` to match actual evosax CMA_ES API (plan comment said 2-arg, which was incorrect)
- Stored mean=zeros as default in init_state; config can override via "mean" key for warm-start
- dummy_key=PRNGKey(0) in tell() is correct — evosax key used only for tie-breaking

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Corrected evosax CMA_ES.init() from 2-arg to 3-arg call**
- **Found during:** Task 2 (CMAESStrategy implementation)
- **Issue:** Plan code comment said `es.init(rng, es_params)` (2 args), but actual evosax API is `es.init(key, mean, params)` (3 args). The 2-arg call raises `TypeError: missing 1 required positional argument: 'params'`
- **Fix:** Updated init_state to call `self._es.init(rng, mean, es_params)` with mean defaulting to jnp.zeros(param_dim), consistent with evolve_envs.py line 166
- **Files modified:** accel_training/es_components/cmaes_strategy.py
- **Verification:** Full ask/tell cycle passes; `init_state(rng, {})` returns state dict with es_state and es_params keys
- **Committed in:** cdfda1c (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug fix)
**Impact on plan:** Essential for correctness — the 2-arg call would fail at runtime. Fix is a direct correction of the evosax API mismatch documented in the plan's interface spec vs actual installed version.

## Issues Encountered

None beyond the evosax init API mismatch above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- ESStrategy Protocol established — NS-ES, SV-CMA-ES can implement this same interface for zero training-loop changes
- CMAESStrategy ready for wiring into the training loop (Phase 3)
- ask/tell shape contract verified: (pop_size, param_dim) candidates, (pop_size,) fitness

---
*Phase: 02-buffer-and-fitness-infrastructure*
*Completed: 2026-02-28*
