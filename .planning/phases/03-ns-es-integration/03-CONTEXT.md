# Phase 3: NS-ES Integration - Context

**Gathered:** 2026-03-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire NS-ES into the ACCEL training loop as the first end-to-end ES with composite fitness. Deliver: NSESStrategy implementation, archive warm-up before training, two-bucket sampling, and WandB metrics. This is the MVP that validates the approach before the more complex Behavioral SV-CMA-ES in Phase 4.

</domain>

<decisions>
## Implementation Decisions

### NS-ES Algorithm Design
- **Fitness for tell():** Composite fitness F = α·Regret + β·Novelty — reuses Phase 2 `compute_fitness()` directly. NS-ES and CMA-ES differ only in that NS-ES uses this composite signal rather than regret-only.
- **ask() behavior:** Identical to CMAESStrategy — sample candidates from current latent distribution. The NS-ES distinction is entirely in fitness computation before tell(). No changes to ask() needed.
- **Novelty source:** NSESStrategy reads behavior signatures from the replay buffer for k-NN novelty computation. No separate novelty archive — one data structure, no duplication.
- **Buffer insertion:** Same criterion as all strategies — regret-based insertion into the PLR replay buffer. Behavior signatures are stored per-entry as established in Phase 2.

### Archive Warm-up
- **Latent distribution:** Sample 256 latents from N(0,I) — same initial distribution as CMA-ES. Warm-up and ES are aligned from the start.
- **Timing:** Synchronous — all 256 warm-up evals complete before step 0 of training. Buffer is pre-populated before any training update.
- **Step budget:** Warm-up is overhead (pre-training), does NOT count toward the training step budget. All strategies get the same N steps of actual training.
- **Solvability gate:** Apply BFS solvability check on each decoded maze before evaluating it. Skip unsolvable latents silently (no eval, no insertion). Check `es/` folder for existing BFS solver before implementing from scratch.
- **Failure handling:** On NaN regret or NaN behavior signature: skip and continue with a warning log. Buffer may end up with slightly fewer than 256 entries — training proceeds regardless.

### Two-Bucket Sampling
- **Default p values:** Match ACCEL's existing sampling split from `maze_plr.py` — use as baseline default. Researcher to confirm the exact value.
- **Schedule:** Fixed split throughout training. No annealing for Phase 3 MVP.
- **Empty buffer guard:** If buffer is empty when two-bucket sampler is called, fall back to 100% frontier sampling. Add a guard clause — this should not happen given synchronous warm-up, but guard for safety.
- **Configuration:** p_replay and p_frontier live in the ES config dict only (consistent with how alpha/beta are handled in Phase 2). No new CLI flags for Phase 3.

### Claude's Discretion
- Exact BFS solver implementation (or reuse from `es/` if it exists)
- WandB metric logging frequency and exact key names (must include: `regret`, `novelty_score`, `replay_buffer_size`, `buffer_occupied`)
- NSESStrategy class structure and file location within `es/`
- How to wire warm-up into `maze_plr.py` (function vs inline block)

</decisions>

<specifics>
## Specific Ideas

- Check `es/` folder for a legacy BFS solver from the old branch before writing a new one
- Phase 3 is NS-ES specifically (composite fitness, novelty from buffer) — Phase 4 is SV-CMA-ES (Stein repulsion in behavior space)
- The success criteria include a vanilla CMA-ES baseline comparison run — same random seed, side-by-side regret curve plot

</specifics>

<deferred>
## Deferred Ideas

- None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-ns-es-integration*
*Context gathered: 2026-03-02*
