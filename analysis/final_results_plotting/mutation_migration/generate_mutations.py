"""Generate mutated levels with 0-50 random tile flips from each LLM original.

For each of the 8 LLM originals, for each flip count 0-50, generates mutations
by randomly toggling wall/non-wall cells. Filters to keep only solvable levels
with SFL > min_buffer (where SFL = solve_rate * (1 - solve_rate)).
Computes embeddings using the t=2250 agent checkpoint.

Usage:
    python analysis/final_results_plotting/mutation_migration/generate_mutations.py --seed 3
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'examples'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

CKPT_DIRS = {
    1: "buffer_dumps/llm_injection_fresh/seed1/checkpoints",
    2: "buffer_dumps/llm_injection_fresh/seed2/seed1/checkpoints",
    3: "buffer_dumps/llm_injection_fresh/seed3/seed1/checkpoints",
}


def mutate_tokens(tokens, n_edits, rng, mutation_weights=(1.0, 1.0, 1.0)):
    """Mutate a single token array matching ACCEL's make_level_mutator_minimax.

    Each edit randomly picks one of: NO_OP, FLIP_WALL, MOVE_GOAL with
    probabilities proportional to mutation_weights = (noop, flip_wall, move_goal).
    FLIP_WALL toggles a random cell (not agent/goal): wall->empty or empty->wall.
    MOVE_GOAL moves goal to a random empty cell.
    NO_OP does nothing.

    Tokens: sorted wall positions + agent_pos + goal_pos (variable # walls).
    """
    tok = tokens.copy()
    agent_pos = int(tok[50])
    goal_pos = int(tok[51])

    # Normalize mutation probabilities
    probs = np.array(mutation_weights, dtype=np.float64)
    probs /= probs.sum()

    # Build wall grid from tokens
    grid = np.zeros(170, dtype=bool)
    for w in tok[:50]:
        grid[w] = True
    grid[agent_pos] = False
    grid[goal_pos] = False

    for _ in range(n_edits):
        mutation_type = rng.choice(3, p=probs)  # 0=noop, 1=flip_wall, 2=move_goal

        if mutation_type == 1:  # FLIP_WALL
            # Pick random cell that isn't agent or goal
            candidates = [c for c in range(169) if c != agent_pos and c != goal_pos]
            cell = rng.choice(candidates)
            grid[cell] = not grid[cell]

        elif mutation_type == 2:  # MOVE_GOAL
            # Move goal to a random empty (non-wall, non-agent) cell
            empty = [c for c in range(169) if not grid[c] and c != agent_pos and c != goal_pos]
            if empty:
                new_goal = rng.choice(empty)
                grid[new_goal] = False
                goal_pos = new_goal

        # NO_OP: do nothing

    # Reconstruct tokens
    new_walls = np.sort(np.where(grid[:169])[0])
    n_walls = len(new_walls)
    if n_walls >= 50:
        tok[:50] = new_walls[:50]
    else:
        tok[:n_walls] = new_walls
        # Pad with last wall value (degenerate but preserves token length)
        tok[n_walls:50] = new_walls[-1] if n_walls > 0 else 0
    tok[51] = goal_pos
    return tok


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--n_target", type=int, default=100,
                        help="Target number of valid mutations per (original, flip_count)")
    parser.add_argument("--oversample", type=float, default=3.0,
                        help="Oversample factor to account for filtering")
    parser.add_argument("--max_flips", type=int, default=50)
    parser.add_argument("--flip_range", type=int, nargs=2, default=None, metavar=("START", "END"),
                        help="Only compute flips START..END (inclusive). For distributed runs.")
    parser.add_argument("--cache_dir", type=str,
                        default="analysis/final_results_plotting/mutation_migration/test_cache")
    parser.add_argument("--injection_cache_dir", type=str,
                        default="analysis/tsne_evolution_investigation/llm_injection_fresh/cache_solved")
    parser.add_argument("--buffer_cache_dir", type=str,
                        default="analysis/final_results_plotting/cache_solved")
    parser.add_argument("--ckpt_step", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_rollouts", type=int, default=10)
    parser.add_argument("--mutation_weights", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        metavar=("NOOP", "FLIP_WALL", "MOVE_GOAL"),
                        help="Relative weights for (no_op, flip_wall, move_goal). Default: equal.")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    # Get min buffer SFL
    buf_path = os.path.join(args.buffer_cache_dir, f"emb_solved_s{args.seed}_t2250.npz")
    buf = np.load(buf_path, allow_pickle=True)
    buf_scores = buf["scores"]
    min_buf_sfl = buf_scores.min()
    print(f"Buffer min SFL: {min_buf_sfl:.6f}")

    # Load LLM originals
    inj_path = os.path.join(args.injection_cache_dir, f"emb_solved_s{args.seed}_t2500.npz")
    inj = np.load(inj_path, allow_pickle=True)
    seed_mask = inj["origins"] == 1
    orig_tokens = inj["tokens"][seed_mask]
    orig_aids = inj["ancestor_ids"][seed_mask]
    # Filter out degenerate originals (duplicate wall tokens)
    valid_mask = np.array([len(set(t[:50].tolist())) == 50 for t in orig_tokens])
    if not valid_mask.all():
        excluded = np.where(~valid_mask)[0]
        print(f"Excluding {len(excluded)} degenerate originals (duplicate wall tokens): "
              f"ancestor_ids={orig_aids[~valid_mask].tolist()}")
        orig_tokens = orig_tokens[valid_mask]
        orig_aids = orig_aids[valid_mask]
    n_originals = len(orig_tokens)
    print(f"Using {n_originals} LLM originals")

    mw = args.mutation_weights
    mw_norm = [w / sum(mw) for w in mw]
    print(f"Mutation weights (noop, flip_wall, move_goal): {mw} -> probs {[f'{p:.2f}' for p in mw_norm]}")

    ckpt_dir = os.path.abspath(CKPT_DIRS[args.seed])

    from utils.compute_embeddings import compute_embeddings_solved

    rng = np.random.RandomState(42)
    n_generate = int(args.n_target * args.oversample)

    # Process each flip count, saving incrementally
    flip_dir = os.path.join(args.cache_dir, "per_flip")
    os.makedirs(flip_dir, exist_ok=True)

    flip_start = args.flip_range[0] if args.flip_range else 0
    flip_end = args.flip_range[1] if args.flip_range else args.max_flips
    print(f"Computing flips {flip_start}..{flip_end}")

    for n_flips in range(flip_start, flip_end + 1):
        flip_path = os.path.join(flip_dir, f"flip_{n_flips:02d}.npz")
        if os.path.exists(flip_path):
            d = np.load(flip_path)
            print(f"  flip={n_flips}: already cached ({len(d['tokens'])} levels), skipping")
            continue

        # Generate candidates for all originals
        cand_tokens = []
        cand_aids = []
        cand_tile_diffs = []

        for oi in range(n_originals):
            for _ in range(n_generate):
                mutated = mutate_tokens(orig_tokens[oi], n_flips, rng,
                                        mutation_weights=tuple(args.mutation_weights))
                # Grid-level hamming distance (actual cell flips)
                def _tokens_to_grid(tok):
                    g = np.zeros(170, dtype=bool)
                    for w in tok[:50]:
                        g[w] = True
                    g[int(tok[50])] = False  # agent
                    g[int(tok[51])] = False  # goal
                    return g
                actual_diffs = int((_tokens_to_grid(mutated) != _tokens_to_grid(orig_tokens[oi])).sum())
                cand_tokens.append(mutated)
                cand_aids.append(orig_aids[oi])
                cand_tile_diffs.append(actual_diffs)

        cand_tokens = np.array(cand_tokens)
        cand_aids = np.array(cand_aids, dtype=np.int32)
        cand_tile_diffs = np.array(cand_tile_diffs, dtype=np.int32)

        # Compute embeddings
        emb_solved, solved, solve_rates, _ = compute_embeddings_solved(
            cand_tokens, ckpt_dir, ckpt_step=args.ckpt_step,
            batch_size=args.batch_size, num_rollouts=args.num_rollouts)

        if emb_solved is None:
            print(f"  flip={n_flips}: agent load failed, skipping")
            continue

        # Compute SFL = solve_rate * (1 - solve_rate)
        sfl = solve_rates * (1 - solve_rates)

        # Filter: solvable (solved at least once) AND SFL > min_buffer
        valid = solved & (sfl > min_buf_sfl)
        n_valid = valid.sum()

        # Per-original, take up to n_target
        kept = np.zeros(len(cand_tokens), dtype=bool)
        for oi in range(n_originals):
            oi_mask = valid & (cand_aids == orig_aids[oi])
            oi_indices = np.where(oi_mask)[0]
            if len(oi_indices) > args.n_target:
                oi_indices = rng.choice(oi_indices, args.n_target, replace=False)
            kept[oi_indices] = True

        n_kept = kept.sum()
        print(f"  flip={n_flips}: {n_valid}/{len(cand_tokens)} valid, "
              f"kept {n_kept} (target {n_originals * args.n_target})")

        np.savez_compressed(flip_path,
            tokens=cand_tokens[kept], ancestor_ids=cand_aids[kept],
            tile_diffs=cand_tile_diffs[kept], flip_counts=np.full(n_kept, n_flips, dtype=np.int32),
            embeddings_solved=emb_solved[kept], solve_rates=solve_rates[kept],
            solved=solved[kept], sfl=sfl[kept])

    # Merge all per-flip files into one
    print("\nMerging per-flip files...")
    results = {k: [] for k in ["tokens", "ancestor_ids", "tile_diffs", "flip_counts",
                                 "embeddings_solved", "solve_rates", "solved", "sfl"]}
    for n_flips in range(0, args.max_flips + 1):
        flip_path = os.path.join(flip_dir, f"flip_{n_flips:02d}.npz")
        if os.path.exists(flip_path):
            d = np.load(flip_path)
            for k in results:
                results[k].append(d[k])

    out = {k: np.concatenate(v, axis=0) for k, v in results.items()}
    out["orig_tokens"] = orig_tokens
    out["orig_ancestor_ids"] = orig_aids

    out_path = os.path.join(args.cache_dir, "mutations_embeddings.npz")
    np.savez_compressed(out_path, **out)
    print(f"\nSaved: {out_path}")
    print(f"Total kept: {len(out['tokens'])}")


if __name__ == "__main__":
    main()
