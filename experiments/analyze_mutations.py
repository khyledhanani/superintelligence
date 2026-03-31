"""Quick analysis: compare SFL scores of LLM seeds vs their wall-flip mutations.

Generates 100 solvable mutants per seed (1000 total from 10 seeds),
scores them, and prints a per-seed comparison.

Usage:
    python experiments/analyze_mutations.py \
        --agent_checkpoint_dir /path/to/checkpoint \
        --seeds_dir /path/to/seeds \
        [--num_edits 5] [--n_per_seed 100] [--n_rollouts 10]
"""
import os
import sys
import argparse

import jax
import jax.numpy as jnp
import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in [
    os.path.join(_project_root, "src"),
    os.path.join(_project_root, "examples"),
    os.path.join(_project_root, "vae"),
    _project_root,
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from vae_level_utils import tokens_to_level
from llm.agent_evaluator import AgentEvaluator
from mutation_strategies import WallFlipMutator


def score_levels_sfl(evaluator, levels, n_rollouts=10):
    scores = np.zeros(len(levels))
    for i, level in enumerate(levels):
        traj = evaluator.evaluate_level_multi_rollout(level, n_rollouts=n_rollouts)
        p = traj["solve_rate"]
        scores[i] = p * (1.0 - p)
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_checkpoint_dir", type=str, required=True)
    parser.add_argument("--seeds_dir", type=str, required=True)
    parser.add_argument("--num_edits", type=int, default=5)
    parser.add_argument("--n_per_seed", type=int, default=100)
    parser.add_argument("--n_rollouts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint_step", type=int, default=-1)
    args = parser.parse_args()

    # Load agent
    print(f"Loading agent from {args.agent_checkpoint_dir}...")
    evaluator = AgentEvaluator.from_checkpoint(
        args.agent_checkpoint_dir,
        checkpoint_step=args.checkpoint_step,
        num_steps=250,
    )

    # Load seeds
    npz_path = os.path.join(args.seeds_dir, "seeds.npz")
    data = np.load(npz_path, allow_pickle=True)
    tokens = data["tokens"]
    seeds = [tokens_to_level(jnp.array(tokens[i])) for i in range(len(tokens))]
    n_seeds = len(seeds)
    print(f"Loaded {n_seeds} LLM seeds")

    # Score original seeds
    print(f"\nScoring {n_seeds} original LLM seeds ({args.n_rollouts} rollouts each)...")
    seed_scores = score_levels_sfl(evaluator, seeds, n_rollouts=args.n_rollouts)

    print(f"\n{'='*70}")
    print(f"ORIGINAL LLM SEEDS")
    print(f"{'='*70}")
    for i, s in enumerate(seed_scores):
        print(f"  Seed {i:2d}: SFL = {s:.4f}")
    print(f"  Mean: {seed_scores.mean():.4f}  Min: {seed_scores.min():.4f}  Max: {seed_scores.max():.4f}")

    # Generate mutations
    print(f"\nGenerating {args.n_per_seed} solvable wall-flip mutants per seed "
          f"({args.num_edits} edits)...")
    mutator = WallFlipMutator(num_edits=args.num_edits)
    rng = jax.random.PRNGKey(args.seed)
    all_mutants = mutator.mutate(seeds, n_per_seed=args.n_per_seed, rng=rng)

    # Group by seed index
    mutants_by_seed = {i: [] for i in range(n_seeds)}
    for level, seed_idx in all_mutants:
        mutants_by_seed[seed_idx].append(level)

    # Score mutations per seed
    print(f"\nScoring {len(all_mutants)} mutants ({args.n_rollouts} rollouts each)...")
    print(f"\n{'='*70}")
    print(f"PER-SEED COMPARISON: Original vs Mutants (num_edits={args.num_edits})")
    print(f"{'='*70}")
    print(f"{'Seed':>6} | {'Original':>10} | {'Mut Mean':>10} | {'Mut Min':>10} | {'Mut Max':>10} | {'Mut Std':>10} | {'Count':>6}")
    print(f"{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*6}")

    all_mut_scores = []
    for i in range(n_seeds):
        mutants = mutants_by_seed[i]
        if not mutants:
            print(f"{i:>6} | {seed_scores[i]:>10.4f} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10} | {0:>6}")
            continue
        mut_scores = score_levels_sfl(evaluator, mutants, n_rollouts=args.n_rollouts)
        all_mut_scores.append(mut_scores)
        print(f"{i:>6} | {seed_scores[i]:>10.4f} | {mut_scores.mean():>10.4f} | "
              f"{mut_scores.min():>10.4f} | {mut_scores.max():>10.4f} | "
              f"{mut_scores.std():>10.4f} | {len(mutants):>6}")

    if all_mut_scores:
        all_mut_scores = np.concatenate(all_mut_scores)
        print(f"\n{'='*70}")
        print(f"AGGREGATE MUTATION STATS")
        print(f"{'='*70}")
        print(f"  Total mutants scored: {len(all_mut_scores)}")
        print(f"  Mean SFL:   {all_mut_scores.mean():.4f}")
        print(f"  Median SFL: {np.median(all_mut_scores):.4f}")
        print(f"  Min SFL:    {all_mut_scores.min():.4f}")
        print(f"  Max SFL:    {all_mut_scores.max():.4f}")
        print(f"  Std SFL:    {all_mut_scores.std():.4f}")
        print(f"  SFL = 0:    {(all_mut_scores == 0).sum()} ({(all_mut_scores == 0).mean():.0%})")
        print(f"  SFL > 0:    {(all_mut_scores > 0).sum()} ({(all_mut_scores > 0).mean():.0%})")
        print(f"  SFL > 0.09: {(all_mut_scores > 0.09).sum()} ({(all_mut_scores > 0.09).mean():.0%})  (above buffer median)")
        print(f"  SFL > 0.16: {(all_mut_scores > 0.16).sum()} ({(all_mut_scores > 0.16).mean():.0%})  (above buffer 75th pct)")


if __name__ == "__main__":
    main()
