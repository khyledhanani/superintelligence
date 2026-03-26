"""Stage 2: Mutate LLM seeds, score by learnability, inject into PLR buffer.

Pipeline:
  1. Load agent checkpoint + buffer + LLM seeds
  2. Generate-until-quota: keep mutating in rounds until we have
     `target_eligible` levels that score above the buffer's minimum SFL
  3. From the eligible set, create merged buffers at multiple injection
     percentages (e.g. 5%, 10%, 15%, 20%, 25% of buffer replaced)
  4. Each injected level carries a provenance tag so it can be tracked
     through subsequent buffer dumps

Provenance tracking:
  Each merged buffer .npz gains an `origins` array (int32, same length as
  buffer). Values:
    0 = organic (original buffer level)
    1 = LLM seed (direct)
    2 = LLM mutation (wall_flip / vae_noise / vae_interpolation descendant)
  Plus `origin_ids`: a unique hash per injected level so you can match it
  across dumps to see if it survived training.

Usage:
    python experiments/run_injection_experiment.py \
        --agent_checkpoint_dir /path/to/checkpoint \
        --buffer_npz /path/to/buffer.npz \
        --seeds_dir /path/to/seeds \
        --mutation_strategy wall_flip \
        --target_eligible 1000 \
        --inject_pcts 5,10,15,20,25 \
        --seed 0
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time

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

from jaxued.environments.maze import Level
from jaxued.level_sampler import LevelSampler
from vae_level_utils import level_to_tokens, tokens_to_level

from experiments.mutation_strategies import get_strategy
from llm.agent_evaluator import AgentEvaluator


# ---------------------------------------------------------------------------
# GCS upload
# ---------------------------------------------------------------------------

def _upload_to_gcs(local_path, gcs_bucket, gcs_path):
    """Upload a local file to GCS."""
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(gcs_bucket)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
    except (ImportError, Exception) as e:
        import subprocess
        dest = f"gs://{gcs_bucket}/{gcs_path}"
        subprocess.run(["gcloud", "storage", "cp", local_path, dest], check=True)
    print(f"[GCS] {os.path.basename(local_path)} -> gs://{gcs_bucket}/{gcs_path}")


def _upload_dir_to_gcs(local_dir, gcs_bucket, gcs_prefix):
    """Upload all files in a directory to GCS."""
    for fname in os.listdir(local_dir):
        fpath = os.path.join(local_dir, fname)
        if os.path.isfile(fpath):
            _upload_to_gcs(fpath, gcs_bucket, f"{gcs_prefix}/{fname}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_seeds(seeds_dir: str):
    """Load pre-harvested LLM seed levels."""
    pkl_path = os.path.join(seeds_dir, "seeds_levels.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            seeds = pickle.load(f)
        print(f"[Seeds] Loaded {len(seeds)} seed levels from {pkl_path}")
        return seeds

    npz_path = os.path.join(seeds_dir, "seeds.npz")
    data = np.load(npz_path, allow_pickle=True)
    tokens = data["tokens"]
    seeds = [tokens_to_level(jnp.array(tokens[i])) for i in range(len(tokens))]
    print(f"[Seeds] Loaded {len(seeds)} seed levels from {npz_path}")
    return seeds


def load_buffer_levels(buffer_npz_path: str):
    """Load buffer levels and scores from .npz dump."""
    data = np.load(buffer_npz_path, allow_pickle=True)
    tokens = data["tokens"]
    scores = data["scores"]
    size = int(data["size"])
    levels = [tokens_to_level(jnp.array(tokens[i])) for i in range(size)]
    print(f"[Buffer] Loaded {size} levels, score range "
          f"[{scores[:size].min():.4f}, {scores[:size].max():.4f}]")
    return levels, scores[:size], size, data


def score_levels_sfl(evaluator, levels, n_rollouts=10):
    """Score levels using SFL learnability: p*(1-p)."""
    scores = np.zeros(len(levels))
    for i, level in enumerate(levels):
        traj = evaluator.evaluate_level_multi_rollout(level, n_rollouts=n_rollouts)
        p = traj["solve_rate"]
        scores[i] = p * (1.0 - p)
    return scores


def token_hash(tokens_array):
    """Deterministic 64-bit hash of a token array for provenance tracking."""
    return int(hashlib.md5(tokens_array.tobytes()).hexdigest()[:16], 16)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(args):
    t0 = time.time()
    inject_pcts = [float(x) / 100.0 for x in args.inject_pcts]

    print(f"\n{'='*60}")
    print(f"Injection Experiment: {args.mutation_strategy}")
    print(f"  target_eligible={args.target_eligible}")
    print(f"  inject_pcts={[f'{p:.0%}' for p in inject_pcts]}")
    print(f"{'='*60}")

    # --- [1/5] Load agent ---
    print(f"\n[1/5] Loading agent from {args.agent_checkpoint_dir}...")
    evaluator = AgentEvaluator.from_checkpoint(
        args.agent_checkpoint_dir,
        checkpoint_step=args.checkpoint_step,
        num_steps=250,
    )

    # --- [2/5] Load buffer ---
    print(f"\n[2/5] Loading buffer from {args.buffer_npz}...")
    buffer_levels, buffer_scores, buffer_size, buffer_raw = load_buffer_levels(args.buffer_npz)
    buffer_min_score = float(buffer_scores.min())
    print(f"  Buffer min SFL: {buffer_min_score:.4f}")

    # --- [3/5] Load seeds ---
    print(f"\n[3/5] Loading LLM seeds from {args.seeds_dir}...")
    seeds = load_seeds(args.seeds_dir)
    n_seeds = len(seeds)

    # --- [4/5] Generate-until-quota ---
    print(f"\n[4/5] Generating mutations until {args.target_eligible} eligible "
          f"(score > {buffer_min_score:.4f})...")

    strategy = get_strategy(
        args.mutation_strategy,
        vae_checkpoint_path=args.vae_checkpoint_path,
        vae_config_path=args.vae_config_path,
        sigma=args.vae_sigma,
        noise_sigma=args.vae_sigma,
        num_edits=args.num_edits,
    )

    rng = jax.random.PRNGKey(args.seed)

    # Accumulate eligible levels across rounds
    eligible_levels = []
    eligible_scores = []
    eligible_seed_indices = []
    total_generated = 0
    total_scored = 0
    round_num = 0
    batch_per_seed = args.batch_per_seed  # mutants per seed per round

    while len(eligible_levels) < args.target_eligible and round_num < args.max_rounds:
        round_num += 1
        rng, mut_rng = jax.random.split(rng)

        print(f"\n  --- Round {round_num} (have {len(eligible_levels)}/{args.target_eligible} eligible) ---")
        print(f"  Generating {batch_per_seed} × {n_seeds} = {batch_per_seed * n_seeds} mutants...")

        mutant_pairs = strategy.mutate(seeds, batch_per_seed, mut_rng)
        n_gen = len(mutant_pairs)
        total_generated += n_gen

        if n_gen == 0:
            print(f"  No solvable mutants this round. Trying again...")
            continue

        # Score this batch
        batch_levels = [pair[0] for pair in mutant_pairs]
        batch_seed_idx = np.array([pair[1] for pair in mutant_pairs])

        print(f"  Scoring {n_gen} mutants ({args.n_scoring_rollouts} rollouts each)...")
        batch_scores = score_levels_sfl(evaluator, batch_levels, n_rollouts=args.n_scoring_rollouts)
        total_scored += n_gen

        # Filter eligible
        mask = batch_scores > buffer_min_score
        n_new = int(mask.sum())
        print(f"  Eligible this round: {n_new}/{n_gen} "
              f"(yield {n_new/max(n_gen,1):.0%})")

        for i in np.where(mask)[0]:
            eligible_levels.append(batch_levels[i])
            eligible_scores.append(batch_scores[i])
            eligible_seed_indices.append(batch_seed_idx[i])

        # Per-seed breakdown this round
        for s in range(n_seeds):
            s_mask = (batch_seed_idx == s) & mask
            if (batch_seed_idx == s).sum() > 0:
                s_total = (batch_seed_idx == s).sum()
                s_eligible = s_mask.sum()
                print(f"    Seed {s}: {s_eligible}/{s_total} eligible")

    n_eligible = len(eligible_levels)
    eligible_scores = np.array(eligible_scores)
    eligible_seed_indices = np.array(eligible_seed_indices)

    print(f"\n  === Generation complete ===")
    print(f"  Rounds: {round_num}, total generated: {total_generated}, "
          f"total scored: {total_scored}")
    print(f"  Eligible: {n_eligible} (overall yield {n_eligible/max(total_scored,1):.0%})")

    if n_eligible == 0:
        print("  WARNING: No eligible mutants found! Saving empty results.")
        os.makedirs(args.output_dir, exist_ok=True)
        json.dump({
            "mutation_strategy": strategy.name,
            "target_eligible": args.target_eligible,
            "n_eligible": 0,
            "total_generated": total_generated,
            "total_scored": total_scored,
            "rounds": round_num,
            "buffer_min_score": buffer_min_score,
            "seed": args.seed,
            "elapsed_s": time.time() - t0,
        }, open(os.path.join(args.output_dir, "experiment_log.json"), "w"), indent=2)
        return

    if n_eligible > 0:
        print(f"  Eligible SFL: mean={eligible_scores.mean():.4f}, "
              f"range=[{eligible_scores.min():.4f}, {eligible_scores.max():.4f}]")

    # Rank all eligible by score (best first)
    rank_order = np.argsort(eligible_scores)[::-1]
    eligible_levels = [eligible_levels[i] for i in rank_order]
    eligible_scores = eligible_scores[rank_order]
    eligible_seed_indices = eligible_seed_indices[rank_order]

    # --- [5/5] Create merged buffers at each injection percentage ---
    print(f"\n[5/5] Creating merged buffers at inject_pcts={[f'{p:.0%}' for p in inject_pcts]}...")
    os.makedirs(args.output_dir, exist_ok=True)

    # Save full eligible pool
    eligible_tokens = np.stack([np.asarray(level_to_tokens(l)) for l in eligible_levels])
    eligible_origin_ids = np.array([token_hash(eligible_tokens[i]) for i in range(len(eligible_tokens))])
    np.savez(
        os.path.join(args.output_dir, "eligible_pool.npz"),
        tokens=eligible_tokens,
        scores=eligible_scores,
        seed_indices=eligible_seed_indices,
        origin_ids=eligible_origin_ids,
    )
    print(f"  Saved eligible_pool.npz: {n_eligible} levels")

    # Original buffer tokens + origins
    orig_tokens = buffer_raw["tokens"].copy()
    orig_scores = buffer_raw["scores"].copy()
    orig_timestamps = buffer_raw.get("timestamps", np.zeros(len(orig_scores), dtype=np.int32))
    orig_origins = np.zeros(len(orig_scores), dtype=np.int32)  # 0 = organic
    orig_origin_ids = np.zeros(len(orig_scores), dtype=np.uint64)

    # Sorted buffer indices (worst first) for replacement
    sorted_buf_idx = np.argsort(orig_scores[:buffer_size])

    injection_summary = {}

    for pct in inject_pcts:
        n_inject = min(int(pct * buffer_size), n_eligible)
        if n_inject == 0:
            print(f"  {pct:.0%}: skipped (not enough eligible)")
            continue

        # Take top n_inject from ranked eligible pool
        inj_tokens = eligible_tokens[:n_inject]
        inj_scores = eligible_scores[:n_inject]
        inj_origin_ids = eligible_origin_ids[:n_inject]

        # Copy buffer and replace worst slots
        merged_tokens = orig_tokens.copy()
        merged_scores = orig_scores.copy()
        merged_timestamps = orig_timestamps.copy()
        merged_origins = orig_origins.copy()
        merged_origin_ids = orig_origin_ids.copy()

        replace_indices = sorted_buf_idx[:n_inject]
        for i, idx in enumerate(replace_indices):
            merged_tokens[idx] = inj_tokens[i]
            merged_scores[idx] = float(inj_scores[i])
            merged_origins[idx] = 2  # 2 = LLM mutation
            merged_origin_ids[idx] = inj_origin_ids[i]

        pct_tag = f"{int(pct * 100)}pct"
        merged_path = os.path.join(args.output_dir, f"merged_buffer_{pct_tag}.npz")
        np.savez_compressed(
            merged_path,
            tokens=merged_tokens,
            scores=merged_scores,
            timestamps=merged_timestamps,
            origins=merged_origins,
            origin_ids=merged_origin_ids,
            size=buffer_size,
            update_num=buffer_raw.get("update_num", 0),
        )

        injection_summary[pct_tag] = {
            "inject_pct": pct,
            "n_injected": n_inject,
            "injected_score_mean": float(inj_scores.mean()),
            "injected_score_min": float(inj_scores.min()),
            "replaced_score_max": float(orig_scores[replace_indices].max()),
            "path": merged_path,
        }
        print(f"  {pct:.0%}: injected {n_inject} levels "
              f"(SFL [{inj_scores.min():.4f}, {inj_scores.max():.4f}], "
              f"replaced buffer slots with score ≤ {orig_scores[replace_indices].max():.4f})")

    # --- Save experiment log ---
    elapsed = time.time() - t0
    experiment_log = {
        "mutation_strategy": strategy.name,
        "target_eligible": args.target_eligible,
        "inject_pcts": [f"{p:.0%}" for p in inject_pcts],
        "n_seeds": n_seeds,
        "batch_per_seed": batch_per_seed,
        "rounds": round_num,
        "total_generated": total_generated,
        "total_scored": total_scored,
        "n_eligible": n_eligible,
        "yield_pct": n_eligible / max(total_scored, 1),
        "buffer_size": buffer_size,
        "buffer_min_score": buffer_min_score,
        "eligible_score_mean": float(eligible_scores.mean()),
        "eligible_score_std": float(eligible_scores.std()),
        "injections": injection_summary,
        "n_scoring_rollouts": args.n_scoring_rollouts,
        "agent_checkpoint": args.agent_checkpoint_dir,
        "buffer_npz": args.buffer_npz,
        "seeds_dir": args.seeds_dir,
        "seed": args.seed,
        "elapsed_s": elapsed,
    }
    with open(os.path.join(args.output_dir, "experiment_log.json"), "w") as f:
        json.dump(experiment_log, f, indent=2)

    print(f"\n[Done] {elapsed:.1f}s")
    print(f"  Output: {args.output_dir}")
    print(f"  eligible_pool.npz: {n_eligible} eligible mutants")
    for pct_tag, info in injection_summary.items():
        print(f"  merged_buffer_{pct_tag}.npz: {info['n_injected']} injected")

    # --- Upload to GCS ---
    if args.gcs_bucket:
        run_tag = os.path.basename(args.output_dir)
        gcs_base = f"{args.gcs_prefix}/{run_tag}"
        print(f"\n[GCS] Uploading to gs://{args.gcs_bucket}/{gcs_base}/...")
        _upload_dir_to_gcs(args.output_dir, args.gcs_bucket, gcs_base)


def parse_inject_pcts(s):
    """Parse comma-separated injection percentages (e.g. '5,10,15,20,25')."""
    return [int(x.strip()) for x in s.split(",")]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM seed injection experiment")

    # Required inputs
    parser.add_argument("--agent_checkpoint_dir", type=str, required=True)
    parser.add_argument("--buffer_npz", type=str, required=True)
    parser.add_argument("--seeds_dir", type=str, required=True,
                        help="Directory from harvest_llm_seeds.py (contains seeds_levels.pkl)")
    parser.add_argument("--checkpoint_step", type=int, default=-1)

    # Mutation config
    parser.add_argument("--mutation_strategy", type=str, default="wall_flip",
                        choices=["wall_flip", "vae_noise", "vae_interpolation"])
    parser.add_argument("--num_edits", type=int, default=3,
                        help="Wall flips per mutation (wall_flip strategy only)")
    parser.add_argument("--target_eligible", type=int, default=1000,
                        help="Keep generating until this many mutants pass the buffer min score")
    parser.add_argument("--batch_per_seed", type=int, default=50,
                        help="Mutants to generate per seed per round")
    parser.add_argument("--max_rounds", type=int, default=20,
                        help="Safety cap on generation rounds")
    parser.add_argument("--inject_pcts", type=parse_inject_pcts, default=[5, 10, 15, 20, 25],
                        help="Comma-separated buffer replacement percentages (e.g. '5,10,15,20,25')")

    # VAE config (for vae_noise, vae_interpolation)
    parser.add_argument("--vae_checkpoint_path", type=str, default="")
    parser.add_argument("--vae_config_path", type=str, default="")
    parser.add_argument("--vae_sigma", type=float, default=0.5,
                        help="Noise sigma for VAE mutation strategies")

    # Scoring
    parser.add_argument("--n_scoring_rollouts", type=int, default=10,
                        help="Rollouts per mutant for SFL scoring")

    # Output
    parser.add_argument("--output_dir", type=str, default="experiments/results/default")
    parser.add_argument("--seed", type=int, default=0)

    # GCS
    parser.add_argument("--gcs_bucket", type=str, default=None,
                        help="GCS bucket for uploading results (e.g. 'ucl-ued-project-bucket')")
    parser.add_argument("--gcs_prefix", type=str, default="llm-exp/injection",
                        help="GCS prefix path within bucket")

    args = parser.parse_args()
    run_experiment(args)
