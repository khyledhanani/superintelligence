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


def _embedding_from_rollout(hstates, actions, dones):
    """Compute 257D embedding from a single rollout's data."""
    done_idx = np.where(dones)[0]
    end = done_idx[0] + 1 if len(done_idx) > 0 else len(dones)
    h_mean = hstates[:end].mean(axis=0)  # (256,)
    a_mean = np.array([actions[:end].astype(np.float32).mean()])  # (1,)
    return np.concatenate([h_mean, a_mean])


def compute_embedding(evaluator, level, n_rollouts=10):
    """Compute 257D behavior embedding for a level, averaged over all rollouts.

    Runs n_rollouts independent rollouts and averages the per-rollout embeddings
    (mean LSTM hidden state + mean action) for a more stable representation.
    """
    batched_levels = Level.stack([level] * n_rollouts)
    results = evaluator._evaluate_batch(batched_levels, n_rollouts)

    # Compute per-rollout embeddings and average
    hstates = results["hstates"]   # (T, N, 256)
    actions = results["actions"]   # (T, N)
    dones = results["dones"]       # (T, N)
    rewards = results["rewards"]   # (T, N)

    per_rollout_emb = []
    returns = np.zeros(n_rollouts)
    solved = np.zeros(n_rollouts, dtype=bool)
    for i in range(n_rollouts):
        emb_i = _embedding_from_rollout(hstates[:, i], actions[:, i], dones[:, i])
        per_rollout_emb.append(emb_i)
        done_idx = np.where(dones[:, i])[0]
        if len(done_idx) > 0:
            end = done_idx[0] + 1
            solved[i] = True
        else:
            end = len(rewards)
        returns[i] = float(np.sum(rewards[:end, i]))

    # --- Mean over all rollouts ---
    mean_emb = np.mean(per_rollout_emb, axis=0)          # (257,) mean embedding
    solve_rate = float(np.mean(solved))                   # fraction solved
    all_returns = returns                                  # (N,) per-rollout returns

    # --- Best single rollout (highest return) ---
    best_idx = int(np.argmax(returns))
    best_return = float(returns[best_idx])

    traj = {
        # all-rollout aggregates
        "solve_rate": solve_rate,
        "all_returns": all_returns,
        "best_return": best_return,
        # single best rollout trajectory
        "observations": results["observations"][:, best_idx],
        "positions": results["positions"][:, best_idx],
        "values": results["values"][:, best_idx],
        "dones": results["dones"][:, best_idx],
        "actions": results["actions"][:, best_idx],
        "rewards": results["rewards"][:, best_idx],
        "entropy": results["entropy"][:, best_idx],
        "hstates": results["hstates"][:, best_idx],
    }
    return mean_emb, traj


def score_and_embed_levels(evaluator, levels, n_rollouts=10):
    """Score levels by SFL and compute 257D embeddings in a single pass.

    Returns (scores, embeddings) — avoids double rollouts when both are needed.
    """
    n = len(levels)
    scores = np.zeros(n)
    embeddings = np.zeros((n, 257), dtype=np.float32)
    for i, level in enumerate(levels):
        emb, traj = compute_embedding(evaluator, level, n_rollouts=n_rollouts)
        p = traj["solve_rate"]
        scores[i] = p * (1.0 - p)
        embeddings[i] = emb
    return scores, embeddings


def token_hash(tokens_array):
    """Deterministic 64-bit hash of a token array for provenance tracking."""
    return int(hashlib.md5(tokens_array.tobytes()).hexdigest()[:16], 16)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(args):
    t0 = time.time()
    inject_pcts = [float(x) / 100.0 for x in args.inject_pcts]

    # --- wandb ---
    if not args.no_wandb:
        import wandb
        wandb_config = vars(args).copy()
        wandb_config["inject_pcts"] = inject_pcts
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            group=args.wandb_group,
            name=args.wandb_name or f"{args.mutation_strategy}_div{args.min_embedding_diversity}_seed{args.seed}",
            config=wandb_config,
        )
    else:
        wandb = None

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

    # --- Proximity gate: compute seed embeddings and per-seed thresholds ---
    proximity_ratio = args.proximity_gate_ratio  # 0 = disabled
    seed_embeddings = None
    proximity_thresholds = None  # per-seed L2 threshold

    if proximity_ratio > 0:
        print(f"\n  Computing seed & buffer embeddings for proximity gate "
              f"(ratio={proximity_ratio:.2f})...")

        # Embed all LLM seeds (mean over n_rollouts)
        seed_embeddings = np.zeros((n_seeds, 257), dtype=np.float32)
        for i, seed_level in enumerate(seeds):
            seed_embeddings[i], _ = compute_embedding(
                evaluator, seed_level, n_rollouts=args.n_scoring_rollouts)

        # Embed organic buffer levels (mean over n_rollouts)
        print(f"  Embedding {buffer_size} buffer levels...")
        buffer_embeddings = np.zeros((buffer_size, 257), dtype=np.float32)
        for i in range(buffer_size):
            buffer_embeddings[i], _ = compute_embedding(
                evaluator, buffer_levels[i], n_rollouts=args.n_scoring_rollouts)
            if (i + 1) % 200 == 0:
                print(f"    {i + 1}/{buffer_size}...")

        # Per-seed: distance to nearest buffer level -> threshold
        from scipy.spatial.distance import cdist
        seed_to_buf = cdist(seed_embeddings, buffer_embeddings, metric="euclidean")
        seed_nn_dist = seed_to_buf.min(axis=1)
        proximity_thresholds = proximity_ratio * seed_nn_dist

        for i in range(n_seeds):
            print(f"    Seed {i}: dist_to_buffer={seed_nn_dist[i]:.4f}, "
                  f"threshold={proximity_thresholds[i]:.4f}")

        print(f"  Proximity gate ENABLED: mutation must have "
              f"dist(mut, parent_seed) < {proximity_ratio:.2f} × dist(seed, nearest_buffer)")
    else:
        print(f"  Proximity gate DISABLED")

    # Accumulate eligible levels across rounds
    eligible_levels = []
    eligible_scores = []
    eligible_seed_indices = []
    eligible_embeddings = []  # 257D embeddings for diversity gating
    total_generated = 0
    total_scored = 0
    total_diversity_rejected = 0
    total_proximity_rejected = 0
    round_num = 0
    batch_per_seed = args.batch_per_seed  # mutants per seed per round
    min_emb_div = args.min_embedding_diversity  # L2 threshold (0 = disabled)

    if min_emb_div > 0:
        print(f"  Embedding diversity gate ENABLED: min L2 distance = {min_emb_div:.3f}")
    else:
        print(f"  Embedding diversity gate DISABLED")

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

        # Score + compute embeddings in a single pass (no double rollouts)
        need_embeddings = min_emb_div > 0 or proximity_ratio > 0
        if need_embeddings:
            print(f"  Scoring + embedding {n_gen} mutants ({args.n_scoring_rollouts} rollouts each)...")
            batch_scores, batch_embeddings = score_and_embed_levels(
                evaluator, batch_levels, n_rollouts=args.n_scoring_rollouts)
        else:
            print(f"  Scoring {n_gen} mutants ({args.n_scoring_rollouts} rollouts each)...")
            batch_scores = score_levels_sfl(evaluator, batch_levels, n_rollouts=args.n_scoring_rollouts)
            batch_embeddings = None
        total_scored += n_gen

        # Filter by SFL score
        sfl_mask = batch_scores > buffer_min_score
        n_sfl_pass = int(sfl_mask.sum())

        # Filter by proximity gate: mutation must stay near its parent seed
        # dist(mutation, parent_seed) < proximity_ratio * dist(parent_seed, nearest_buffer)
        n_prox_rejected = 0
        mask = sfl_mask.copy()
        if proximity_ratio > 0 and n_sfl_pass > 0:
            for i in np.where(sfl_mask)[0]:
                parent_idx = batch_seed_idx[i]
                dist_to_parent = np.linalg.norm(batch_embeddings[i] - seed_embeddings[parent_idx])
                if dist_to_parent > proximity_thresholds[parent_idx]:
                    mask[i] = False
                    n_prox_rejected += 1
            total_proximity_rejected += n_prox_rejected

        n_after_prox = int(mask.sum())

        # Filter by embedding diversity (greedy: each accepted must be > threshold from all prior)
        n_div_rejected = 0
        if min_emb_div > 0 and n_after_prox > 0:
            accepted_arr = np.stack(eligible_embeddings) if eligible_embeddings else None
            for i in np.where(mask)[0]:
                emb_i = batch_embeddings[i]
                if accepted_arr is not None:
                    dists = np.linalg.norm(accepted_arr - emb_i, axis=1)
                    min_dist = dists.min()
                    if min_dist < min_emb_div:
                        mask[i] = False
                        n_div_rejected += 1
                        continue
                # Accepted — append to running array for within-batch checks
                accepted_arr = (np.vstack([accepted_arr, emb_i[None]])
                                if accepted_arr is not None
                                else emb_i[None])

            total_diversity_rejected += n_div_rejected

        n_new = int(mask.sum())
        gate_parts = []
        if proximity_ratio > 0:
            gate_parts.append(f"{n_prox_rejected} proximity-rejected")
        if min_emb_div > 0:
            gate_parts.append(f"{n_div_rejected} diversity-rejected")
        gate_str = f", {', '.join(gate_parts)}" if gate_parts else ""
        print(f"  Eligible this round: {n_new}/{n_gen} "
              f"(SFL pass: {n_sfl_pass}{gate_str}, yield {n_new/max(n_gen,1):.0%})")

        for i in np.where(mask)[0]:
            eligible_levels.append(batch_levels[i])
            eligible_scores.append(batch_scores[i])
            eligible_seed_indices.append(batch_seed_idx[i])
            if need_embeddings:
                eligible_embeddings.append(batch_embeddings[i])

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
    if proximity_ratio > 0:
        print(f"  Proximity-rejected: {total_proximity_rejected} "
              f"(ratio: {proximity_ratio:.2f})")
    if min_emb_div > 0:
        print(f"  Diversity-rejected: {total_diversity_rejected} "
              f"(threshold: {min_emb_div:.3f} L2)")

    if wandb is not None:
        wandb.log({
            "generation/rounds": round_num,
            "generation/total_generated": total_generated,
            "generation/total_scored": total_scored,
            "generation/n_eligible": n_eligible,
            "generation/yield_pct": n_eligible / max(total_scored, 1),
            "generation/total_diversity_rejected": total_diversity_rejected,
            "generation/total_proximity_rejected": total_proximity_rejected,
        })

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
        if wandb is not None:
            wandb.finish()
        return

    if n_eligible > 0:
        print(f"  Eligible SFL: mean={eligible_scores.mean():.4f}, "
              f"range=[{eligible_scores.min():.4f}, {eligible_scores.max():.4f}]")

    # Round-robin per-ancestor selection (best SFL first within each ancestor)
    # This ensures equal representation across LLM seeds rather than letting
    # one high-scoring ancestor dominate the mutation pool.
    unique_ancestors = np.unique(eligible_seed_indices)
    per_ancestor = {}
    for a in unique_ancestors:
        mask = eligible_seed_indices == a
        a_indices = np.where(mask)[0]
        # Sort this ancestor's mutations by score (best first)
        a_sorted = a_indices[np.argsort(eligible_scores[a_indices])[::-1]]
        per_ancestor[a] = list(a_sorted)

    rank_order = []
    while any(len(v) > 0 for v in per_ancestor.values()):
        for a in unique_ancestors:
            if per_ancestor[a]:
                rank_order.append(per_ancestor[a].pop(0))
    rank_order = np.array(rank_order)

    eligible_levels = [eligible_levels[i] for i in rank_order]
    eligible_scores = eligible_scores[rank_order]
    eligible_seed_indices = eligible_seed_indices[rank_order]

    # --- [5/5] Create merged buffers at each injection percentage ---
    print(f"\n[5/5] Creating merged buffers at inject_pcts={[f'{p:.0%}' for p in inject_pcts]}...")
    os.makedirs(args.output_dir, exist_ok=True)

    # Save full eligible pool
    eligible_tokens = np.stack([np.asarray(level_to_tokens(l)) for l in eligible_levels])
    eligible_origin_ids = np.array([token_hash(eligible_tokens[i]) for i in range(len(eligible_tokens))])
    pool_data = dict(
        tokens=eligible_tokens,
        scores=eligible_scores,
        seed_indices=eligible_seed_indices,
        origin_ids=eligible_origin_ids,
    )
    if eligible_embeddings:
        pool_data["embeddings"] = np.stack(eligible_embeddings)
    np.savez(
        os.path.join(args.output_dir, "eligible_pool.npz"),
        **pool_data,
    )
    print(f"  Saved eligible_pool.npz: {n_eligible} levels"
          f"{' (with embeddings)' if eligible_embeddings else ''}")

    # Original buffer tokens + origins
    orig_tokens = buffer_raw["tokens"].copy()
    orig_scores = buffer_raw["scores"].copy()
    orig_timestamps = buffer_raw.get("timestamps", np.zeros(len(orig_scores), dtype=np.int32))
    orig_origins = np.zeros(len(orig_scores), dtype=np.int32)  # 0 = organic
    orig_origin_ids = np.zeros(len(orig_scores), dtype=np.uint64)
    orig_ancestor_ids = np.full(len(orig_scores), -1, dtype=np.int32)  # -1 = organic

    # Score original LLM seeds so they can also be injected
    print(f"\n  Scoring {n_seeds} original LLM seeds...")
    seed_tokens = np.stack([np.asarray(level_to_tokens(s)) for s in seeds])
    seed_scores = score_levels_sfl(evaluator, seeds, n_rollouts=args.n_scoring_rollouts)
    seed_origin_ids = np.array([token_hash(seed_tokens[i]) for i in range(n_seeds)])
    print(f"  LLM seed SFL: mean={seed_scores.mean():.4f}, "
          f"range=[{seed_scores.min():.4f}, {seed_scores.max():.4f}]")

    # Sorted buffer indices (worst first) for replacement
    sorted_buf_idx = np.argsort(orig_scores[:buffer_size])

    injection_summary = {}

    for pct in inject_pcts:
        n_inject_total = int(pct * buffer_size)
        if n_inject_total == 0:
            print(f"  {pct:.0%}: skipped (zero slots)")
            continue

        # Copy buffer and replace worst slots
        merged_tokens = orig_tokens.copy()
        merged_scores = orig_scores.copy()
        merged_timestamps = orig_timestamps.copy()
        merged_origins = orig_origins.copy()
        merged_origin_ids = orig_origin_ids.copy()
        merged_ancestor_ids = orig_ancestor_ids.copy()

        # First inject original LLM seeds (origin=1)
        n_originals = min(n_seeds, n_inject_total)
        replace_idx = 0
        for i in range(n_originals):
            idx = sorted_buf_idx[replace_idx]
            merged_tokens[idx] = seed_tokens[i]
            merged_scores[idx] = float(seed_scores[i])
            merged_origins[idx] = 1  # 1 = LLM seed (original)
            merged_origin_ids[idx] = seed_origin_ids[i]
            merged_ancestor_ids[idx] = i  # seed index
            replace_idx += 1

        # Then inject mutations (origin=2) in remaining slots
        n_mutations = min(n_inject_total - n_originals, n_eligible)
        inj_tokens = eligible_tokens[:n_mutations]
        inj_scores = eligible_scores[:n_mutations]
        inj_origin_ids_mut = eligible_origin_ids[:n_mutations]
        inj_seed_indices = eligible_seed_indices[:n_mutations]

        for i in range(n_mutations):
            idx = sorted_buf_idx[replace_idx]
            merged_tokens[idx] = inj_tokens[i]
            merged_scores[idx] = float(inj_scores[i])
            merged_origins[idx] = 2  # 2 = LLM mutation
            merged_origin_ids[idx] = inj_origin_ids_mut[i]
            merged_ancestor_ids[idx] = int(inj_seed_indices[i])
            replace_idx += 1

        n_injected = n_originals + n_mutations
        all_inj_scores = np.concatenate([seed_scores[:n_originals], inj_scores])
        pct_tag = f"{int(pct * 100)}pct"
        merged_path = os.path.join(args.output_dir, f"merged_buffer_{pct_tag}.npz")
        np.savez_compressed(
            merged_path,
            tokens=merged_tokens,
            scores=merged_scores,
            timestamps=merged_timestamps,
            origins=merged_origins,
            origin_ids=merged_origin_ids,
            ancestor_ids=merged_ancestor_ids,
            size=buffer_size,
            update_num=buffer_raw.get("update_num", 0),
        )

        injection_summary[pct_tag] = {
            "inject_pct": pct,
            "n_injected": n_injected,
            "n_originals": n_originals,
            "n_mutations": n_mutations,
            "injected_score_mean": float(all_inj_scores.mean()),
            "injected_score_min": float(all_inj_scores.min()),
            "replaced_score_max": float(orig_scores[sorted_buf_idx[:n_injected]].max()),
            "path": merged_path,
        }
        print(f"  {pct:.0%}: injected {n_injected} levels "
              f"({n_originals} originals + {n_mutations} mutations, "
              f"SFL [{all_inj_scores.min():.4f}, {all_inj_scores.max():.4f}], "
              f"replaced buffer slots with score ≤ {orig_scores[sorted_buf_idx[:n_injected]].max():.4f})")

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
        "min_embedding_diversity": min_emb_div,
        "total_diversity_rejected": total_diversity_rejected,
        "proximity_gate_ratio": proximity_ratio,
        "total_proximity_rejected": total_proximity_rejected,
        "proximity_thresholds": proximity_thresholds.tolist() if proximity_thresholds is not None else None,
        "n_scoring_rollouts": args.n_scoring_rollouts,
        "agent_checkpoint": args.agent_checkpoint_dir,
        "buffer_npz": args.buffer_npz,
        "seeds_dir": args.seeds_dir,
        "seed": args.seed,
        "elapsed_s": elapsed,
    }
    with open(os.path.join(args.output_dir, "experiment_log.json"), "w") as f:
        json.dump(experiment_log, f, indent=2)

    if wandb is not None:
        wandb.log(experiment_log)
        wandb.finish()

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

    # Scoring & diversity
    parser.add_argument("--n_scoring_rollouts", type=int, default=10,
                        help="Rollouts per mutant for SFL scoring")
    parser.add_argument("--min_embedding_diversity", type=float, default=0.35,
                        help="Min L2 distance in 257D embedding space between accepted "
                             "mutations. Set to 0 to disable. Default 0.35 matches organic "
                             "buffer 1-NN p25.")
    parser.add_argument("--proximity_gate_ratio", type=float, default=0.0,
                        help="Proximity gate: mutation must have dist(mut, parent_seed) < "
                             "ratio * dist(parent_seed, nearest_buffer_level). "
                             "Set to 0 to disable. E.g. 0.5 means mutations must stay "
                             "within half the seed-to-buffer distance.")

    # Output
    parser.add_argument("--output_dir", type=str, default="experiments/results/default")
    parser.add_argument("--seed", type=int, default=0)

    # wandb
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="JAXUED_LLM_INJECTION",
                        help="wandb project name")
    parser.add_argument("--wandb_entity", type=str,
                        default="romain-hautier-university-college-london-ucl-",
                        help="wandb entity/team")
    parser.add_argument("--wandb_group", type=str, default=None,
                        help="wandb group name (e.g. 'diversity_gate_v1')")
    parser.add_argument("--wandb_name", type=str, default=None,
                        help="wandb run name (auto-generated if omitted)")

    # GCS
    parser.add_argument("--gcs_bucket", type=str, default=None,
                        help="GCS bucket for uploading results (e.g. 'ucl-ued-project-bucket')")
    parser.add_argument("--gcs_prefix", type=str, default="llm-exp/injection",
                        help="GCS prefix path within bucket")

    args = parser.parse_args()
    run_experiment(args)
