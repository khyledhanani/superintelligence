"""Evaluate LLM-generated seed mazes: difficulty, diversity, and deep-dive plots.

Loads agent checkpoint + buffer references + LLM seeds, rolls out the agent,
and produces:
  1. Difficulty comparison: seed SFL scores vs buffer score distribution
  2. Diversity heatmap: pairwise TD error EMD between seeds and references
  3. Per-seed summary: solve rate, return, episode length, regret, entropy
  4. Deep-dive panels: maze renders with agent paths, per-step metrics

Uses the metrics infrastructure from the LLM-diversity-injection branch.

Usage:
    python experiments/evaluate_seeds.py \
        --agent_checkpoint_dir /path/to/checkpoint_10k \
        --buffer_npz /path/to/buffer_dump_10000.npz \
        --seeds_dir /path/to/seeds_10k \
        --output_dir /path/to/seeds_10k/evaluation
"""
import argparse
import json
import os
import pickle
import sys
import time

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

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

from jaxued.environments import Maze
from jaxued.environments.maze import Level
from jaxued.environments.maze.renderer import MazeRenderer

from llm.agent_evaluator import AgentEvaluator

# Metrics from colleague's branch
from metrics.pairwise.td_error_distribution import compute_td_errors, td_error_divergence
from metrics.standalone.regret import compute_regret
from metrics.standalone.per_step_entropy import compute_per_step_entropy


def load_seeds(seeds_dir):
    """Load seed levels from pkl or npz."""
    pkl_path = os.path.join(seeds_dir, "seeds_levels.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            levels = pickle.load(f)
        print(f"  Loaded {len(levels)} seed levels from pkl")
        return levels

    npz_path = os.path.join(seeds_dir, "seeds.npz")
    if os.path.exists(npz_path):
        from vae_level_utils import tokens_to_level
        data = np.load(npz_path, allow_pickle=True)
        levels = [tokens_to_level(jnp.array(t)) for t in data["tokens"]]
        print(f"  Loaded {len(levels)} seed levels from npz")
        return levels

    raise FileNotFoundError(f"No seeds found in {seeds_dir}")


def load_buffer_reference_levels(buffer_npz, n_refs=5):
    """Load the top-n highest-scoring levels from the buffer."""
    from vae_level_utils import tokens_to_level
    data = np.load(buffer_npz, allow_pickle=True)
    tokens = data["tokens"]
    scores = data["scores"]
    size = int(data["size"])
    tokens = tokens[:size]
    scores = scores[:size]

    # Pick highest-scoring levels as references
    top_idx = np.argsort(scores)[-n_refs:][::-1]
    ref_levels = [tokens_to_level(jnp.array(tokens[i])) for i in top_idx]
    ref_scores = scores[top_idx]
    print(f"  Buffer: {size} levels, selected {n_refs} refs (scores: {ref_scores})")
    return ref_levels, ref_scores, scores


def evaluate(args):
    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load agent ---
    print("[1/5] Loading agent...")
    evaluator = AgentEvaluator.from_checkpoint(
        args.agent_checkpoint_dir,
        checkpoint_step=args.checkpoint_step,
        num_steps=250,
    )

    # --- Load seeds ---
    print("[2/5] Loading seeds...")
    seed_levels = load_seeds(args.seeds_dir)
    n_seeds = len(seed_levels)

    # --- Load buffer references ---
    print("[3/5] Loading buffer references...")
    ref_levels, ref_scores, all_buffer_scores = load_buffer_reference_levels(
        args.buffer_npz, n_refs=args.n_references
    )

    # --- Roll out agent on seeds + references ---
    print("[4/5] Rolling out agent...")
    print(f"  Rolling out on {n_seeds} seeds (multi-rollout for SFL)...")
    seed_trajectories = []
    seed_stats = []
    for i, level in enumerate(seed_levels):
        traj = evaluator.evaluate_level_multi_rollout(level, n_rollouts=args.n_rollouts)
        seed_trajectories.append(traj)
        seed_stats.append({
            "index": i,
            "solve_rate": traj["solve_rate"],
            "best_return": traj["best_return"],
            "sfl_score": traj["solve_rate"] * (1 - traj["solve_rate"]),
        })
        p = traj["solve_rate"]
        sfl = p * (1 - p)
        print(f"    Seed {i}: solve={p:.2f}, SFL={sfl:.4f}, return={traj['best_return']:.3f}")

    print(f"  Rolling out on {len(ref_levels)} buffer references...")
    ref_trajectories = []
    ref_stats = []
    for i, level in enumerate(ref_levels):
        traj = evaluator.evaluate_level_multi_rollout(level, n_rollouts=args.n_rollouts)
        ref_trajectories.append(traj)
        p = traj["solve_rate"]
        ref_stats.append({
            "index": i,
            "buffer_score": float(ref_scores[i]),
            "solve_rate": p,
            "sfl_score": p * (1 - p),
            "best_return": traj["best_return"],
        })
        print(f"    Ref {i}: solve={p:.2f}, SFL={p*(1-p):.4f}, buffer_score={ref_scores[i]:.4f}")

    # --- Compute metrics ---
    print("[5/5] Computing metrics and plotting...")

    # Pairwise TD error EMD: seeds vs refs
    td_matrix = np.zeros((n_seeds, len(ref_levels)))
    for i, straj in enumerate(seed_trajectories):
        for j, rtraj in enumerate(ref_trajectories):
            td_matrix[i, j] = td_error_divergence(straj, rtraj)

    # Per-seed regret and entropy
    seed_regrets = []
    seed_entropies = []
    for traj in seed_trajectories:
        reg = compute_regret(traj, max_steps=250)
        ent = compute_per_step_entropy(traj)
        seed_regrets.append(reg)
        seed_entropies.append(ent)

    ref_regrets = []
    ref_entropies = []
    for traj in ref_trajectories:
        reg = compute_regret(traj, max_steps=250)
        ent = compute_per_step_entropy(traj)
        ref_regrets.append(reg)
        ref_entropies.append(ent)

    # === PLOT 1: SFL score comparison ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: histogram of buffer scores + seed scores
    ax = axes[0]
    ax.hist(all_buffer_scores, bins=50, alpha=0.6, color="steelblue", label="Buffer levels")
    seed_sfls = [s["sfl_score"] for s in seed_stats]
    for i, sfl in enumerate(seed_sfls):
        ax.axvline(sfl, color="red", linewidth=2, alpha=0.7,
                   label="LLM seeds" if i == 0 else None)
    ax.set_xlabel("SFL Score [p*(1-p)]")
    ax.set_ylabel("Count")
    ax.set_title("Buffer Score Distribution vs LLM Seed Scores")
    ax.legend()

    # Right: bar chart of seed solve rates
    ax = axes[1]
    x = np.arange(n_seeds)
    colors = ["green" if s["solve_rate"] > 0 else "red" for s in seed_stats]
    ax.bar(x, [s["solve_rate"] for s in seed_stats], color=colors, alpha=0.7)
    ax.set_xlabel("Seed Index")
    ax.set_ylabel("Solve Rate")
    ax.set_title("Agent Solve Rate on LLM Seeds")
    ax.set_xticks(x)
    ax.axhline(0.5, color="grey", linestyle="--", alpha=0.5, label="50% solve")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "sfl_comparison.png"), dpi=150)
    plt.close(fig)
    print("  Saved sfl_comparison.png")

    # === PLOT 2: TD Error EMD Heatmap ===
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(td_matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xlabel("Buffer Reference")
    ax.set_ylabel("LLM Seed")
    ax.set_xticks(range(len(ref_levels)))
    ax.set_xticklabels([f"Ref {j}\n({ref_scores[j]:.3f})" for j in range(len(ref_levels))],
                       fontsize=8)
    ax.set_yticks(range(n_seeds))
    ax.set_yticklabels([f"Seed {i}" for i in range(n_seeds)], fontsize=8)
    ax.set_title("TD Error EMD: LLM Seeds vs Buffer References\n(higher = more diverse)")
    plt.colorbar(im, ax=ax, label="EMD")

    # Annotate cells
    for i in range(n_seeds):
        for j in range(len(ref_levels)):
            ax.text(j, i, f"{td_matrix[i,j]:.3f}", ha="center", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "td_emd_heatmap.png"), dpi=150)
    plt.close(fig)
    print("  Saved td_emd_heatmap.png")

    # === PLOT 3: Per-seed deep-dive (maze + path + metrics) ===
    env = Maze(max_height=13, max_width=13, agent_view_size=5, normalize_obs=True)
    renderer = MazeRenderer(env, tile_size=32)
    env_params = env.default_params

    ncols = 5
    nrows = (n_seeds + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 5 * nrows))
    if nrows == 1:
        axes = [axes] if ncols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i in range(n_seeds):
        ax = axes[i]
        level = seed_levels[i]
        traj = seed_trajectories[i]

        # Render maze
        img = np.asarray(renderer.render_level(level, env_params))
        ax.imshow(img)

        # Overlay agent path
        pos = traj["positions"]
        dones = traj["dones"]
        done_idx = np.where(dones)[0]
        end = int(done_idx[0] + 1) if len(done_idx) > 0 else len(pos)
        pos_trunc = pos[:end]

        # Scale positions to pixel coordinates (tile_size=32, +1 for border)
        ts = 32
        for t in range(len(pos_trunc) - 1):
            alpha = 0.3 + 0.7 * (t / max(len(pos_trunc) - 1, 1))
            ax.plot(
                [(pos_trunc[t, 0] + 1) * ts + ts // 2, (pos_trunc[t+1, 0] + 1) * ts + ts // 2],
                [(pos_trunc[t, 1] + 1) * ts + ts // 2, (pos_trunc[t+1, 1] + 1) * ts + ts // 2],
                color="cyan", alpha=alpha, linewidth=1.5,
            )

        p = traj["solve_rate"]
        sfl = p * (1 - p)
        reg = seed_regrets[i]
        ent = seed_entropies[i]
        min_td = td_matrix[i].min()
        ax.set_title(
            f"Seed {i}\nsolve={p:.0%}  SFL={sfl:.3f}\n"
            f"regret={reg['regret']:.3f}  entropy={ent['mean']:.2f}\n"
            f"min_TD_EMD={min_td:.3f}",
            fontsize=8,
        )
        ax.axis("off")

    for j in range(n_seeds, len(axes)):
        axes[j].axis("off")

    fig.suptitle("LLM Seed Evaluation: Agent Paths + Metrics", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "seed_deepdive.png"), dpi=150)
    plt.close(fig)
    print("  Saved seed_deepdive.png")

    # === PLOT 4: Per-step regret curves (seeds vs refs) ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for i, reg in enumerate(seed_regrets):
        if reg["episode_length"] > 0:
            ax.plot(reg["regret_curve"], alpha=0.6, label=f"Seed {i}")
    ax.set_title("Per-step Regret: LLM Seeds")
    ax.set_xlabel("Step")
    ax.set_ylabel("Regret")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for i, reg in enumerate(ref_regrets):
        if reg["episode_length"] > 0:
            ax.plot(reg["regret_curve"], alpha=0.6, label=f"Ref {i} (score={ref_scores[i]:.3f})")
    ax.set_title("Per-step Regret: Buffer References")
    ax.set_xlabel("Step")
    ax.set_ylabel("Regret")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "regret_curves.png"), dpi=150)
    plt.close(fig)
    print("  Saved regret_curves.png")

    # === Save summary JSON ===
    summary = {
        "n_seeds": n_seeds,
        "n_references": len(ref_levels),
        "n_rollouts": args.n_rollouts,
        "seed_stats": seed_stats,
        "ref_stats": ref_stats,
        "td_emd_matrix": td_matrix.tolist(),
        "td_emd_summary": {
            "mean": float(td_matrix.mean()),
            "min": float(td_matrix.min()),
            "max": float(td_matrix.max()),
        },
        "buffer_score_summary": {
            "mean": float(all_buffer_scores.mean()),
            "std": float(all_buffer_scores.std()),
            "max": float(all_buffer_scores.max()),
        },
        "elapsed_seconds": time.time() - t0,
    }
    with open(os.path.join(args.output_dir, "evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("  Saved evaluation_summary.json")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"{'Seed':>6} {'Solve%':>8} {'SFL':>8} {'Regret':>8} {'MinTD':>8}")
    print(f"{'='*60}")
    for i in range(n_seeds):
        p = seed_stats[i]["solve_rate"]
        sfl = seed_stats[i]["sfl_score"]
        reg = seed_regrets[i]["regret"]
        min_td = td_matrix[i].min()
        print(f"{'S'+str(i):>6} {p:>8.2f} {sfl:>8.4f} {reg:>8.4f} {min_td:>8.4f}")
    print(f"{'='*60}")
    print(f"Buffer mean SFL: {all_buffer_scores.mean():.4f}")
    print(f"Elapsed: {time.time() - t0:.1f}s")
    print(f"Output: {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM seed mazes")
    parser.add_argument("--agent_checkpoint_dir", type=str, required=True)
    parser.add_argument("--buffer_npz", type=str, required=True)
    parser.add_argument("--seeds_dir", type=str, required=True)
    parser.add_argument("--checkpoint_step", type=int, default=-1)
    parser.add_argument("--n_references", type=int, default=5)
    parser.add_argument("--n_rollouts", type=int, default=10,
                        help="Rollouts per level for SFL scoring")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join(args.seeds_dir, "evaluation")
    evaluate(args)
