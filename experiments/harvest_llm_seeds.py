"""Stage 1: Harvest LLM seed levels with decision gate evaluation.

Loads a trained agent and its PLR buffer, queries the LLM for new maze levels
(using the buffer's reference mazes as context), evaluates each candidate
through the decision gate (difficulty + diversity checks), retries with
feedback on rejection, and produces evaluation plots.

Full pipeline per seed:
  1. LLM generates a candidate maze
  2. Agent rolls out on it (multi-rollout for SFL)
  3. Decision gate checks:
     - Difficulty: SFL learnability p*(1-p) > threshold
     - Diversity: pairwise TD error EMD vs reference trajectories > threshold
  4. If rejected: feedback issues go back to LLM for retry
  5. If accepted: seed is stored

Usage:
    python experiments/harvest_llm_seeds.py \
        --agent_checkpoint_dir /path/to/checkpoint_10k \
        --buffer_npz /path/to/buffer_dump_10000.npz \
        --llm_provider claude-code \
        --llm_model claude-sonnet-4-20250514 \
        --n_seeds 10 \
        --output_dir /path/to/seeds_10k

Output:
    seeds.npz           — tokens (N,52), grids (N,), metadata
    seeds_levels.pkl     — pickled list of Level objects
    harvest_log.json     — generation stats, gate results, config, timing
    renders/             — individual seed PNGs + overview grid
    evaluation/          — SFL comparison, TD EMD heatmap, deep-dive plots
"""
import argparse
import json
import os
import pickle
import sys
import time
import yaml

import jax
import jax.numpy as jnp
import numpy as np

# Add project paths
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
from vae_level_utils import level_to_tokens, tokens_to_level

from llm.maze_generator import MazeGenerator, GenerationConfig
from llm.prompt_builder import ReferenceMaze, MetricEntry, build_generation_prompt
from llm.buffer_stats import BufferStatsExtractor
from llm.agent_evaluator import AgentEvaluator
from llm.decision_gate import evaluate_candidate, DiversityThresholds

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_buffer_as_sampler(buffer_npz_path: str) -> dict:
    """Load a buffer .npz dump and reconstruct a sampler-like dict."""
    data = np.load(buffer_npz_path, allow_pickle=True)
    tokens = data["tokens"]
    scores = data["scores"]
    size = int(data["size"])

    levels_list = []
    for i in range(size):
        level = tokens_to_level(jnp.array(tokens[i]))
        levels_list.append(level)

    if levels_list:
        levels_pytree = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs), *levels_list
        )
    else:
        raise ValueError(f"Buffer at {buffer_npz_path} is empty (size=0)")

    sampler = {
        "levels": levels_pytree,
        "scores": jnp.array(scores[:size], dtype=jnp.float32),
        "size": size,
    }
    print(f"[Buffer] Loaded {size} levels from {buffer_npz_path}")
    print(f"[Buffer] Score range: [{scores[:size].min():.4f}, {scores[:size].max():.4f}], "
          f"mean={scores[:size].mean():.4f}")
    return sampler, scores[:size]


def load_llm_config(config_path: str) -> dict:
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def render_all_seeds(seed_levels, output_dir):
    """Render individual seed PNGs + grid overview."""
    from jaxued.environments import Maze
    from jaxued.environments.maze.renderer import MazeRenderer

    os.makedirs(output_dir, exist_ok=True)
    env = Maze(max_height=13, max_width=13, agent_view_size=5, normalize_obs=True)
    renderer = MazeRenderer(env, tile_size=32)
    env_params = env.default_params

    images = []
    for i, level in enumerate(seed_levels):
        img = np.asarray(renderer.render_level(level, env_params))
        images.append(img)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img); ax.set_title(f"Seed {i}"); ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"seed_{i:02d}.png"), dpi=150)
        plt.close(fig)

    # Grid overview
    n = len(images)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if nrows == 1:
        axes = [axes] if ncols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]
    for i, (img, ax) in enumerate(zip(images, axes)):
        ax.imshow(img); ax.set_title(f"Seed {i}", fontsize=10); ax.axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"LLM-Generated Seed Mazes ({n} seeds)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "seeds_overview.png"), dpi=150)
    plt.close(fig)
    return images


def plot_evaluation(seed_stats, ref_stats, td_matrix, all_buffer_scores,
                    seed_levels, seed_trajectories, ref_scores, output_dir):
    """Generate evaluation plots: SFL comparison, TD EMD heatmap, deep-dive."""
    from jaxued.environments import Maze
    from jaxued.environments.maze.renderer import MazeRenderer
    from metrics.standalone.regret import compute_regret
    from metrics.standalone.per_step_entropy import compute_per_step_entropy

    os.makedirs(output_dir, exist_ok=True)
    n_seeds = len(seed_stats)

    # --- Plot 1: SFL comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.hist(all_buffer_scores, bins=50, alpha=0.6, color="steelblue", label="Buffer levels")
    for i, s in enumerate(seed_stats):
        ax.axvline(s["sfl_score"], color="red", linewidth=2, alpha=0.7,
                   label="LLM seeds" if i == 0 else None)
    ax.set_xlabel("SFL Score [p*(1-p)]"); ax.set_ylabel("Count")
    ax.set_title("Buffer Score Distribution vs LLM Seed Scores"); ax.legend()

    ax = axes[1]
    x = np.arange(n_seeds)
    colors_bar = ["green" if s["solve_rate"] > 0 else "red" for s in seed_stats]
    ax.bar(x, [s["solve_rate"] for s in seed_stats], color=colors_bar, alpha=0.7)
    ax.set_xlabel("Seed Index"); ax.set_ylabel("Solve Rate")
    ax.set_title("Agent Solve Rate on LLM Seeds"); ax.set_xticks(x)
    ax.axhline(0.5, color="grey", linestyle="--", alpha=0.5, label="50% solve"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "sfl_comparison.png"), dpi=150)
    plt.close(fig)
    print("    Saved sfl_comparison.png")

    # --- Plot 2: TD EMD heatmap ---
    if td_matrix is not None and td_matrix.size > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(td_matrix, cmap="YlOrRd", aspect="auto")
        ax.set_xlabel("Buffer Reference"); ax.set_ylabel("LLM Seed")
        ax.set_xticks(range(td_matrix.shape[1]))
        ax.set_xticklabels([f"Ref {j}\n({ref_scores[j]:.3f})" for j in range(td_matrix.shape[1])],
                           fontsize=8)
        ax.set_yticks(range(n_seeds))
        ax.set_yticklabels([f"Seed {i}" for i in range(n_seeds)], fontsize=8)
        ax.set_title("TD Error EMD: LLM Seeds vs Buffer References\n(higher = more diverse)")
        plt.colorbar(im, ax=ax, label="EMD")
        for i in range(n_seeds):
            for j in range(td_matrix.shape[1]):
                ax.text(j, i, f"{td_matrix[i,j]:.3f}", ha="center", va="center", fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "td_emd_heatmap.png"), dpi=150)
        plt.close(fig)
        print("    Saved td_emd_heatmap.png")

    # --- Plot 3: Deep-dive with agent paths + metrics ---
    env = Maze(max_height=13, max_width=13, agent_view_size=5, normalize_obs=True)
    renderer = MazeRenderer(env, tile_size=32)
    env_params = env.default_params

    ncols = min(5, n_seeds)
    nrows = (n_seeds + ncols - 1) // ncols
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(4 * ncols, 5 * nrows))
    if nrows == 1:
        axes_flat = [axes_grid] if ncols == 1 else list(axes_grid)
    else:
        axes_flat = [ax for row in axes_grid for ax in row]

    for i in range(n_seeds):
        ax = axes_flat[i]
        level = seed_levels[i]
        traj = seed_trajectories[i]

        img = np.asarray(renderer.render_level(level, env_params))
        ax.imshow(img)

        # Overlay agent path
        pos = traj["positions"]
        dones = traj["dones"]
        done_idx = np.where(dones)[0]
        end = int(done_idx[0] + 1) if len(done_idx) > 0 else len(pos)
        pos_trunc = pos[:end]
        ts = 32
        for t in range(len(pos_trunc) - 1):
            alpha = 0.3 + 0.7 * (t / max(len(pos_trunc) - 1, 1))
            ax.plot(
                [(pos_trunc[t, 0] + 1) * ts + ts // 2, (pos_trunc[t+1, 0] + 1) * ts + ts // 2],
                [(pos_trunc[t, 1] + 1) * ts + ts // 2, (pos_trunc[t+1, 1] + 1) * ts + ts // 2],
                color="cyan", alpha=alpha, linewidth=1.5,
            )

        s = seed_stats[i]
        min_td = td_matrix[i].min() if td_matrix is not None else 0
        ax.set_title(
            f"Seed {i} {'PASS' if s.get('gate_accepted', False) else 'FAIL'}\n"
            f"solve={s['solve_rate']:.0%}  SFL={s['sfl_score']:.3f}\n"
            f"min_TD_EMD={min_td:.3f}",
            fontsize=8,
            color="green" if s.get("gate_accepted", False) else "red",
        )
        ax.axis("off")

    for j in range(n_seeds, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle("LLM Seed Evaluation: Agent Paths + Gate Results", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "seed_deepdive.png"), dpi=150)
    plt.close(fig)
    print("    Saved seed_deepdive.png")

    # --- Plot 4: Per-step regret curves ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, traj in enumerate(seed_trajectories):
        reg = compute_regret(traj, max_steps=250)
        if reg.episode_length > 0:
            status = "PASS" if seed_stats[i].get("gate_accepted", False) else "FAIL"
            ax.plot(reg.regret_curve, alpha=0.6, label=f"Seed {i} ({status})")
    ax.set_title("Per-step Regret: LLM Seeds")
    ax.set_xlabel("Step"); ax.set_ylabel("Regret")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "regret_curves.png"), dpi=150)
    plt.close(fig)
    print("    Saved regret_curves.png")


def harvest(args):
    t0 = time.time()

    # --- Load agent ---
    print(f"\n[1/6] Loading agent from {args.agent_checkpoint_dir}...")
    evaluator = AgentEvaluator.from_checkpoint(
        args.agent_checkpoint_dir,
        checkpoint_step=args.checkpoint_step,
        num_steps=250,
    )

    # --- Load buffer ---
    print(f"\n[2/6] Loading buffer from {args.buffer_npz}...")
    sampler, all_buffer_scores = load_buffer_as_sampler(args.buffer_npz)

    # --- Extract references + roll out agent on them ---
    print(f"\n[3/6] Extracting references and rolling out agent...")
    extractor = BufferStatsExtractor(
        n_references=args.n_references,
        strategy=args.ref_strategy,
    )
    references, ref_levels = extractor.extract_references_with_levels(sampler)
    buffer_summary = extractor.extract_buffer_summary(sampler)
    global_metrics = extractor.extract_global_metrics(buffer_summary)

    print(f"  Selected {len(references)} reference mazes ({args.ref_strategy})")
    for ref in references:
        print(f"    {ref.label}: {ref.metrics[0].value:.4f} regret")

    # Roll out agent on reference levels for diversity comparison
    print(f"  Rolling out agent on {len(ref_levels)} references...")
    ref_trajectories = []
    ref_stats = []
    ref_scores_arr = np.array([float(r.metrics[0].value) for r in references])
    for i, level in enumerate(ref_levels):
        traj = evaluator.evaluate_level_multi_rollout(level, n_rollouts=args.n_rollouts)
        ref_trajectories.append(traj)
        p = traj["solve_rate"]
        ref_stats.append({
            "label": references[i].label,
            "buffer_score": ref_scores_arr[i],
            "solve_rate": p,
            "sfl_score": p * (1 - p),
        })
        print(f"    {references[i].label}: solve={p:.2f}, SFL={p*(1-p):.4f}")

    # --- Configure gate ---
    thresholds = DiversityThresholds(
        difficulty_threshold=args.difficulty_threshold,
        difficulty_metric="sfl",
        min_diversity=args.diversity_threshold,
        diversity_metric="td_error_emd",
    )
    print(f"\n  Gate config: difficulty_threshold={args.difficulty_threshold}, "
          f"diversity_threshold={args.diversity_threshold}")

    # --- Configure LLM generator ---
    llm_cfg = load_llm_config(args.llm_config)
    gen_config = GenerationConfig(
        provider=args.llm_provider,
        model=args.llm_model,
        base_url=llm_cfg.get("base_url", ""),
        api_key=llm_cfg.get("api_key", ""),
        temperature=llm_cfg.get("temperature", 0.8),
        max_retries=llm_cfg.get("max_retries", 3),
        timeout=llm_cfg.get("timeout", 120),
        max_tokens=llm_cfg.get("max_tokens", 4096),
        min_walls=llm_cfg.get("min_walls", 15),
        min_path_distance=llm_cfg.get("min_path_distance", 8),
        validate_solvable=True,
    )
    generator = MazeGenerator(gen_config)

    # --- Generate + gate seeds ---
    print(f"\n[4/6] Generating {args.n_seeds} seed levels with decision gate...")
    seed_levels = []
    seed_grids = []
    seed_tokens = []
    seed_trajectories = []
    seed_stats = []
    generation_stats = []
    td_matrix_rows = []

    ref_labels = [r.label for r in references]

    for i in range(args.n_seeds):
        print(f"\n  === Seed {i+1}/{args.n_seeds} ===")
        prior_messages = []
        accepted = False
        best_result = None
        best_gate = None
        best_traj = None

        for gate_attempt in range(1, args.max_gate_retries + 1):
            print(f"    Gate attempt {gate_attempt}/{args.max_gate_retries}...", end=" ", flush=True)

            # Generate maze (LLM call with retry context)
            gen_result = generator.generate(
                references=references,
                global_metrics=global_metrics,
                prior_messages=prior_messages if prior_messages else None,
            )

            if not gen_result.success or gen_result.level is None:
                print(f"LLM FAILED ({gen_result.attempts} attempts)")
                for err in gen_result.errors:
                    print(f"      {err}")
                continue

            level = gen_result.level
            grid = gen_result.grid
            print(f"maze generated ({gen_result.attempts} LLM attempts, {gen_result.latency_ms:.0f}ms)")

            # Roll out agent on candidate
            print(f"    Rolling out agent ({args.n_rollouts} rollouts)...", end=" ", flush=True)
            traj = evaluator.evaluate_level_multi_rollout(level, n_rollouts=args.n_rollouts)
            p = traj["solve_rate"]
            sfl = p * (1 - p)
            print(f"solve={p:.2f}, SFL={sfl:.4f}")

            # Decision gate
            gate_result = evaluate_candidate(
                candidate_trajectory=traj,
                reference_trajectories=ref_trajectories,
                reference_labels=ref_labels,
                thresholds=thresholds,
            )

            # Track best attempt (even if rejected)
            if best_gate is None or sfl > (best_gate.summary.get("learnability", 0)):
                best_result = gen_result
                best_gate = gate_result
                best_traj = traj

            if gate_result.accepted:
                print(f"    ACCEPTED: SFL={sfl:.4f}, "
                      f"min_diversity={gate_result.summary.get('min_diversity', 0):.4f}")
                best_result = gen_result
                best_gate = gate_result
                best_traj = traj
                accepted = True
                break
            else:
                print(f"    REJECTED:")
                for issue in gate_result.issues:
                    issue_short = issue.split('\n')[0]
                    print(f"      {issue_short}")

                # Build feedback for LLM retry
                prior_messages.append({"role": "assistant", "content": gen_result.raw_responses[-1]})
                feedback = (
                    "Your maze was rejected. Issues:\n"
                    + "\n".join(f"- {iss}" for iss in gate_result.issues)
                    + "\n\nPlease generate a new maze that addresses these issues."
                )
                prior_messages.append({"role": "user", "content": feedback})

        # Store the best attempt (accepted or best rejected)
        if best_result is not None and best_result.level is not None:
            seed_levels.append(best_result.level)
            seed_grids.append(best_result.grid)
            seed_tokens.append(np.asarray(level_to_tokens(best_result.level)))
            seed_trajectories.append(best_traj)

            # Compute TD EMD row for this seed
            from metrics.pairwise.td_error_distribution import td_error_divergence
            td_row = []
            for rtraj in ref_trajectories:
                td_res = td_error_divergence(best_traj, best_traj["dones"],
                                             rtraj, rtraj["dones"])
                td_row.append(td_res["emd"])
            td_matrix_rows.append(td_row)

            p = best_traj["solve_rate"]
            stat = {
                "index": i,
                "gate_accepted": accepted,
                "gate_attempts": gate_attempt,
                "solve_rate": p,
                "sfl_score": p * (1 - p),
                "min_diversity": best_gate.summary.get("min_diversity", 0),
                "mean_diversity": best_gate.summary.get("mean_diversity", 0),
                "gate_issues": best_gate.issues,
                "gen_attempts": best_result.attempts,
                "gen_latency_ms": best_result.latency_ms,
            }
            seed_stats.append(stat)
            generation_stats.append(stat)

            status = "ACCEPTED" if accepted else "BEST-REJECTED"
            print(f"  --> Stored as {status}: SFL={p*(1-p):.4f}, "
                  f"min_TD={td_row[-1] if td_row else 0:.4f}")
        else:
            print(f"  --> FAILED: no valid maze generated")
            generation_stats.append({"index": i, "gate_accepted": False, "failed": True})

    n_ok = len(seed_levels)
    n_accepted = sum(1 for s in seed_stats if s["gate_accepted"])
    print(f"\n[Result] {n_ok} seeds stored ({n_accepted} accepted, "
          f"{n_ok - n_accepted} best-rejected)")

    if n_ok == 0:
        print("No seeds generated. Exiting.")
        return

    # --- Save ---
    print(f"\n[5/6] Saving to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)

    np.savez(
        os.path.join(args.output_dir, "seeds.npz"),
        tokens=np.stack(seed_tokens),
        grids=np.array(seed_grids, dtype=object),
    )

    with open(os.path.join(args.output_dir, "seeds_levels.pkl"), "wb") as f:
        pickle.dump(seed_levels, f)

    # Build TD matrix
    td_matrix = np.array(td_matrix_rows) if td_matrix_rows else None

    # Save log
    elapsed = time.time() - t0
    log = {
        "n_requested": args.n_seeds,
        "n_generated": n_ok,
        "n_accepted": n_accepted,
        "n_best_rejected": n_ok - n_accepted,
        "difficulty_threshold": args.difficulty_threshold,
        "diversity_threshold": args.diversity_threshold,
        "max_gate_retries": args.max_gate_retries,
        "n_rollouts": args.n_rollouts,
        "agent_checkpoint": args.agent_checkpoint_dir,
        "buffer_npz": args.buffer_npz,
        "llm_provider": args.llm_provider,
        "llm_model": args.llm_model,
        "ref_strategy": args.ref_strategy,
        "n_references": args.n_references,
        "elapsed_seconds": elapsed,
        "seed_stats": seed_stats,
        "ref_stats": ref_stats,
        "td_emd_matrix": td_matrix.tolist() if td_matrix is not None else None,
        "buffer_summary": {k: float(v) if isinstance(v, (float, np.floating)) else int(v)
                          for k, v in buffer_summary.items()},
    }
    with open(os.path.join(args.output_dir, "harvest_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    # --- Render + plots ---
    print(f"\n[6/6] Rendering and plotting...")
    print("  Rendering mazes...")
    render_all_seeds(seed_levels, os.path.join(args.output_dir, "renders"))

    print("  Generating evaluation plots...")
    plot_evaluation(
        seed_stats, ref_stats, td_matrix, all_buffer_scores,
        seed_levels, seed_trajectories, ref_scores_arr,
        os.path.join(args.output_dir, "evaluation"),
    )

    # --- Summary table ---
    print(f"\n{'='*70}")
    print(f"{'Seed':>6} {'Gate':>6} {'Solve%':>8} {'SFL':>8} {'MinTD':>8} {'Attempts':>8}")
    print(f"{'='*70}")
    for s in seed_stats:
        status = "PASS" if s["gate_accepted"] else "FAIL"
        print(f"  S{s['index']:>4} {status:>6} {s['solve_rate']:>8.2f} "
              f"{s['sfl_score']:>8.4f} {s['min_diversity']:>8.4f} "
              f"{s['gate_attempts']:>8}")
    print(f"{'='*70}")
    print(f"Buffer mean SFL: {all_buffer_scores.mean():.4f}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"\nOutput: {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harvest LLM seed levels with decision gate")
    parser.add_argument("--agent_checkpoint_dir", type=str, required=True)
    parser.add_argument("--buffer_npz", type=str, required=True)
    parser.add_argument("--checkpoint_step", type=int, default=-1)
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--n_references", type=int, default=5)
    parser.add_argument("--n_rollouts", type=int, default=10,
                        help="Rollouts per level for SFL scoring")
    parser.add_argument("--ref_strategy", type=str, default="hardest",
                        choices=["hardest", "random", "diverse"])
    parser.add_argument("--llm_provider", type=str, default="claude-code",
                        choices=["claude-code", "openrouter", "ollama"])
    parser.add_argument("--llm_model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--llm_config", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="experiments/seeds/default")
    # Gate thresholds
    parser.add_argument("--difficulty_threshold", type=float, default=0.05,
                        help="Min SFL learnability to accept (0.05 = ~25%% or ~75%% solve)")
    parser.add_argument("--diversity_threshold", type=float, default=0.02,
                        help="Min TD error EMD vs references to accept")
    parser.add_argument("--max_gate_retries", type=int, default=3,
                        help="Max gate retry attempts per seed (with LLM feedback)")
    args = parser.parse_args()
    harvest(args)
