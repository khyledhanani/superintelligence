"""Per-maze performance table at 30k updates on the friend's 8 prefab mazes.

For each eval maze, shows per-seed solve rate and mean across seeds, then
prints an aggregate row at the bottom (mean over mazes per seed / mean over
all mazes and seeds). Also produces a matplotlib visualisation of the table.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONDITION_LABEL = {
    "final-gpt41mini": "GPT-4.1-mini (LLM)",
    "final-accel-baseline": "ACCEL baseline",
}
CONDITION_COLOR = {
    "final-gpt41mini": "#d62728",
    "final-accel-baseline": "#1f77b4",
}


def load(json_path: str):
    with open(json_path, "r") as f:
        return json.load(f)


def build_matrices(rows):
    """Concatenate per-level arrays across all eval sets (in deterministic
    eval-set order) and return: condition -> {level_names, seeds, matrix,
    extras}. Matrix shape is (n_levels, n_seeds). `extras` holds per-seed
    aggregate metrics (iqm_solve_rate, mean_return, mean_optimality_gap,
    mean_path_ratio) — averaged across eval sets for each seed."""
    # Group by (condition, seed) -> list of rows (one per eval set)
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in rows:
        if int(r["training_updates"]) != 30000:
            continue
        grouped[(r["condition"], int(r["seed"]))].append(r)

    # Consistent eval-set ordering: sort by eval_set name
    # so every (cond, seed) pair concatenates levels in the same order.
    def _sorted(rs):
        return sorted(rs, key=lambda r: r["eval_set"])

    # Figure out canonical level-name ordering from any one (cond, seed)
    canonical_names = None
    for (cond, seed), rs in grouped.items():
        names = []
        for r in _sorted(rs):
            names.extend(r.get("level_names") or [f"{r['eval_set']}:{i}" for i in range(len(r['per_level_solve_rates']))])
        if canonical_names is None:
            canonical_names = names
        else:
            # Assume all match; if not, warn but keep the first seen
            if names != canonical_names:
                print(f"[WARN] level order mismatch for {cond}/seed{seed}")

    by_cond = {}
    for (cond, seed), rs in grouped.items():
        rs = _sorted(rs)
        per_level = np.concatenate(
            [np.asarray(r["per_level_solve_rates"], dtype=np.float64) for r in rs]
        )
        # Extras: weighted mean across eval sets (weighted by level count)
        n_levels_list = [len(r["per_level_solve_rates"]) for r in rs]
        total_n = float(sum(n_levels_list))
        def _wmean(key):
            return float(
                sum(r[key] * n for r, n in zip(rs, n_levels_list)) / total_n
            )
        extras = {
            "iqm_solve_rate": _wmean("iqm_solve_rate") if "iqm_solve_rate" in rs[0] else float("nan"),
            "mean_return": _wmean("mean_return") if "mean_return" in rs[0] else float("nan"),
            "mean_optimality_gap": _wmean("mean_optimality_gap") if "mean_optimality_gap" in rs[0] else float("nan"),
            "mean_path_ratio": _wmean("mean_path_ratio") if "mean_path_ratio" in rs[0] else float("nan"),
        }
        if cond not in by_cond:
            by_cond[cond] = {
                "level_names": canonical_names,
                "data": {},     # seed -> per-level array
                "extras": {},   # seed -> dict of aggregate metrics
            }
        by_cond[cond]["data"][seed] = per_level
        by_cond[cond]["extras"][seed] = extras

    result = {}
    for cond, info in by_cond.items():
        seeds = sorted(info["data"].keys())
        mat = np.stack([info["data"][s] for s in seeds], axis=1)  # (n_levels, n_seeds)
        extras_mat = {
            k: np.array([info["extras"][s][k] for s in seeds], dtype=np.float64)
            for k in ("iqm_solve_rate", "mean_return", "mean_optimality_gap", "mean_path_ratio")
        }
        result[cond] = {
            "level_names": info["level_names"],
            "seeds": seeds,
            "matrix": mat,
            "extras": extras_mat,
        }
    return result


def print_table(matrices, condition):
    info = matrices[condition]
    level_names = info["level_names"]
    seeds = info["seeds"]
    mat = info["matrix"]

    print()
    print("=" * 90)
    print(f"  {CONDITION_LABEL[condition]}  @ 30k updates  (per-maze solve rate)")
    print("=" * 90)
    header = f"  {'Level':<22}" + "".join(f"{'seed '+str(s):>10}" for s in seeds) + f"{'mean':>10}" + f"{'std':>10}"
    print(header)
    print("-" * len(header))
    level_means = mat.mean(axis=1)
    level_stds = mat.std(axis=1, ddof=0)
    for i, name in enumerate(level_names):
        row = f"  {name:<22}"
        for j in range(len(seeds)):
            row += f"{mat[i, j]:>10.2f}"
        row += f"{level_means[i]:>10.2f}"
        row += f"{level_stds[i]:>10.2f}"
        print(row)
    print("-" * len(header))
    seed_means = mat.mean(axis=0)
    overall = float(mat.mean())
    overall_std = float(seed_means.std(ddof=0))
    row = f"  {'MEAN over mazes':<22}"
    for v in seed_means:
        row += f"{v:>10.2f}"
    row += f"{overall:>10.2f}"
    row += f"{overall_std:>10.2f}"
    print(row)
    print("=" * 90)


def make_table_figure(matrices, output_path):
    """Render a styled matplotlib table per condition + an aggregate table."""
    conditions = [c for c in CONDITION_LABEL if c in matrices]
    n_cond = len(conditions)

    fig, axes = plt.subplots(
        n_cond + 1, 1,
        figsize=(11, 3.0 + 0.45 * (len(matrices[conditions[0]]["level_names"]) + 1) * n_cond),
        gridspec_kw={"height_ratios": [1.0] * n_cond + [0.55]},
    )
    if n_cond + 1 == 1:
        axes = [axes]

    for ax, cond in zip(axes[:-1], conditions):
        info = matrices[cond]
        level_names = info["level_names"]
        seeds = info["seeds"]
        mat = info["matrix"]

        level_means = mat.mean(axis=1)
        level_stds = mat.std(axis=1, ddof=0)

        col_headers = [f"seed {s}" for s in seeds] + ["mean", "std"]
        row_headers = list(level_names) + ["mean over mazes"]

        cell_text = []
        for i in range(len(level_names)):
            row = [f"{mat[i, j]:.2f}" for j in range(len(seeds))]
            row += [f"{level_means[i]:.2f}", f"{level_stds[i]:.2f}"]
            cell_text.append(row)

        seed_means = mat.mean(axis=0)
        overall = float(mat.mean())
        overall_std = float(seed_means.std(ddof=0))
        agg_row = [f"{v:.2f}" for v in seed_means] + [f"{overall:.2f}", f"{overall_std:.2f}"]
        cell_text.append(agg_row)

        ax.axis("off")
        ax.set_title(
            f"{CONDITION_LABEL[cond]}  —  overall mean = {overall:.3f}  ±  {overall_std:.3f}",
            fontsize=12,
            color=CONDITION_COLOR[cond],
            fontweight="bold",
            loc="left",
        )
        table = ax.table(
            cellText=cell_text,
            rowLabels=row_headers,
            colLabels=col_headers,
            cellLoc="center",
            loc="center",
            rowLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.4)

        # Highlight header row + aggregate row
        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor("#444")
                cell.set_text_props(color="white", fontweight="bold")
            elif r == len(cell_text):  # aggregate row
                cell.set_facecolor("#e9eef5")
                cell.set_text_props(fontweight="bold")
            # Mean / std columns
            if c in (len(seeds) - 1 + 1, len(seeds) - 1 + 2):
                if r > 0 and r < len(cell_text):
                    cell.set_facecolor("#f5f5f5")

    # ---------- Comparison table ----------
    ax = axes[-1]
    ax.axis("off")
    compare_rows = []
    compare_row_labels = []
    level_names_ref = matrices[conditions[0]]["level_names"]
    for i, name in enumerate(level_names_ref):
        gpt_mean = matrices.get("final-gpt41mini", {}).get("matrix")
        acc_mean = matrices.get("final-accel-baseline", {}).get("matrix")
        row = []
        if gpt_mean is not None:
            row.append(f"{gpt_mean[i].mean():.3f}")
        if acc_mean is not None:
            row.append(f"{acc_mean[i].mean():.3f}")
        if gpt_mean is not None and acc_mean is not None:
            delta = gpt_mean[i].mean() - acc_mean[i].mean()
            row.append(f"{delta:+.3f}")
        compare_rows.append(row)
        compare_row_labels.append(name)

    # Add overall row
    overall_row = []
    if "final-gpt41mini" in matrices:
        overall_row.append(f"{matrices['final-gpt41mini']['matrix'].mean():.3f}")
    if "final-accel-baseline" in matrices:
        overall_row.append(f"{matrices['final-accel-baseline']['matrix'].mean():.3f}")
    if "final-gpt41mini" in matrices and "final-accel-baseline" in matrices:
        delta = (
            matrices["final-gpt41mini"]["matrix"].mean()
            - matrices["final-accel-baseline"]["matrix"].mean()
        )
        overall_row.append(f"{delta:+.3f}")
    compare_rows.append(overall_row)
    compare_row_labels.append("OVERALL")

    compare_cols = ["GPT-4.1-mini", "ACCEL base", "Δ (LLM − base)"]
    table = ax.table(
        cellText=compare_rows,
        rowLabels=compare_row_labels,
        colLabels=compare_cols,
        cellLoc="center",
        loc="center",
        rowLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.4)
    ax.set_title(
        "Per-maze mean solve rate (over 5 seeds) — GPT-4.1-mini vs ACCEL baseline",
        fontsize=12,
        fontweight="bold",
        loc="left",
    )
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#444")
            cell.set_text_props(color="white", fontweight="bold")
        elif r == len(compare_rows):  # overall row
            cell.set_facecolor("#ffe9c0")
            cell.set_text_props(fontweight="bold")
        if c == 2 and r > 0 and r <= len(compare_rows):
            # Colour the delta column by sign
            try:
                val = float(compare_rows[r - 1][-1])
                cell.set_facecolor("#d5f3d5" if val >= 0 else "#f6d6d6")
            except (ValueError, IndexError):
                pass

    n_mazes = len(level_names_ref)
    fig.suptitle(
        f"Final performance on {n_mazes} hand-designed prefab mazes",
        fontsize=14,
        y=1.0,
    )
    fig.tight_layout(rect=[0.03, 0.02, 1, 0.97])
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"Saved table figure: {output_path}")


def make_agg_metrics_figure(matrices, output_path):
    """Render a small matplotlib table of aggregate metrics:
    mean solve rate, IQM, mean return, mean optimality gap, mean path ratio.
    One row per condition, with ± std across seeds."""
    conditions = [c for c in CONDITION_LABEL if c in matrices]
    if not conditions:
        return

    metric_cols = [
        ("Mean SR", "matrix", True),            # solve-rate mean (%)
        ("IQM SR", "iqm_solve_rate", True),
        ("Mean Return", "mean_return", False),
        ("Opt Gap", "mean_optimality_gap", False),
        ("Path Ratio", "mean_path_ratio", False),
    ]

    cell_text = []
    row_labels = []
    for cond in conditions:
        info = matrices[cond]
        row = []
        for label, key, as_pct in metric_cols:
            if key == "matrix":
                # mean solve rate per seed from full matrix
                per_seed = info["matrix"].mean(axis=0)
            else:
                per_seed = info["extras"][key]
            m = float(np.nanmean(per_seed))
            s = float(np.nanstd(per_seed, ddof=0))
            if as_pct:
                row.append(f"{m*100:.1f}% ± {s*100:.1f}")
            else:
                row.append(f"{m:.3f} ± {s:.3f}")
        cell_text.append(row)
        row_labels.append(CONDITION_LABEL[cond])

    # Add delta row (LLM - baseline) if both conditions present
    if "final-gpt41mini" in matrices and "final-accel-baseline" in matrices:
        gpt = matrices["final-gpt41mini"]
        acc = matrices["final-accel-baseline"]
        delta_row = []
        for label, key, as_pct in metric_cols:
            if key == "matrix":
                g_per = gpt["matrix"].mean(axis=0)
                a_per = acc["matrix"].mean(axis=0)
            else:
                g_per = gpt["extras"][key]
                a_per = acc["extras"][key]
            d = float(np.nanmean(g_per) - np.nanmean(a_per))
            # For solve rates / IQM / return: higher is better. For opt gap &
            # path ratio: lower is better.
            if as_pct:
                delta_row.append(f"{d*100:+.1f}")
            else:
                delta_row.append(f"{d:+.3f}")
        cell_text.append(delta_row)
        row_labels.append("Δ (LLM − base)")

    col_headers = [c[0] for c in metric_cols]

    fig, ax = plt.subplots(figsize=(11, 2 + 0.55 * len(cell_text)))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_headers,
        cellLoc="center",
        loc="center",
        rowLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.7)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#444")
            cell.set_text_props(color="white", fontweight="bold")
        elif r == len(cell_text):  # delta row
            cell.set_facecolor("#ffe9c0")
            cell.set_text_props(fontweight="bold")

    ax.set_title(
        "Aggregate metrics — GPT-4.1-mini vs ACCEL baseline (mean ± std across seeds)",
        fontsize=12,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    print(f"Saved aggregate-metrics figure: {output_path}")


def print_aggregate_metrics(matrices):
    conditions = [c for c in CONDITION_LABEL if c in matrices]
    if not conditions:
        return
    print()
    print("=" * 100)
    print("  Aggregate metrics (mean ± std across seeds)")
    print("=" * 100)
    header = f"  {'Condition':<30}{'Mean SR':>12}{'IQM SR':>12}{'Mean Return':>14}{'Opt Gap':>12}{'Path Ratio':>14}"
    print(header)
    print("-" * len(header))
    for cond in conditions:
        info = matrices[cond]
        sr_per_seed = info["matrix"].mean(axis=0)
        iqm_per_seed = info["extras"]["iqm_solve_rate"]
        ret_per_seed = info["extras"]["mean_return"]
        og_per_seed = info["extras"]["mean_optimality_gap"]
        pr_per_seed = info["extras"]["mean_path_ratio"]
        label = CONDITION_LABEL[cond]
        row = (
            f"  {label:<30}"
            f"{np.nanmean(sr_per_seed)*100:>10.1f}% "
            f"{np.nanmean(iqm_per_seed)*100:>10.1f}% "
            f"{np.nanmean(ret_per_seed):>13.3f} "
            f"{np.nanmean(og_per_seed):>11.3f} "
            f"{np.nanmean(pr_per_seed):>13.2f}"
        )
        print(row)

    if "final-gpt41mini" in matrices and "final-accel-baseline" in matrices:
        gpt, acc = matrices["final-gpt41mini"], matrices["final-accel-baseline"]
        d_sr = np.nanmean(gpt["matrix"]) - np.nanmean(acc["matrix"])
        d_iqm = np.nanmean(gpt["extras"]["iqm_solve_rate"]) - np.nanmean(acc["extras"]["iqm_solve_rate"])
        d_ret = np.nanmean(gpt["extras"]["mean_return"]) - np.nanmean(acc["extras"]["mean_return"])
        d_og = np.nanmean(gpt["extras"]["mean_optimality_gap"]) - np.nanmean(acc["extras"]["mean_optimality_gap"])
        d_pr = np.nanmean(gpt["extras"]["mean_path_ratio"]) - np.nanmean(acc["extras"]["mean_path_ratio"])
        print("-" * len(header))
        print(
            f"  {'Δ (LLM − base)':<30}"
            f"{d_sr*100:>+10.1f}  "
            f"{d_iqm*100:>+10.1f}  "
            f"{d_ret:>+13.3f} "
            f"{d_og:>+11.3f} "
            f"{d_pr:>+13.2f}"
        )
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        type=str,
        default="results/ood_eval_friend_v2/ood_eval_results.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/ood_eval_friend_v2/ood_friend_v2_table.png",
    )
    parser.add_argument(
        "--agg_output",
        type=str,
        default="results/ood_eval_friend_v2/ood_friend_v2_agg_metrics.png",
    )
    args = parser.parse_args()

    rows = load(args.json)
    matrices = build_matrices(rows)

    for cond in ("final-gpt41mini", "final-accel-baseline"):
        if cond in matrices:
            print_table(matrices, cond)

    # Comparison block
    print()
    print("=" * 90)
    print("  GPT-4.1-mini vs ACCEL baseline — per-maze MEAN over seeds")
    print("=" * 90)
    lnames = matrices["final-gpt41mini"]["level_names"]
    print(f"  {'Level':<22}{'GPT-4.1-mini':>16}{'ACCEL base':>16}{'Δ (LLM-base)':>16}")
    print("-" * 72)
    for i, name in enumerate(lnames):
        gpt = matrices["final-gpt41mini"]["matrix"][i].mean()
        acc = matrices["final-accel-baseline"]["matrix"][i].mean()
        delta = gpt - acc
        print(f"  {name:<22}{gpt:>16.3f}{acc:>16.3f}{delta:>+16.3f}")
    print("-" * 72)
    gpt_overall = matrices["final-gpt41mini"]["matrix"].mean()
    acc_overall = matrices["final-accel-baseline"]["matrix"].mean()
    print(
        f"  {'OVERALL':<22}{gpt_overall:>16.3f}{acc_overall:>16.3f}"
        f"{gpt_overall - acc_overall:>+16.3f}"
    )
    print("=" * 90)

    print_aggregate_metrics(matrices)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    make_table_figure(matrices, args.output)
    make_agg_metrics_figure(matrices, args.agg_output)


if __name__ == "__main__":
    main()
