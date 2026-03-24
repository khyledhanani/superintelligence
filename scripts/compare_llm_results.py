#!/usr/bin/env python3
"""Compare ACCEL+LLM injection vs ACCEL-only control runs from JAXUED_LLM WandB project."""
import argparse
import numpy as np
import wandb


GROUPS = ["accel-llm", "accel-only"]

LLM_METRICS = [
    "llm/acceptance_rate",
    "llm/total_injected",
    "llm/diversity_score_mean",
]


def get_final_value(run, key):
    """Return the last non-NaN value for a metric key from a run's history."""
    try:
        hist = run.history(keys=[key], pandas=False)
        values = [row[key] for row in hist if key in row and row[key] is not None]
        if not values:
            return None
        return float(values[-1])
    except Exception:
        return None


def collect_group_stats(runs):
    """Collect per-run stats grouped by run_name."""
    groups = {}
    for run in runs:
        group_name = run.config.get("run_name", run.group or "unknown")
        if group_name not in groups:
            groups[group_name] = []
        groups[group_name].append(run)
    return groups


def print_summary_table(group_stats):
    """Print the formatted comparison summary table."""
    header = f"{'Group':<20} | {'N':>3} | {'Solve Rate (mean ± std)':<24} | {'Acceptance Rate':<18} | {'Injected Count':<16} | {'Diversity Score':<16}"
    separator = "-" * len(header)
    print()
    print(separator)
    print(header)
    print(separator)

    for group_name in GROUPS:
        runs = group_stats.get(group_name, [])
        if not runs:
            print(f"{'No runs found for: ' + group_name}")
            continue

        solve_rates = []
        acceptance_rates = []
        injected_counts = []
        diversity_scores = []

        for run in runs:
            sr = get_final_value(run, "solve_rate/mean")
            if sr is not None:
                solve_rates.append(sr)

            if group_name == "accel-llm":
                ar = get_final_value(run, "llm/acceptance_rate")
                ic = get_final_value(run, "llm/total_injected")
                ds = get_final_value(run, "llm/diversity_score_mean")
                if ar is not None:
                    acceptance_rates.append(ar)
                if ic is not None:
                    injected_counts.append(ic)
                if ds is not None:
                    diversity_scores.append(ds)

        n = len(runs)

        if solve_rates:
            sr_str = f"{np.mean(solve_rates):.3f} +/- {np.std(solve_rates):.3f}"
        else:
            sr_str = "N/A"

        if acceptance_rates:
            ar_str = f"{np.mean(acceptance_rates):.3f} +/- {np.std(acceptance_rates):.3f}"
        else:
            ar_str = "-"

        if injected_counts:
            ic_str = f"{np.mean(injected_counts):.1f} +/- {np.std(injected_counts):.1f}"
        else:
            ic_str = "-"

        if diversity_scores:
            ds_str = f"{np.mean(diversity_scores):.4f} +/- {np.std(diversity_scores):.4f}"
        else:
            ds_str = "-"

        print(f"{group_name:<20} | {n:>3} | {sr_str:<24} | {ar_str:<18} | {ic_str:<16} | {ds_str:<16}")

    print(separator)


def print_per_run_table(group_stats):
    """Print per-run detail table."""
    print()
    print("=== Per-Run Details ===")
    header2 = f"{'Run Name':<20} | {'Seed':>4} | {'Final Solve Rate':>16} | {'State':<10}"
    print("-" * len(header2))
    print(header2)
    print("-" * len(header2))

    for group_name in GROUPS:
        runs = group_stats.get(group_name, [])
        for run in runs:
            seed = run.config.get("seed", "?")
            sr = get_final_value(run, "solve_rate/mean")
            sr_str = f"{sr:.4f}" if sr is not None else "N/A"
            state = run.state
            print(f"{group_name:<20} | {str(seed):>4} | {sr_str:>16} | {state:<10}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare ACCEL+LLM injection vs ACCEL-only control from JAXUED_LLM WandB project."
    )
    parser.add_argument("--project", type=str, default="JAXUED_LLM",
                        help="WandB project name (default: JAXUED_LLM)")
    parser.add_argument("--entity", type=str, default=None,
                        help="WandB entity/team name (default: None, uses default entity)")
    args = parser.parse_args()

    project_path = f"{args.entity}/{args.project}" if args.entity else args.project
    print(f"=== JAXUED_LLM Comparison ===")
    print(f"Project: {project_path}")

    api = wandb.Api()
    runs = api.runs(project_path)
    run_list = list(runs)

    if not run_list:
        print(f"No runs found in project '{project_path}'.")
        return

    print(f"Found {len(run_list)} run(s) total.")

    group_stats = collect_group_stats(run_list)

    found_groups = list(group_stats.keys())
    print(f"Groups found: {found_groups}")

    # Warn about any unexpected or missing groups
    for g in GROUPS:
        if g not in group_stats:
            print(f"  WARNING: No runs found for group '{g}'")

    print_summary_table(group_stats)
    print_per_run_table(group_stats)


if __name__ == "__main__":
    main()
