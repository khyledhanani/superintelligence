import numpy as np
import jax.numpy as jnp
from collections import deque


def is_reachable_bfs(grid_size, walls, start_idx, goal_idx):
    """
    Checks if a path exists from start to goal on a grid.

    Args:
        grid_size (int): Dimension of the grid (e.g., 13).
        walls (array-like): List of wall indices. 0 is treated as padding.
        start_idx (int): 1-based index of agent.
        goal_idx (int): 1-based index of goal.
    """
    def to_coord(idx):
        idx = int(idx) - 1
        return idx // grid_size, idx % grid_size

    wall_coords = set()
    for w in np.array(walls).flatten():
        if w != 0:
            wall_coords.add(to_coord(w))

    start_pos = to_coord(start_idx)
    goal_pos = to_coord(goal_idx)

    if start_pos == goal_pos:
        return True
    if start_pos in wall_coords or goal_pos in wall_coords:
        return False

    queue = deque([start_pos])
    visited = {start_pos}
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        r, c = queue.popleft()
        if (r, c) == goal_pos:
            return True
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                if (nr, nc) not in visited and (nr, nc) not in wall_coords:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return False


def evaluate_cluttr_metrics(original_batch, reconstructed_batch, pad_token=169,
                            grid_size=13, check_reachability=False):
    """
    Computes validity and accuracy metrics for CLUTTR VAE reconstructions.

    Args:
        original_batch (array-like): Batch of ground truth sequences.
        reconstructed_batch (array-like): Batch of reconstructed sequences.
        pad_token (int): The token value used for padding (default 169).
        grid_size (int): Grid dimension for reachability checks (default 13).
        check_reachability (bool): Whether to run BFS reachability (slower).

    Returns:
        dict: Metric values. Keys ending in '_pct' are percentages [0–100].
              Others are raw counts or averages.
    """
    original_batch = np.array(original_batch)
    reconstructed_batch = np.array(reconstructed_batch)
    batch_size = len(original_batch)

    # --- Counters ---
    counts = {
        "agent_correct": 0,
        "goal_correct": 0,
        "valid_structure": 0,
        "fail_agent_on_wall": 0,
        "fail_goal_on_wall": 0,
        "fail_agent_goal_overlap": 0,
        "wall_count_error_sum": 0.0,
        # NEW metrics
        "reachable": 0,
        "total_reachability_checked": 0,
        "duplicate_wall_samples": 0,
        "total_duplicate_walls": 0,
        "per_token_correct": 0,
        "per_token_total": 0,
        "wall_iou_sum": 0.0,
        "out_of_range_tokens": 0,
    }

    vocab_size = 170  # matches CONFIG; adjust if needed

    for orig, recon in zip(original_batch, reconstructed_batch):
        orig = orig.flatten()
        recon = recon.flatten()

        # --- Key objects ---
        orig_agent = orig[-1]
        orig_goal = orig[-2]
        recon_agent = recon[-1]
        recon_goal = recon[-2]

        if orig_agent == recon_agent:
            counts["agent_correct"] += 1
        if orig_goal == recon_goal:
            counts["goal_correct"] += 1

        # --- Per-token accuracy (full sequence) ---
        seq_len = len(orig)
        matches = np.sum(orig == recon)
        counts["per_token_correct"] += int(matches)
        counts["per_token_total"] += seq_len

        # --- Walls ---
        raw_recon_walls = recon[:-2]
        raw_orig_walls = orig[:-2]

        recon_walls = raw_recon_walls[raw_recon_walls != pad_token]
        orig_walls = raw_orig_walls[raw_orig_walls != pad_token]

        # Wall count error
        counts["wall_count_error_sum"] += abs(len(recon_walls) - len(orig_walls))

        # --- NEW: Duplicate walls ---
        unique_recon_walls = np.unique(recon_walls)
        n_duplicates = len(recon_walls) - len(unique_recon_walls)
        if n_duplicates > 0:
            counts["duplicate_wall_samples"] += 1
            counts["total_duplicate_walls"] += n_duplicates

        # --- NEW: Wall IoU (set overlap) ---
        if len(orig_walls) > 0 or len(recon_walls) > 0:
            orig_set = set(orig_walls.tolist())
            recon_set = set(recon_walls.tolist())
            intersection = len(orig_set & recon_set)
            union = len(orig_set | recon_set)
            counts["wall_iou_sum"] += intersection / union if union > 0 else 1.0
        else:
            counts["wall_iou_sum"] += 1.0  # both empty = perfect match

        # --- NEW: Out-of-range token check ---
        oob = np.sum((recon < 0) | (recon >= vocab_size))
        counts["out_of_range_tokens"] += int(oob)

        # --- Structural validity ---
        is_agent_on_wall = np.any(recon_walls == recon_agent)
        is_goal_on_wall = np.any(recon_walls == recon_goal)
        is_agent_on_goal = (recon_agent == recon_goal)

        if is_agent_on_wall:
            counts["fail_agent_on_wall"] += 1
        if is_goal_on_wall:
            counts["fail_goal_on_wall"] += 1
        if is_agent_on_goal:
            counts["fail_agent_goal_overlap"] += 1

        is_valid = not (is_agent_on_wall or is_goal_on_wall or is_agent_on_goal)
        if is_valid:
            counts["valid_structure"] += 1

        # --- NEW: Reachability (optional, slow) ---
        if check_reachability and is_valid:
            counts["total_reachability_checked"] += 1
            if is_reachable_bfs(grid_size, recon_walls, int(recon_agent), int(recon_goal)):
                counts["reachable"] += 1

    # --- Summary ---
    summary = {
        # Percentages (0–100)
        "validity_pct": (counts["valid_structure"] / batch_size) * 100.0,
        "agent_acc_pct": (counts["agent_correct"] / batch_size) * 100.0,
        "goal_acc_pct": (counts["goal_correct"] / batch_size) * 100.0,
        "per_token_acc_pct": (counts["per_token_correct"] / counts["per_token_total"]) * 100.0,
        "duplicate_wall_sample_pct": (counts["duplicate_wall_samples"] / batch_size) * 100.0,

        # Raw values (NOT percentages)
        "avg_wall_count_error": counts["wall_count_error_sum"] / batch_size,
        "avg_wall_iou": counts["wall_iou_sum"] / batch_size,
        "avg_duplicate_walls_per_sample": counts["total_duplicate_walls"] / batch_size,
        "out_of_range_tokens_total": counts["out_of_range_tokens"],

        # Failure breakdown (percentages)
        "fail_agent_wall_pct": (counts["fail_agent_on_wall"] / batch_size) * 100.0,
        "fail_goal_wall_pct": (counts["fail_goal_on_wall"] / batch_size) * 100.0,
        "fail_overlap_pct": (counts["fail_agent_goal_overlap"] / batch_size) * 100.0,
    }

    # Reachability (only if computed)
    if counts["total_reachability_checked"] > 0:
        summary["reachability_pct"] = (
            counts["reachable"] / counts["total_reachability_checked"]
        ) * 100.0
        summary["reachability_n_checked"] = counts["total_reachability_checked"]
    else:
        summary["reachability_pct"] = None

    # --- BACK-COMPAT aliases (so old plotting code doesn't break) ---
    summary["validity_score"] = summary["validity_pct"]
    summary["agent_accuracy"] = summary["agent_acc_pct"]
    summary["goal_accuracy"] = summary["goal_acc_pct"]
    summary["avg_wall_error"] = summary["avg_wall_count_error"]

    return summary


def compute_latent_stats(mean, logvar):
    """
    Compute diagnostic statistics about the latent space.
    Useful for detecting posterior collapse and monitoring training health.

    Args:
        mean: (batch, latent_dim) array of posterior means.
        logvar: (batch, latent_dim) array of posterior log-variances.

    Returns:
        dict with latent space diagnostics.
    """
    mean = np.array(mean)
    logvar = np.array(logvar)
    std = np.exp(0.5 * logvar)

    # Active units: dimensions where Var_x[E[z_d|x]] > 0.01
    # (Burda et al., 2015 — a dim is "active" if the encoder actually uses it)
    mean_variance_per_dim = np.var(mean, axis=0)  # (latent_dim,)
    active_units = int(np.sum(mean_variance_per_dim > 0.01))

    return {
        "latent_mean_avg": float(np.mean(mean)),
        "latent_mean_std": float(np.std(mean)),
        "latent_mean_abs_max": float(np.max(np.abs(mean))),
        "latent_std_avg": float(np.mean(std)),
        "latent_std_min": float(np.min(std)),
        "latent_std_max": float(np.max(std)),
        "active_units": active_units,
        "active_units_pct": (active_units / mean.shape[-1]) * 100.0,
        "mean_variance_per_dim": mean_variance_per_dim,  # (latent_dim,) for plotting
    }


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    t1 = [10, 12, 169, 169, 168, 167]
    p1 = [10, 12, 169, 169, 168, 167]

    t2 = [10, 12, 169, 169, 168, 167]
    p2 = [10, 12, 169, 169, 168, 10]

    batch_true = [t1, t2]
    batch_pred = [p1, p2]

    results = evaluate_cluttr_metrics(batch_true, batch_pred, pad_token=169)

    print("--- Test Results ---")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")