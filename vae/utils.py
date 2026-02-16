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
    # 1. Helper to convert 1-based index to (row, col)
    #    Index 1 -> (0,0), Index 13 -> (0,12), Index 14 -> (1,0)
    def to_coord(idx):
        idx = int(idx) - 1  # Shift to 0-based
        return idx // grid_size, idx % grid_size

    # 2. Filter Walls: Create a set of coordinates, IGNORING 0s
    wall_coords = set()
    for w in np.array(walls).flatten():
        if w != 0: # 0 is padding, not a wall
            wall_coords.add(to_coord(w))
            
    start_pos = to_coord(start_idx)
    goal_pos = to_coord(goal_idx)
    
    # 3. Quick Checks
    if start_pos == goal_pos: return True
    if start_pos in wall_coords or goal_pos in wall_coords: return False

    # 4. Standard BFS
    queue = deque([start_pos])
    visited = {start_pos}
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right

    while queue:
        r, c = queue.popleft()
        
        if (r, c) == goal_pos:
            return True
            
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            # Check Bounds (0 to grid_size-1)
            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                # Check Obstacles & Visited
                if (nr, nc) not in visited and (nr, nc) not in wall_coords:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

    return False


def evaluate_cluttr_metrics(original_batch, reconstructed_batch, pad_token=169):
    """
    Computes validity and accuracy metrics for CLUTTR VAE reconstructions.
    
    Args:
        original_batch (array-like): Batch of ground truth sequences.
        reconstructed_batch (array-like): Batch of reconstructed sequences.
        pad_token (int): The token value used for padding (default 169).
        
    Returns:
        dict: A dictionary containing percentage metrics.
    """
    # ensure inputs are numpy arrays for fast CPU processing
    original_batch = np.array(original_batch)
    reconstructed_batch = np.array(reconstructed_batch)
    
    batch_size = len(original_batch)
    
    # Initialize counters
    metrics = {
        "agent_accuracy": 0,
        "goal_accuracy": 0,
        "valid_structure_count": 0,
        "failure_agent_on_wall": 0,
        "failure_goal_on_wall": 0,
        "failure_agent_goal_overlap": 0,
        "wall_count_error": 0.0
    }

    for orig, recon in zip(original_batch, reconstructed_batch):
        # Flatten to ensure 1D processing per sample
        orig = orig.flatten()
        recon = recon.flatten()

        # --- 1. Key Object Identifiers ---
        # Per your spec: Last element is Agent, Second to last is Goal
        orig_agent = orig[-1]
        orig_goal = orig[-2]
        
        recon_agent = recon[-1]
        recon_goal = recon[-2]
        
        # Accuracy Check
        if orig_agent == recon_agent:
            metrics["agent_accuracy"] += 1
        if orig_goal == recon_goal:
            metrics["goal_accuracy"] += 1

        # --- 2. Structural Integrity (Validity) ---
        # Extract walls: All elements except the last two
        raw_recon_walls = recon[:-2]
        raw_orig_walls = orig[:-2]
        
        # Filter out PAD tokens to get actual wall indices
        # IMPORTANT: We filter pad_token, NOT 0. (0 is a valid grid location)
        recon_walls = raw_recon_walls[raw_recon_walls != pad_token]
        orig_walls = raw_orig_walls[raw_orig_walls != pad_token]
        
        # Check for Overlaps (Failures)
        # 1. Agent on top of a wall
        is_agent_on_wall = np.any(recon_walls == recon_agent)
        
        # 2. Goal on top of a wall
        is_goal_on_wall = np.any(recon_walls == recon_goal)
        
        # 3. Agent on top of Goal (Overlap)
        is_agent_on_goal = (recon_agent == recon_goal)
        
        # Record Failures
        if is_agent_on_wall:
            metrics["failure_agent_on_wall"] += 1
        if is_goal_on_wall:
            metrics["failure_goal_on_wall"] += 1
        if is_agent_on_goal:
            metrics["failure_agent_goal_overlap"] += 1
            
        # A structure is VALID if none of these failures occur
        if not (is_agent_on_wall or is_goal_on_wall or is_agent_on_goal):
            metrics["valid_structure_count"] += 1
            
        # --- 3. Wall Count Error ---
        # How many extra or missing walls did we generate?
        metrics["wall_count_error"] += abs(len(recon_walls) - len(orig_walls))

    # --- 4. Final Percentages ---
    summary = {
        "validity_score": (metrics["valid_structure_count"] / batch_size) * 100.0,
        "agent_accuracy": (metrics["agent_accuracy"] / batch_size) * 100.0,
        "goal_accuracy": (metrics["goal_accuracy"] / batch_size) * 100.0,
        "avg_wall_error": metrics["wall_count_error"] / batch_size,
        # Detailed failure rates (useful for debugging)
        "fail_agent_wall": (metrics["failure_agent_on_wall"] / batch_size) * 100.0,
        "fail_goal_wall": (metrics["failure_goal_on_wall"] / batch_size) * 100.0,
        "fail_overlap": (metrics["failure_agent_goal_overlap"] / batch_size) * 100.0,
    }
    
    return summary

# Example Usage Block (for testing)
if __name__ == "__main__":
    # Mock data to test the function
    # 52 length (50 walls + 2 objects)
    # Using 169 as PAD token
    
    # Case 1: Perfect match
    t1 = [10, 12, 169, 169, 168, 167] # Walls at 10,12. Goal 168, Agent 167
    p1 = [10, 12, 169, 169, 168, 167]
    
    # Case 2: Agent on Wall (Invalid)
    t2 = [10, 12, 169, 169, 168, 167]
    p2 = [10, 12, 169, 169, 168, 10]  # Agent at 10 (which is a wall)

    batch_true = [t1, t2]
    batch_pred = [p1, p2]
    
    results = evaluate_cluttr_metrics(batch_true, batch_pred, pad_token=169)
    
    print("--- Test Results ---")
    for k, v in results.items():
        print(f"{k}: {v:.2f}")