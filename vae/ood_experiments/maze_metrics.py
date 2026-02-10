from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class MazeSpec:
    inner_dim: int = 13
    max_obs_tokens: int = 50

    @property
    def n_cells(self) -> int:
        return self.inner_dim * self.inner_dim


def token_to_xy(token: int, inner_dim: int) -> Tuple[int, int]:
    idx = token - 1
    x = idx % inner_dim
    y = idx // inner_dim
    return int(x), int(y)


def _valid_spatial_token(token: int, n_cells: int) -> bool:
    return 1 <= token <= n_cells


def sequence_to_layout(sequence: np.ndarray, spec: MazeSpec) -> Dict[str, object]:
    seq = np.asarray(sequence, dtype=np.int32).reshape(-1)
    expected_len = spec.max_obs_tokens + 2
    if seq.shape[0] != expected_len:
        raise ValueError(f"Expected sequence length {expected_len}, got {seq.shape[0]}")

    obstacles_raw = seq[: spec.max_obs_tokens]
    goal_token = int(seq[spec.max_obs_tokens])
    agent_token = int(seq[spec.max_obs_tokens + 1])

    n_cells = spec.n_cells

    obstacles = [int(tok) for tok in obstacles_raw if tok != 0]
    invalid_obstacles = [tok for tok in obstacles if not _valid_spatial_token(tok, n_cells)]
    valid_obstacles = [tok for tok in obstacles if _valid_spatial_token(tok, n_cells)]
    unique_obstacles = sorted(set(valid_obstacles))

    has_duplicate_obstacles = len(unique_obstacles) != len(valid_obstacles)

    wall_map = np.zeros((spec.inner_dim, spec.inner_dim), dtype=np.bool_)
    for tok in unique_obstacles:
        x, y = token_to_xy(tok, spec.inner_dim)
        wall_map[y, x] = True

    goal_valid = _valid_spatial_token(goal_token, n_cells)
    agent_valid = _valid_spatial_token(agent_token, n_cells)

    goal_xy = token_to_xy(goal_token, spec.inner_dim) if goal_valid else None
    agent_xy = token_to_xy(agent_token, spec.inner_dim) if agent_valid else None

    agent_goal_distinct = goal_valid and agent_valid and (goal_token != agent_token)

    goal_on_wall = False
    agent_on_wall = False
    if goal_xy is not None:
        goal_on_wall = bool(wall_map[goal_xy[1], goal_xy[0]])
    if agent_xy is not None:
        agent_on_wall = bool(wall_map[agent_xy[1], agent_xy[0]])

    parse_valid = (
        (len(invalid_obstacles) == 0)
        and goal_valid
        and agent_valid
        and agent_goal_distinct
        and (not goal_on_wall)
        and (not agent_on_wall)
    )

    return {
        "wall_map": wall_map,
        "goal_xy": goal_xy,
        "agent_xy": agent_xy,
        "goal_token": goal_token,
        "agent_token": agent_token,
        "n_obstacles": len(unique_obstacles),
        "n_invalid_obstacle_tokens": len(invalid_obstacles),
        "has_duplicate_obstacles": has_duplicate_obstacles,
        "goal_valid": goal_valid,
        "agent_valid": agent_valid,
        "agent_goal_distinct": agent_goal_distinct,
        "goal_on_wall": goal_on_wall,
        "agent_on_wall": agent_on_wall,
        "parse_valid": parse_valid,
    }


def _neighbors(y: int, x: int, h: int, w: int) -> Iterable[Tuple[int, int]]:
    if y > 0:
        yield y - 1, x
    if y + 1 < h:
        yield y + 1, x
    if x > 0:
        yield y, x - 1
    if x + 1 < w:
        yield y, x + 1


def _bfs_shortest_path(wall_map: np.ndarray, start_xy: Tuple[int, int], goal_xy: Tuple[int, int]) -> Tuple[int, np.ndarray]:
    h, w = wall_map.shape
    visited = np.zeros_like(wall_map, dtype=np.bool_)
    dist = np.full((h, w), -1, dtype=np.int32)

    sx, sy = start_xy
    gx, gy = goal_xy

    queue_y = [sy]
    queue_x = [sx]
    head = 0

    visited[sy, sx] = True
    dist[sy, sx] = 0

    while head < len(queue_y):
        y = queue_y[head]
        x = queue_x[head]
        head += 1

        if x == gx and y == gy:
            return int(dist[y, x]), visited

        for ny, nx in _neighbors(y, x, h, w):
            if visited[ny, nx] or wall_map[ny, nx]:
                continue
            visited[ny, nx] = True
            dist[ny, nx] = dist[y, x] + 1
            queue_y.append(ny)
            queue_x.append(nx)

    return -1, visited


def _branching_and_loops(wall_map: np.ndarray, reachable: np.ndarray) -> Tuple[int, int, int]:
    h, w = wall_map.shape
    node_count = 0
    edge_sum = 0
    branching = 0

    for y in range(h):
        for x in range(w):
            if wall_map[y, x] or (not reachable[y, x]):
                continue
            node_count += 1
            degree = 0
            for ny, nx in _neighbors(y, x, h, w):
                if (not wall_map[ny, nx]) and reachable[ny, nx]:
                    degree += 1
            edge_sum += degree
            if degree >= 3:
                branching += 1

    edges = edge_sum // 2
    loops = 0
    if node_count > 0:
        loops = max(0, edges - node_count + 1)

    return int(branching), int(loops), int(node_count)


def evaluate_sequence(sequence: np.ndarray, spec: MazeSpec) -> Dict[str, float]:
    layout = sequence_to_layout(sequence, spec)
    wall_map = layout["wall_map"]

    h, w = wall_map.shape
    free_cells = int(np.size(wall_map) - np.count_nonzero(wall_map))
    wall_density = float(np.count_nonzero(wall_map) / np.size(wall_map))

    path_len = -1
    reachable = np.zeros_like(wall_map, dtype=np.bool_)

    if layout["parse_valid"]:
        path_len, reachable = _bfs_shortest_path(
            wall_map,
            layout["agent_xy"],
            layout["goal_xy"],
        )

    branching, loops, reachable_nodes = _branching_and_loops(wall_map, reachable)

    valid = bool(layout["parse_valid"] and (path_len >= 0))

    # Score-ready feature vector values are normalized and always finite.
    max_path = float(spec.n_cells)
    path_norm = float(path_len / max_path) if path_len >= 0 else 1.0
    branching_density = float(branching / max(free_cells, 1))
    loops_density = float(loops / max(free_cells, 1))
    reachable_ratio = float(reachable_nodes / max(free_cells, 1))

    return {
        "valid": float(valid),
        "path_len": float(path_len),
        "branching": float(branching),
        "loops": float(loops),
        "wall_density": wall_density,
        "free_cells": float(free_cells),
        "reachable_nodes": float(reachable_nodes),
        "reachable_ratio": reachable_ratio,
        "n_obstacles": float(layout["n_obstacles"]),
        "n_invalid_obstacle_tokens": float(layout["n_invalid_obstacle_tokens"]),
        "duplicate_obstacles": float(layout["has_duplicate_obstacles"]),
        "feature_path_norm": path_norm,
        "feature_branching_density": branching_density,
        "feature_loops_density": loops_density,
    }


def feature_vector_from_metrics(metrics: Dict[str, float]) -> np.ndarray:
    return np.array(
        [
            metrics["wall_density"],
            metrics["feature_path_norm"],
            metrics["feature_branching_density"],
            metrics["feature_loops_density"],
            metrics["reachable_ratio"],
        ],
        dtype=np.float64,
    )


def compute_feature_stats(metrics: List[Dict[str, float]], eps: float = 1e-6) -> Dict[str, np.ndarray]:
    if not metrics:
        raise ValueError("No metrics provided for feature stats.")

    valid_mask = np.array([m["valid"] > 0.5 for m in metrics], dtype=np.bool_)
    selected = [metrics[i] for i in range(len(metrics)) if valid_mask[i]]
    if len(selected) < 10:
        selected = metrics

    features = np.stack([feature_vector_from_metrics(m) for m in selected], axis=0)

    mean = features.mean(axis=0)
    cov = np.cov(features, rowvar=False)
    cov = np.atleast_2d(cov)
    cov += np.eye(cov.shape[0], dtype=np.float64) * eps
    cov_inv = np.linalg.pinv(cov)

    return {
        "mean": mean,
        "cov": cov,
        "cov_inv": cov_inv,
        "n_reference": np.array([features.shape[0]], dtype=np.int64),
    }


def mahalanobis_distance_batch(
    vectors: np.ndarray,
    mean: np.ndarray,
    cov_inv: np.ndarray,
) -> np.ndarray:
    x = np.asarray(vectors, dtype=np.float64)
    centered = x - mean[None, :]
    left = centered @ cov_inv
    d2 = np.sum(left * centered, axis=1)
    d2 = np.clip(d2, a_min=0.0, a_max=None)
    return np.sqrt(d2)
