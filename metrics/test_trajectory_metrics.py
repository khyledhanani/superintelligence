"""Tests for trajectory diversity metrics."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from trajectory_metrics import (
    dtw_with_path,
    observation_dtw,
    position_trace_dtw,
    value_trajectory_dtw,
    spatial_footprint_jaccard,
    compute_pairwise_metrics,
)


def test_dtw_identical_sequences():
    """Two identical sequences should have zero distance."""
    seq = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]])
    dist, path, local_costs = dtw_with_path(seq, seq)
    assert dist < 1e-8, f"Expected ~0 distance for identical sequences, got {dist}"
    assert len(path) == len(seq), f"Path length should match sequence length, got {len(path)}"
    assert np.allclose(local_costs, 0.0), "Local costs should be zero for identical sequences"
    print("PASS: dtw_identical_sequences")


def test_dtw_different_sequences():
    """Very different sequences should have high distance."""
    seq_a = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    seq_b = np.array([[0.0, 10.0], [1.0, 10.0], [2.0, 10.0], [3.0, 10.0]])
    dist, path, local_costs = dtw_with_path(seq_a, seq_b)
    assert dist > 5.0, f"Expected high distance, got {dist}"
    assert all(c > 0 for c in local_costs), "All local costs should be positive"
    print("PASS: dtw_different_sequences")


def test_dtw_different_speeds():
    """Same path at different speeds -- DTW should align correctly."""
    seq_fast = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    seq_slow = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    dist, path, local_costs = dtw_with_path(seq_fast, seq_slow)
    assert dist < 0.5, f"Same path at different speeds should have low distance, got {dist}"
    print(f"PASS: dtw_different_speeds (distance={dist:.4f})")


def test_observation_dtw():
    """Test observation DTW with truncation at done."""
    np.random.seed(100)
    obs_a = np.random.randn(10, 5, 5, 3).astype(np.float32)
    dones_a = np.zeros(10, dtype=bool)
    dones_a[6] = True

    obs_b = obs_a.copy()
    dones_b = dones_a.copy()

    result = observation_dtw(obs_a, dones_a, obs_b, dones_b)
    assert result["distance"] < 1e-6, f"Identical obs should give ~0 distance, got {result['distance']}"

    obs_c = np.random.randn(8, 5, 5, 3).astype(np.float32)
    dones_c = np.zeros(8, dtype=bool)
    dones_c[5] = True

    result2 = observation_dtw(obs_a, dones_a, obs_c, dones_c)
    assert result2["distance"] > 0.1, f"Different obs should give non-zero distance, got {result2['distance']}"
    print("PASS: observation_dtw")


def test_position_trace_dtw():
    """Test position trace DTW."""
    pos_a = np.array([[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]])
    dones_a = np.array([False, False, False, False, True])

    pos_b = np.array([[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]])
    dones_b = np.array([False, False, False, False, True])

    result = position_trace_dtw(pos_a, dones_a, pos_b, dones_b)
    assert result["distance"] < 1e-6, f"Same path should give ~0 distance, got {result['distance']}"

    # Different shape: pos_a goes right then down, pos_c goes down then left
    pos_c = np.array([[3, 3], [3, 4], [3, 5], [3, 6], [2, 6], [1, 6]])
    dones_c = np.array([False, False, False, False, False, True])

    result2 = position_trace_dtw(pos_a, dones_a, pos_c, dones_c)
    assert result2["distance"] > 0.5, f"Different-shape path should give non-trivial distance, got {result2['distance']}"
    print("PASS: position_trace_dtw")


def test_value_trajectory_dtw():
    """Test value trajectory DTW."""
    values = np.linspace(0, 1, 20).astype(np.float64)
    dones = np.zeros(20, dtype=bool)
    dones[19] = True

    result = value_trajectory_dtw(values, dones, values, dones)
    assert result["distance"] < 1e-6, f"Same curves should have ~0 DTW distance, got {result['distance']}"
    assert "local_costs" in result, "Should return local_costs from DTW"
    assert "path" in result, "Should return warping path from DTW"

    values_inv = 1.0 - values
    result2 = value_trajectory_dtw(values, dones, values_inv, dones)
    assert result2["distance"] > 0.1, f"Inverted curves should have non-zero DTW distance, got {result2['distance']}"

    values_const = np.ones(15) * 0.5
    dones_const = np.zeros(15, dtype=bool)
    dones_const[14] = True
    result3 = value_trajectory_dtw(values, dones, values_const, dones_const)
    assert result3["distance"] > 0, f"Ramp vs constant should have non-zero DTW distance, got {result3['distance']}"
    print("PASS: value_trajectory_dtw")


def test_spatial_footprint_jaccard():
    """Test Jaccard index of visited cells."""
    pos_a = np.array([[0, 0], [0, 1], [0, 2], [1, 2]])
    dones_a = np.array([False, False, False, True])

    result = spatial_footprint_jaccard(pos_a, dones_a, pos_a, dones_a)
    assert result["jaccard"] == 1.0, f"Same cells should give Jaccard 1.0, got {result['jaccard']}"

    pos_b = np.array([[5, 5], [5, 6], [6, 6], [7, 7]])
    dones_b = np.array([False, False, False, True])
    result2 = spatial_footprint_jaccard(pos_a, dones_a, pos_b, dones_b)
    assert result2["jaccard"] == 0.0, f"Disjoint cells should give Jaccard 0.0, got {result2['jaccard']}"

    pos_c = np.array([[0, 0], [0, 1], [1, 1], [2, 1]])
    dones_c = np.array([False, False, False, True])
    result3 = spatial_footprint_jaccard(pos_a, dones_a, pos_c, dones_c)
    expected = 2 / 6
    assert abs(result3["jaccard"] - expected) < 1e-6, f"Expected Jaccard {expected}, got {result3['jaccard']}"
    print("PASS: spatial_footprint_jaccard")


def test_compute_pairwise_metrics():
    """Test the batch pairwise metric computation."""
    np.random.seed(42)
    trajectories = []
    for _ in range(4):
        t = np.random.randint(8, 15)
        dones = np.zeros(t, dtype=bool)
        dones[-1] = True
        trajectories.append({
            "observations": np.random.randn(t, 5, 5, 3).astype(np.float32),
            "positions": np.random.randint(0, 13, size=(t, 2)),
            "values": np.random.randn(t).astype(np.float64),
            "dones": dones,
        })

    pairwise = compute_pairwise_metrics(trajectories)
    n_pairs = 4 * 3 // 2  # 6 pairs
    assert len(pairwise["obs_dtw_distances"]) == n_pairs
    assert len(pairwise["pos_dtw_distances"]) == n_pairs
    assert len(pairwise["value_dtw_distances"]) == n_pairs
    assert len(pairwise["jaccard_indices"]) == n_pairs
    print(f"PASS: compute_pairwise_metrics (obs_dtw_mean={pairwise['obs_dtw_distances'].mean():.4f}, "
          f"pos_dtw_mean={pairwise['pos_dtw_distances'].mean():.4f})")


def test_truncation():
    """Test that truncation at first done works correctly."""
    pos = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
    dones_early = np.array([False, False, True, False, False])
    dones_never = np.zeros(5, dtype=bool)

    result_early = position_trace_dtw(pos, dones_early, pos, dones_early)
    assert result_early["path"].shape[0] == 3, f"Expected 3-step path, got {result_early['path'].shape[0]}"

    result_full = position_trace_dtw(pos, dones_never, pos, dones_never)
    assert result_full["path"].shape[0] == 5, f"Expected 5-step path, got {result_full['path'].shape[0]}"
    print("PASS: truncation")


if __name__ == "__main__":
    test_dtw_identical_sequences()
    test_dtw_different_sequences()
    test_dtw_different_speeds()
    test_observation_dtw()
    test_position_trace_dtw()
    test_value_trajectory_dtw()
    test_spatial_footprint_jaccard()
    test_compute_pairwise_metrics()
    test_truncation()
    print("\nAll trajectory metric tests passed!")
