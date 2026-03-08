"""Tests for trajectory cache."""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from trajectory_cache import TrajectoryCache


def test_basic_operations():
    """Test insert, get, evict, contains."""
    cache = TrajectoryCache()
    assert len(cache) == 0

    obs = np.random.randn(10, 5, 5, 3)
    pos = np.random.randint(0, 13, (10, 2))
    vals = np.random.randn(10)
    dones = np.zeros(10, dtype=bool)
    dones[7] = True
    actions = np.random.randint(0, 4, 10)

    cache.update(0, obs, pos, vals, dones, actions)
    assert len(cache) == 1
    assert 0 in cache

    entry = cache.get(0)
    assert entry is not None
    assert entry["observations"].shape[0] == 8
    assert entry["positions"].shape[0] == 8
    assert entry["values"].shape[0] == 8
    assert entry["dones"][-1] == True

    cache.evict(0)
    assert len(cache) == 0
    assert 0 not in cache
    print("PASS: basic_operations")


def test_batch_update():
    """Test batch update."""
    cache = TrajectoryCache()
    T, N = 15, 5
    obs = np.random.randn(T, N, 5, 5, 3)
    pos = np.random.randint(0, 13, (T, N, 2))
    vals = np.random.randn(T, N)
    dones = np.zeros((T, N), dtype=bool)
    for i in range(N):
        dones[np.random.randint(5, T), i] = True
    actions = np.random.randint(0, 4, (T, N))
    indices = np.array([10, 20, 30, 40, 50])

    cache.update_batch(indices, obs, pos, vals, dones, actions)
    assert len(cache) == 5
    for idx in indices:
        assert idx in cache
        entry = cache.get(idx)
        assert entry["dones"][-1] == True
    print("PASS: batch_update")


def test_sync_with_buffer():
    """Test eviction of stale entries."""
    cache = TrajectoryCache()
    for i in range(10):
        cache.update(i, np.zeros((3, 2)), np.zeros((3, 2)), np.zeros(3),
                     np.array([False, False, True]), np.zeros(3))

    assert len(cache) == 10

    cache.sync_with_buffer({0, 1, 2, 3, 4})
    assert len(cache) == 5
    for i in range(5):
        assert i in cache
    for i in range(5, 10):
        assert i not in cache
    print("PASS: sync_with_buffer")


def test_sample():
    """Test random sampling."""
    cache = TrajectoryCache()
    for i in range(20):
        cache.update(i, np.zeros((3, 2)), np.zeros((3, 2)), np.zeros(3),
                     np.array([False, False, True]), np.zeros(3))

    rng = np.random.default_rng(42)
    samples = cache.sample(5, rng)
    assert len(samples) == 5
    assert all("level_idx" in s for s in samples)
    assert len(set(s["level_idx"] for s in samples)) == 5
    print("PASS: sample")


def test_no_done_full_trajectory():
    """When no done flag, should keep full trajectory."""
    cache = TrajectoryCache()
    obs = np.random.randn(10, 2)
    pos = np.random.randint(0, 13, (10, 2))
    vals = np.random.randn(10)
    dones = np.zeros(10, dtype=bool)
    actions = np.zeros(10)

    cache.update(0, obs, pos, vals, dones, actions)
    entry = cache.get(0)
    assert entry["observations"].shape[0] == 10
    print("PASS: no_done_full_trajectory")


if __name__ == "__main__":
    test_basic_operations()
    test_batch_update()
    test_sync_with_buffer()
    test_sample()
    test_no_done_full_trajectory()
    print("\nAll trajectory cache tests passed!")
