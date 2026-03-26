# Testing Patterns

**Analysis Date:** 2025-03-23

## Test Framework

**Runner:**
- `pytest` — configured in `tox.ini`
- Version: Not pinned in `pyproject.toml` (uses latest compatible)

**Config:**
- File: `tox.ini`
- Test environments: Python 3.9, 3.10, 3.11, 3.12, 3.13

**Assertion Library:**
- Python built-in `assert` statements
- No pytest-specific assertions or fixtures detected

**Run Commands:**
```bash
pytest -v -s --cov=src/jaxued                 # Run all tests with coverage
tox                                            # Run tests across all Python versions
```

From `tox.ini`:
```ini
[testenv]
deps =
    pytest
    pytest-cov
commands = pytest -v -s --cov=src/jaxued
```

## Test File Organization

**Location:**
- Single test file: `tests/test_examples_kinda.py`
- Separate from source (not co-located with code)

**Naming:**
- File pattern: `test_*.py`
- Test function pattern: `test_*` (parameterized test names)

**Structure:**
```
tests/
├── test_examples_kinda.py
└── (no other test files)
```

**Note:** Minimal test coverage — only example execution test detected. No unit tests for library code.

## Test Structure

**Suite Organization:**

From `tests/test_examples_kinda.py`:
```python
import subprocess
import pytest
import os
import sys

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "../examples")
EXAMPLE_SCRIPTS = [
    "maze_dr.py",
    "maze_plr.py",
    "maze_paired.py",
]

@pytest.mark.parametrize("script", EXAMPLE_SCRIPTS)
def test_run_example(script):
    # Test implementation
```

**Patterns:**
- Uses `@pytest.mark.parametrize` for data-driven tests
- Setup: Tests configure environment variables (`WANDB_MODE=disabled`)
- Execution: Uses `subprocess.run()` to execute scripts as separate processes
- Assertions: Checks return code: `assert process.returncode in [None, 0]`
- Timeout: 30-second timeout per example

## Mocking

**Framework:** No mocking detected in test suite

**Patterns:**
- Environment variable manipulation instead of mocking: `env = os.environ.copy(); env["WANDB_MODE"] = "disabled"`
- External system interaction: Tests use real environment objects, not mocks
- No `unittest.mock`, `pytest-mock`, or `pytest-freezegun` detected

## Fixtures and Factories

**Test Data:**
Not detected — tests execute real example scripts rather than using fixtures

**Location:**
No conftest.py file; no shared fixtures

**Note:** Testing approach is integration-level (full example execution) rather than unit-level with fixtures

## Coverage

**Requirements:** `--cov=src/jaxued` flag in tox commands, but no coverage threshold enforcement

**View Coverage:**
```bash
pytest --cov=src/jaxued
```

**Observed:** Coverage report generates but no minimum threshold set in `pyproject.toml` or `tox.ini`

## Test Types

**Unit Tests:**
Not observed — only integration tests present

**Integration Tests:**
- Type: Script execution tests
- Scope: Full example scripts (maze_plr.py, maze_dr.py, maze_paired.py)
- Approach: Subprocess execution with timeout

From test:
```python
process = subprocess.run(
    [sys.executable, script_path],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    timeout=30,
    env=env,
)
```

**E2E Tests:**
Not detected as separate category — integration tests serve as end-to-end verification

**Snapshot Tests:**
Not present

**Property-based Tests:**
Not detected (no hypothesis library)

## Common Patterns

**Error Handling in Tests:**

From `test_examples_kinda.py`:
```python
try:
    process = subprocess.run(
        [sys.executable, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        env=env,
    )
    assert process.returncode in [None, 0], f"Script {script} failed:\n{process.stderr.decode()}"
except subprocess.TimeoutExpired:
    pass  # Allow timeout (examples may be long-running)
except Exception as e:
    pytest.fail(f"Error running {script}: {e}")
```

**Timeout Handling:**
- 30-second timeout allows example to run but prevents hanging
- TimeoutExpired exception silently passed (acceptable for long-running examples)

**Environment Setup:**
```python
env = os.environ.copy()
env["WANDB_MODE"] = "disabled"  # Disable WandB during testing
```

## Testing Gaps

**Not Tested:**
- Core library functions: `LevelSampler`, `Maze`, wrappers
- Utility functions: `accumulate_rollout_stats()`, `compute_max_returns()`
- JAX operations: scan, vmap, jit behavior
- Error conditions: Invalid configurations, boundary cases
- Type safety: Type hints not validated at runtime

**Why:**
- Focus is on research code (examples are primary deliverable)
- Library components tested implicitly through example scripts
- JAX code difficult to test in isolation (heavy use of functional transformations)

## Test Infrastructure

**CI/CD:**
No GitHub Actions or CI pipeline detected in repository

**Pre-commit Hooks:**
Not observed in `.git/hooks/` or configuration files

**Test Database:**
Not applicable (no state dependencies)

**Parallelization:**
Not configured — tests run sequentially

## Recommendations for New Tests

**Unit Test Pattern:**
When adding core functionality, follow this pattern:

```python
# tests/test_level_sampler.py
import pytest
import jax
import jax.numpy as jnp
from jaxued.level_sampler import LevelSampler
from jaxued.environments.maze.level import Level

def create_test_level():
    """Fixture factory for Level objects"""
    return Level(
        wall_map=jnp.zeros((5, 5), dtype=bool),
        goal_pos=jnp.array([4, 4], dtype=jnp.uint32),
        agent_pos=jnp.array([0, 0], dtype=jnp.uint32),
        agent_dir=0,
        width=5,
        height=5,
    )

@pytest.mark.parametrize("capacity", [10, 100, 1000])
def test_level_sampler_insert(capacity):
    level_sampler = LevelSampler(capacity=capacity)
    pholder = create_test_level()
    sampler = level_sampler.initialize(pholder)

    new_level = create_test_level()
    sampler, idx = level_sampler.insert(sampler, new_level, score=1.0)

    assert idx >= 0, "Level should be inserted at valid index"
    assert sampler["size"] == 1, "Sampler size should increase after insert"
```

**Integration Test Pattern:**
When adding new examples or workflows:

```python
# Extend test_examples_kinda.py
EXAMPLE_SCRIPTS = [
    "maze_dr.py",
    "maze_plr.py",
    "maze_paired.py",
    "new_workflow.py",  # New test
]
```

---

*Testing analysis: 2025-03-23*
