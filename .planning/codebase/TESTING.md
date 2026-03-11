# Testing Patterns

**Analysis Date:** 2026-03-11

## Test Framework

**Runner:**
- pytest
- Config: `tox.ini` (not pytest.ini)
- Tox envlist: py39, py310, py311, py312, py313

**Assertion Library:**
- pytest built-in assertions (no custom assertion helpers found)

**Run Commands:**
```bash
pytest -v -s --cov=src/jaxued                # Run all tests with coverage
tox                                           # Run tests across all Python versions
tox -e py311                                  # Run tests for specific Python version
```

**Coverage:**
- Coverage tracking on `src/jaxued` package specifically
- No coverage thresholds enforced (not configured in tox.ini)

## Test File Organization

**Location:**
- Co-located with source in `tests/` directory at project root
- Location: `/cs/student/project_msc/2025/csml/gmaralla/superintelligence/tests/`

**Naming:**
- Test files use `test_*.py` pattern: `test_examples_kinda.py`
- Test functions use `test_*` prefix: `test_run_example()`

**Structure:**
```
tests/
└── test_examples_kinda.py      # Integration tests for example scripts
```

## Test Structure

**Suite Organization:**
```python
import subprocess
import pytest
import os
import sys
import time

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "../examples")
EXAMPLE_SCRIPTS = [
    "maze_dr.py",
    "maze_plr.py",
    "maze_paired.py",
]

@pytest.mark.parametrize("script", EXAMPLE_SCRIPTS)
def test_run_example(script):
    script_path = os.path.join(EXAMPLES_DIR, script)
    assert os.path.exists(script_path), f"Script {script} not found."

    # Setup environment
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    # Run subprocess test
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
        pass
    except Exception as e:
        pytest.fail(f"Error running {script}: {e}")
```

**Patterns:**
- Parametrized tests using `@pytest.mark.parametrize()` for running same test with multiple inputs
- Setup via environment variables: `env["WANDB_MODE"] = "disabled"` to disable external logging
- No explicit teardown (subprocess handles cleanup)
- Assertions on return codes: `assert process.returncode in [None, 0]`
- Graceful timeout handling: `except subprocess.TimeoutExpired: pass` for long-running tests
- Assertion on subprocess stderr for debugging: `assert process.returncode..., f"Script {script} failed:\n{process.stderr.decode()}"`

## Mocking

**Framework:** No mocking library detected (unittest.mock not imported)

**What NOT to Mock:**
- Environment subprocess execution is real, not mocked
- Tests verify actual script execution end-to-end
- JAX/Flax operations run in real computation (no JAX tracing in tests)

**Real Subprocess Tests:**
- Each test script is executed as a subprocess with actual dependencies
- Timeout protection: 30-second timeout prevents hanging
- Standard output/error captured for debugging

## Fixtures and Factories

**Test Data:**
- No pytest fixtures defined
- Environment variables used for configuration:
  ```python
  env = os.environ.copy()
  env["WANDB_MODE"] = "disabled"
  ```

**Location:**
- Shared test utilities in `tests/test_examples_kinda.py`
- Example scripts in `examples/` used as test inputs

## Coverage

**Requirements:** No coverage minimum enforced

**View Coverage:**
```bash
pytest -v -s --cov=src/jaxued --cov-report=html    # Generate HTML coverage report
pytest --cov=src/jaxued --cov-report=term-missing  # Terminal coverage with missing lines
```

## Test Types

**Integration Tests:**
- `test_run_example()`: Runs example scripts as subprocess to verify end-to-end functionality
- Parametrized over: `maze_dr.py`, `maze_plr.py`, `maze_paired.py`
- Checks: script existence, successful execution (return code 0), error output
- 30-second timeout per script

**Unit Tests:**
- Not detected in current test suite
- Main focus is integration testing of examples

**E2E Tests:**
- Effectively E2E via subprocess execution of training scripts
- Verifies complete pipeline from environment setup through training initialization

## Common Patterns

**Subprocess Testing:**
```python
@pytest.mark.parametrize("script", EXAMPLE_SCRIPTS)
def test_run_example(script):
    script_path = os.path.join(EXAMPLES_DIR, script)
    assert os.path.exists(script_path), f"Script {script} not found."

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"  # Disable external logging for tests

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
        pass  # Timeout is acceptable (script may run indefinitely)
    except Exception as e:
        pytest.fail(f"Error running {script}: {e}")
```

**Key Pattern Elements:**
- Environment variable override to disable wandb during testing
- Subprocess stdout/stderr capture for debugging failures
- Timeout handling for long-running training scripts
- Graceful timeout acceptance (training scripts may run longer than test timeout)

## Test Execution

**Running Tests:**
```bash
# Run with pytest directly
pytest -v -s --cov=src/jaxued

# Run with tox (tests all supported Python versions)
tox

# Run single environment
tox -e py311

# Run with verbose output
pytest -v tests/test_examples_kinda.py

# Run single parametrized test
pytest -v tests/test_examples_kinda.py::test_run_example[maze_dr.py]
```

## Testing Philosophy

**Current approach:**
- Minimal unit tests; emphasis on integration tests
- Subprocess-based testing ensures real dependencies work
- Environment isolation via `WANDB_MODE=disabled`
- Timeout-based smoke testing (30 seconds per script)

**Gaps detected:**
- No unit tests for individual functions (`compute_gae()`, `update_actor_critic_rnn()`, etc.)
- No tests for JAX-traced operations
- No tests for environment mechanics (Maze, Level, etc.)
- No tests for dataclass invariants or error cases
- No tests for error handling paths (GCS upload fallback, parsing errors, etc.)

---

*Testing analysis: 2026-03-11*
