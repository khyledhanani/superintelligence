"""
CNN-VAE Checkpoint Verification Script
=======================================
Verifies CKPT-01, CKPT-02, and CKPT-03 requirements for Phase 1.

Requirements:
  CKPT-01: CNN-VAE Orbax checkpoint files exist locally at vae/checkpoints/cnn_vae/
  CKPT-02: decoder_params dict contains all 8 expected sub-keys
  CKPT-03: decode_fn(z_zeros) returns (wall, goal, agent) logits each (1,13,13) / (13,13)

Run on sideswipe or prowl (CUDA 12 nodes), NOT on blaze (CUDA 11.7 incompatible).
Usage:
    cd /cs/student/project_msc/2025/csml/gmaralla/superintelligence
    /cs/student/project_msc/2025/csml/gmaralla/miniconda3/envs/jax_env/bin/python scripts/verify_checkpoint.py
"""

import sys
import os

# Ensure the project root is on the Python path for vae.cnn_vae_model import
sys.path.insert(0, '/cs/student/project_msc/2025/csml/gmaralla/superintelligence')

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

# Checkpoint directory (relative to project root)
CKPT_DIR = 'vae/checkpoints/cnn_vae'
CKPT_DIR_ABS = '/cs/student/project_msc/2025/csml/gmaralla/superintelligence/vae/checkpoints/cnn_vae'

EXPECTED_DECODER_KEYS = {
    'dec_lstm', 'dec_proj', 'dec_conv1', 'dec_conv2', 'dec_conv3',
    'wall_head', 'goal_head', 'agent_head'
}

all_passed = True


def fail(msg: str) -> None:
    """Print a failure message and mark the run as failed."""
    global all_passed
    print(f'FAIL: {msg}', file=sys.stderr)
    all_passed = False


def check(condition: bool, pass_msg: str, fail_msg: str) -> bool:
    """Assert a condition, print pass/fail, return True if passed."""
    if condition:
        print(f'PASS: {pass_msg}')
        return True
    else:
        fail(fail_msg)
        return False


# ---------------------------------------------------------------------------
# CKPT-01: Checkpoint files exist locally
# ---------------------------------------------------------------------------
print('=' * 60)
print('CKPT-01: Verifying checkpoint files exist locally')
print('=' * 60)

ckpt_exists = os.path.isdir(CKPT_DIR)
check(ckpt_exists,
      f'Checkpoint directory exists: {CKPT_DIR}',
      f'Checkpoint directory not found: {CKPT_DIR}')

if ckpt_exists:
    entries = os.listdir(CKPT_DIR_ABS)
    print(f'  Directory contents: {sorted(entries)}')
    non_empty = len(entries) > 0
    check(non_empty,
          'Checkpoint directory is non-empty',
          'Checkpoint directory is empty — download may have failed')
else:
    print('  Skipping subsequent checks — checkpoint directory missing.')
    sys.exit(1)

# ---------------------------------------------------------------------------
# Determine which Orbax loader to use
# ---------------------------------------------------------------------------
# Inspect entries: if any entry is a digit-only subdirectory (e.g. '200000'),
# use CheckpointManager; otherwise use PyTreeCheckpointer.
has_step_dirs = any(e.isdigit() for e in entries)
print(f'\nCheckpoint format detection:')
print(f'  Has step subdirectories (digit-only dirs): {has_step_dirs}')

# Special case: if 'default/' subdir is present but no digit dirs, the checkpoint
# was saved by StandardCheckpointHandler — PyTreeCheckpointer on the 'default/' subdir.
has_default_subdir = 'default' in entries

if has_default_subdir and not has_step_dirs:
    print('  Format: StandardCheckpointHandler (default/ subdir present)')
    print('  Loader: PyTreeCheckpointer on default/ subdir')
    loader_type = 'pytree_default'
elif has_step_dirs:
    print('  Format: CheckpointManager (step-indexed subdirs)')
    print('  Loader: CheckpointManager')
    loader_type = 'manager'
else:
    print('  Format: Flat PyTreeCheckpointer directory')
    print('  Loader: PyTreeCheckpointer on root dir')
    loader_type = 'pytree_root'

# ---------------------------------------------------------------------------
# CKPT-02: Load checkpoint and verify decoder params
# ---------------------------------------------------------------------------
print()
print('=' * 60)
print('CKPT-02: Loading checkpoint and verifying decoder param tree')
print('=' * 60)

restored = None
try:
    if loader_type == 'pytree_default':
        # Orbax requires absolute path for tensorstore
        ckpt_default_abs = os.path.join(CKPT_DIR_ABS, 'default')
        checkpointer = ocp.PyTreeCheckpointer()
        restored = checkpointer.restore(ckpt_default_abs)
        print(f'  Loaded with PyTreeCheckpointer from: {ckpt_default_abs}')

    elif loader_type == 'manager':
        mgr = ocp.CheckpointManager(
            CKPT_DIR_ABS,
            options=ocp.CheckpointManagerOptions()
        )
        restored = mgr.restore(200000)
        print(f'  Loaded with CheckpointManager, step=200000')

    else:  # pytree_root
        checkpointer = ocp.PyTreeCheckpointer()
        restored = checkpointer.restore(CKPT_DIR_ABS)
        print(f'  Loaded with PyTreeCheckpointer from: {CKPT_DIR_ABS}')

    check(restored is not None,
          'Checkpoint restored successfully',
          'Checkpoint restore returned None')

except Exception as e:
    fail(f'Failed to load checkpoint: {e}')
    print('  Cannot proceed without loaded checkpoint.')
    sys.exit(1)

# Verify top-level structure
top_keys = set(restored.keys()) if isinstance(restored, dict) else set()
print(f'  Top-level keys in restored checkpoint: {sorted(top_keys)}')
check('params' in top_keys,
      "Checkpoint has 'params' top-level key",
      f"Missing 'params' key; got: {sorted(top_keys)}")

if 'params' not in top_keys:
    fail('Cannot extract decoder_params — missing params key')
    sys.exit(1)

params = restored['params']
params_keys = set(params.keys()) if isinstance(params, dict) else set()
print(f'  params sub-keys: {sorted(params_keys)}')

check('decoder' in params_keys,
      "params has 'decoder' sub-key",
      f"Missing 'decoder' in params; got: {sorted(params_keys)}")

if 'decoder' not in params_keys:
    fail('Cannot extract decoder_params — missing decoder key')
    sys.exit(1)

decoder_params = params['decoder']
actual_decoder_keys = set(decoder_params.keys()) if isinstance(decoder_params, dict) else set()
print(f'\n  Decoder sub-keys: {sorted(actual_decoder_keys)}')

missing_keys = EXPECTED_DECODER_KEYS - actual_decoder_keys
extra_keys = actual_decoder_keys - EXPECTED_DECODER_KEYS

check(len(missing_keys) == 0,
      f'All 8 expected decoder sub-keys present: {sorted(EXPECTED_DECODER_KEYS)}',
      f'Missing decoder sub-keys: {sorted(missing_keys)}')

if extra_keys:
    print(f'  Note: Extra decoder keys (unexpected but not blocking): {sorted(extra_keys)}')

# Print full param tree shapes
print('\n  Decoder param shapes:')
shapes = jax.tree_util.tree_map(lambda x: x.shape, decoder_params)
import pprint
for key, val in sorted(shapes.items()):
    print(f'    {key}: {val}')

ckpt02_passed = len(missing_keys) == 0

# ---------------------------------------------------------------------------
# CKPT-03: Run decode_fn(z_zeros) and verify output shapes
# ---------------------------------------------------------------------------
print()
print('=' * 60)
print('CKPT-03: Running decode_fn(z_zeros) and verifying output shapes')
print('=' * 60)

if not ckpt02_passed:
    fail('Skipping CKPT-03 — decoder_params failed CKPT-02 verification')
else:
    try:
        from vae.cnn_vae_model import CnnLstmDecoder

        decoder = CnnLstmDecoder(latent_dim=64)
        z_zeros = jnp.zeros((1, 64))
        wall_logits, goal_logits, agent_logits = decoder.apply(
            {'params': decoder_params}, z_zeros
        )

        print(f'  wall_logits  shape: {wall_logits.shape}')
        print(f'  goal_logits  shape: {goal_logits.shape}')
        print(f'  agent_logits shape: {agent_logits.shape}')

        check(wall_logits.shape == (1, 13, 13),
              'wall_logits shape is (1, 13, 13)',
              f'wall_logits shape mismatch: expected (1,13,13) got {wall_logits.shape}')
        check(goal_logits.shape == (1, 13, 13),
              'goal_logits shape is (1, 13, 13)',
              f'goal_logits shape mismatch: expected (1,13,13) got {goal_logits.shape}')
        check(agent_logits.shape == (1, 13, 13),
              'agent_logits shape is (1, 13, 13)',
              f'agent_logits shape mismatch: expected (1,13,13) got {agent_logits.shape}')

        # Check for NaN values
        wall_nan = bool(jnp.any(jnp.isnan(wall_logits)))
        goal_nan = bool(jnp.any(jnp.isnan(goal_logits)))
        agent_nan = bool(jnp.any(jnp.isnan(agent_logits)))

        check(not wall_nan, 'wall_logits contains no NaN values',
              'wall_logits contains NaN values')
        check(not goal_nan, 'goal_logits contains no NaN values',
              'goal_logits contains NaN values')
        check(not agent_nan, 'agent_logits contains no NaN values',
              'agent_logits contains NaN values')

        # Squeeze to single-sample (13, 13) as per CKPT-03 spec
        wl = wall_logits[0]
        gl = goal_logits[0]
        al = agent_logits[0]

        check(wl.shape == (13, 13),
              'wall_logits squeezed shape is (13, 13)',
              f'wall_logits squeezed shape mismatch: {wl.shape}')
        check(gl.shape == (13, 13),
              'goal_logits squeezed shape is (13, 13)',
              f'goal_logits squeezed shape mismatch: {gl.shape}')
        check(al.shape == (13, 13),
              'agent_logits squeezed shape is (13, 13)',
              f'agent_logits squeezed shape mismatch: {al.shape}')

    except Exception as e:
        fail(f'Exception during decode_fn: {e}')

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print()
print('=' * 60)
if all_passed:
    print('ALL CHECKS PASSED: CKPT-01, CKPT-02, CKPT-03')
    print('  Checkpoint: vae/checkpoints/cnn_vae/ (run10, step 200000)')
    print('  Loader:     PyTreeCheckpointer on default/ subdir')
    print('  Decoder:    CnnLstmDecoder(latent_dim=64) — 8 sub-keys present')
    print('  Output:     (wall, goal, agent) logits each (1,13,13) / (13,13), no NaN')
    sys.exit(0)
else:
    print('SOME CHECKS FAILED — see FAIL messages above')
    sys.exit(1)
