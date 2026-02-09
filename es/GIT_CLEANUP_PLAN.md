# Git Branch Cleanup Plan for `feat/es-implementation-Giacomo`

## Current Status

```
On branch feat/es-implementation-Giacomo
Changes not staged for commit:
    modified:   vae/vae_train_config.yml

Untracked files:
    nlp/                          ← REMOVE (not related to ES)
    vae/evolve_config.yml         ← ADD
    vae/evolve_envs.py            ← ADD
    vae/evolved/                  ← GITIGNORE (generated outputs)
    vae/miniconda_old.sh          ← REMOVE (not needed)
    vae/vae_decoder.py            ← ADD
    vae/visualize_envs.py         ← ADD (new)
    vae/test_evolution.sh         ← ADD (new)
    vae/EVOLUTION_README.md       ← ADD (new)
```

## Cleanup Steps

### 1. Add Generated Outputs to .gitignore

Create/update `.gitignore` in the repo root:

```bash
cd /cs/student/msc/csml/2025/gmaralla/superintelligence
```

Add these lines to `.gitignore`:
```
# VAE evolution outputs
vae/evolved/
vae/*.png
vae/__pycache__/
vae/.ipynb_checkpoints/

# Python cache
*.pyc
__pycache__/

# Misc
vae/miniconda_old.sh
```

### 2. Remove Unrelated Files

```bash
# Remove the nlp folder (not part of ES implementation)
git clean -fd nlp/

# Remove miniconda installer
rm vae/miniconda_old.sh
```

### 3. Review vae_train_config.yml Changes

Check what was modified:
```bash
git diff vae/vae_train_config.yml
```

If the changes are just path updates or unrelated, you can either:
- Keep them if they're improvements
- Revert with `git restore vae/vae_train_config.yml`

### 4. Stage New ES Files

```bash
git add vae/vae_decoder.py
git add vae/evolve_envs.py
git add vae/evolve_config.yml
git add vae/visualize_envs.py
git add vae/test_evolution.sh
git add vae/EVOLUTION_README.md
git add vae/GIT_CLEANUP_PLAN.md
git add .gitignore  # if you created/modified it
```

### 5. Review Staged Changes

```bash
git status
git diff --cached
```

Verify only ES-related files are staged.

### 6. Commit with Clear Message

```bash
git commit -m "feat: Add CMA-ES evolution for CLUTTR environments

Implement evolutionary environment generation using CMA-ES in VAE latent space:

- Add vae_decoder.py: Standalone VAE decoder extraction and repair utilities
- Add evolve_envs.py: Main CMA-ES evolution script using evosax
- Add evolve_config.yml: Evolution hyperparameters configuration
- Add visualize_envs.py: Environment visualization tool (random vs evolved)
- Add test_evolution.sh: End-to-end testing script
- Add EVOLUTION_README.md: Complete documentation and usage guide

Evolution approach:
- Search in 64-dim continuous VAE latent space (not discrete sequences)
- Use CMA-ES for covariance-adaptive optimization
- Warm-start with VAE-encoded random environments
- Placeholder fitness: obstacle density + agent-goal distance
- Outputs saved to vae/evolved/ (gitignored)

Future work:
- Replace placeholder fitness with RL-based evaluation
- Integrate with jaxued LevelSampler for curriculum learning

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

### 7. Push to Remote

```bash
# First time pushing this branch
git push -u origin feat/es-implementation-Giacomo

# Or if branch already exists on remote
git push
```

### 8. Create Pull Request (when ready)

On GitHub:
1. Go to your repository
2. Click "Compare & pull request" for `feat/es-implementation-Giacomo`
3. Set base branch to `main` (or `dev` if you have one)
4. Title: "Add CMA-ES Evolution for CLUTTR Environments"
5. Description:

```markdown
## Overview
Implements evolutionary environment generation for CLUTTR using CMA-ES in the VAE latent space.

## Key Changes
- **Evolution Pipeline**: CMA-ES search in 64-dim VAE latent space
- **Decoder Extraction**: Standalone VAE decoder from trained checkpoint
- **Visualization**: Tool to compare random vs evolved environments
- **Documentation**: Complete README with usage examples

## Testing
Run `bash vae/test_evolution.sh` to verify the pipeline.

## Future Work
- [ ] Replace placeholder fitness with RL-based evaluation (agent failure, regret)
- [ ] Integrate with jaxued `LevelSampler` for curriculum learning
- [ ] Multi-objective optimization (MAP-Elites, NSGA-II)

## Dependencies
Requires `evosax==0.2.0` (install instructions in EVOLUTION_README.md)
```

## Files to Commit

**New files (should be committed):**
- ✅ `vae/vae_decoder.py` — Core decoder utilities
- ✅ `vae/evolve_envs.py` — Main evolution script
- ✅ `vae/evolve_config.yml` — Configuration
- ✅ `vae/visualize_envs.py` — Visualization tool
- ✅ `vae/test_evolution.sh` — Test script
- ✅ `vae/EVOLUTION_README.md` — Documentation
- ✅ `.gitignore` (if modified)

**Generated outputs (should NOT be committed):**
- ❌ `vae/evolved/` — Generated environments, fitness history, etc.
- ❌ `vae/*.png` — Visualization outputs
- ❌ `vae/__pycache__/` — Python cache

**Unrelated (should be removed):**
- ❌ `nlp/` — Not part of ES implementation
- ❌ `vae/miniconda_old.sh` — Installer script

## Pre-Commit Checklist

Before committing, verify:

1. **All tests pass**:
   ```bash
   bash vae/test_evolution.sh
   ```

2. **No accidental large files**:
   ```bash
   git status
   du -sh vae/evolved/  # Should NOT be staged
   ```

3. **Clean diff**:
   ```bash
   git diff --cached --stat
   ```

4. **Python runs without errors**:
   ```bash
   python vae/evolve_envs.py --num_generations 5 --pop_size 8
   python vae/visualize_envs.py --random --num_envs 4
   ```

5. **Documentation is complete**:
   - README has usage examples
   - All new files have docstrings
   - Configuration is documented

## Post-Merge Tasks

After the PR is merged to main:

1. **Update README.md** (repo root) to mention evolution capability
2. **Add to examples/** if the project has an examples directory
3. **Create a release tag** if this is a major feature
4. **Update requirements.txt** or `pyproject.toml` to list `evosax` as optional dependency

## Contact

For questions about this implementation, contact:
- Giacomo Maralla (gmaralla@...)
- Branch: `feat/es-implementation-Giacomo`
