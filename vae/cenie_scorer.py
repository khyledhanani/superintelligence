"""
CENIE: Coverage-based Evaluation of Novelty In Environments.

Implements the novelty scoring component from:
"Improving Environment Novelty Quantification for Effective Unsupervised Environment Design"
(Teoh, Li, Varakantham, 2025) - https://arxiv.org/abs/2502.05726

Maintains a FIFO buffer of (state, action) pairs from training trajectories.
Periodically fits a diagonal-covariance GMM on the buffer.
GMM parameters are extracted as JAX arrays for pure-JAX NLL computation
inside the JIT'd training loop (avoids host callbacks that segfault on TPU).

The final score combines novelty and regret via rank-based weighting:
  P_replay = alpha * P_novelty + (1 - alpha) * P_regret
"""
import numpy as np
import jax.numpy as jnp


class CENIEScorer:
    def __init__(self, buffer_size=50000, n_components=10, alpha=0.5, temperature=0.3):
        self.buffer = []
        self.buffer_size = buffer_size
        self.n_components = n_components
        self.alpha = alpha
        self.temperature = temperature
        self.gmm = None
        self._step_count = 0

    def add_to_buffer(self, obs_actions):
        """Add trajectory (state, action) pairs to the FIFO coverage buffer.

        Args:
            obs_actions: (T, N, D) array — T timesteps, N envs, D = obs_dim + act_dim.
        """
        data = np.asarray(obs_actions).reshape(-1, obs_actions.shape[-1])
        self.buffer.extend(data.tolist())
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]
        self._step_count += 1

    def refit_gmm(self):
        """Refit the GMM on the current coverage buffer."""
        min_samples = self.n_components * 20
        if len(self.buffer) < min_samples:
            print(f"[CENIE] Skipping GMM refit: {len(self.buffer)} < {min_samples} samples")
            return
        from sklearn.mixture import GaussianMixture
        data = np.array(self.buffer)
        n_components = min(self.n_components, len(data) // 20)
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='diag',
            max_iter=100,
            random_state=42,
        )
        self.gmm.fit(data)
        print(f"[CENIE] GMM refit: {n_components} components, {len(data)} samples, "
              f"buffer={len(self.buffer)}/{self.buffer_size}")

    def get_jax_params(self, max_components):
        """Extract fitted GMM parameters as JAX arrays for pure-JAX NLL computation.

        Args:
            max_components: Fixed array size (padded if GMM has fewer components).

        Returns:
            Dict with 'means' (K, D), 'log_vars' (K, D), 'log_weights' (K,), 'fitted' (bool).
        """
        if self.gmm is None:
            return None
        K_actual = self.gmm.n_components
        D = self.gmm.means_.shape[1]
        K = max_components

        means = np.zeros((K, D), dtype=np.float32)
        log_vars = np.zeros((K, D), dtype=np.float32)
        log_weights = np.full(K, -1e10, dtype=np.float32)  # near-zero weight for padding

        means[:K_actual] = self.gmm.means_
        log_vars[:K_actual] = np.log(self.gmm.covariances_ + 1e-30)
        log_weights[:K_actual] = np.log(self.gmm.weights_ + 1e-30)

        return {
            'means': jnp.array(means),
            'log_vars': jnp.array(log_vars),
            'log_weights': jnp.array(log_weights),
            'fitted': jnp.bool_(True),
        }

    def compute_combined_score(self, obs_actions, regret_scores):
        """Compute CENIE combined score: alpha * novelty_rank + (1-alpha) * regret_rank.

        Called from JIT via jax.pure_callback.

        Args:
            obs_actions: (T, N, D) trajectory data.
            regret_scores: (N,) regret scores from MaxMC/PVL.

        Returns:
            (N,) combined scores (higher = higher priority).
        """
        obs_actions = np.asarray(obs_actions)
        regret_scores = np.asarray(regret_scores)
        N = regret_scores.shape[0]

        if self.gmm is None:
            # No GMM yet — fall back to pure regret
            return regret_scores.astype(np.float32)

        T = obs_actions.shape[0]
        # Compute per-env novelty (average NLL under GMM)
        novelty = np.zeros(N, dtype=np.float64)
        for i in range(N):
            log_probs = self.gmm.score_samples(obs_actions[:, i, :])  # (T,)
            novelty[i] = -log_probs.mean()

        # Rank-based combination (CENIE Eq. 4-5)
        from scipy.stats import rankdata
        regret_ranks = rankdata(-regret_scores, method='ordinal')
        novelty_ranks = rankdata(-novelty, method='ordinal')

        p_r = (1.0 / regret_ranks) ** (1.0 / self.temperature)
        p_r /= p_r.sum()
        p_n = (1.0 / novelty_ranks) ** (1.0 / self.temperature)
        p_n /= p_n.sum()

        combined = self.alpha * p_n + (1 - self.alpha) * p_r
        return combined.astype(np.float32)
