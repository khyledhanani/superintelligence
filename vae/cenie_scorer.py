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
        self.buffer_size = buffer_size
        self.n_components = n_components
        self.alpha = alpha
        self.temperature = temperature
        self.gmm = None
        self._step_count = 0
        # Pre-allocated ring buffer (lazily initialized on first add)
        self._buffer = None
        self._buf_ptr = 0
        self._buf_count = 0

    def add_to_buffer(self, obs_actions):
        """Add trajectory (state, action) pairs to the FIFO coverage buffer.

        Uses a pre-allocated numpy ring buffer to avoid Python list overhead.

        Args:
            obs_actions: (T, N, D) array — T timesteps, N envs, D = obs_dim + act_dim.
        """
        data = np.asarray(obs_actions).reshape(-1, obs_actions.shape[-1])
        n_new = data.shape[0]
        D = data.shape[1]

        # Lazy init on first call
        if self._buffer is None:
            self._buffer = np.zeros((self.buffer_size, D), dtype=np.float32)
            self._buf_ptr = 0
            self._buf_count = 0

        # Write into ring buffer
        if n_new >= self.buffer_size:
            # More data than buffer can hold — just keep the last buffer_size entries
            self._buffer[:] = data[-self.buffer_size:]
            self._buf_ptr = 0
            self._buf_count = self.buffer_size
        else:
            end = self._buf_ptr + n_new
            if end <= self.buffer_size:
                self._buffer[self._buf_ptr:end] = data
            else:
                # Wrap around
                first = self.buffer_size - self._buf_ptr
                self._buffer[self._buf_ptr:] = data[:first]
                self._buffer[:n_new - first] = data[first:]
            self._buf_ptr = end % self.buffer_size
            self._buf_count = min(self._buf_count + n_new, self.buffer_size)

        self._step_count += 1

    def refit_gmm(self):
        """Refit the GMM on the current coverage buffer."""
        min_samples = self.n_components * 20
        if self._buf_count < min_samples:
            print(f"[CENIE] Skipping GMM refit: {self._buf_count} < {min_samples} samples")
            return
        from sklearn.mixture import GaussianMixture
        data = self._buffer[:self._buf_count]
        n_components = min(self.n_components, self._buf_count // 20)
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='diag',
            max_iter=100,
            random_state=42,
        )
        self.gmm.fit(data)
        print(f"[CENIE] GMM refit: {n_components} components, {len(data)} samples, "
              f"buffer={self._buf_count}/{self.buffer_size}")

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
        # Clip log-variances for numerical stability in JAX NLL computation
        log_vars[:K_actual] = np.clip(np.log(self.gmm.covariances_ + 1e-30), -20.0, 20.0)
        log_weights[:K_actual] = np.log(self.gmm.weights_ + 1e-30)

        return {
            'means': jnp.array(means),
            'log_vars': jnp.array(log_vars),
            'log_weights': jnp.array(log_weights),
            'fitted': jnp.bool_(True),
        }
