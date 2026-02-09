import jax
import jax.numpy as jnp
import optax
import numpy as np
import os
import yaml
import pickle
import matplotlib
matplotlib.use('Agg')  # MUST be the first matplotlib call
import matplotlib.pyplot as plt
from flax import linen as nn
from flax.training import train_state
from tqdm.auto import tqdm

# Loading the config file.
with open('vae_train_config.yml', 'r') as f:
    CONFIG = yaml.safe_load(f)

# ==========================================
# MODEL ARCHITECTURE (Bowman et al.)
# ==========================================
class HighwayStage(nn.Module):
    dim: int = 300
    @nn.compact
    def __call__(self, x):
        g = nn.Dense(self.dim)(x)
        g = nn.relu(g)
        f_g_x = nn.relu(nn.Dense(self.dim)(g))
        q_x = nn.Dense(self.dim)(nn.relu(nn.Dense(self.dim)(x)))
        gate = nn.sigmoid(nn.Dense(self.dim)(x))
        return gate * f_g_x + (1.0 - gate) * q_x

class CluttrVAE(nn.Module):
    @nn.compact
    def __call__(self, x, z_rng, train: bool = True):
        # Encoder
        x = nn.Embed(CONFIG["vocab_size"], CONFIG["embed_dim"])(x)
        x = nn.Dropout(rate = 0.1, deterministic= not train)(x)
        x = HighwayStage(CONFIG["embed_dim"])(x)
        x = HighwayStage(CONFIG["embed_dim"])(x)
        outputs = nn.Bidirectional(
            nn.RNN(nn.LSTMCell(300)), 
            nn.RNN(nn.LSTMCell(300))
            )(x)
        outputs = nn.Dropout(rate=0.1, deterministic=not train)(outputs)

        fwd_out = outputs[:, :, :300]
        bwd_out = outputs[:, :, 300:]
        h = jnp.concatenate([fwd_out[:, -1, :], bwd_out[:, 0, :]], axis = -1)
        
        # Bottleneck
        mean = nn.Dense(CONFIG['latent_dim'], name="mean_layer")(h)
        logvar = nn.Dense(CONFIG["latent_dim"], name = "logvar_layer")(h)
        #mean, logvar = jnp.split(stats, 2, axis=-1)
        mean = jnp.tanh(mean) * 4.0 # Standard Objective Scaling
        
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(z_rng, mean.shape)
        z = mean + eps * std
        
        # Decoder
        z_seq = jnp.tile(z[:, jnp.newaxis, :], (1, CONFIG["seq_len"], 1))
        d_out = nn.Bidirectional(nn.RNN(nn.LSTMCell(400)), nn.RNN(nn.LSTMCell(400)))(z_seq)
        d_out = nn.Bidirectional(nn.RNN(nn.LSTMCell(400)), nn.RNN(nn.LSTMCell(400)))(d_out)
        logits = nn.Dense(CONFIG["vocab_size"])(d_out)
        
        return logits, mean, logvar

# ==========================================
# TRAINING UTILITIES
# ==========================================
def get_kl_weight(step):
    return jnp.minimum(1.0, step / CONFIG["anneal_steps"])

@jax.jit
def train_step(state, batch, z_rng, kl_weight):

    z_key, dropout_key = jax.random.split(z_rng)

    def loss_fn(params):
        logits, mean, logvar = CluttrVAE().apply({'params': params}, batch, z_key, train =True, rngs={'dropout': dropout_key})
        labels_onehot = jax.nn.one_hot(batch, num_classes=CONFIG["vocab_size"])
        recon_loss = optax.softmax_cross_entropy(logits, labels_onehot).mean()
        kl_loss = -0.5 * jnp.mean(jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar), axis=-1))
        weighted_reconstruction_loss = CONFIG["recon_weight"] * recon_loss
        total_loss = weighted_reconstruction_loss + (kl_weight * kl_loss)
        return total_loss, (recon_loss, kl_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (recon, kl)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, recon, kl

def save_checkpoint(state, step):
    checkpoint_dir = os.path.join(CONFIG["working_path"],
                                 CONFIG["vae_folder"],
                                 CONFIG["checkpoint_dir"])
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pkl")
    with open(path, "wb") as f:
        pickle.dump({'params': state.params, 'step': step}, f)

def load_checkpoint(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data['params'], data['step']

# ==========================================
# MAIN TRAINING ENGINE
# ==========================================
def run_training():
    # 1. Setup Data
    data_path = os.path.join(CONFIG["working_path"],
                            CONFIG["vae_folder"],
                            CONFIG["data_path"]
                        )
    if not os.path.exists(data_path):
        print(f"Error: Dataset {data_path} not found.")
        return
    dataset = jnp.array(np.load(data_path))
    print(f"Loaded dataset: {dataset.shape}")

    # 2. Initialize State
    key = jax.random.PRNGKey(0)
    model = CluttrVAE()
    key, init_key, z_key = jax.random.split(key, 3)
    
    # Check for Resuming
    start_step = 0
    if CONFIG["resume_path"]:
        params, start_step = load_checkpoint(CONFIG["resume_path"])
        print(f"Resuming from step {start_step}")
    else:
        params = model.init(init_key, jnp.zeros((1, 52), dtype=jnp.int32), z_key, train=False)['params']

    state = train_state.TrainState.create(
        apply_fn=model.apply, 
        params=params, 
        tx=optax.adam(CONFIG["learning_rate"])
    )

    # 3. Training Loop
    history = {'recon': [], 'kl': []}
    
    # Define the plot path
    plot_path = os.path.join(CONFIG["working_path"], CONFIG["vae_folder"], "training_metrics.png")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    progress_bar = tqdm(range(start_step, CONFIG["num_steps"]), desc="ACCEL VAE Training")

    for step in progress_bar:
        # ... [Keep your Batch Sampling and train_step logic] ...
        
        # 1. Update PRNG Key and split for sampling
        key, subkey = jax.random.split(key)
        
        # 2. Sample random indices for the batch
        # randint(key, shape, minval, maxval)
        idx = jax.random.randint(subkey, (CONFIG["batch_size"],), 0, len(dataset))
        
        # 3. Define the batch using the indices
        batch = dataset[idx]  # <--- This line defines 'batch'
        
        # 4. Split key again for the VAE's internal sampling (z_rng)
        key, z_rng = jax.random.split(key)
        
        # 5. Get current KL weight
        kl_w = get_kl_weight(step)
        
        # 6. Execute the training step
        state, loss, recon, kl = train_step(state, batch, z_rng, kl_w)
        
        history['recon'].append(float(recon))
        history['kl'].append(float(kl))

        if step % CONFIG["plot_freq"] == 0:
            progress_bar.set_postfix({'Recon': f"{recon:.3f}", 'KL': f"{kl:.3f}", 'Beta': f"{kl_w:.2f}"})
            
            # --- FIXED PLOTTING FOR HEADLESS SERVER ---
            ax.clear()
            ax.plot(history['recon'], label='Reconstruction', color='blue')
            ax.plot(history['kl'], label='KL Div', color='red', alpha=0.6)
            ax.set_yscale('log')
            ax.set_title(f"Step {step} | KL Weight: {kl_w:.2f}")
            ax.legend()
            
            # Save the figure to disk instead of trying to display it
            fig.savefig(plot_path)
            # ------------------------------------------

        if step % CONFIG["save_freq"] == 0 and step > 0:
            save_checkpoint(state, step)

    save_checkpoint(state, "final")
    plt.close(fig) # Cleanup
    print(f"Training Complete. Final plot saved to {plot_path}")
if __name__ == "__main__":
    run_training()