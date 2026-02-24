import jax
import jax.numpy as jnp
import optax
import numpy as np
import os
import yaml
import pickle
import matplotlib
# Use Agg backend for headless plotting (server/colab)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from flax import linen as nn
from flax.training import train_state
from tqdm.auto import tqdm
from utils import evaluate_cluttr_metrics
import shutil

# Loading the config file.
with open('vae_train_config.yml', 'r') as f:
    CONFIG = yaml.safe_load(f)

# ==========================================
# MODEL ARCHITECTURE
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
        mean = jnp.tanh(mean) * 4.0 
        
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(z_rng, mean.shape)
        z = mean + eps * std
        
        # Decoder
        z_seq = jnp.tile(z[:, jnp.newaxis, :], (1, CONFIG["seq_len"], 1))
        d_out = nn.Bidirectional(nn.RNN(nn.LSTMCell(400)), nn.RNN(nn.LSTMCell(400)))(z_seq)
        d_out = nn.Bidirectional(nn.RNN(nn.LSTMCell(400)), nn.RNN(nn.LSTMCell(400)))(d_out)
        logits = nn.Dense(CONFIG["vocab_size"])(d_out)
        
        return logits, mean, logvar
    
    #@nn.compact
    #def decode(self, z):
        # z: (latent_dim,) -> logits: (seq_len, vocab_size)
        #z_seq  = jnp.tile(z[jnp.newaxis, :], (CONFIG["seq_len"], 1))[jnp.newaxis, :, :]
        #d_out  = nn.Bidirectional(nn.RNN(nn.LSTMCell(400)), nn.RNN(nn.LSTMCell(400)))(z_seq)
        #d_out  = nn.Bidirectional(nn.RNN(nn.LSTMCell(400)), nn.RNN(nn.LSTMCell(400)))(d_out)
        #logits = nn.Dense(CONFIG["vocab_size"])(d_out)
        
        #return logits[0]  # (seq_len, vocab_size)
# ==========================================
# TRAINING UTILITIES
# ==========================================
def get_kl_weight(step):
    return jnp.minimum(0.5, step / CONFIG["anneal_steps"])

@jax.jit
def eval_inference(state, batch, z_rng):
    """Runs inference to get hard token predictions (Argmax)."""
    logits, _, _ = CluttrVAE().apply({'params': state.params}, batch, z_rng, train=False)
    return jnp.argmax(logits, axis=1)

@jax.jit
def eval_loss_step(state, batch, z_rng):
    """Calculates Loss on Validation Set (No Gradients)."""
    logits, mean, logvar = CluttrVAE().apply({'params': state.params}, batch, z_rng, train=False)
    labels_onehot = jax.nn.one_hot(batch, num_classes=CONFIG["vocab_size"])
    recon_loss = optax.softmax_cross_entropy(logits, labels_onehot).mean()
    kl_loss = -0.5 * jnp.mean(jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar), axis=-1))
    return recon_loss, kl_loss

@jax.jit
def train_step(state, batch, z_rng, kl_weight):
    z_key, dropout_key = jax.random.split(z_rng)
    def loss_fn(params):
        logits, mean, logvar = CluttrVAE().apply({'params': params}, batch, z_key, train =True, rngs={'dropout': dropout_key})
        labels_onehot = jax.nn.one_hot(batch, num_classes=CONFIG["vocab_size"])
        per_token_loss = optax.softmax_cross_entropy(logits, labels_onehot)
        weights = jnp.ones_like(per_token_loss)

        recon_added_weight_factor = CONFIG['recon_added_weight_factor']
        weights = weights.at[: -2:].set(recon_added_weight_factor)
        #recon_loss = optax.softmax_cross_entropy(logits, labels_onehot).mean()
        recon_loss = jnp.mean(per_token_loss*weights)
        kl_loss = -0.5 * jnp.mean(jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar), axis=-1))
        weighted_reconstruction_loss = CONFIG["recon_weight"] * recon_loss
        total_loss = weighted_reconstruction_loss + (kl_weight * kl_loss)
        return total_loss, (recon_loss, kl_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (recon, kl)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, recon, kl

def save_checkpoint(state, step):
    checkpoint_dir = os.path.join(CONFIG["working_path"], CONFIG["vae_folder"], CONFIG["checkpoint_dir"])
    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
    path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pkl")
    with open(path, "wb") as f:
        pickle.dump({'params': state.params, 'step': step}, f)

def load_checkpoint(path):
    with open(path, "rb") as f: data = pickle.load(f)
    return data['params'], data['step']

# ==========================================
# MAIN TRAINING ENGINE
# ==========================================
def run_training():

    # 0. Save config file used to run training
    config_save_dir = os.path.join(CONFIG["working_path"], CONFIG["vae_folder"], "configs")
    if not os.path.exists(config_save_dir):
        os.makedirs(config_save_dir)
    config_save_path = os.path.join(config_save_dir, f"config_{CONFIG['seq_len']}.yaml")
    with open(config_save_path, 'w') as f:
        yaml.safe_dump(CONFIG, f)

    print(f'config saved to {config_save_path}')

    # 1. Setup Data
    data_path = os.path.join(CONFIG["working_path"], CONFIG["vae_folder"], "datasets", CONFIG["train_data_path"])
    validation_data_path = os.path.join(CONFIG["working_path"], CONFIG["vae_folder"], "datasets", CONFIG["validation_data_path"])
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset {data_path} not found.")
        return
    if not os.path.exists(validation_data_path):
        print(f"Error: Dataset {validation_data_path} not found.")
        return
    
    # Load Data
    dataset = jnp.array(np.load(data_path))
    print(f"Loaded Train dataset: {dataset.shape}")

    val_dataset_full = jnp.array(np.load(validation_data_path))
    # We take a subset for fast loss checks
    val_dataset_small = val_dataset_full[:1000] 
    print(f"Loaded Val dataset (subset for loss): {val_dataset_small.shape}")

    # 2. Initialize State
    key = jax.random.PRNGKey(0)
    model = CluttrVAE()
    key, init_key, z_key = jax.random.split(key, 3)
    
    start_step = 0
    if CONFIG["resume_path"]:
        params, start_step = load_checkpoint(CONFIG["resume_path"])
        print(f"Resuming from step {start_step}")
    else:
        params = model.init(init_key, jnp.zeros((1, CONFIG["seq_len"]), dtype=jnp.int32), z_key, train=False)['params']

    state = train_state.TrainState.create(
        apply_fn=model.apply, 
        params=params, 
        tx=optax.adam(CONFIG["learning_rate"])
    )

    # 3. Training Loop Data Structures
    # Fast history (Losses)
    history = {
        'train_recon': [], 'train_kl': [], 
        'val_recon': [], 'val_kl': [], 
        'steps': [],       # Steps for train data
        'val_steps': []    # Steps for val loss data
    }

    # Slow history (Generative Metrics)
    metric_history = {
        'steps': [],
        'train': {'validity': [], 'agent_acc': [], 'goal_acc': [], 'wall_err': []},
        'val':   {'validity': [], 'agent_acc': [], 'goal_acc': [], 'wall_err': []}
    }

    EVAL_METRIC_FREQ = 10000 
    
    plot_path = os.path.join(CONFIG["working_path"], CONFIG["vae_folder"], CONFIG["KL_recon_plot_name"])
    
    # Create 4 subplots: KL, Recon, Percentages, Wall Error
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 20), sharex=True)
    
    progress_bar = tqdm(range(start_step, CONFIG["num_steps"]), desc="Training")

    for step in progress_bar:
        # --- TRAINING STEP ---
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(subkey, (CONFIG["batch_size"],), 0, len(dataset))
        batch = dataset[idx]
        
        key, z_rng = jax.random.split(key)
        kl_w = get_kl_weight(step)
        
        state, loss, recon, kl = train_step(state, batch, z_rng, kl_w)
        
        history['train_recon'].append(float(recon))
        history['train_kl'].append(float(kl))
        history['steps'].append(step)

        # --- METRICS EVALUATION (Slow, Less Frequent) ---
        if step % EVAL_METRIC_FREQ == 0 and step > 0:
            # 1. Run inference on a subset of TRAIN and VAL
            # Note: We use dataset[:1000] to be comparable in size to val_dataset_small
            train_preds = eval_inference(state, dataset[:1000], z_rng)
            val_preds = eval_inference(state, val_dataset_small, z_rng)
            
            # 2. Compute metrics
            # Ensure pad_token matches your data (0 or 169)
            train_m = evaluate_cluttr_metrics(dataset[:1000], train_preds, pad_token=0)
            val_m = evaluate_cluttr_metrics(val_dataset_small, val_preds, pad_token=0)
            
            # 3. Store Results
            metric_history['steps'].append(step)
            
            # Train
            metric_history['train']['validity'].append(train_m['validity_score'])
            metric_history['train']['agent_acc'].append(train_m['agent_accuracy'])
            metric_history['train']['goal_acc'].append(train_m['goal_accuracy'])
            metric_history['train']['wall_err'].append(train_m['avg_wall_error'])
            
            # Val
            metric_history['val']['validity'].append(val_m['validity_score'])
            metric_history['val']['agent_acc'].append(val_m['agent_accuracy'])
            metric_history['val']['goal_acc'].append(val_m['goal_accuracy'])
            metric_history['val']['wall_err'].append(val_m['avg_wall_error'])

        # --- PLOTTING & VALIDATION LOSS (Frequent) ---
        if step % CONFIG["plot_freq"] == 0 and step > 0:
            # 1. Validation Loss (Fast)
            val_recon, val_kl = eval_loss_step(state, val_dataset_small, z_rng)
            history['val_recon'].append(float(val_recon))
            history['val_kl'].append(float(val_kl))
            history['val_steps'].append(step)
            
            progress_bar.set_postfix({'Recon': f"{recon:.2f}", 'ValRecon': f"{val_recon:.2f}", 'KL': f"{kl:.2f}"})
            
            # 2. Plotting
            ax1.clear(); ax2.clear(); ax3.clear(); ax4.clear()

            # --- Plot 1: KL Divergence (Train Only) ---
            ax1.plot(history['steps'], history['train_kl'], label='Train KL', color='red', alpha=0.7)
            ax1.set_ylabel("Nats")
            ax1.set_title(f"KL Divergence (Beta: {kl_w:.2f})")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # --- Plot 2: Reconstruction Loss (Train & Val) ---
            ax2.plot(history['steps'], history['train_recon'], label='Train Recon', color='blue', alpha=0.4, linewidth=1)
            ax2.plot(history['val_steps'], history['val_recon'], label='Val Recon', color='darkblue', linewidth=2, linestyle='--')
            ax2.set_ylabel("Cross Entropy")
            ax2.set_title("Reconstruction Loss")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # --- Plot 3: Percentages (Validity, Agent Acc, Goal Acc) ---
            # --- Plot 3: Percentages (Validity, Agent Acc, Goal Acc) ---
            if len(metric_history['steps']) > 0:
                steps = metric_history['steps']
                # Train (Solid lines with markers)
                ax3.plot(steps, metric_history['train']['validity'], label='Train Validity', color='green', marker='o')
                ax3.plot(steps, metric_history['train']['agent_acc'], label='Train Agent Acc', color='orange', marker='x')
                ax3.plot(steps, metric_history['train']['goal_acc'], label='Train Goal Acc', color='purple', marker='^')
                
                # Val (Dashed lines WITH MARKERS added)
                # FIX: Added marker='o', marker='x', etc. so single points are visible
                ax3.plot(steps, metric_history['val']['validity'], label='Val Validity', color='green', linestyle='--', marker='o', alpha=0.7)
                ax3.plot(steps, metric_history['val']['agent_acc'], label='Val Agent Acc', color='orange', linestyle='--', marker='x', alpha=0.7)
                ax3.plot(steps, metric_history['val']['goal_acc'], label='Val Goal Acc', color='purple', linestyle='--', marker='^', alpha=0.7)
                
                ax3.set_ylabel("Percentage (%)")
                ax3.set_ylim([-5, 105])
                ax3.set_title("Accuracy & Validity Metrics")
                ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
                ax3.grid(True, alpha=0.3)
                
                # --- Plot 4: Wall Count Error (Scalar) ---
                # FIX: Added marker='.' to both so they show up immediately
                ax4.plot(steps, metric_history['train']['wall_err'], label='Train Wall Err', color='brown', marker='.')
                ax4.plot(steps, metric_history['val']['wall_err'], label='Val Wall Err', color='brown', linestyle='--', marker='.')
                ax4.set_ylabel("Avg Count Error")
                ax4.set_title("Wall Count Error (Lower is Better)")
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, "Metrics not computed yet...", ha='center', transform=ax3.transAxes)
                ax4.text(0.5, 0.5, "Metrics not computed yet...", ha='center', transform=ax4.transAxes)

            ax4.set_xlabel("Training Steps")
            
            fig.tight_layout()
            fig.savefig(plot_path)

        if step % CONFIG["save_freq"] == 0 and step > 0:
            save_checkpoint(state, step)

    save_checkpoint(state, "final")
    plt.close(fig)
    print(f"Training Complete. Plot saved to {plot_path}")

if __name__ == "__main__":
    run_training()