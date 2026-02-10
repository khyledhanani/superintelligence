from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from checkpointing import load_model_params
from maze_metrics import (
    MazeSpec,
    compute_feature_stats,
    evaluate_sequence,
    feature_vector_from_metrics,
    mahalanobis_distance_batch,
)
from modeling import CluttrVAE, VAEConfig


def parse_float_list(raw: str) -> List[float]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError(f"No valid float values in: {raw}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OOD maze-generation experiments for a trained VAE.")

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--decode_mode", type=str, choices=["sample", "argmax"], default="sample")

    parser.add_argument("--n_samples_prior", type=int, default=2000)
    parser.add_argument("--alphas", type=str, default="1.5,2,3,4")
    parser.add_argument("--n_samples_shift", type=int, default=2000)
    parser.add_argument("--betas", type=str, default="0.5,1.0,1.5,2.0")
    parser.add_argument("--pca_component", type=int, default=0)

    parser.add_argument("--max_train_for_stats", type=int, default=20000)

    parser.add_argument("--seq_len", type=int, default=52)
    parser.add_argument("--vocab_size", type=int, default=170)
    parser.add_argument("--embed_dim", type=int, default=300)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--dropout_rate", type=float, default=0.1)

    parser.add_argument("--inner_dim", type=int, default=13)
    parser.add_argument("--max_obs_tokens", type=int, default=50)

    return parser.parse_args()


def save_rows_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def batched_indices(n: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    start = 0
    while start < n:
        end = min(start + batch_size, n)
        yield start, end
        start = end


def encode_means(
    model: CluttrVAE,
    params,
    sequences: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    outputs = []
    for start, end in batched_indices(len(sequences), batch_size):
        batch = jnp.array(sequences[start:end], dtype=jnp.int32)
        mean, _ = model.apply(
            {"params": params},
            batch,
            train=False,
            method=CluttrVAE.encode_stats,
        )
        outputs.append(np.asarray(mean))
    return np.concatenate(outputs, axis=0)


def decode_latents(
    model: CluttrVAE,
    params,
    latents: np.ndarray,
    batch_size: int,
    mode: str,
    rng_key: jax.Array,
) -> np.ndarray:
    all_tokens = []
    key = rng_key

    for start, end in batched_indices(len(latents), batch_size):
        z_batch = jnp.array(latents[start:end], dtype=jnp.float32)
        logits = model.apply(
            {"params": params},
            z_batch,
            train=False,
            method=CluttrVAE.decode,
        )

        if mode == "argmax":
            tokens = jnp.argmax(logits, axis=-1)
        else:
            key, subkey = jax.random.split(key)
            tokens = jax.random.categorical(subkey, logits, axis=-1)

        all_tokens.append(np.asarray(tokens, dtype=np.int32))

    return np.concatenate(all_tokens, axis=0)


def compute_latent_stats(latents: np.ndarray, eps: float = 1e-6) -> Dict[str, np.ndarray]:
    mean = latents.mean(axis=0)
    cov = np.cov(latents, rowvar=False)
    cov = np.atleast_2d(cov)
    cov += np.eye(cov.shape[0], dtype=np.float64) * eps
    cov_inv = np.linalg.pinv(cov)
    return {"mean": mean, "cov": cov, "cov_inv": cov_inv}


def pca_components(latents: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    centered = latents - latents.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt
    total_var = np.sum(s**2)
    explained = (s**2 / total_var) if total_var > 0 else np.zeros_like(s)
    return components, explained


def evaluate_group(
    *,
    group: str,
    family: str,
    latents: np.ndarray,
    tokens: np.ndarray,
    spec: MazeSpec,
    feature_stats: Dict[str, np.ndarray],
    latent_stats: Dict[str, np.ndarray],
) -> List[Dict[str, object]]:
    metrics = [evaluate_sequence(seq, spec=spec) for seq in tokens]
    feature_vectors = np.stack([feature_vector_from_metrics(m) for m in metrics], axis=0)

    feature_ood = mahalanobis_distance_batch(
        feature_vectors,
        mean=feature_stats["mean"],
        cov_inv=feature_stats["cov_inv"],
    )
    latent_ood = mahalanobis_distance_batch(
        latents,
        mean=latent_stats["mean"],
        cov_inv=latent_stats["cov_inv"],
    )

    rows: List[Dict[str, object]] = []
    for idx, metric in enumerate(metrics):
        row = {
            "group": group,
            "family": family,
            "sample_id": idx,
            "valid": int(metric["valid"] > 0.5),
            "path_len": float(metric["path_len"]),
            "branching": float(metric["branching"]),
            "loops": float(metric["loops"]),
            "wall_density": float(metric["wall_density"]),
            "reachable_ratio": float(metric["reachable_ratio"]),
            "feature_ood": float(feature_ood[idx]),
            "latent_ood": float(latent_ood[idx]),
        }
        rows.append(row)

    return rows


def summarize_groups(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group"])].append(row)

    summary: List[Dict[str, object]] = []
    for group, items in grouped.items():
        valid = np.array([int(item["valid"]) for item in items], dtype=np.float64)
        feature_ood = np.array([float(item["feature_ood"]) for item in items], dtype=np.float64)
        latent_ood = np.array([float(item["latent_ood"]) for item in items], dtype=np.float64)
        path_len = np.array([float(item["path_len"]) for item in items], dtype=np.float64)

        valid_mask = valid > 0.5
        path_len_valid = path_len[valid_mask]

        summary.append(
            {
                "group": group,
                "family": str(items[0]["family"]),
                "n": int(len(items)),
                "validity_rate": float(valid.mean()),
                "feature_ood_mean": float(feature_ood.mean()),
                "feature_ood_median": float(np.median(feature_ood)),
                "latent_ood_mean": float(latent_ood.mean()),
                "latent_ood_median": float(np.median(latent_ood)),
                "path_len_mean_valid": float(path_len_valid.mean()) if len(path_len_valid) > 0 else float("nan"),
            }
        )

    return summary


def mark_pareto(summary: List[Dict[str, object]], ood_key: str, out_key: str) -> None:
    points = np.array([[float(x[ood_key]), float(x["validity_rate"])] for x in summary], dtype=np.float64)
    n = len(summary)
    pareto = np.ones(n, dtype=np.bool_)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dominates = (
                points[j, 0] >= points[i, 0]
                and points[j, 1] >= points[i, 1]
                and (points[j, 0] > points[i, 0] or points[j, 1] > points[i, 1])
            )
            if dominates:
                pareto[i] = False
                break

    for i in range(n):
        summary[i][out_key] = bool(pareto[i])


def add_tradeoff_scores(summary: List[Dict[str, object]], ood_key: str, out_key: str) -> None:
    validity = np.array([float(x["validity_rate"]) for x in summary], dtype=np.float64)
    ood = np.array([float(x[ood_key]) for x in summary], dtype=np.float64)

    ood_min, ood_max = float(np.min(ood)), float(np.max(ood))
    if ood_max > ood_min:
        ood_norm = (ood - ood_min) / (ood_max - ood_min)
    else:
        ood_norm = np.zeros_like(ood)

    score = 0.5 * validity + 0.5 * ood_norm
    for idx in range(len(summary)):
        summary[idx][out_key] = float(score[idx])


def plot_tradeoff(summary: List[Dict[str, object]], out_path: Path) -> None:
    families = sorted({str(item["family"]) for item in summary})
    cmap = plt.get_cmap("tab10")
    colors = {family: cmap(i % 10) for i, family in enumerate(families)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=120)

    for item in summary:
        family = str(item["family"])
        color = colors[family]

        axes[0].scatter(item["feature_ood_mean"], item["validity_rate"], color=color, s=60)
        axes[1].scatter(item["latent_ood_mean"], item["validity_rate"], color=color, s=60)

        axes[0].annotate(str(item["group"]), (item["feature_ood_mean"], item["validity_rate"]), fontsize=7)
        axes[1].annotate(str(item["group"]), (item["latent_ood_mean"], item["validity_rate"]), fontsize=7)

    axes[0].set_title("Validity vs Feature OOD")
    axes[0].set_xlabel("Feature OOD (Mahalanobis)")
    axes[0].set_ylabel("Validity Rate")

    axes[1].set_title("Validity vs Latent OOD")
    axes[1].set_xlabel("Latent OOD (Mahalanobis)")
    axes[1].set_ylabel("Validity Rate")

    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[f], markersize=8, label=f) for f in families]
    fig.legend(handles=handles, loc="lower center", ncol=max(1, len(families)))

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    alphas = parse_float_list(args.alphas)
    betas = parse_float_list(args.betas)

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(str(Path(args.data_path).expanduser().resolve()))
    if data.ndim != 2:
        raise ValueError(f"Expected data shape [N, seq_len], got {data.shape}")
    if data.shape[1] != args.seq_len:
        raise ValueError(f"Expected seq_len={args.seq_len}, dataset has {data.shape[1]}")

    cfg = VAEConfig(
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        latent_dim=args.latent_dim,
        dropout_rate=args.dropout_rate,
    )
    model = CluttrVAE(cfg)

    params, ckpt_meta = load_model_params(args.checkpoint_path)

    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)

    n_train = data.shape[0]
    stats_n = min(args.max_train_for_stats, n_train)
    stats_indices = rng.choice(n_train, size=stats_n, replace=False)
    data_stats = data[stats_indices]

    # Training-set feature stats
    train_metrics = [evaluate_sequence(seq, MazeSpec(args.inner_dim, args.max_obs_tokens)) for seq in data_stats]
    feature_stats = compute_feature_stats(train_metrics)

    # Training-set latent stats
    train_means = encode_means(model, params, data_stats, batch_size=args.batch_size)
    latent_stats = compute_latent_stats(train_means)

    components, explained = pca_components(train_means)
    if args.pca_component < 0 or args.pca_component >= components.shape[0]:
        raise ValueError(
            f"pca_component {args.pca_component} out of range [0, {components.shape[0] - 1}]"
        )

    pca_dir = components[args.pca_component]
    pca_dir = pca_dir / (np.linalg.norm(pca_dir) + 1e-12)

    spec = MazeSpec(inner_dim=args.inner_dim, max_obs_tokens=args.max_obs_tokens)

    all_rows: List[Dict[str, object]] = []

    # Baseline from train data for comparison.
    baseline_latents = train_means[: min(len(train_means), args.n_samples_prior)]
    baseline_tokens = data_stats[: len(baseline_latents)]
    all_rows.extend(
        evaluate_group(
            group="train_data",
            family="train",
            latents=baseline_latents,
            tokens=baseline_tokens,
            spec=spec,
            feature_stats=feature_stats,
            latent_stats=latent_stats,
        )
    )

    # z ~ N(0, I)
    key, subkey = jax.random.split(key)
    prior_z = np.asarray(jax.random.normal(subkey, (args.n_samples_prior, cfg.latent_dim)))
    key, decode_key = jax.random.split(key)
    prior_tokens = decode_latents(model, params, prior_z, args.batch_size, args.decode_mode, decode_key)
    all_rows.extend(
        evaluate_group(
            group="prior_alpha_1.0",
            family="prior",
            latents=prior_z,
            tokens=prior_tokens,
            spec=spec,
            feature_stats=feature_stats,
            latent_stats=latent_stats,
        )
    )

    # z ~ N(0, alpha^2 I)
    for alpha in alphas:
        key, subkey = jax.random.split(key)
        scaled_z = np.asarray(jax.random.normal(subkey, (args.n_samples_prior, cfg.latent_dim))) * float(alpha)
        key, decode_key = jax.random.split(key)
        scaled_tokens = decode_latents(model, params, scaled_z, args.batch_size, args.decode_mode, decode_key)
        all_rows.extend(
            evaluate_group(
                group=f"prior_scaled_alpha_{alpha}",
                family="scaled_prior",
                latents=scaled_z,
                tokens=scaled_tokens,
                spec=spec,
                feature_stats=feature_stats,
                latent_stats=latent_stats,
            )
        )

    # z = mu(x) + beta * v, with v from PCA direction
    for beta in betas:
        base_idx = rng.integers(0, len(train_means), size=args.n_samples_shift)
        base_mu = train_means[base_idx]

        z_pca = base_mu + float(beta) * pca_dir[None, :]
        key, decode_key = jax.random.split(key)
        tokens_pca = decode_latents(model, params, z_pca, args.batch_size, args.decode_mode, decode_key)
        all_rows.extend(
            evaluate_group(
                group=f"mu_shift_pca_beta_{beta}",
                family="shift_pca",
                latents=z_pca,
                tokens=tokens_pca,
                spec=spec,
                feature_stats=feature_stats,
                latent_stats=latent_stats,
            )
        )

        # z = mu(x) + beta * v, with random unit direction per sample
        random_dir = rng.normal(size=(args.n_samples_shift, cfg.latent_dim))
        random_dir /= np.linalg.norm(random_dir, axis=1, keepdims=True) + 1e-12
        z_rand = base_mu + float(beta) * random_dir
        key, decode_key = jax.random.split(key)
        tokens_rand = decode_latents(model, params, z_rand, args.batch_size, args.decode_mode, decode_key)
        all_rows.extend(
            evaluate_group(
                group=f"mu_shift_random_beta_{beta}",
                family="shift_random",
                latents=z_rand,
                tokens=tokens_rand,
                spec=spec,
                feature_stats=feature_stats,
                latent_stats=latent_stats,
            )
        )

    summary = summarize_groups(all_rows)
    mark_pareto(summary, ood_key="feature_ood_mean", out_key="pareto_feature")
    mark_pareto(summary, ood_key="latent_ood_mean", out_key="pareto_latent")
    add_tradeoff_scores(summary, ood_key="feature_ood_mean", out_key="tradeoff_feature_score")
    add_tradeoff_scores(summary, ood_key="latent_ood_mean", out_key="tradeoff_latent_score")

    save_rows_csv(out_dir / "sample_scores.csv", all_rows)
    save_rows_csv(out_dir / "group_summary.csv", summary)
    plot_tradeoff(summary, out_dir / "validity_vs_ood.png")

    np.savez(
        out_dir / "feature_stats.npz",
        mean=feature_stats["mean"],
        cov=feature_stats["cov"],
        cov_inv=feature_stats["cov_inv"],
    )
    np.savez(
        out_dir / "latent_stats.npz",
        mean=latent_stats["mean"],
        cov=latent_stats["cov"],
        cov_inv=latent_stats["cov_inv"],
        pca_components=components,
        pca_explained=explained,
    )

    metadata = {
        "checkpoint_path": str(Path(args.checkpoint_path).expanduser().resolve()),
        "checkpoint_backend": ckpt_meta.get("backend"),
        "checkpoint_step": ckpt_meta.get("step"),
        "data_path": str(Path(args.data_path).expanduser().resolve()),
        "n_train_samples": int(data.shape[0]),
        "n_stats_reference": int(stats_n),
        "decode_mode": args.decode_mode,
        "alphas": alphas,
        "betas": betas,
        "pca_component": int(args.pca_component),
        "pca_component_explained": float(explained[args.pca_component]),
    }
    with (out_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    # Print a compact recommendation for quick inspection.
    feature_best = max(summary, key=lambda x: float(x["tradeoff_feature_score"]))
    latent_best = max(summary, key=lambda x: float(x["tradeoff_latent_score"]))

    print("Saved:")
    print(f"  - {out_dir / 'sample_scores.csv'}")
    print(f"  - {out_dir / 'group_summary.csv'}")
    print(f"  - {out_dir / 'validity_vs_ood.png'}")
    print("Best tradeoff groups:")
    print(
        "  - feature: "
        f"{feature_best['group']} (validity={feature_best['validity_rate']:.3f}, "
        f"feature_ood={feature_best['feature_ood_mean']:.3f})"
    )
    print(
        "  - latent: "
        f"{latent_best['group']} (validity={latent_best['validity_rate']:.3f}, "
        f"latent_ood={latent_best['latent_ood_mean']:.3f})"
    )


if __name__ == "__main__":
    main()
