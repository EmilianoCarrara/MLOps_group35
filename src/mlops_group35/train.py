"""Training pipeline

This module handles:
- data loading (currently synthetic data)
- model training and evaluation
- saving trained model and metrics
"""

import cProfile
import json
import logging
import pstats
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mlops_group35.model import Model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    """Configuration object for the training pipeline."""

    seed: int = 42
    n_samples: int = 5_000
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 20
    hidden_dim: int = 32
    val_split: float = 0.2
    noise_std: float = 0.1
    out_dir: str = "models"
    metrics_path: str = "reports/metrics.json"
    model_path: str = "models/model.pt"
    profile: bool = False
    profile_path: str = "reports/profile.pstats"

    # CSV data (optional)
    use_csv: bool = False
    csv_path: str = "data/processed/combined.csv"

    # Clustering config (used when use_csv=True)
    id_col: str = "ScanDir ID"
    feature_cols: tuple[str, ...] = (
        "Age",
        "Gender",
        "Handedness",
        "Verbal IQ",
        "Performance IQ",
        "Full4 IQ",
        "ADHD Index",
        "Inattentive",
        "Hyper/Impulsive",
    )
    n_clusters: int = 5


def set_seed(seed: int) -> None:
    logger.info("Setting seed=%d", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_synthetic_regression(n: int, noise_std: float) -> tuple[torch.Tensor, torch.Tensor]:
    logger.info("Generating synthetic regression data: n=%d, noise_std=%.6f", n, noise_std)
    x = 2 * torch.rand(n, 1) - 1
    noise = noise_std * torch.randn(n, 1)
    y = 3.0 * x - 0.5 + noise
    logger.debug("Generated tensors: x_shape=%s y_shape=%s", tuple(x.shape), tuple(y.shape))
    return x, y


def load_csv_for_clustering(
    csv_path: str,
    id_col: str,
    feature_cols: tuple[str, ...],
) -> tuple[pd.Series, pd.DataFrame]:
    df = pd.read_csv(csv_path)

    needed = [id_col, *feature_cols]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df = df[needed].replace(-999, pd.NA).dropna()
    return df[id_col], df[list(feature_cols)]


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    n_batches = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        total_loss += float(loss.item())
        n_batches += 1
    avg_loss = total_loss / max(n_batches, 1)
    logger.debug("Evaluation finished: avg_mse=%.6f over %d batches", avg_loss, n_batches)
    return avg_loss


def train(cfg: TrainConfig, run: wandb.sdk.wandb_run.Run | None = None) -> dict[str, Any]:
    logger.info("Starting train()")
    logger.info(
        "Config: epochs=%d lr=%s batch_size=%d hidden_dim=%d val_split=%s n_samples=%d",
        cfg.epochs,
        cfg.lr,
        cfg.batch_size,
        cfg.hidden_dim,
        cfg.val_split,
        cfg.n_samples,
    )

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device=%s | torch=%s", device, torch.__version__)

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    Path(Path(cfg.metrics_path).parent).mkdir(parents=True, exist_ok=True)

    # ---- Clustering mode (unsupervised) ----
    if cfg.use_csv:
        logger.info("Clustering from CSV: %s", cfg.csv_path)
        ids, feats = load_csv_for_clustering(cfg.csv_path, cfg.id_col, cfg.feature_cols)
        logger.info("Clustering data shape: n_samples=%d n_features=%d", len(feats), feats.shape[1])

        x_scaled = StandardScaler().fit_transform(feats.to_numpy(dtype=float))
        kmeans = KMeans(n_clusters=cfg.n_clusters, random_state=cfg.seed, n_init="auto")
        clusters = kmeans.fit_predict(x_scaled)

        sil = float("nan")
        if cfg.n_clusters >= 2 and len(feats) > cfg.n_clusters:
            sil = float(silhouette_score(x_scaled, clusters))
        inertia = float(kmeans.inertia_)

        assignments_path = "reports/cluster_assignments.csv"
        pd.DataFrame({cfg.id_col: ids.to_numpy(), "cluster": clusters}).to_csv(assignments_path, index=False)
        logger.info("Saved cluster assignments to %s", assignments_path)

        metrics: dict[str, Any] = {
            "silhouette": sil,
            "inertia": inertia,
            "n_clusters": cfg.n_clusters,
            "n_samples": int(len(feats)),
            "n_features": int(feats.shape[1]),
            "csv_path": cfg.csv_path,
        }
        with open(cfg.metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved metrics to %s", cfg.metrics_path)

        if run is not None:
            run.log({"silhouette": sil, "inertia": inertia, "n_clusters": cfg.n_clusters})
            run.summary["silhouette"] = sil
            run.summary["inertia"] = inertia
            run.summary["n_clusters"] = cfg.n_clusters

        return metrics

    # ---- Original regression mode (unchanged) ----
    x, y = make_synthetic_regression(cfg.n_samples, cfg.noise_std)
    n_val = int(cfg.n_samples * cfg.val_split)
    n_train = cfg.n_samples - n_val

    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train:], y[n_train:]

    logger.info("Split data: n_train=%d n_val=%d", n_train, n_val)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=cfg.batch_size, shuffle=False)

    model = Model(hidden_dim=cfg.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    logger.info("Initialized model and optimizer")

    train_mse = 0.0
    val_mse = 0.0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        train_mse = epoch_loss / max(n_batches, 1)
        val_mse = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:02d}/{cfg.epochs} | train_mse={train_mse:.6f} | val_mse={val_mse:.6f}")
        logger.info("Epoch %02d/%d | train_mse=%.6f | val_mse=%.6f", epoch, cfg.epochs, train_mse, val_mse)

        if run is not None:
            run.log({"epoch": epoch, "train_mse": float(train_mse), "val_mse": float(val_mse)}, step=epoch)

    payload = {
        "state_dict": model.state_dict(),
        "config": asdict(cfg),
        "pytorch_version": torch.__version__,
        "device": str(device),
    }
    torch.save(payload, cfg.model_path)
    logger.info("Saved model to %s", cfg.model_path)

    metrics = {
        "train_mse": float(train_mse),
        "val_mse": float(val_mse),
        "epochs": cfg.epochs,
        "n_samples": cfg.n_samples,
        "seed": cfg.seed,
        "model_path": cfg.model_path,
    }
    with open(cfg.metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", cfg.metrics_path)

    if run is not None:
        run.summary["final_train_mse"] = float(train_mse)
        run.summary["final_val_mse"] = float(val_mse)
        run.log({"final_val_mse": float(val_mse)}, step=cfg.epochs)

    return metrics


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train_baseline")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    print(OmegaConf.to_yaml(cfg))
    logger.info("Hydra config resolved:\n%s", OmegaConf.to_yaml(cfg))

    Path("reports").mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, "reports/config.yaml")
    logger.info("Saved Hydra config to reports/config.yaml")

    use_wandb = bool(cfg.get("use_wandb", False))
    wandb_project = str(cfg.get("wandb_project", "mlops_group35"))
    wandb_mode = str(cfg.get("wandb_mode", "offline"))
    wandb_run_name = cfg.get("wandb_run_name", None)

    run: wandb.sdk.wandb_run.Run | None = None
    if use_wandb:
        run = wandb.init(
            project=wandb_project,
            name=str(wandb_run_name) if wandb_run_name is not None else None,
            mode=wandb_mode,
        )
        logger.info("W&B initialized: project=%s mode=%s", wandb_project, wandb_mode)

        sweep_overrides = dict(run.config)
        allowed = set(TrainConfig.__dataclass_fields__.keys())
        sweep_overrides = {k: v for k, v in sweep_overrides.items() if k in allowed}
        if sweep_overrides:
            logger.info("Applying sweep overrides to Hydra cfg: %s", sweep_overrides)
            cfg = OmegaConf.merge(cfg, OmegaConf.create(sweep_overrides))

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb_keys = {"use_wandb", "wandb_project", "wandb_mode", "wandb_run_name"}
    cfg_dict = {k: v for k, v in cfg_dict.items() if k not in wandb_keys}
    train_cfg = TrainConfig(**cfg_dict)

    try:
        if train_cfg.profile:
            Path(train_cfg.profile_path).parent.mkdir(parents=True, exist_ok=True)
            logger.info("Profiling enabled: profile_path=%s", train_cfg.profile_path)

            profiler = cProfile.Profile()
            profiler.enable()
            train(train_cfg, run=run)
            profiler.disable()

            profiler.dump_stats(train_cfg.profile_path)
            logger.info("Saved profiling stats to %s", train_cfg.profile_path)

            stats = pstats.Stats(train_cfg.profile_path)
            stats.strip_dirs().sort_stats("cumtime")
            stats.print_stats(25)

            txt_path = Path(train_cfg.profile_path).with_suffix(".txt")
            with txt_path.open("w", encoding="utf-8") as f:
                stats.stream = f
                stats.print_stats(50)
            logger.info("Saved profiling report to %s", txt_path)

        else:
            logger.info("Profiling disabled")
            train(train_cfg, run=run)

        if run is not None:
            artifact = wandb.Artifact(name="training-artifacts", type="model")
            if Path("reports/config.yaml").exists():
                artifact.add_file("reports/config.yaml")
            if Path(train_cfg.metrics_path).exists():
                artifact.add_file(train_cfg.metrics_path)
            if Path(train_cfg.model_path).exists():
                artifact.add_file(train_cfg.model_path)
            if Path("reports/cluster_assignments.csv").exists():
                artifact.add_file("reports/cluster_assignments.csv")

            if train_cfg.profile and Path(train_cfg.profile_path).exists():
                artifact.add_file(train_cfg.profile_path)
                txt_path = Path(train_cfg.profile_path).with_suffix(".txt")
                if txt_path.exists():
                    artifact.add_file(str(txt_path))

            run.log_artifact(artifact)
            logger.info("Logged W&B artifact (config/metrics/model + optional profiling)")

    except Exception:
        logger.exception("Training run failed with an exception")
        raise
    finally:
        if run is not None:
            run.finish()
            logger.info("W&B run finished")


if __name__ == "__main__":
    main()
