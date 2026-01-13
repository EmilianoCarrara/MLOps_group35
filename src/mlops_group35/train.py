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
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mlops_group35.model import Model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    """Configuration object for the training pipeline.

    Attributes:
        seed: Random seed for reproducibility.
        n_samples: Number of training samples (synthetic data).
        batch_size: Batch size for training.
        lr: Learning rate.
        epochs: Number of training epochs.
        hidden_dim: Hidden layer size.
        val_split: Fraction of data used for validation.
        noise_std: Noise level for synthetic data.
        out_dir: Directory where models are saved.
        metrics_path: Path to metrics JSON file.
        model_path: Path to trained model file.
        profile: Enable cProfile profiling.
        profile_path: Output path for profiling stats (.pstats).
    """

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


def set_seed(seed: int) -> None:
    logger.info("Setting seed=%d", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_synthetic_regression(n: int, noise_std: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create a simple regression dataset: y = 3x - 0.5 + noise
    x in [-1, 1].
    """
    logger.info("Generating synthetic regression data: n=%d, noise_std=%.6f", n, noise_std)
    x = 2 * torch.rand(n, 1) - 1
    noise = noise_std * torch.randn(n, 1)
    y = 3.0 * x - 0.5 + noise
    logger.debug("Generated tensors: x_shape=%s y_shape=%s", tuple(x.shape), tuple(y.shape))
    return x, y


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
    """Train the model using the provided configuration.

    Args:
        cfg: Training configuration.
        run: Optional W&B run to log metrics/artifacts.

    Returns:
        Dictionary containing final training and validation metrics.
    """
    start_time = time.time()
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

    # Ensure output dirs exist
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    Path(Path(cfg.metrics_path).parent).mkdir(parents=True, exist_ok=True)
    logger.debug("Ensured output directories exist: out_dir=%s metrics_dir=%s", cfg.out_dir, Path(cfg.metrics_path).parent)

    # Data
    x, y = make_synthetic_regression(cfg.n_samples, cfg.noise_std)
    n_val = int(cfg.n_samples * cfg.val_split)
    n_train = cfg.n_samples - n_val

    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train:], y[n_train:]

    logger.info("Split data: n_train=%d n_val=%d", n_train, n_val)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=cfg.batch_size, shuffle=False)

    # Model
    model = Model(hidden_dim=cfg.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    logger.info("Initialized model and optimizer")

    # Train loop
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

        # Keep original behavior (print) + add logging
        print(f"Epoch {epoch:02d}/{cfg.epochs} | train_mse={train_mse:.6f} | val_mse={val_mse:.6f}")
        logger.info(
            "Epoch %02d/%d | train_mse=%.6f | val_mse=%.6f",
            epoch,
            cfg.epochs,
            train_mse,
            val_mse,
        )

        # W&B metrics
        if run is not None:
            run.log(
                {"epoch": epoch, "train_mse": float(train_mse), "val_mse": float(val_mse)},
                step=epoch,
            )

    # Save model
    payload = {
        "state_dict": model.state_dict(),
        "config": asdict(cfg),
        "pytorch_version": torch.__version__,
        "device": str(device),
    }
    torch.save(payload, cfg.model_path)
    logger.info("Saved model to %s", cfg.model_path)

    # Save metrics
    metrics: dict[str, Any] = {
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

    # W&B summary (final numbers)
    if run is not None:
        run.summary["final_train_mse"] = float(train_mse)
        run.summary["final_val_mse"] = float(val_mse)

    elapsed = time.time() - start_time
    logger.info("train() completed in %.2fs", elapsed)

    return metrics


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train_baseline")
def main(cfg: DictConfig) -> None:
    """Train entrypoint using Hydra config."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Keep original behavior (print config) + add logging
    print(OmegaConf.to_yaml(cfg))
    logger.info("Hydra config resolved:\n%s", OmegaConf.to_yaml(cfg))

    Path("reports").mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, "reports/config.yaml")
    logger.info("Saved Hydra config to reports/config.yaml")

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb_keys = {"use_wandb", "wandb_project", "wandb_mode", "wandb_run_name"}
    cfg_dict = {k: v for k, v in cfg_dict.items() if k not in wandb_keys}

    train_cfg = TrainConfig(**cfg_dict)


    # ----- W&B init (minimal, controlled via config) -----
    use_wandb = bool(cfg.get("use_wandb", False))
    wandb_project = str(cfg.get("wandb_project", "mlops_group35"))
    wandb_mode = str(cfg.get("wandb_mode", "offline"))
    wandb_run_name = cfg.get("wandb_run_name", None)

    run: wandb.sdk.wandb_run.Run | None = None
    if use_wandb:
        run = wandb.init(
            project=wandb_project,
            name=str(wandb_run_name) if wandb_run_name is not None else None,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=wandb_mode,
        )
        logger.info("W&B initialized: project=%s mode=%s", wandb_project, wandb_mode)

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

            stats.print_stats(25)  # console

            txt_path = Path(train_cfg.profile_path).with_suffix(".txt")
            with txt_path.open("w", encoding="utf-8") as f:
                stats.stream = f
                stats.print_stats(50)
            logger.info("Saved profiling report to %s", txt_path)

        else:
            logger.info("Profiling disabled")
            train(train_cfg, run=run)

        # ----- W&B artifacts (model + metrics + config + optional profile) -----
        if run is not None:
            artifact = wandb.Artifact(name="training-artifacts", type="model")
            if Path("reports/config.yaml").exists():
                artifact.add_file("reports/config.yaml")
            if Path(train_cfg.metrics_path).exists():
                artifact.add_file(train_cfg.metrics_path)
            if Path(train_cfg.model_path).exists():
                artifact.add_file(train_cfg.model_path)

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
