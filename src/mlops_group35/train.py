"""Training pipeline

This module handles:
- data loading (currently synthetic data)
- model training and evaluation
- saving trained model and metrics
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import hydra
from omegaconf import DictConfig, OmegaConf

from mlops_group35.model import Model


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


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_synthetic_regression(n: int, noise_std: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create a simple regression dataset: y = 3x - 0.5 + noise
    x in [-1, 1].
    """
    x = 2 * torch.rand(n, 1) - 1
    noise = noise_std * torch.randn(n, 1)
    y = 3.0 * x - 0.5 + noise
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
    return total_loss / max(n_batches, 1)


def train(cfg: TrainConfig) -> dict:

    """Train the model using the provided configuration.

    Args:
        cfg: Training configuration.

    Returns:
        Dictionary containing final training and validation metrics.
    """

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure output dirs exist
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    Path(Path(cfg.metrics_path).parent).mkdir(parents=True, exist_ok=True)

    # Data
    x, y = make_synthetic_regression(cfg.n_samples, cfg.noise_std)
    n_val = int(cfg.n_samples * cfg.val_split)
    n_train = cfg.n_samples - n_val

    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train:], y[n_train:]

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=cfg.batch_size, shuffle=False)

    # Model
    model = Model(hidden_dim=cfg.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    # Train loop
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

    # Save model
    payload = {
        "state_dict": model.state_dict(),
        "config": asdict(cfg),
        "pytorch_version": torch.__version__,
        "device": str(device),
    }
    torch.save(payload, cfg.model_path)

    # Save metrics
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

    return metrics


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train_baseline")
def main(cfg: DictConfig) -> None:
    """Train entrypoint using Hydra config."""
    print(OmegaConf.to_yaml(cfg))

    Path("reports").mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, "reports/config.yaml")
    
    # Hydra -> dict -> dataclass
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    train_cfg = TrainConfig(**cfg_dict)  # type: ignore[arg-type]
    train(train_cfg)

if __name__ == "__main__":
    main()