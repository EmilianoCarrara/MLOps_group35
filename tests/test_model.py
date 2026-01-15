

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from omegaconf import OmegaConf

from mlops_group35.cluster_train import (
    build_train_config,
    init_wandb,
    setup_logging_and_dirs,
    train,
)
from mlops_group35.config import TrainConfig


def test_build_train_config_filters_wandb_keys():
    cfg = OmegaConf.create({
        "csv_path": "data.csv",
        "id_col": "id",
        "feature_cols": ["a", "b"],
        "n_clusters": 3,
        "seed": 42,
        "metrics_path": "reports/metrics.json",
        "use_wandb": True,
        "wandb_project": "test",
    })

    train_cfg = build_train_config(cfg)

    assert isinstance(train_cfg, TrainConfig)
    assert train_cfg.csv_path == "data.csv"
    assert not hasattr(train_cfg, "use_wandb")
