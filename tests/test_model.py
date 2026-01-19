

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from omegaconf import OmegaConf

import mlops_group35.cluster_train as c_train
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

    train_cfg = c_train.build_train_config(cfg)

    assert isinstance(train_cfg, TrainConfig)
    assert train_cfg.csv_path == "data.csv"
    assert not hasattr(train_cfg, "use_wandb")


@patch("mlops_group35.cluster_train.generate_and_save_metrics")
@patch("mlops_group35.cluster_train.load_csv_for_clustering")
def test_train_calls_metrics_function(mock_loader, mock_metrics, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    ids = pd.Series([1, 2, 3])
    feats = pd.DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]})
    mock_loader.return_value = (ids, feats)

    cfg = TrainConfig(
        csv_path="data.csv",
        id_col="id",
        feature_cols=["a", "b"],
        n_clusters=2,
        seed=42,
        metrics_path="reports/metrics.json",
        profile=False,
        profile_path="reports/profile.prof",
    )

    c_train.train_config(cfg, ids, feats, run=None)
    mock_metrics.assert_called_once()



@patch("mlops_group35.cluster_train.generate_and_save_metrics")
@patch("mlops_group35.cluster_train.load_csv_for_clustering")
def test_train_returns_clusters(mock_loader, mock_metrics, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # Fake dataset
    ids = pd.Series([1, 2, 3, 4])
    feats = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 3, 4, 5]})
    mock_loader.return_value = (ids, feats)

    cfg = TrainConfig(
        csv_path="dummy.csv",
        id_col="id",
        feature_cols=["a", "b"],
        n_clusters=2,
        seed=0,
        metrics_path="reports/metrics.json",
        profile=False,
        profile_path="reports/profile.prof",
    )

    clusters = c_train.train_config(cfg, ids, feats, run=None)

    # Assertions
    assert len(clusters) == 4
    assert set(clusters) <= {0, 1}
