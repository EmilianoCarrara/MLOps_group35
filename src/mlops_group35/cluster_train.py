"""Clustering pipeline (Unsupervised Learning)

This module handles:
- loading CSV data
- K-Means clustering
- evaluation with silhouette score
- saving cluster assignments and metrics
"""

import cProfile
import json
import logging
import pstats
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.impute import SimpleImputer
import hydra
import pandas as pd
import wandb
from omegaconf import DictConfig, OmegaConf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from mlops_group35.config import TrainConfig
from mlops_group35.data import load_preprocessed_data

CONFIG_YAML_FILE = "reports/cluster_config.yaml"
logger = logging.getLogger(__name__)


def setup_logging_and_dirs(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    logger.info("Hydra config resolved:\n%s", OmegaConf.to_yaml(cfg))

    Path("reports").mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, CONFIG_YAML_FILE)


def init_wandb(cfg: DictConfig) -> wandb.sdk.wandb_run.Run | None:
    use_wandb = bool(cfg.get("use_wandb", False))
    if not use_wandb:
        return None

    return wandb.init(
        project=str(cfg.get("wandb_project", "mlops_group35")),
        name=str(cfg.get("wandb_run_name")) if cfg.get("wandb_run_name") else None,
        mode=str(cfg.get("wandb_mode", "offline")),
    )


def build_train_config(cfg: DictConfig) -> TrainConfig:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    wandb_keys = {"use_wandb", "wandb_project", "wandb_mode", "wandb_run_name"}
    cfg_dict = {k: v for k, v in cfg_dict.items() if k not in wandb_keys}

    return TrainConfig(**cfg_dict)


def generate_and_save_metrics(cfg, feats, kmeans, x_scaled, clusters, ids, run):
    sil = float("nan")
    if cfg.n_clusters >= 2 and len(feats) > cfg.n_clusters:
        sil = float(silhouette_score(x_scaled, clusters))

    inertia = float(kmeans.inertia_)

    metrics: dict[str, Any] = {
        "silhouette": sil,
        "inertia": inertia,
        "n_clusters": cfg.n_clusters,
        "n_samples": int(len(feats)),
        "n_features": int(feats.shape[1]),
        "csv_path": cfg.csv_path,
    }

    # ---- Save outputs ----
    assignments_path = "reports/cluster_assignments.csv"
    pd.DataFrame({cfg.id_col: ids.to_numpy(), "cluster": clusters}).to_csv(assignments_path, index=False)

    logger.info("Saved cluster assignments to %s", assignments_path)

    with open(cfg.metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Saved metrics to %s", cfg.metrics_path)

    # ---- Log to W&B ----
    if run is not None:
        run.log(metrics)
        run.summary.update(metrics)

    return metrics


def train_with_config(cfg, df, run: None = None) -> dict[str, Any]:
    logger.info("Starting clustering pipeline")
    logger.info("CSV path: %s | n_clusters=%d", cfg.csv_path, cfg.n_clusters)

    Path("reports").mkdir(parents=True, exist_ok=True)
    Path(Path(cfg.metrics_path).parent).mkdir(parents=True, exist_ok=True)

    df_out, kmeans, X_scaled = train(df, cfg.n_clusters, cfg.seed)
    clusters = df_out["cluster"]
    ids = df_out["scandir_id"]

    generate_and_save_metrics(cfg, df, kmeans, X_scaled, clusters, ids, run)
    return clusters


def train(df, n_clusters, seed):
    X_scaled = _process_data(df)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=seed,
        n_init="auto",
    )
    clusters = kmeans.fit_predict(X_scaled)

    df_out = df.copy()
    df_out["cluster"] = clusters
    return df_out, kmeans, X_scaled


def _process_data(df):
    # TODO apply one-hot-encode for gender
    df2 = df.drop(
        columns=["scandir_id", "site", "full2_iq", "qc_rest_3", "qc_rest_4", "qc_anatomical_2", "secondary_dx"],
        errors="ignore",
    )

    X = df2.select_dtypes(include="number")
    if len(X) != len(df):
        print("Error: Data type is not numeric")

    X = X.replace(-999, np.nan)

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns, index=X.index)

    return X_scaled


def run_training_with_optional_profiling(cfg: TrainConfig, run: wandb.sdk.wandb_run.Run | None) -> None:
    # ---- Load data ----
    df = load_preprocessed_data(cfg.csv_path, cfg.feature_cols)
    logger.info(
        "Clustering data shape: n_samples=%d n_features=%d",
        len(df),
        df.shape[1],
    )

    if not cfg.profile:
        train_with_config(cfg, df, run=run)
        return

    profiler = cProfile.Profile()
    profiler.enable()

    train_with_config(cfg, df, run=run)

    profiler.disable()
    profiler.dump_stats(cfg.profile_path)

    stats = pstats.Stats(cfg.profile_path)
    stats.sort_stats("cumtime").print_stats(25)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="cluster")
def main(cfg: DictConfig) -> None:
    setup_logging_and_dirs(cfg)

    run = init_wandb(cfg)
    train_cfg = build_train_config(cfg)

    try:
        run_training_with_optional_profiling(train_cfg, run)
    except Exception:
        logger.exception("Clustering run failed")
        raise
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
