
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

import hydra
import pandas as pd
import wandb
from omegaconf import DictConfig, OmegaConf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from mlops_group35.config import TrainConfig
from mlops_group35.data import load_csv_for_clustering

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
    pd.DataFrame(
        {cfg.id_col: ids.to_numpy(), "cluster": clusters}
    ).to_csv(assignments_path, index=False)

    logger.info("Saved cluster assignments to %s", assignments_path)

    with open(cfg.metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Saved metrics to %s", cfg.metrics_path)

    # ---- Log to W&B ----
    if run is not None:
        run.log(metrics)
        run.summary.update(metrics)

    return metrics



def train(cfg: TrainConfig, run: wandb.sdk.wandb_run.Run | None = None) -> dict[str, Any]:
    logger.info("Starting clustering pipeline")
    logger.info("CSV path: %s | n_clusters=%d", cfg.csv_path, cfg.n_clusters)

    Path("reports").mkdir(parents=True, exist_ok=True)
    Path(Path(cfg.metrics_path).parent).mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    ids, feats = load_csv_for_clustering(cfg.csv_path, cfg.id_col, cfg.feature_cols)
    logger.info(
        "Clustering data shape: n_samples=%d n_features=%d",
        len(feats),
        feats.shape[1],
    )

    # ---- Preprocessing ----
    x_scaled = StandardScaler().fit_transform(feats.to_numpy(dtype=float))

    # ---- Clustering ----
    kmeans = KMeans(
        n_clusters=cfg.n_clusters,
        random_state=cfg.seed,
        n_init="auto",
    )
    clusters = kmeans.fit_predict(x_scaled)

    generate_and_save_metrics(cfg, feats, kmeans, x_scaled, clusters, ids, run)
    return clusters



def run_training_with_optional_profiling(
    train_cfg: TrainConfig,
    run: wandb.sdk.wandb_run.Run | None,
) -> None:
    if not train_cfg.profile:
        train(train_cfg, run=run)
        return

    profiler = cProfile.Profile()
    profiler.enable()

    train(train_cfg, run=run)

    profiler.disable()
    profiler.dump_stats(train_cfg.profile_path)

    stats = pstats.Stats(train_cfg.profile_path)
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
