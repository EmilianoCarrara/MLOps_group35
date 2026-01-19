from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    """Configuration object for the clustering pipeline."""

    seed: int = 42
    metrics_path: str = "reports/metrics.json"
    profile: bool = False
    profile_path: str = "reports/profile.pstats"

    # CSV data
    csv_path: str = "data/processed/combined.csv"

    # Clustering config
    id_col: str = "scandir_id"
    feature_cols: tuple[str, ...] = (
        "age",
        "gender",
        "handedness",
        "verbal_iq",
        "performance_iq",
        "full4_iq",
        "adhd_index",
        "inattentive",
        "hyper/impulsive",
    )
    n_clusters: int = 5
