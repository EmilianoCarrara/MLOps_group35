from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DriftResult:
    drift_detected: bool
    n_features_drifted: int
    threshold: float


def _safe_pct(x: np.ndarray) -> np.ndarray:
    # Avoid log(0) by clipping probabilities.
    eps = 1e-6
    return np.clip(x, eps, 1.0)


def psi(
    baseline: pd.Series,
    current: pd.Series,
    n_bins: int = 10,
) -> float:
    """Population Stability Index (PSI) between baseline and current distributions."""
    b = baseline.dropna().to_numpy(dtype=float)
    c = current.dropna().to_numpy(dtype=float)

    if len(b) == 0 or len(c) == 0:
        return float("nan")

    # Use baseline quantiles to define bins (robust and common practice).
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(b, quantiles))
    if len(edges) < 3:  # not enough unique values to bin
        return 0.0

    b_counts, _ = np.histogram(b, bins=edges)
    c_counts, _ = np.histogram(c, bins=edges)

    b_pct = _safe_pct(b_counts / max(1, b_counts.sum()))
    c_pct = _safe_pct(c_counts / max(1, c_counts.sum()))

    return float(np.sum((c_pct - b_pct) * np.log(c_pct / b_pct)))


def psi_table(
    df_baseline: pd.DataFrame,
    df_current: pd.DataFrame,
    feature_cols: Iterable[str],
    n_bins: int = 10,
) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        if col not in df_baseline.columns or col not in df_current.columns:
            continue
        rows.append(
            {
                "feature": col,
                "psi": psi(df_baseline[col], df_current[col], n_bins=n_bins),
                "baseline_mean": float(pd.to_numeric(df_baseline[col], errors="coerce").mean()),
                "current_mean": float(pd.to_numeric(df_current[col], errors="coerce").mean()),
            }
        )
    out = pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="last")
    return out.reset_index(drop=True)


def detect_drift(
    psi_df: pd.DataFrame,
    threshold: float = 0.2,
    min_features: int = 2,
) -> DriftResult:
    n = int((psi_df["psi"] > threshold).sum())
    return DriftResult(drift_detected=(n >= min_features), n_features_drifted=n, threshold=threshold)
