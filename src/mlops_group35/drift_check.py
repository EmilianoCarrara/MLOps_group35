from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlops_group35.config import TrainConfig
from mlops_group35.data import load_preprocessed_data
from mlops_group35.drift import detect_drift, psi_table


def simulate_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Create a 'drifted' copy of the data for demonstration."""
    out = df.copy()

    # Example drift: older population + scaled ADHD index
    if "age" in out.columns:
        out["age"] = pd.to_numeric(out["age"], errors="coerce") + 10

    if "adhd_index" in out.columns:
        out["adhd_index"] = pd.to_numeric(out["adhd_index"], errors="coerce") * 1.25

    # Example drift: more missingness on inattentive
    if "inattentive" in out.columns:
        mask = np.random.RandomState(42).rand(len(out)) < 0.10
        out.loc[mask, "inattentive"] = np.nan

    return out


def main() -> None:
    cfg = TrainConfig()
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    df_base = load_preprocessed_data(cfg.csv_path, cfg.feature_cols)
    df_base = df_base.replace(-999, np.nan)
    # Drop id col for drift calculations if present
    feature_cols = [c for c in cfg.feature_cols if c in df_base.columns]

    df_new = simulate_drift(df_base)
    df_new = df_new.replace(-999, np.nan)

    psi_df = psi_table(df_base, df_new, feature_cols=feature_cols, n_bins=10)
    psi_path = reports_dir / "drift_psi_table.csv"
    psi_df.to_csv(psi_path, index=False)

    res = detect_drift(psi_df, threshold=0.2, min_features=2)
    report = {
        "drift_detected": res.drift_detected,
        "n_features_drifted": res.n_features_drifted,
        "threshold": res.threshold,
        "top_features": psi_df.head(5).to_dict(orient="records"),
        "notes": "Drift was simulated for demonstration (age shift, ADHD scale, missingness).",
    }
    report_path = reports_dir / "drift_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote: {psi_path}")
    print(f"Wrote: {report_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
