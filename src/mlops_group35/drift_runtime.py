import json
from pathlib import Path
import pandas as pd

from mlops_group35.drift import psi_table  # <-- usa la tua PSI table


def _read_last_predictions(jsonl_path: Path, n: int) -> pd.DataFrame:
    if not jsonl_path.exists():
        return pd.DataFrame()

    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") == "prediction" and "input" in rec:
                rows.append(rec["input"])

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.tail(n)


def run_drift_report(
    baseline_csv: Path,
    requests_jsonl: Path,
    features: list[str],
    n: int = 200,
    psi_threshold: float = 0.2,
) -> dict:
    baseline = pd.read_csv(baseline_csv)[features]
    current = _read_last_predictions(requests_jsonl, n=n)

    if current.empty:
        return {
            "drift_detected": False,
            "reason": "no_requests_logged",
            "n_samples_current": 0,
            "psi_threshold": psi_threshold,
        }

    current = current[features]

    psi_df = psi_table(baseline, current)  # <-- deve tornare columns: feature, psi (o simile)
    # Adatta i nomi colonne se necessari:
    psi_col = "psi" if "psi" in psi_df.columns else psi_df.columns[-1]
    feat_col = "feature" if "feature" in psi_df.columns else psi_df.columns[0]

    psi_df_sorted = psi_df.sort_values(psi_col, ascending=False)
    drifted = psi_df_sorted[psi_df_sorted[psi_col] >= psi_threshold]

    return {
        "drift_detected": len(drifted) > 0,
        "n_samples_current": int(len(current)),
        "psi_threshold": psi_threshold,
        "n_features_drifted": int(len(drifted)),
        "top_features": psi_df_sorted.head(5)[[feat_col, psi_col]].to_dict(orient="records"),
    }
