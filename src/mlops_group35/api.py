from fastapi import FastAPI
from http import HTTPStatus
from hydra import compose
import pandas as pd
from pydantic import BaseModel

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from mlops_group35 import train
from mlops_group35.train import build_train_config
from mlops_group35.data import load_preprocessed_data

# --- ADDED: drift endpoint support ---
from hydra import initialize_config_dir
from mlops_group35.drift_runtime import run_drift_report

# --- ADDED: M28 metrics instrumentation ---
from prometheus_fastapi_instrumentator import Instrumentator
from mlops_group35.metrics import update_system_metrics

# --- ADDED: robust paths for Cloud Run / containers ---
APP_DIR = Path(__file__).resolve().parent  # .../src/mlops_group35


def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "configs").is_dir():
            return p
    return start  # fallback


REPO_DIR = _find_repo_root(APP_DIR)

CONFIGS_DIR = REPO_DIR / "configs"
DATA_CSV = REPO_DIR / "data" / "processed" / "combined.csv"
LOGS_DIR = REPO_DIR / "logs"
REQUESTS_JSONL = LOGS_DIR / "requests.jsonl"


class PredictionInput(BaseModel):
    age: int
    gender: int
    handedness: int
    verbal_iq: int
    performance_iq: int
    full4_iq: float
    adhd_index: float
    inattentive: float
    hyper_impulsive: float


app = FastAPI()

# --- ADDED: expose Prometheus metrics endpoint ---
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

logger = logging.getLogger("mlops_group35.api")
logging.basicConfig(level=logging.INFO)


@app.get("/")
def root():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}


# --- ADDED: Drift endpoint (M27) ---
@app.get("/drift")
def drift(n: int = 200, psi_threshold: float = 0.2):
    # --- ADDED: update system metrics on request ---
    update_system_metrics()

    required_feats = list(PredictionInput.model_fields.keys())

    report = run_drift_report(
        baseline_csv=DATA_CSV,
        requests_jsonl=REQUESTS_JSONL,
        features=required_feats,
        n=n,
        psi_threshold=psi_threshold,
    )
    return report


@app.post("/predict")
def predict(data: PredictionInput):
    # --- ADDED: update system metrics on request ---
    update_system_metrics()

    t0 = time.perf_counter()
    request_id = str(uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    required_feats = data.model_fields.keys()

    # --- CHANGED MINIMALLY: robust Hydra config path ---
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base="1.3"):
        cfg = compose(config_name="cluster")

    train_cfg = build_train_config(cfg)

    # --- CHANGED MINIMALLY: robust data path ---
    csv_path = str(DATA_CSV)
    df = load_preprocessed_data(csv_path, required_feats)

    # Convert input to DataFrame
    new_row = pd.DataFrame([data.model_dump()])
    new_row = new_row[required_feats]

    # Append to dataset
    df_with_new = pd.concat([df, new_row], ignore_index=True)

    df_out, kmeans, X_scaled = train.train(df_with_new, train_cfg.n_clusters, train_cfg.seed)

    # Get user's cluster
    user_cluster = df_out.iloc[-1]["cluster"]

    latency_ms = (time.perf_counter() - t0) * 1000.0
    log_record = {
        "event": "prediction",
        "request_id": request_id,
        "timestamp": ts,
        "model_type": "kmeans_clustering",
        "n_clusters": int(train_cfg.n_clusters),
        "input": data.model_dump(),
        "output": {"cluster": int(user_cluster)},
        "latency_ms": latency_ms,
    }
    logger.info(json.dumps(log_record))

    # --- CHANGED MINIMALLY: robust logs path ---
    LOGS_DIR.mkdir(exist_ok=True)
    with open(REQUESTS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_record) + "\n")

    # TODO ATM it returns the cluster number, but it should return some interpretations
    return {"Group": int(user_cluster)}
