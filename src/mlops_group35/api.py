from fastapi import FastAPI
from http import HTTPStatus
from hydra import initialize, compose
import pandas as pd
from pydantic import BaseModel

from mlops_group35 import train
from mlops_group35.train import build_train_config
from mlops_group35.data import load_preprocessed_data


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


@app.post("/predict")
def predict(data: PredictionInput):
    required_feats = data.model_fields.keys()

    with initialize(config_path="../../configs", version_base="1.3"):
        cfg = compose(config_name="cluster")

    train_cfg = build_train_config(cfg)

    csv_path = "data/processed/combined.csv"
    df = load_preprocessed_data(csv_path, required_feats)

    # Convert input to DataFrame
    new_row = pd.DataFrame([data.model_dump()])
    new_row = new_row[required_feats]

    # Append to dataset
    df_with_new = pd.concat([df, new_row], ignore_index=True)

    df_out, kmeans, X_scaled = train.train(df_with_new, train_cfg.n_clusters, train_cfg.seed)

    # Get user's cluster
    user_cluster = df_out.iloc[-1]["cluster"]
    # TODO ATM it returns the cluster number, but it should return some interpretations
    return {"Group": int(user_cluster)}
