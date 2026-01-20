from fastapi import FastAPI
from http import HTTPStatus
from hydra import initialize, compose
import pandas as pd
from pydantic import BaseModel

from mlops_group35 import train
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

print("Api is starting...")


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


# @app.post("/predict")
# def predict(data: PredictionInput, n_clusters: int):
#     required_feats = data.model_fields.keys()
#
#     with initialize(config_path="../../configs", version_base="1.3"):
#         cfg = compose(config_name="cluster")
#
#     train_cfg = train.build_train_config(cfg)
#
#     csv_path = "data/processed/combined.csv"
#     df = load_preprocessed_data(csv_path, required_feats)
#
#     # Convert input to DataFrame
#     new_row = pd.DataFrame([data.model_dump()])
#     new_row = new_row[required_feats]
#
#     # Append to dataset
#     df_with_new = pd.concat([df, new_row], ignore_index=True)
#
#     if n_clusters == 1 or n_clusters == 0:
#         n_clusters = train_cfg.n_clusters
#     df_out, kmeans, X_scaled = train.train(df_with_new, n_clusters, train_cfg.seed)
#
#     # Get user's cluster
#     user_cluster = df_out.iloc[-1]["cluster"]
#     # TODO ATM it returns the cluster number, but it should return some interpretations
#     return {"Group": int(user_cluster)}


@app.post("/predict")
def predict(data: PredictionInput, n_clusters: int):
    required_feats = data.model_fields.keys()

    with initialize(config_path="../../configs", version_base="1.3"):
        cfg = compose(config_name="cluster")

    train_cfg = train.build_train_config(cfg)

    csv_path = "data/processed/combined.csv"
    df = load_preprocessed_data(csv_path, required_feats)

    # Convert input to DataFrame
    new_row = pd.DataFrame([data.model_dump()])
    new_row = new_row[required_feats]

    # Append to dataset
    df_with_new = pd.concat([df, new_row], ignore_index=True)

    if n_clusters == 1 or n_clusters == 0:
        n_clusters = train_cfg.n_clusters

    df_out, kmeans, X_scaled = train.train(df_with_new, n_clusters, train_cfg.seed)

    # Get user's cluster
    user_cluster = int(df_out.iloc[-1]["cluster"])

    # Compute cluster profiles (means)
    cluster_profiles = df_out.groupby("cluster")[list(required_feats)].mean().round(2).to_dict(orient="index")

    user_values = new_row.iloc[0].to_dict()
    cluster_mean = cluster_profiles[user_cluster]

    # Compare user to cluster mean
    differences = {feat: round(user_values[feat] - cluster_mean[feat], 2) for feat in required_feats}

    # Simple interpretation logic
    interpretation = []

    if user_values["adhd_index"] > cluster_mean["adhd_index"]:
        interpretation.append("Higher ADHD index than typical for this group.")
    else:
        interpretation.append("Lower ADHD index than typical for this group.")

    if user_values["verbal_iq"] > cluster_mean["verbal_iq"]:
        interpretation.append("Stronger verbal reasoning compared to peers.")
    else:
        interpretation.append("Weaker verbal reasoning compared to peers.")

    if user_values["inattentive"] > cluster_mean["inattentive"]:
        interpretation.append("More inattentive traits than average.")

    if user_values["hyper_impulsive"] > cluster_mean["hyper_impulsive"]:
        interpretation.append("More hyperactive/impulsive traits than average.")

    return {
        "group": user_cluster,
        "cluster_profile": cluster_mean,
        "user_values": user_values,
        "differences": differences,
        "interpretation": interpretation,
    }
