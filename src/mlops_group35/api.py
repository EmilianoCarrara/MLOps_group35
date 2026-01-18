from fastapi import FastAPI
from http import HTTPStatus
from hydra import initialize, compose

from pydantic import BaseModel

from mlops_group35.cluster_train import build_train_config
from mlops_group35.data import load_csv_for_clustering


class PredictionInput(BaseModel):
    Age: int
    Gender: int
    Handedness: int
    Verbal_IQ: int
    Performance_IQ: int
    Full4_IQ: float
    ADHD_Index: float
    Inattentive: float
    Hyper_Impulsive: float


app = FastAPI()



@app.get("/")
def root():
    """ Health check."""
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
    data_dict = {
        "Age": data.Age,
        "Gender": data.Gender,
        "Handedness": data.Handedness,
        "Verbal IQ": data.Verbal_IQ,
        "Performance IQ": data.Performance_IQ,
        "Full4 IQ": data.Full4_IQ,
        "ADHD Index": data.ADHD_Index,
        "Inattentive": data.Inattentive,
        "Hyper/Impulsive": data.Hyper_Impulsive,
    }



    with initialize(config_path="../../configs", version_base="1.3"):
        cfg = compose(config_name="cluster")

    train_cfg = build_train_config(cfg)

    csv_path = "data/processed/combined.csv"
    id_col = "ScanDir ID"
    ids, feats = load_csv_for_clustering(csv_path, id_col, )


    return {"features_used"}
