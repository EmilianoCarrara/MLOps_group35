
from fastapi.testclient import TestClient

from mlops_group35.api import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    print(response.json())
    assert response.json() == {'message': 'OK', 'status-code': 200}



def test_predict_endpoint_frontend_style():
    payload = {
        "age": 12,
        "gender": 0,
        "handedness": 1,
        "verbal_iq": 100,
        "performance_iq": 100,
        "full4_iq": 100,
        "adhd_index": 50,
        "inattentive": 50,
        "hyper_impulsive": 50,
    }

    response = client.post(
        "/predict",
        params={"n_clusters": 4},
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    # Core response structure
    assert "group" in data
    assert "cluster_profile" in data
    assert "user_values" in data
    assert "differences" in data
    assert "interpretation" in data

    # Basic type checks
    assert isinstance(data["group"], int)
    assert isinstance(data["cluster_profile"], dict)
    assert isinstance(data["user_values"], dict)
    assert isinstance(data["differences"], dict)
    assert isinstance(data["interpretation"], list)

    assert "adhd_index" in data["cluster_profile"]
    assert "verbal_iq" in data["user_values"]
    assert "hyper_impulsive" in data["differences"]
