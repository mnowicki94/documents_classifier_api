from fastapi.testclient import TestClient
from ing_mle_task_api.main import app

client = TestClient(app)

def test_predict_one():
    response = client.post("/predict-one", json={"title": "More apps for Android wearable devices coming soon"})
    assert response.status_code == 200
    assert "predicted_category" in response.json()
    assert "predicted_category_label" in response.json()

def test_predict_multiple():
    response = client.post("/predict-multiple", json={"titles": {
        "11": "US open: Stocks fall after Fed official hints at accelerated tapering",
        "22": "Robin Thicke writes new song dedicated to Paula Patton",
        "34": "EBay rejects Icahn slate of directors"
    }})
    assert response.status_code == 200
    json_response = response.json()
    assert "predictions" in json_response
    for key in ["11", "22", "34"]:
        assert key in json_response["predictions"]
        assert "predicted_category" in json_response["predictions"][key]
        assert "predicted_category_label" in json_response["predictions"][key]