from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200


def test_predict_category_get():
    response = client.get("/predict_get?headline=some+text")
    assert response.status_code == 200
    assert "category" in response.json()


def test_predict_category_post():
    response = client.post("/predict", json={"headline": "some text"})
    assert response.status_code == 200
    assert "category" in response.json()


def test_predict_medical_headline():
    response = client.post("/predict", json={"headline": "hospital are about health"})
    assert response.status_code == 200
    assert response.json() == {"category": "m"}


def test_predict_tech_headline():
    response = client.post(
        "/predict", json={"headline": "computers are about technology"}
    )
    assert response.status_code == 200
    assert response.json() == {"category": "t"}


def test_predict_business_headline():
    response = client.post("/predict", json={"headline": "stocks are getting higher"})
    assert response.status_code == 200
    assert response.json() == {"category": "b"}


def test_predict_entertainment_headline():
    response = client.post("/predict", json={"headline": "actors are getting married"})
    assert response.status_code == 200
    assert response.json() == {"category": "e"}
