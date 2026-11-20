from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_recommender_v1_returns_all_districts_and_models():
    response = client.get("/api/recommender/district-forecasts")
    assert response.status_code == 200
    payload = response.json()
    assert payload["version"] == "v1"
    assert payload["district_count"] == 25
    assert set(payload["models"]) == {
        "apart_sale",
        "apart_rent",
        "child_environment",
        "crime_safety",
        "traffic_accident",
    }
    assert len(payload["forecasts"]) == 25
    row = payload["forecasts"][0]
    assert {
        "apart_sale",
        "apart_rent",
        "child_environment_1y",
        "child_environment_3y",
        "crime_score",
        "traffic_score",
        "traffic_accidents",
    }.issubset(row)


def test_recommender_v1_filters_a_district():
    response = client.get("/api/recommender/district-forecasts", params={"district": "강남구"})
    assert response.status_code == 200
    assert response.json()["forecasts"][0]["district"] == "강남구"
