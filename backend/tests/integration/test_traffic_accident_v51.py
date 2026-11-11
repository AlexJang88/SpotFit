from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_traffic_v51_status_is_ready():
    response = client.get("/api/recommender/model-status")
    assert response.status_code == 200
    traffic = response.json()["traffic_accident"]
    assert traffic["ready"] is True
    assert [item["horizon_years"] for item in traffic["horizons"]] == [2, 3]


def test_traffic_v51_returns_2027_and_2028_forecasts():
    for horizon, year in ((2, 2027), (3, 2028)):
        response = client.get(
            "/api/recommender/traffic-accident/district-forecasts",
            params={"horizon": horizon},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["temporary_model"] is True
        assert payload["prediction_year"] == year
        assert payload["district_count"] == 25
        assert sorted(row["predicted_safety_rank"] for row in payload["forecasts"]) == list(range(1, 26))
        assert {row["predicted_safety_grade"] for row in payload["forecasts"]} == {1, 2, 3, 4, 5}
        assert payload["grade_definition"] == "1등급 가장 안전, 5등급 가장 위험"


def test_traffic_v51_filters_one_district():
    response = client.get(
        "/api/recommender/traffic-accident/district-forecasts",
        params={"horizon": 3, "district": "강남구"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["district_count"] == 1
    assert payload["forecasts"][0]["district"] == "강남구"
