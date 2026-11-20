from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_returns_all_25_seoul_district_forecasts() -> None:
    response = client.get("/api/safety/crime/district-forecasts")
    assert response.status_code == 200
    payload = response.json()
    assert payload["district_count"] == 25
    assert len(payload["forecasts"]) == 25
    assert {row["predicted_grade"] for row in payload["forecasts"]} == set("ABCDE")
    assert sorted(row["predicted_rank"] for row in payload["forecasts"]) == list(range(1, 26))


def test_filters_one_district_without_recalculating_relative_grade() -> None:
    all_response = client.get("/api/safety/crime/district-forecasts").json()
    expected = next(row for row in all_response["forecasts"] if row["district"] == "강남구")
    response = client.get(
        "/api/safety/crime/district-forecasts", params={"district": "강남구"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["district_count"] == 1
    assert payload["forecasts"] == [expected]


def test_unknown_district_returns_404() -> None:
    response = client.get(
        "/api/safety/crime/district-forecasts", params={"district": "없는구"}
    )
    assert response.status_code == 404

