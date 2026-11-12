import json
from datetime import date

from fastapi import FastAPI, HTTPException, Query
from fastapi.testclient import TestClient

from app.core.error_handling import install_error_handling
from app.services import admin_logs


def error_client() -> TestClient:
    test_app = FastAPI()
    install_error_handling(test_app)

    @test_app.get("/unexpected")
    def unexpected():
        raise RuntimeError("secret internal path: /models/private.joblib")

    @test_app.get("/unavailable")
    def unavailable():
        raise HTTPException(503, "secret artifact path: /models/private.joblib")

    @test_app.get("/validated")
    def validated(value: int = Query(gt=0)):
        return {"value": value}

    return TestClient(test_app, raise_server_exceptions=False)


def test_unexpected_error_is_sanitized_and_traceable():
    response = error_client().get("/unexpected", headers={"X-Request-ID": "test-request-500"})
    payload = response.json()
    assert response.status_code == 500
    assert response.headers["X-Request-ID"] == "test-request-500"
    assert payload["error"]["code"] == "INTERNAL_SERVER_ERROR"
    assert payload["error"]["request_id"] == "test-request-500"
    assert "private.joblib" not in response.text


def test_known_server_error_does_not_expose_internal_detail():
    response = error_client().get("/unavailable")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "HTTP_503"
    assert "private.joblib" not in response.text


def test_validation_error_has_common_error_contract():
    response = error_client().get("/validated", params={"value": 0})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert response.headers["X-Request-ID"] == response.json()["error"]["request_id"]


def test_log_reader_filters_date_and_level(tmp_path, monkeypatch):
    records = [
        {"timestamp": "2026-11-12T10:00:00+09:00", "level": "INFO", "message": "started"},
        {"timestamp": "2026-11-12T10:01:00+09:00", "level": "ERROR", "message": "failed"},
        {"timestamp": "2026-11-13T10:01:00+09:00", "level": "CRITICAL", "message": "crashed"},
    ]
    (tmp_path / admin_logs.log_file_name()).write_text(
        "\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(admin_logs, "log_directory", lambda: tmp_path)
    result = admin_logs.read_logs(limit=10, minimum_level="ERROR", log_date=date(2026, 11, 12))
    assert result["count"] == 1
    assert result["logs"][0]["message"] == "failed"
