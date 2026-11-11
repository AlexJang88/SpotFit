import math

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import neighborhood_recommendations as service


@pytest.fixture
def client(monkeypatch):
    names = [f"구{i:02}" for i in range(25)]
    def crime(horizon):
        return {"data_year": 2025, "prediction_year": 2025 + horizon,
                "forecasts": [{"district": name, "predicted_score": 100 - (i // 5) * 20,
                               "predicted_exposure_y": i / 10} for i, name in enumerate(names)]}
    monkeypatch.setattr(service, "forecast_payload", crime)
    monkeypatch.setattr(service, "_traffic_predictions", lambda: (
        pd.DataFrame({"district": names, "traffic_score": list(range(100, 75, -1))}),
        {"prediction_year": 2025}))
    neighborhoods = [
        {"id": f"{name}|중앙1동", "district": name, "neighborhood": "중앙1동",
         "apart_sale_amount": 40000 + i * 1000, "apart_sale_per_pyeong": 2000,
         "apart_rent_amount": 20000 + i * 500, "apart_rent_per_pyeong": 1000}
        for i, name in enumerate(names)
    ]
    monkeypatch.setattr(service, "property_catalog", lambda: {
        "neighborhoods": neighborhoods,
        "property_models": {"sale": {"data_year": 2025, "is_future_forecast": False},
                            "rent": {"data_year": 2024, "is_future_forecast": False}}})
    return TestClient(app)


def test_budget_is_total_amount_and_top_five(client):
    result = client.get('/api/neighborhoods/recommendations', params={"budget_manwon": 45000}).json()
    assert result['candidate_count'] == 6
    assert len(result['recommendations']) == 5
    assert [r['recommendation_rank'] for r in result['recommendations']] == list(range(1, 6))
    assert all(r['estimated_amount_manwon'] <= 45000 for r in result['recommendations'])
    assert result['recommendations'][0]['neighborhood'] == '중앙1동'


def test_small_budget_does_not_return_unaffordable_rows(client):
    result = client.get('/api/neighborhoods/recommendations', params={"budget_manwon": 100}).json()
    assert result['recommendations'] == []
    assert result['candidate_count'] == 0


def test_rent_and_area_conversion(client):
    # 400/121 square meters is one pyeong, so 10 pyeong is 10,000만원 rent.
    result = client.get('/api/neighborhoods/recommendations', params={
        "budget_manwon": 10001, "transaction": "rent", "area_m2": 4000 / 121}).json()
    assert len(result['recommendations']) == 5
    assert math.isclose(result['recommendations'][0]['estimated_amount_manwon'], 10000)
    sale = client.get('/api/neighborhoods/recommendations', params={
        "budget_manwon": 10001, "transaction": "sale", "area_m2": 4000 / 121}).json()
    assert sale['recommendations'] == []


def test_fewer_than_five_and_missing_prices(client, monkeypatch):
    original = service.property_catalog()
    original['neighborhoods'][0]['apart_sale_amount'] = None
    result = client.get('/api/neighborhoods/recommendations', params={"budget_manwon": 42000}).json()
    assert len(result['recommendations']) == 2
    assert result['excluded_missing_count'] == 1


@pytest.mark.parametrize('parameters', [
    {'budget_manwon': -1}, {'budget_manwon': 0}, {'budget_manwon': 'nan'},
    {'budget_manwon': 'inf'}, {'budget_manwon': 50000, 'horizon': 1},
    {'budget_manwon': 50000, 'transaction': 'monthly'}, {'budget_manwon': 50000, 'limit': 6},
    {'budget_manwon': 50000, 'area_m2': 0}, {'budget_manwon': 50000, 'area_m2': 'nan'},
])
def test_invalid_input_rejected(client, parameters):
    assert client.get('/api/neighborhoods/recommendations', params=parameters).status_code == 422


def test_catalog_has_fixed_two_domain_score_and_truthful_years(client):
    for horizon in (2, 3):
        data = client.get('/api/neighborhoods/catalog', params={'horizon': horizon}).json()
        assert data['weights'] == {'crime': .5, 'traffic': .5}
        assert data['crime_prediction_year'] == 2025 + horizon
        assert data['traffic_prediction_year'] == 2025
        assert len(data['districts']) == 25
        assert [r['grade'] for r in data['districts']] == list('AAAAABBBBBCCCCCDDDDDEEEEE')
        assert all(r['score'] == .5 * r['crime_score'] + .5 * r['traffic_score'] for r in data['districts'])
        assert any('참고' in w for w in data['warnings'])


def test_detail_matches_district_and_dong_alias(client):
    result = client.get('/api/neighborhoods/detail', params={
        'district': '구05', 'neighborhood': '중앙제1동', 'horizon': 3})
    assert result.status_code == 200
    assert result.json()['district_score']['district'] == '구05'
    assert result.json()['apart_sale_amount'] == 45000
    assert client.get('/api/neighborhoods/detail', params={
        'district': '구05', 'neighborhood': '없는동'}).status_code == 404


def test_invalid_traffic_artifact_fails_without_fake_score(client, monkeypatch):
    monkeypatch.setattr(service, '_traffic_predictions', lambda: (pd.DataFrame({
        'district': ['구00'], 'traffic_score': [100]}), {'prediction_year': 2025}))
    assert client.get('/api/neighborhoods/catalog').status_code == 503


@pytest.mark.parametrize('left,right', [
    ('상계3.4동', '상계3·4동'), ('상계6.7동', '상계6·7동'),
    ('중계2.3동', '중계2·3동'), ('홍제제1동', '홍제1동'),
    ('홍제제2동', '홍제2동'), ('홍제제3동', '홍제3동'),
    ('금호2.3가동', '금호2·3가동'), ('종로1.2.3.4가동', '종로1·2·3·4가동'),
    ('종로5.6가동', '종로5·6가동'), ('면목제3.8동', '면목3·8동'),
])
def test_official_display_aliases(left, right):
    assert service.normalize_neighborhood(left) == service.normalize_neighborhood(right)
