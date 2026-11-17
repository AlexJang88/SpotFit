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
                "score_method": "Feature별 5분위 안전점수 × 관련도 가중치 합",
                "score_formula": [
                    {"feature": "feature_a", "direction": "higher_risk", "weight_pct": 60.0},
                    {"feature": "feature_b", "direction": "lower_risk", "weight_pct": 40.0},
                ],
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


def test_one_cheapest_dong_per_district_in_score_rank_order(client):
    rows = service.property_catalog()['neighborhoods']
    # Several cheap dongs in the best district must not crowd out other districts.
    rows.extend([
        {**rows[0], 'id': '구00|저렴동', 'neighborhood': '저렴동', 'apart_sale_amount': 30000},
        {**rows[0], 'id': '구00|비싼동', 'neighborhood': '비싼동', 'apart_sale_amount': 44000},
        {**rows[1], 'id': '구01|최저가동', 'neighborhood': '최저가동', 'apart_sale_amount': 20000},
    ])
    result = client.get('/api/neighborhoods/recommendations', params={'budget_manwon': 45000}).json()
    selected = result['recommendations']
    assert [r['district'] for r in selected] == [f'구{i:02}' for i in range(5)]
    assert selected[0]['neighborhood'] == '저렴동'
    assert selected[1]['neighborhood'] == '최저가동'
    assert result['candidate_count'] == 9
    assert result['eligible_district_count'] == 6


def test_unique_districts_can_return_less_than_five(client):
    rows = service.property_catalog()['neighborhoods']
    rows.extend([{**rows[0], 'id': f'구00|추가{i}동', 'neighborhood': f'추가{i}동',
                  'apart_sale_amount': 40000} for i in range(8)])
    result = client.get('/api/neighborhoods/recommendations', params={'budget_manwon': 41000}).json()
    assert len(result['recommendations']) == 2
    assert result['eligible_district_count'] == 2


def test_budget_range_filters_before_choosing_district_cheapest(client):
    rows = service.property_catalog()['neighborhoods']
    rows.append({**rows[0], 'id': '구00|범위안동', 'neighborhood': '범위안동', 'apart_sale_amount': 43500})
    result = client.get('/api/neighborhoods/recommendations', params={
        'min_budget_manwon': 43000, 'budget_manwon': 45000}).json()
    assert [r['district'] for r in result['recommendations']] == ['구00', '구03', '구04', '구05']
    assert result['recommendations'][0]['neighborhood'] == '범위안동'
    assert all(43000 <= r['estimated_amount_manwon'] <= 45000 for r in result['recommendations'])
    exact = client.get('/api/neighborhoods/recommendations', params={
        'min_budget_manwon': 45000, 'budget_manwon': 45000}).json()
    assert len(exact['recommendations']) == 1


@pytest.mark.parametrize('lower', [-1, 50001, 'nan', 'inf'])
def test_invalid_budget_range(client, lower):
    assert client.get('/api/neighborhoods/recommendations', params={
        'min_budget_manwon': lower, 'budget_manwon': 50000}).status_code == 422


def test_area_presets_use_last_completed_year_and_jeonse_only(client, monkeypatch, tmp_path):
    monkeypatch.setattr(service, 'RAW_ROOT', tmp_path)
    service.popular_areas.cache_clear()
    year = service.date.today().year - 1
    pd.DataFrame({
        '계약연도': [year] * 7 + [year + 1],
        '전월세구분': ['전세'] * 5 + ['월세'] * 3,
        '임대면적(㎡)': [84.9, 84.8, 59.9, 49.5, 49.8, 20, 20, 20],
    }).to_csv(tmp_path / '아파트_전월세.csv', index=False, encoding='utf-8-sig')
    try:
        data = client.get('/api/neighborhoods/area-presets?transaction=rent').json()
        assert data['data_year'] == year
        assert [r['label'] for r in data['options']] == ['49㎡대', '84㎡대', '59㎡대']
        assert [r['transaction_count'] for r in data['options']] == [2, 2, 1]
        assert data['options'][0]['area_m2'] == 49.65
        fallback = client.get('/api/neighborhoods/area-presets?transaction=sale').json()
        assert fallback['basis'] == 'default'
        assert all(r['transaction_count'] is None for r in fallback['options'])
    finally:
        service.popular_areas.cache_clear()


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
        assert sum(row['weight_pct'] for row in data['crime_score_formula']) == 100
        assert 'Feature' in data['crime_score_method']
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
