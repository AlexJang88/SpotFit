import math

import numpy as np
import pandas as pd
import pytest

from app.services import final_model_artifacts as final


@pytest.fixture(autouse=True)
def empty_caches():
    final.clear_caches()
    yield
    final.clear_caches()


@pytest.fixture
def monthly(monkeypatch):
    buildings = pd.DataFrame({
        '동': [123] * 4, 'band': [2] * 4, 'key': ['a', 'b', 'c', 'd'],
        'J': [20000., 30000., 40000., 50000.], 'rate': [.06] * 4,
        'b_cnt': [1., 2., 1., 0.],
    })
    meta = {'min_buildings': 3, 'deposit_to_jeonse_max': .8, 'trained_until': '2026-09-30'}
    monkeypatch.setattr(final, 'wolse_buildings', lambda: (buildings, meta))
    return buildings


def test_monthly_uses_building_specific_jeonse_and_weighted_quantiles(monthly):
    quote = final.monthly_quotes(5000, 84)[0][123]
    # .06 * (J - 5000) / 12 -> 75,125,175,225, with weights 1,2,1,0.
    assert quote['available']
    assert quote['monthly_rent_manwon'] == 125
    assert (quote['p25_manwon'], quote['p75_manwon']) == (75, 125)
    assert quote['conversion_rate_pct'] == 6
    assert quote['area_band'] == '20평대'


def test_larger_deposit_lowers_monthly_rent(monthly):
    before = final.monthly_quotes(5000, 84)[0][123]['monthly_rent_manwon']
    after = final.monthly_quotes(10000, 84)[0][123]['monthly_rent_manwon']
    assert before - after == 25


def test_excess_deposit_and_missing_band_are_not_zero_price(monthly):
    quote = final.monthly_quotes(40000, 84)[0][123]
    assert not quote['available']
    assert quote['monthly_rent_manwon'] is None
    assert final.monthly_quotes(5000, 59)[0] == {}


def test_minimum_counts_physical_buildings_not_area_variants(monthly):
    monthly['key'] = 'same building'
    assert not final.monthly_quotes(5000, 84)[0][123]['available']


@pytest.mark.parametrize('deposit,area', [(-1,84), (math.nan,84), (5000,0), (5000,math.inf)])
def test_invalid_monthly_inputs(monthly, deposit, area):
    with pytest.raises(ValueError):
        final.monthly_quotes(deposit, area)


def test_property_target_year_and_units(tmp_path, monkeypatch):
    monkeypatch.setattr(final, 'MODEL_ROOT', tmp_path)
    for folder in ('apart_sale_final', 'apart_jeonse_final'):
        path = tmp_path / folder
        path.mkdir()
        for h in (1,2):
            (path / f'model_{h}y.joblib').write_bytes(b'present')
        pd.DataFrame({'행정동_코드': [123], '자치구': ['중구'], '행정동': ['중앙동'],
                      '1년 뒤 예측(만원/평)': [1000], '2년 뒤 예측(만원/평)': [1500]}).to_csv(
                          path / 'neighborhood_forecasts.csv', index=False)
    first, second = final.property_forecasts(2), final.property_forecasts(3)
    assert first['property_models']['sale']['prediction_year'] == 2027
    assert first['property_models']['sale']['horizon_years'] == 1
    assert first['neighborhoods'][0]['apart_sale_per_pyeong'] == 1000
    assert second['neighborhoods'][0]['apart_sale_per_pyeong'] == 1500
    assert first['neighborhoods'][0]['apart_sale_amount'] == pytest.approx(1000 * 84 / (400 / 121))


def test_weighted_mean_of_safety_points_equals_reverse_risk():
    weights = np.array([.5, .3, .2])
    risk_points = np.array([100, 20, 80])
    assert (weights * (120 - risk_points)).sum() == pytest.approx(120 - (weights * risk_points).sum())
