"""Serving adapters for the delivered property bundles and traffic v94.

Never import training scripts here: several execute training at import time.
"""
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from app.services.recommender_v1 import RecommenderV1Error

ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = ROOT / 'models/recommender/v1'
M2_PER_PYEONG = 400 / 121
DEFAULT_AREA = 84.0
PROPERTY_BASE_YEAR = 2026


def number(value):
    return float(value) if pd.notna(value) and math.isfinite(float(value)) else None


@lru_cache(maxsize=2)
def property_forecasts(horizon: int) -> dict:
    # Crime/traffic origin is 2025, property origin is 2026-09.
    target_year = 2025 + horizon
    property_horizon = target_year - PROPERTY_BASE_YEAR
    if property_horizon not in (1, 2):
        raise ValueError('2027년 또는 2028년 가격만 지원합니다.')
    frames, models = [], {}
    for transaction, folder, prefix in (
        ('sale', 'apart_sale_final', 'apart_sale'),
        ('rent', 'apart_jeonse_final', 'apart_rent'),
    ):
        directory = MODEL_ROOT / folder
        if not (directory / f'model_{property_horizon}y.joblib').is_file():
            raise RecommenderV1Error(f'{folder} 모델이 없습니다.')
        frame = pd.read_csv(directory / 'neighborhood_forecasts.csv', encoding='utf-8-sig')
        column = f'{property_horizon}년 뒤 예측(만원/평)'
        frame = frame[['행정동_코드', '자치구', '행정동', column]].rename(columns={
            '행정동_코드': 'dong_code', '자치구': 'district', '행정동': 'neighborhood',
            column: f'{prefix}_per_pyeong'})
        prices = pd.to_numeric(frame[f'{prefix}_per_pyeong'], errors='raise')
        if ((prices.notna()) & (~np.isfinite(prices) | (prices <= 0))).any():
            raise RecommenderV1Error('가격 예측에 유효하지 않은 금액이 있습니다.')
        frames.append(frame)
        models[transaction] = {'data_year': 2026, 'base_year_month': '2026-09',
            'prediction_year': target_year, 'prediction_year_month': f'{target_year}-09',
            'horizon_years': property_horizon, 'is_future_forecast': True,
            'method': '현재 평당가격 × exp(예측 로그변화율)', 'source': folder}
    merged = frames[0].merge(frames[1], on=['dong_code', 'district', 'neighborhood'],
                             how='outer', validate='one_to_one')
    # A delivered snapshot uses the same old dong code for different names
    # (e.g. sale=용두동, jeonse=신설동). Do not attach that price to either area.
    conflicts = merged.dong_code.duplicated(keep=False)
    excluded = sorted(merged.loc[conflicts, 'dong_code'].astype(int).unique().tolist())
    merged = merged.loc[~conflicts].copy()
    if merged[['district', 'neighborhood']].duplicated().any():
        raise RecommenderV1Error('행정동 가격 식별자가 중복됩니다.')
    rows = []
    for item in merged.to_dict('records'):
        row = {'id': f"{item['district']}|{item['neighborhood']}",
               'dong_code': int(item['dong_code']), 'district': item['district'],
               'neighborhood': item['neighborhood'], 'default_area_m2': DEFAULT_AREA}
        for prefix in ('apart_sale', 'apart_rent'):
            price = number(item[f'{prefix}_per_pyeong'])
            row[f'{prefix}_per_pyeong'] = price
            row[f'{prefix}_amount'] = price * DEFAULT_AREA / M2_PER_PYEONG if price else None
        rows.append(row)
    return {'neighborhoods': rows, 'property_models': models, 'excluded_ambiguous_dong_codes': excluded,
            'price_source': {'type': 'delivered_model_forecasts', 'base_year_month': '2026-09'}}


@lru_cache(maxsize=1)
def wolse_buildings():
    """Predict once at the bundle's snapshot date, retaining each building's J/r."""
    bundle = joblib.load(MODEL_ROOT / 'apart_wolse_final/model_bundle.joblib')
    tables = bundle['tables']
    x = tables['building_info'].copy()
    x['key'] = (x['자치구'].astype(str) + '|' + x['법정동'].astype(str) + '|' +
                x['지번'].astype(str).str.extract(r'(\d+)')[0].fillna(''))
    x['age'] = int(bundle['trained_until'][:4]) - x['Y0']
    for name, key in (('dong', '동'), ('building', 'bk'), ('kapt', 'key'),
                      ('jeonse', 'bk'), ('anchor', '동')):
        x = x.merge(tables[name], on=key, how='left', validate='many_to_one')
    x['j_src'] = 2
    x['J'] = x.j_b * x.A
    x['logJ'] = np.log(x.J.where(x.J > 0))
    x = x.loc[(x.J > 0) & (x.j_n >= 2) & x.an_d.notna() & x.dong_rs.notna()].copy()
    with threadpool_limits(limits=1):
        residual = bundle['model'].predict(x.reindex(columns=bundle['features']))
    x['rate'] = np.clip(x.an_d + residual, *bundle['rate_clip'])
    if not np.isfinite(x[['rate', 'J']].to_numpy()).all():
        raise RecommenderV1Error('월세 전환율 또는 기준 전세가가 유효하지 않습니다.')
    meta = {'data_year': int(bundle['trained_until'][:4]), 'trained_until': bundle['trained_until'],
            'is_future_forecast': False, 'method': '건물별 예측 전환율 × (기준 전세가 − 입력 보증금) ÷ 12',
            'min_buildings': int(bundle.get('min_buildings', 3)),
            'deposit_to_jeonse_max': float(bundle['deposit_to_jeonse_max'])}
    return x, meta


def weighted_quantile(values, weights, fraction):
    order = np.argsort(values, kind='stable')
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    return float(values[min(np.searchsorted(np.cumsum(weights), fraction * weights.sum()), len(values) - 1)])


@lru_cache(maxsize=32)
def monthly_quotes(deposit: float, area_m2: float):
    if not math.isfinite(deposit) or deposit < 0 or not math.isfinite(area_m2) or not 1 <= area_m2 <= 1000:
        raise ValueError('보증금과 면적을 확인해 주세요.')
    buildings, meta = wolse_buildings()
    # Use the same tested area bands as the delivered model; do not scale a
    # building-specific prediction linearly to arbitrary areas.
    band = int(np.clip(area_m2 / 3.3058 // 10, 1, 4))
    band_label = {1: '20평 미만', 2: '20평대', 3: '30평대', 4: '40평 이상'}[band]
    quotes = {}
    for code, all_rows in buildings[buildings.band.eq(band)].groupby('동'):
        eligible = all_rows[deposit <= meta['deposit_to_jeonse_max'] * all_rows.J].copy()
        # bk includes a 5m2 band; count physical buildings for the minimum support.
        n = eligible['key'].nunique()
        quote = {'deposit_manwon': deposit, 'area_band': band_label,
                 'buildings_used': int(n), 'available': n >= meta['min_buildings'],
                 'monthly_rent_manwon': None, 'p25_manwon': None, 'p75_manwon': None,
                 'conversion_rate_pct': None, 'reason': None, 'trained_until': meta['trained_until']}
        if not quote['available']:
            quote['reason'] = '보증금 조건에 맞는 전세 이력 보유 건물이 부족합니다.'
        else:
            values = eligible.rate * (eligible.J - deposit) / 12
            weights = eligible.b_cnt.fillna(0).clip(lower=0).to_numpy()
            if not weights.sum():
                weights = np.ones(len(eligible))
            quote.update(monthly_rent_manwon=round(weighted_quantile(values, weights, .5), 2),
                p25_manwon=round(weighted_quantile(values, weights, .25), 2),
                p75_manwon=round(weighted_quantile(values, weights, .75), 2),
                conversion_rate_pct=round(weighted_quantile(eligible.rate, weights, .5) * 100, 3))
        quotes[int(code)] = quote
    return quotes, meta


@lru_cache(maxsize=2)
def traffic_forecasts(horizon: int):
    if horizon not in (2, 3):
        raise ValueError('2년 또는 3년 모델을 선택해 주세요.')
    directory = MODEL_ROOT / 'traffic_accident_v94'
    metadata = json.loads((directory / 'model_metadata.json').read_text(encoding='utf-8'))
    path = directory / f'traffic_accident_{horizon}y.joblib'
    expected_hash = metadata['artifacts'][str(horizon)]['sha256']
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise RecommenderV1Error('교통 v94 모델 파일의 해시가 일치하지 않습니다.')
    bundle = joblib.load(path)
    if bundle['version'] != 94 or bundle['horizon'] != horizon:
        raise RecommenderV1Error('교통 모델 버전 또는 예측기간이 다릅니다.')
    features = bundle['features']
    weights = np.array([bundle['weights'][f] for f in features], dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any() or not math.isclose(weights.sum(), 1, abs_tol=1e-6):
        raise RecommenderV1Error('교통 가중치 합계가 100%가 아닙니다.')
    frame = pd.read_csv(directory / 'service_forecast.csv', encoding='utf-8-sig')
    frame = frame[frame.horizon_years.eq(horizon)].copy()
    if len(frame) != 25 or frame.district.nunique() != 25 or not frame.base_year.eq(metadata['base_year']).all():
        raise RecommenderV1Error('교통 예측은 같은 기준연도의 25개 자치구여야 합니다.')
    if not (frame.prediction_year - frame.base_year).eq(horizon).all():
        raise RecommenderV1Error('교통 예측연도가 맞지 않습니다.')
    if not frame.predicted_safety_score.between(20 - 1e-6, 100 + 1e-6).all() or not np.allclose(
            frame.predicted_safety_score, 120 - frame.predicted_risk_score):
        raise RecommenderV1Error('교통 Feature 가중합 안전점수 형식이 올바르지 않습니다.')
    # The publisher has already calculated sum(weight * feature quintile).
    # 120 - sum(w * risk_points) == sum(w * (120 - risk_points)), sum(w)=1.
    frame['traffic_score'] = frame.predicted_safety_score
    formula = [{'feature': f, 'weight_pct': float(w * 100),
                'direction': 'higher_risk' if bundle['coefficients'][f] >= 0 else 'lower_risk'}
               for f, w in zip(features, weights)]
    return frame, {'prediction_year': int(frame.prediction_year.iloc[0]), 'base_year': int(metadata['base_year']),
        'source_model': 'traffic_accident_v94', 'score_method': 'Feature별 5분위 안전점수 × Poisson 계수 가중치 합',
        'score_formula': sorted(formula, key=lambda r: -r['weight_pct'])}


def clear_caches():
    property_forecasts.cache_clear()
    wolse_buildings.cache_clear()
    monthly_quotes.cache_clear()
    traffic_forecasts.cache_clear()
