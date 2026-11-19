"""Budget-filtered neighborhood recommendations, with explicit data years.

Only property, child-crime exposure and traffic outputs are dependencies here.
The older crime/child-environment bundles are not required by the final service.
"""
from __future__ import annotations

import math
import json
import re
from datetime import date
from functools import lru_cache
from threading import Lock

import pandas as pd

from app.services.child_crime_exposure import required_paths
from app.services.recommender_v1 import RecommenderV1Error, RAW_ROOT
from app.services.final_model_artifacts import property_forecasts, monthly_quotes, traffic_forecasts, clear_caches, DEFAULT_AREA
from app.services.historical_safety_reference import reference_series_by_district

WEIGHTS = {"crime": 0.5, "traffic": 0.5}
_PROPERTY_LOCK = Lock()


def score_weights(crime_weight: float | None = None, traffic_weight: float | None = None) -> dict:
    """Resolve request-local weights; never change shared model weights."""
    if crime_weight is None and traffic_weight is None:
        return dict(WEIGHTS)
    if crime_weight is None:
        crime_weight = 1 - traffic_weight
    if traffic_weight is None:
        traffic_weight = 1 - crime_weight
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (crime_weight, traffic_weight)):
        raise ValueError("점수 반영 비율은 0~100% 사이여야 합니다.")
    if not math.isclose(crime_weight + traffic_weight, 1, rel_tol=0, abs_tol=1e-9):
        raise ValueError("범죄·교통 반영 비율의 합계는 100%여야 합니다.")
    return {"crime": crime_weight, "traffic": 1 - crime_weight}


def forecast_payload(horizon: int) -> dict:
    """Read the final 2y/3y artifacts independently of the legacy 1y API."""
    if horizon not in (2, 3):
        raise ValueError("2년 또는 3년 모델을 선택해 주세요.")
    model, csv, metadata = required_paths(horizon)
    directory = model.parent
    formula_path = directory / "feature_score_formula.csv"
    breakdown_path = directory / "feature_score_breakdown.csv"
    if not all(path.is_file() for path in (model, csv, metadata, formula_path, breakdown_path)):
        raise RecommenderV1Error(f"{horizon}년 범죄모델 산출물이 없습니다.")
    data = pd.read_csv(csv, encoding="utf-8-sig")
    if len(data) != 25 or data.district.nunique() != 25:
        raise RecommenderV1Error("범죄 결과가 서울 25개 자치구 구조가 아닙니다.")
    if data.base_year.nunique() != 1 or not (data.prediction_year - data.base_year).eq(horizon).all():
        raise RecommenderV1Error("범죄 예측의 기준연도와 예측연도가 맞지 않습니다.")
    if not all(math.isfinite(float(y)) for y in data.predicted_y):
        raise RecommenderV1Error("범죄 예측값이 유효하지 않습니다.")
    formula = pd.read_csv(formula_path, encoding="utf-8-sig")
    breakdown = pd.read_csv(breakdown_path, encoding="utf-8-sig")
    required_formula = {"feature", "direction", "weight_pct"}
    required_breakdown = {"district", "feature", "quintile_points", "weight_pct", "weighted_contribution"}
    if not required_formula.issubset(formula.columns) or not required_breakdown.issubset(breakdown.columns):
        raise RecommenderV1Error("범죄 Feature 점수 산출물 형식이 올바르지 않습니다.")
    if not math.isclose(float(formula.weight_pct.sum()), 100.0, abs_tol=1e-6):
        raise RecommenderV1Error("범죄 Feature 가중치 합계가 100%가 아닙니다.")
    if set(breakdown.district) != set(data.district) or breakdown.feature.nunique() != len(formula):
        raise RecommenderV1Error("범죄 Feature별 자치구 점수 내역이 완전하지 않습니다.")
    recomputed = breakdown.groupby("district").weighted_contribution.sum()
    actual = data.set_index("district").safety_score
    if not all(math.isclose(float(actual[d]), float(v), abs_tol=1e-6) for d, v in recomputed.items()):
        raise RecommenderV1Error("범죄 Feature 가중합과 서비스 점수가 일치하지 않습니다.")
    if not data.safety_score.between(20, 100).all() or data.safety_score.nunique() <= 5:
        raise RecommenderV1Error("범죄 Feature 가중합 점수가 연속 점수 구조가 아닙니다.")
    score_formula = [{
        "feature": r.feature, "direction": r.direction, "weight_pct": float(r.weight_pct),
    } for r in formula.sort_values("weight_pct", ascending=False).itertuples(index=False)]
    return {"data_year": int(data.base_year.iloc[0]), "prediction_year": int(data.prediction_year.iloc[0]),
            "score_method": "Feature별 5분위 안전점수 × 관련도 가중치 합",
            "score_formula": score_formula,
            "forecasts": [{"district": r.district, "predicted_score": float(r.safety_score),
                           "predicted_exposure_y": float(r.predicted_y)} for r in data.itertuples()]}


def normalize_neighborhood(name: str) -> str:
    # Common official display aliases, always matched WITH the district name.
    value = re.sub(r"제+(?=\d)", "", re.sub(r"\s+", "", name))
    return re.sub(r"(?<=\d)[.·ㆍ](?=\d)", "·", value)


def _number(value):
    if value is None or pd.isna(value):
        return None
    value = float(value)
    return round(value, 4) if math.isfinite(value) else None


def property_catalog(horizon: int = 2, target_year: int | None = None) -> dict:
    with _PROPERTY_LOCK:
        return property_forecasts(horizon, target_year=target_year)


def clear_neighborhood_cache() -> None:
    clear_caches()
    popular_areas.cache_clear()


@lru_cache(maxsize=2)
def popular_areas(transaction: str) -> dict:
    """Top three 1m² bands in the latest completed year of local transactions."""
    name = "아파트_매매.csv" if transaction == "sale" else "아파트_전월세.csv"
    path = next((p for p in (RAW_ROOT / name, RAW_ROOT / "부동산" / name) if p.is_file()), None)
    if path is None:
        # These are explicitly generic shortcuts, never claimed as measured rankings.
        return {"transaction": transaction, "basis": "default", "data_year": None,
                "description": "거래 원자료가 없어 기본 면적을 제공합니다.",
                "options": [{"label": f"{a}㎡", "area_m2": a, "transaction_count": None} for a in (59, 84, 114)]}
    column = "전용면적(㎡)" if transaction == "sale" else "임대면적(㎡)"
    columns = ["계약연도", column] + (["전월세구분"] if transaction != "sale" else [])
    frame = pd.read_csv(path, usecols=columns, encoding="utf-8-sig")
    if transaction in ("rent", "wolse"):
        frame = frame.loc[frame["전월세구분"].astype(str).str.strip().eq("월세" if transaction == "wolse" else "전세")].copy()
    frame["year"] = pd.to_numeric(frame["계약연도"], errors="coerce")
    frame["area"] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[frame["year"].lt(date.today().year) & frame["area"].between(1, 1000)].copy()
    if frame.empty:
        return {"transaction": transaction, "basis": "unavailable", "data_year": None,
                "description": "완료 연도의 거래 면적 자료가 없습니다.", "options": []}
    year = int(frame["year"].max())
    frame = frame.loc[frame["year"].eq(year)].copy()
    frame["band"] = frame["area"].astype(int)
    grouped = frame.groupby("band")["area"].agg(["size", "median"]).reset_index()
    grouped = grouped.sort_values(["size", "band"], ascending=[False, True]).head(3)
    return {"transaction": transaction, "basis": "transactions", "data_year": year,
            "source_file": str(path.relative_to(RAW_ROOT)),
            "description": f"{year}년 거래 건수 상위 3개 · 1㎡ 구간별 집계, 선택 시 거래면적 중앙값 적용",
            "options": [{"label": f"{int(r.band)}㎡대", "area_m2": round(float(r.median), 2),
                         "transaction_count": int(r.size)} for r in grouped.itertuples()]}


def district_scores(horizon: int, crime_weight: float | None = None,
                    traffic_weight: float | None = None) -> dict:
    weights = score_weights(crime_weight, traffic_weight)
    crime = forecast_payload(horizon)
    traffic, traffic_meta = traffic_forecasts(horizon)
    if len(traffic) != 25 or traffic.district.nunique() != 25:
        raise RecommenderV1Error("교통안전 결과가 서울 25개 자치구 구조가 아닙니다.")
    traffic = traffic.copy()
    traffic["traffic_score"] = pd.to_numeric(traffic.traffic_score, errors="raise")
    if not all(math.isfinite(float(v)) for v in traffic.traffic_score):
        raise RecommenderV1Error("교통안전 점수에 유효하지 않은 값이 있습니다.")
    # Preserve the delivered weighted Feature score without re-binning it.
    traffic["points"] = traffic.traffic_score
    traffic_by = {r.district: r for r in traffic.itertuples()}
    rows = []
    for c in crime["forecasts"]:
        t = traffic_by.get(c["district"])
        if t is None:
            raise RecommenderV1Error("범죄·교통 결과의 자치구가 일치하지 않습니다.")
        rows.append({
            "district": c["district"],
            "crime_score": float(c["predicted_score"]),
            "traffic_score": float(t.points),
            "score": round(weights["crime"] * c["predicted_score"] + weights["traffic"] * t.points, 4),
            "crime_prediction_year": crime["prediction_year"],
            "traffic_prediction_year": int(traffic_meta["prediction_year"]),
            "predicted_exposure_y": c["predicted_exposure_y"],
        })
    rows.sort(key=lambda row: (-row["score"], -row["crime_score"], row["district"]))
    for index, row in enumerate(rows):
        row["rank"] = index + 1
        row["grade"] = "ABCDE"[index // 5]
    return {
        "districts": rows,
        "horizon": horizon,
        "crime_base_year": crime["data_year"],
        "crime_prediction_year": crime["prediction_year"],
        "traffic_prediction_year": int(traffic_meta["prediction_year"]),
        "weights": weights,
        "score_method": f"범죄 안전점수 {weights['crime'] * 100:g}% + 교통 안전점수 {weights['traffic'] * 100:g}%",
        "traffic_score_method": traffic_meta["score_method"],
        "traffic_score_formula": traffic_meta["score_formula"],
        "crime_score_method": crime["score_method"],
        "crime_score_formula": crime["score_formula"],
        "tie_rule": "종합점수 → 범죄안전 점수 → 자치구명 순으로 동점 정렬",
    }


def service_catalog(horizon: int = 2, transaction: str = "sale", deposit_manwon: float | None = None,
                    area_m2: float | None = None, crime_weight: float | None = None,
                    traffic_weight: float | None = None) -> dict:
    scores = district_scores(horizon, crime_weight, traffic_weight)
    if scores["crime_prediction_year"] != scores["traffic_prediction_year"]:
        raise RecommenderV1Error(
            "범죄·교통 모델의 예측연도가 달라 통합 추천을 제공할 수 없습니다."
        )
    target_year = scores["crime_prediction_year"]
    properties = property_catalog(horizon, target_year=target_year)
    other_horizon = 3 if horizon == 2 else 2
    other_scores = district_scores(other_horizon, crime_weight, traffic_weight)
    other_by_district = {row['district']: row for row in other_scores['districts']}
    historical = reference_series_by_district(scores['weights']['crime'], scores['weights']['traffic'])
    for row in scores['districts']:
        row['historical_reference_series'] = historical.get(row['district'], [])
        counterpart = other_by_district.get(row['district'])
        row['score_series'] = sorted([
            {'year': row['crime_prediction_year'], 'score': row['score'], 'kind': 'forecast'},
            *([{'year': counterpart['crime_prediction_year'], 'score': counterpart['score'],
                'kind': 'forecast'}] if counterpart else []),
        ], key=lambda point: point['year'])
    for property_type in ("sale", "rent"):
        model = properties.get("property_models", {}).get(property_type, {})
        if model.get("prediction_year") != target_year:
            raise RecommenderV1Error(
                "부동산 모델의 예측연도가 범죄·교통 모델과 달라 통합 추천을 제공할 수 없습니다."
            )
    warnings = [f"매매·전세는 2026년 9월 기준 {target_year}년 9월 평당 예측가격입니다. 면적 미입력 시 전용 84㎡로 환산합니다.",
                "동네점수는 자치구 단위이며 같은 구의 행정동에 동일하게 적용됩니다."]
    if properties.get('excluded_ambiguous_dong_codes'):
        warnings.append('매매·전세 자료의 행정동 코드와 이름이 서로 다른 지역은 가격 추천에서 제외합니다.')
    if transaction == "wolse":
        if deposit_manwon is None or area_m2 is None:
            raise ValueError('월세 조회에는 보증금과 면적이 필요합니다.')
        quotes, meta = monthly_quotes(deposit_manwon, area_m2)
        properties = {**properties,
            'neighborhoods': [{**row, 'wolse': quotes.get(row['dong_code'], {
                'available': False, 'monthly_rent_manwon': None,
                'reason': '해당 동·면적대의 전세 이력이 부족해 월세를 계산할 수 없습니다.'})}
                for row in properties['neighborhoods']],
            'property_models': {**properties['property_models'], 'wolse': meta}}
        warnings.append(f"월세는 {meta['trained_until']}까지의 자료로 계산한 현재 조건 추정값이며 미래 연도 예측이 아닙니다. 표시 범위는 건물별 추정 월세의 25~75% 구간입니다.")
    return {"version": "v1", **properties, **scores, "warnings": warnings,
            "transaction": transaction, "deposit_manwon": deposit_manwon, "area_m2": area_m2}


def priced_neighborhood(item: dict, transaction: str, area_m2: float | None) -> dict:
    if transaction == 'wolse':
        quote = item.get('wolse', {})
        return {**item, 'estimated_amount_manwon': quote.get('monthly_rent_manwon'),
                'transaction': transaction, 'area_m2': area_m2,
                'price_basis': '입력 보증금 기준, 같은 면적대 건물별 예상 월세의 거래량 가중 중앙값'}
    prefix = "apart_sale" if transaction == "sale" else "apart_rent"
    amount = item.get(f"{prefix}_amount")
    if area_m2 is not None:
        price = item.get(f"{prefix}_per_pyeong")
        # Convert to pyeong before multiplying; never compare a per-area price to a budget.
        amount = price * area_m2 / (400 / 121) if price is not None else None
    return {
        **item, "estimated_amount_manwon": _number(amount),
        "transaction": transaction, "area_m2": area_m2,
        "price_basis": "전용면적 × 행정동 미래 평당 예측가" if area_m2 is not None else "전용 84㎡ × 행정동 미래 평당 예측가",
    }


def recommend(budget_manwon: float, transaction: str = "sale", horizon: int = 2,
              area_m2: float | None = None, limit: int = 5, min_budget_manwon: float = 0,
              deposit_manwon: float | None = None, crime_weight: float | None = None,
              traffic_weight: float | None = None) -> dict:
    catalog = service_catalog(horizon, transaction, deposit_manwon, area_m2, crime_weight, traffic_weight)
    by_district = {row["district"]: row for row in catalog["districts"]}
    candidates = []
    excluded = 0
    for item in catalog["neighborhoods"]:
        price = priced_neighborhood(item, transaction, area_m2)
        amount = price["estimated_amount_manwon"]
        score = by_district.get(item["district"])
        if amount is None or amount <= 0 or score is None:
            excluded += 1
            continue
        if min_budget_manwon <= amount <= budget_manwon:
            candidates.append({**price, "district_score": score,
                               "budget_remaining_manwon": round(budget_manwon - amount, 4)})
    # One affordable, cheapest neighborhood per district, then district ranking.
    # Use the same tie-breaks as district_scores so map grades and results agree.
    cheapest_by_district = {}
    for row in sorted(candidates, key=lambda row: (row["estimated_amount_manwon"], row["neighborhood"], row["id"])):
        cheapest_by_district.setdefault(row["district"], row)
    selected = sorted(cheapest_by_district.values(), key=lambda row: row["district_score"]["rank"])[:limit]
    return {
        "budget_manwon": budget_manwon, "min_budget_manwon": min_budget_manwon, "transaction": transaction, "horizon": horizon,
        "area_m2": area_m2, "limit": limit, "candidate_count": len(candidates),
        "excluded_missing_count": excluded,
        "eligible_district_count": len(cheapest_by_district),
        "recommendations": [{**row, "recommendation_rank": i + 1} for i, row in enumerate(selected)],
        "warnings": catalog["warnings"], "weights": catalog["weights"],
        "crime_prediction_year": catalog["crime_prediction_year"],
        "traffic_prediction_year": catalog["traffic_prediction_year"],
        "sort_rule": "예산 내 구별 최저가 행정동 1곳 → 동네점수 → 범죄안전 점수 → 자치구명",
    }


def neighborhood_detail(district: str, neighborhood: str, horizon: int,
                        transaction: str = "sale", area_m2: float | None = None,
                        deposit_manwon: float | None = None, crime_weight: float | None = None,
                        traffic_weight: float | None = None) -> dict:
    catalog = service_catalog(horizon, transaction, deposit_manwon, area_m2, crime_weight, traffic_weight)
    found = [row for row in catalog["neighborhoods"]
             if row["district"] == district.strip()
             and normalize_neighborhood(row["neighborhood"]) == normalize_neighborhood(neighborhood)]
    if len(found) > 1:
        raise RecommenderV1Error("행정동 이름이 중복 매칭되어 가격을 확정할 수 없습니다.")
    score = next((row for row in catalog["districts"] if row["district"] == district.strip()), None)
    if not score:
        raise LookupError("자치구를 찾을 수 없습니다.")
    if not found:
        raise LookupError("이 행정동의 가격 데이터가 없습니다. 지도에서 소속 자치구 점수는 확인할 수 있습니다.")
    return {**priced_neighborhood(found[0], transaction, area_m2), "district_score": score,
            "property_models": catalog["property_models"], "warnings": catalog["warnings"],
            "weights": catalog["weights"]}
