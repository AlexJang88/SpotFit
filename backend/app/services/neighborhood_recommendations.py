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
from app.services.recommender_v1 import _property_predictions, _traffic_predictions, RecommenderV1Error, MODEL_ROOT, RAW_ROOT

WEIGHTS = {"crime": 0.5, "traffic": 0.5}
_PROPERTY_LOCK = Lock()


def forecast_payload(horizon: int) -> dict:
    """Read the final 2y/3y artifacts independently of the legacy 1y API."""
    if horizon not in (2, 3):
        raise ValueError("2년 또는 3년 모델을 선택해 주세요.")
    model, csv, metadata = required_paths(horizon)
    if not all(path.is_file() for path in (model, csv, metadata)):
        raise RecommenderV1Error(f"{horizon}년 범죄모델 산출물이 없습니다.")
    data = pd.read_csv(csv, encoding="utf-8-sig")
    if len(data) != 25 or data.district.nunique() != 25:
        raise RecommenderV1Error("범죄 결과가 서울 25개 자치구 구조가 아닙니다.")
    if data.base_year.nunique() != 1 or not (data.prediction_year - data.base_year).eq(horizon).all():
        raise RecommenderV1Error("범죄 예측의 기준연도와 예측연도가 맞지 않습니다.")
    if not all(math.isfinite(float(y)) for y in data.predicted_y):
        raise RecommenderV1Error("범죄 예측값이 유효하지 않습니다.")
    points = 100 - ((data.predicted_y.rank(ascending=True, method="min").astype(int) - 1) // 5) * 20
    if not data.safety_score.eq(points).all():
        raise RecommenderV1Error("범죄 점수와 예측 순위의 환산 규칙이 일치하지 않습니다.")
    return {"data_year": int(data.base_year.iloc[0]), "prediction_year": int(data.prediction_year.iloc[0]),
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


@lru_cache(maxsize=1)
def _property_catalog_cached() -> dict:
    # Allow a provenance-bearing snapshot exported from the same model API on
    # machines without the large raw transaction files. Never invent prices.
    if not all((RAW_ROOT / name).is_file() for name in ("아파트_매매.csv", "아파트_전월세.csv")):
        snapshot = MODEL_ROOT / "property_service_forecasts.json"
        if not snapshot.is_file():
            raise RecommenderV1Error("아파트 원본 CSV 또는 property_service_forecasts.json이 필요합니다.")
        payload = json.loads(snapshot.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or not payload.get("provenance"):
            raise RecommenderV1Error("가격 스냅샷의 형식 또는 출처 정보가 없습니다.")
        rows = payload["neighborhoods"]
        identities = [(r["district"], r["neighborhood"]) for r in rows]
        if not rows or len(set(identities)) != len(identities):
            raise RecommenderV1Error("가격 스냅샷에 중복되거나 빈 행정동 자료가 있습니다.")
        for row in rows:
            for key in ("apart_sale_amount", "apart_rent_amount", "apart_sale_per_pyeong", "apart_rent_per_pyeong"):
                value = row[key]
                if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0):
                    raise RecommenderV1Error("가격 스냅샷에 유효하지 않은 값이 있습니다.")
        return {"neighborhoods": rows, "property_models": payload["property_models"],
                "price_source": payload["provenance"]}
    _, sale, sale_meta = _property_predictions("apart_sale")
    _, rent, rent_meta = _property_predictions("apart_rent")
    merged = sale.merge(rent, on=["district", "neighborhood"], how="outer", validate="one_to_one")
    rows = []
    for item in merged.sort_values(["district", "neighborhood"]).to_dict("records"):
        rows.append({
            "id": f"{item['district']}|{item['neighborhood']}",
            "district": item["district"], "neighborhood": item["neighborhood"],
            **{k: _number(v) for k, v in item.items() if k not in ("district", "neighborhood")},
        })
    return {"neighborhoods": rows, "property_models": {"sale": sale_meta, "rent": rent_meta},
            "price_source": {"type": "local_model_inference"}}


def property_catalog() -> dict:
    with _PROPERTY_LOCK:
        return _property_catalog_cached()


def clear_neighborhood_cache() -> None:
    _property_catalog_cached.cache_clear()
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
    columns = ["계약연도", column] + (["전월세구분"] if transaction == "rent" else [])
    frame = pd.read_csv(path, usecols=columns, encoding="utf-8-sig")
    if transaction == "rent":
        frame = frame.loc[frame["전월세구분"].astype(str).str.strip().eq("전세")].copy()
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


def district_scores(horizon: int) -> dict:
    crime = forecast_payload(horizon)
    traffic, traffic_meta = _traffic_predictions()
    if len(traffic) != 25 or traffic.district.nunique() != 25:
        raise RecommenderV1Error("교통안전 결과가 서울 25개 자치구 구조가 아닙니다.")
    traffic = traffic.copy()
    traffic["traffic_score"] = pd.to_numeric(traffic.traffic_score, errors="raise")
    if not all(math.isfinite(float(v)) for v in traffic.traffic_score):
        raise RecommenderV1Error("교통안전 점수에 유효하지 않은 값이 있습니다.")
    # Both domains use the same 20/40/60/80/100 scale, higher = safer.
    # Equal traffic values receive the same rank/points.
    ranks = traffic.traffic_score.rank(ascending=False, method="min").astype(int)
    traffic["points"] = 100 - ((ranks - 1) // 5) * 20
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
            "score": round(WEIGHTS["crime"] * c["predicted_score"] + WEIGHTS["traffic"] * t.points, 4),
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
        "weights": WEIGHTS,
        "score_method": "각 영역 5분위 점수(높을수록 낮은 위험)를 50%씩 합산",
        "tie_rule": "종합점수 → 범죄안전 점수 → 자치구명 순으로 동점 정렬",
    }


def service_catalog(horizon: int = 2) -> dict:
    properties = property_catalog()
    scores = district_scores(horizon)
    warnings = ["가격은 미래 연도 가격이 아닌 최신 거래 특성 기반 대표 추정값입니다.",
                "동네점수는 자치구 단위이며 같은 구의 행정동에 동일하게 적용됩니다."]
    if scores["crime_prediction_year"] != scores["traffic_prediction_year"]:
        warnings.append(
            f"범죄노출 {scores['crime_prediction_year']}년 예측과 교통안전 {scores['traffic_prediction_year']}년 참고 결과를 결합합니다."
        )
    return {"version": "v1", **properties, **scores, "warnings": warnings}


def priced_neighborhood(item: dict, transaction: str, area_m2: float | None) -> dict:
    prefix = "apart_sale" if transaction == "sale" else "apart_rent"
    amount = item.get(f"{prefix}_amount")
    if area_m2 is not None:
        price = item.get(f"{prefix}_per_pyeong")
        # Convert to pyeong before multiplying; never compare a per-area price to a budget.
        amount = price * area_m2 / (400 / 121) if price is not None else None
    return {
        **item, "estimated_amount_manwon": _number(amount),
        "transaction": transaction, "area_m2": area_m2,
        "price_basis": "입력면적 × 행정동 대표 평당 추정가" if area_m2 is not None else "행정동 최신 거래 특성의 대표 총액",
    }


def recommend(budget_manwon: float, transaction: str = "sale", horizon: int = 2,
              area_m2: float | None = None, limit: int = 5, min_budget_manwon: float = 0) -> dict:
    catalog = service_catalog(horizon)
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
        "warnings": catalog["warnings"], "weights": WEIGHTS,
        "crime_prediction_year": catalog["crime_prediction_year"],
        "traffic_prediction_year": catalog["traffic_prediction_year"],
        "sort_rule": "예산 내 구별 최저가 행정동 1곳 → 동네점수 → 범죄안전 점수 → 자치구명",
    }


def neighborhood_detail(district: str, neighborhood: str, horizon: int,
                        transaction: str = "sale", area_m2: float | None = None) -> dict:
    catalog = service_catalog(horizon)
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
            "property_models": catalog["property_models"], "warnings": catalog["warnings"]}
