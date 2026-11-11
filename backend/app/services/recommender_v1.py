"""Serve the versioned recommender-v1 model outputs to the application."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from threading import Lock

import joblib
import numpy as np
import pandas as pd

from app.services.traffic_accident_v51 import forecast_payload as traffic_v51_forecast


ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = ROOT / "models" / "recommender" / "v1"
RAW_ROOT = ROOT / "data" / "raw"
_LOAD_LOCK = Lock()


class RecommenderV1Error(RuntimeError):
    pass


def _grade(rank: int) -> str:
    return ("A", "B", "C", "D", "E")[min(4, (rank - 1) // 5)]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise RecommenderV1Error(f"모델 예측 산출물이 없습니다: {path}")
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def _property_predictions(kind: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    is_sale = kind == "apart_sale"
    bundle_path = MODEL_ROOT / kind / "model_bundle.joblib"
    raw_path = RAW_ROOT / ("아파트_매매.csv" if is_sale else "아파트_전월세.csv")
    if not bundle_path.is_file() or not raw_path.is_file():
        raise RecommenderV1Error(f"{kind} 모델 또는 입력 데이터가 없습니다.")

    bundle = joblib.load(bundle_path)
    pipeline = bundle["pipeline"]
    metadata = bundle["metadata"]
    features = metadata["features"]
    columns = ["자치구", "행정동", *features]
    if not is_sale:
        columns.append("전월세구분")
    frame = pd.read_csv(raw_path, encoding="utf-8-sig", usecols=columns, low_memory=False)
    if not is_sale:
        frame = frame.loc[frame["전월세구분"].astype(str).str.strip().eq("전세")]
    for feature in features:
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")

    year_counts = frame["계약연도"].value_counts()
    eligible = year_counts[year_counts >= 100].index
    latest_year = int(max(eligible))
    latest = frame.loc[frame["계약연도"].eq(latest_year)]
    profiles = latest.dropna(subset=["자치구", "행정동"]).groupby(
        ["자치구", "행정동"], as_index=False
    )[features].median(numeric_only=True)
    profiles["계약연도"] = latest_year
    profiles["prediction"] = pipeline.predict(profiles[features])
    area_feature = "전용면적(평)" if is_sale else "임대면적(평)"
    profiles["amount"] = profiles["prediction"] * profiles[area_feature]
    per_pyeong = f"{kind}_per_pyeong"
    amount = f"{kind}_amount"
    neighborhoods = profiles[["자치구", "행정동", "prediction", "amount"]].rename(
        columns={
            "자치구": "district",
            "행정동": "neighborhood",
            "prediction": per_pyeong,
            "amount": amount,
        }
    )
    # District prices are intentionally the unweighted mean of child-dong
    # estimates, as required by the map's district-level display contract.
    districts = neighborhoods.groupby("district", as_index=False)[[per_pyeong, amount]].mean()
    districts = districts.rename(columns={per_pyeong: kind})
    return districts, neighborhoods, {
        "model": metadata["model_name"],
        "data_year": latest_year,
        "unit": "만원/평",
        "is_future_forecast": False,
        "method": "행정동별 최신 거래 특성 중앙값 예측 후 하위 행정동 단순평균",
        "district_aggregation": "하위 행정동 예측값의 산술평균",
    }


def _child_predictions() -> tuple[pd.DataFrame, dict]:
    path = MODEL_ROOT / "child_environment" / "forecasts_2025_2027.csv"
    frame = _read_csv(path)
    pivot = frame.pivot(index="gu", columns="horizon_years", values="predicted_score_100")
    pivot = pivot.rename(columns={1: "child_environment_1y", 3: "child_environment_3y"})
    result = pivot.reset_index().rename(columns={"gu": "district"})
    return result, {
        "models": ["child_environment_1y.joblib", "child_environment_3y.joblib"],
        "base_year": int(frame["base_year"].max()),
        "prediction_years": sorted(int(value) for value in frame["forecast_year"].unique()),
        "unit": "점",
    }


def _crime_predictions() -> tuple[pd.DataFrame, dict]:
    path = MODEL_ROOT / "crime_safety" / "2024_예측.csv"
    frame = _read_csv(path)
    result = frame[["자치구", "pred_안전점수", "pred_순위", "pred_등급문자"]].rename(
        columns={
            "자치구": "district",
            "pred_안전점수": "crime_raw_score",
            "pred_순위": "crime_rank",
            "pred_등급문자": "crime_grade",
        }
    )
    result["crime_score"] = result["crime_raw_score"].rank(
        ascending=True, method="average", pct=True
    ).mul(100.0)
    return result, {
        "model": "standard_1y_models.joblib",
        "prediction_year": int(frame["예측연도"].max()),
        "unit": "점",
    }


def _traffic_predictions() -> tuple[pd.DataFrame, dict]:
    payload_2027 = traffic_v51_forecast(2)
    payload_2028 = traffic_v51_forecast(3)
    result_2027 = pd.DataFrame(payload_2027["forecasts"]).rename(
        columns={
            "predicted_safety_score": "traffic_score_2027",
            "predicted_safety_rank": "traffic_rank_2027",
            "predicted_safety_grade": "traffic_grade_2027",
        }
    )[["district", "traffic_score_2027", "traffic_rank_2027", "traffic_grade_2027"]]
    result_2028 = pd.DataFrame(payload_2028["forecasts"]).rename(
        columns={
            "predicted_safety_score": "traffic_score_2028",
            "predicted_safety_rank": "traffic_rank_2028",
            "predicted_safety_grade": "traffic_grade_2028",
        }
    )[["district", "traffic_score_2028", "traffic_rank_2028", "traffic_grade_2028"]]
    result = result_2027.merge(result_2028, on="district", validate="one_to_one")
    # Keep the original combined endpoint contract during the presentation.
    result["traffic_score"] = result["traffic_score_2027"]
    result["traffic_rank"] = result["traffic_rank_2027"]
    result["traffic_grade"] = result["traffic_grade_2027"]
    result["traffic_accidents"] = np.nan
    return result, {
        "models": ["traffic_accident_2y.joblib", "traffic_accident_3y.joblib"],
        "temporary_model": True,
        "base_year": 2025,
        "prediction_years": [2027, 2028],
        "unit": {"traffic_score_2027": "점", "traffic_score_2028": "점"},
    }


def _clean(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return round(float(value), 4)
    return value


@lru_cache(maxsize=1)
def _get_district_forecasts_cached() -> dict:
    sale, sale_neighborhoods, sale_meta = _property_predictions("apart_sale")
    rent, rent_neighborhoods, rent_meta = _property_predictions("apart_rent")
    child, child_meta = _child_predictions()
    crime, crime_meta = _crime_predictions()
    traffic, traffic_meta = _traffic_predictions()

    combined = sale.merge(rent, on="district", how="outer")
    for frame in (child, crime, traffic):
        combined = combined.merge(frame, on="district", how="outer")
    combined = combined.sort_values("district")
    records = [
        {key: _clean(value) for key, value in row.items()}
        for row in combined.to_dict(orient="records")
    ]
    neighborhoods = sale_neighborhoods.merge(
        rent_neighborhoods, on=["district", "neighborhood"], how="outer"
    ).sort_values(["district", "neighborhood"])
    neighborhood_records = [
        {key: _clean(value) for key, value in row.items()}
        for row in neighborhoods.to_dict(orient="records")
    ]
    return {
        "version": "v1",
        "district_count": len(records),
        "models": {
            "apart_sale": sale_meta,
            "apart_rent": rent_meta,
            "child_environment": child_meta,
            "crime_safety": crime_meta,
            "traffic_accident": traffic_meta,
        },
        "warnings": [
            "매매·전세 값은 미래가격이 아니라 최신 거래 특성의 자치구 대표 추정값입니다.",
            "범죄 모델 산출물의 예측연도는 2024년이므로 최신 데이터 갱신 전까지 참고값입니다.",
        ],
        "forecasts": records,
        "neighborhood_count": len(neighborhood_records),
        "neighborhoods": neighborhood_records,
    }


def get_district_forecasts() -> dict:
    # The frontend requests the combined and category endpoints together. Guard
    # the first cache miss so the ~900 MB property bundles are never loaded by
    # several worker threads at the same time.
    with _LOAD_LOCK:
        return _get_district_forecasts_cached()
