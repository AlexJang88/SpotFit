from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = ROOT / "models" / "recommender" / "v1" / "child_crime_exposure"
GRADE_DEFINITION = {"A": "노출위험 하위 20%", "B": "노출위험 하위 20~40%", "C": "중간 20%", "D": "노출위험 상위 20~40%", "E": "노출위험 상위 20%"}


class ChildCrimeExposureError(RuntimeError):
    pass


def required_paths(horizon: int) -> tuple[Path, Path, Path]:
    directory = MODEL_ROOT / f"{horizon}y"
    return directory / f"child_crime_exposure_{horizon}y.joblib", directory / "service_forecast.csv", directory / "model_metadata.json"


def model_status() -> dict:
    horizons = []
    for horizon in (1, 2):
        model, forecast, metadata = required_paths(horizon)
        horizons.append({"horizon_years": horizon, "ready": all(path.is_file() for path in (model, forecast, metadata)), "model_path": str(model), "forecast_path": str(forecast), "missing": [str(path) for path in (model, forecast, metadata) if not path.is_file()]})
    return {"service": "child_regional_crime_exposure", "ready": all(item["ready"] for item in horizons), "horizons": horizons}


@lru_cache(maxsize=2)
def forecast_payload(horizon: int) -> dict:
    if horizon not in (1, 2):
        raise ChildCrimeExposureError("예측기간은 1년 또는 2년만 지원합니다.")
    model, forecast, metadata_path = required_paths(horizon)
    missing = [path for path in (model, forecast, metadata_path) if not path.is_file()]
    if missing:
        raise ChildCrimeExposureError(f"범죄노출 모델 산출물이 없습니다: {', '.join(map(str, missing))}")
    frame = pd.read_csv(forecast, encoding="utf-8-sig")
    if len(frame) != 25 or frame.district.nunique() != 25:
        raise ChildCrimeExposureError("범죄노출 예측 결과가 서울 25개 자치구 구조가 아닙니다.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = []
    for row in frame.sort_values("safety_rank").itertuples(index=False):
        rows.append({"district": row.district, "data_year": int(row.base_year), "prediction_year": int(row.prediction_year), "predicted_score": float(row.safety_score), "predicted_rank": int(row.safety_rank), "predicted_grade": row.safety_grade, "previous_rank": int(row.safety_rank), "previous_grade": row.safety_grade, "grade_change_expected": False, "grade_change_direction": "유지", "predicted_exposure_y": round(float(row.predicted_y), 6), "predicted_exposure_rate_per_10k": round(float(row.predicted_exposure_rate_per_10k), 4), "risk_rank": int(row.risk_rank)})
    winner = metadata.get("winner", metadata.get("strict_winner_cv", {}))
    return {"model_name": f"child_crime_exposure_{horizon}y", "model_version": "v1", "algorithm": f"Ridge(alpha={winner.get('alpha', 10)})", "target": "아동인구 대비 5대범죄 지역노출지수", "target_formula": "log(1 + 5대범죄 발생건수 / 0~17세 아동인구 × 10,000)", "horizon_years": horizon, "data_year": int(frame.base_year.max()), "prediction_year": int(frame.prediction_year.max()), "district_count": len(rows), "grade_definition": GRADE_DEFINITION, "warning": "실제 아동 대상 범죄율이 아니라 지역의 5대범죄를 아동인구로 표준화한 상대 노출지수입니다.", "metrics": winner, "forecasts": rows}


def clear_forecast_cache() -> None:
    forecast_payload.cache_clear()
