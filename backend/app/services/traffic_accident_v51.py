"""Temporary service adapter for the presentation-time traffic model v51.

The final traffic model is not locked yet.  Keep every v51-specific path in this
module so replacing it later does not require changes across the API.
"""

from __future__ import annotations

import json
import os
import shutil
from functools import lru_cache
from pathlib import Path
from threading import Lock

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SERVICE_DIR = ROOT / "models" / "recommender" / "v1" / "traffic_accident_v51"
SOURCE_MODEL_DIR = Path(
    os.getenv("TRAFFIC_V51_SOURCE_MODEL_DIR", ROOT / "models" / "교통사고" / "51차")
)
SOURCE_PROCESSED = Path(
    os.getenv(
        "TRAFFIC_V51_PROCESSED_PATH",
        ROOT / "data" / "processed" / "교통사고_51차_최종모델데이터.xlsx",
    )
)
MODEL_FILES = {
    2: "traffic_accident_2y.joblib",
    3: "traffic_accident_3y.joblib",
}
SOURCE_MODEL_FILES = {
    2: "교통사고_51차_2027년_2년후_직접예측모델.joblib",
    3: "교통사고_51차_2028년_3년후_직접예측모델.joblib",
}
FORECAST_FILE = "service_forecast.csv"
METADATA_FILE = "model_metadata.json"
_ENSURE_LOCK = Lock()


class TrafficAccidentV51Error(RuntimeError):
    pass


def required_paths() -> list[Path]:
    return [
        *(SERVICE_DIR / name for name in MODEL_FILES.values()),
        SERVICE_DIR / FORECAST_FILE,
        SERVICE_DIR / METADATA_FILE,
    ]


def model_status() -> dict:
    paths = required_paths()
    horizons = []
    for horizon, name in MODEL_FILES.items():
        path = SERVICE_DIR / name
        horizons.append(
            {
                "horizon_years": horizon,
                "prediction_year": 2025 + horizon,
                "ready": path.is_file(),
                "model_path": str(path),
            }
        )
    missing = [str(path) for path in paths if not path.is_file()]
    return {
        "service": "traffic_accident_v51_temporary",
        "temporary_model": True,
        "ready": not missing,
        "service_dir": str(SERVICE_DIR),
        "missing": missing,
        "horizons": horizons,
    }


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def ensure_models(force: bool = False) -> dict:
    """Create recommender/v1 service artifacts from the current v51 outputs."""
    with _ENSURE_LOCK:
        if not force and model_status()["ready"]:
            return {**model_status(), "created": False}

        source_paths = {
            horizon: SOURCE_MODEL_DIR / name
            for horizon, name in SOURCE_MODEL_FILES.items()
        }
        missing_sources = [str(path) for path in source_paths.values() if not path.is_file()]
        if not SOURCE_PROCESSED.is_file():
            missing_sources.append(str(SOURCE_PROCESSED))
        if missing_sources:
            raise TrafficAccidentV51Error(
                "51차 교통사고 서비스 모델을 생성할 원본이 없습니다: "
                + ", ".join(missing_sources)
            )

        SERVICE_DIR.mkdir(parents=True, exist_ok=True)
        for horizon, source in source_paths.items():
            _atomic_copy(source, SERVICE_DIR / MODEL_FILES[horizon])

        frame = pd.read_excel(SOURCE_PROCESSED, sheet_name="51차_2027_2028예측")
        expected = {
            "자치구",
            "현재연도_종합위험점수",
            "현재연도_위험등급",
            "예측연도",
            "예측위험점수",
            "예측위험등급",
            "예측위험순위",
        }
        missing_columns = sorted(expected.difference(frame.columns))
        if missing_columns:
            raise TrafficAccidentV51Error(
                "51차 예측 시트에 필요한 컬럼이 없습니다: " + ", ".join(missing_columns)
            )
        if len(frame) != 50 or frame.groupby("예측연도")["자치구"].nunique().to_dict() != {2027: 25, 2028: 25}:
            raise TrafficAccidentV51Error("51차 예측 결과가 2027·2028년 각 25개 자치구 구조가 아닙니다.")

        service = frame.rename(
            columns={
                "자치구": "district",
                "현재연도_종합위험점수": "base_risk_score",
                "현재연도_위험등급": "base_risk_grade",
                "예측연도": "prediction_year",
                "예측위험점수": "predicted_risk_score",
                "예측위험등급": "predicted_risk_grade",
                "예측위험순위": "predicted_risk_rank",
            }
        )
        forecast_path = SERVICE_DIR / FORECAST_FILE
        forecast_tmp = forecast_path.with_suffix(".csv.tmp")
        service.to_csv(forecast_tmp, index=False, encoding="utf-8-sig")
        os.replace(forecast_tmp, forecast_path)

        source_info_path = SOURCE_MODEL_DIR / "model_info.json"
        source_info = (
            json.loads(source_info_path.read_text(encoding="utf-8"))
            if source_info_path.is_file()
            else {}
        )
        metadata = {
            "version": "v1",
            "source_model": "교통사고 51차 임시 발표 모델",
            "temporary_model": True,
            "base_year": 2025,
            "prediction_years": [2027, 2028],
            "horizons": [2, 3],
            "grade_definition": "1등급 고위험, 5등급 저위험",
            "source_info": source_info,
        }
        metadata_path = SERVICE_DIR / METADATA_FILE
        metadata_tmp = metadata_path.with_suffix(".json.tmp")
        metadata_tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(metadata_tmp, metadata_path)
        clear_forecast_cache()
        return {**model_status(), "created": True}


@lru_cache(maxsize=2)
def forecast_payload(horizon: int) -> dict:
    if horizon not in MODEL_FILES:
        raise TrafficAccidentV51Error("교통사고 51차는 2년 후 또는 3년 후 예측만 지원합니다.")
    if not model_status()["ready"]:
        ensure_models()
    frame = pd.read_csv(SERVICE_DIR / FORECAST_FILE, encoding="utf-8-sig")
    year = 2025 + horizon
    selected = frame.loc[frame["prediction_year"].eq(year)].sort_values("predicted_risk_rank")
    metadata = json.loads((SERVICE_DIR / METADATA_FILE).read_text(encoding="utf-8"))
    rows = [
        {
            "district": row.district,
            "base_year": 2025,
            "prediction_year": int(row.prediction_year),
            "horizon_years": horizon,
            "predicted_risk_score": round(float(row.predicted_risk_score), 6),
            "predicted_risk_rank": int(row.predicted_risk_rank),
            "predicted_risk_grade": int(row.predicted_risk_grade),
        }
        for row in selected.itertuples(index=False)
    ]
    return {
        "model_name": f"traffic_accident_v51_{horizon}y",
        "model_version": "v1",
        "temporary_model": True,
        "base_year": 2025,
        "prediction_year": year,
        "horizon_years": horizon,
        "district_count": len(rows),
        "grade_definition": metadata["grade_definition"],
        "forecasts": rows,
    }


def clear_forecast_cache() -> None:
    forecast_payload.cache_clear()
