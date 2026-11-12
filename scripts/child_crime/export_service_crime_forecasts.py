"""Export reproducible 1y/2y child crime-exposure forecasts for the API."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SERVICE = ROOT / "models" / "recommender" / "v1" / "child_crime_exposure"
TWO_YEAR_EXPERIMENT = ROOT / "models" / "experiments" / "child_crime_exposure_2y_2027_v1"
TWO_YEAR_ARTIFACT = ROOT / "artifacts" / "child_crime_exposure_2y_2027_v1"
THREE_YEAR_EXPERIMENT = ROOT / "models" / "experiments" / "child_crime_exposure_integrated_v2"
THREE_YEAR_ARTIFACT = ROOT / "artifacts" / "child_crime_exposure_integrated_v2"


def add_service_columns(frame: pd.DataFrame, prediction: str) -> pd.DataFrame:
    result = frame.copy()
    result["predicted_exposure_rate_per_10k"] = np.expm1(result[prediction])
    result["risk_rank"] = result[prediction].rank(ascending=False, method="min").astype(int)
    result["safety_rank"] = result[prediction].rank(ascending=True, method="min").astype(int)
    result["safety_score"] = 100 - ((result["safety_rank"] - 1) // 5) * 20
    result["safety_grade"] = result["safety_rank"].map(lambda rank: "ABCDE"[(rank - 1) // 5])
    return result.sort_values("risk_rank")


def export_one_year() -> None:
    target = SERVICE / "1y"
    bundle_path = target / "child_crime_exposure_1y.joblib"
    joblib.load(bundle_path)
    raw_path = target / "raw_forecast.csv"
    if not raw_path.is_file():
        raise FileNotFoundError("1년 서비스 예측 산출물이 없습니다. train_service_crime_exposure_1y.py를 먼저 실행하세요.")
    frame = pd.read_csv(raw_path, encoding="utf-8-sig")
    add_service_columns(frame, "predicted_y").to_csv(target / "service_forecast.csv", index=False, encoding="utf-8-sig")


def export_two_year() -> None:
    target = SERVICE / "2y"
    target.mkdir(parents=True, exist_ok=True)
    source_model = TWO_YEAR_EXPERIMENT / "operational_exposure_2y_2027_strict.joblib"
    source_forecast = TWO_YEAR_ARTIFACT / "forecast_2027.csv"
    source_metadata = TWO_YEAR_ARTIFACT / "result.json"
    if not source_model.is_file() or not source_forecast.is_file() or not source_metadata.is_file():
        raise FileNotFoundError("2년 범죄노출 모델 실험 산출물이 없습니다. optimize_exposure_2y_2027.py를 먼저 실행하세요.")
    shutil.copy2(source_model, target / "child_crime_exposure_2y.joblib")
    shutil.copy2(source_metadata, target / "model_metadata.json")
    raw = pd.read_csv(source_forecast, encoding="utf-8-sig")
    frame = raw.rename(columns={"gu": "district", "year": "base_year", "predicted_y_2027": "predicted_y"})
    frame = frame[["district", "base_year", "target_year", "predicted_y"]].rename(columns={"target_year": "prediction_year"})
    add_service_columns(frame, "predicted_y").to_csv(target / "service_forecast.csv", index=False, encoding="utf-8-sig")


def export_three_year() -> None:
    target = SERVICE / "3y"
    target.mkdir(parents=True, exist_ok=True)
    source_model = THREE_YEAR_EXPERIMENT / "operational_exposure_integrated_blend_3y.joblib"
    source_forecast = THREE_YEAR_ARTIFACT / "forecast_2028_blended.csv"
    source_metadata = THREE_YEAR_ARTIFACT / "result.json"
    if not all(path.is_file() for path in (source_model, source_forecast, source_metadata)):
        raise FileNotFoundError("3년 범죄노출 앙상블 산출물이 없습니다.")
    shutil.copy2(source_model, target / "child_crime_exposure_3y.joblib")
    shutil.copy2(source_metadata, target / "model_metadata.json")
    raw = pd.read_csv(source_forecast, encoding="utf-8-sig")
    frame = raw.rename(columns={"gu": "district", "blended_y_2028": "predicted_y"})
    frame["base_year"], frame["prediction_year"] = 2025, 2028
    frame = frame[["district", "base_year", "prediction_year", "predicted_y"]]
    add_service_columns(frame, "predicted_y").to_csv(target / "service_forecast.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    export_one_year()
    export_two_year()
    export_three_year()
    manifest = {
        "service": "child_regional_crime_exposure",
        "target_formula": "log(1 + 5대범죄 발생건수 / 0~17세 아동인구 * 10,000)",
        "horizons": [1, 2, 3],
        "score_rule": "서울 25개 자치구 위험도 오름차순 5개 구씩 A=100, B=80, C=60, D=40, E=20",
    }
    (SERVICE / "service_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
