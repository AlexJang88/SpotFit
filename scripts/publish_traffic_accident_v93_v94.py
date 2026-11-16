"""Publish traffic-accident v93/v94 as recommender/v1 service artifacts.

The layout mirrors ``traffic_accident_v51``:

* ``traffic_accident_2y.joblib``
* ``traffic_accident_3y.joblib``
* ``service_forecast.csv``
* ``model_metadata.json``
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.train_traffic_accident_v71_fixed_rubric import score_frame
from ml.train_traffic_accident_v72_indicator_forecast import deterministic, feature_frame
from ml.train_traffic_accident_v74_all_available import raw_feature_frame
from ml.traffic_accident_pipeline_paths import (
    MODEL_V93_REBUILT,
    MODEL_V94_REBUILT,
    PROCESSED_V93,
    PROCESSED_V94,
)


VERSIONS = {
    93: {
        "source_dir": MODEL_V93_REBUILT,
        "processed": PROCESSED_V93,
        "panel_sheet": "전체정제패널",
        "info_file": "reproduction_report.json",
    },
    94: {
        "source_dir": MODEL_V94_REBUILT,
        "processed": PROCESSED_V94,
        "panel_sheet": "94차전체입력패널",
        "info_file": "model_info.json",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_model(source_dir: Path, horizon: int) -> Path:
    matches = sorted(source_dir.glob(f"*_{horizon}년후_*.joblib"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{source_dir}에서 {horizon}년 후 모델을 하나만 찾을 수 있어야 합니다: {matches}"
        )
    return matches[0]


def model_forecasts(model: dict) -> dict:
    horizon = int(model["horizon"])
    return model["forecast_models"][2] if horizon == 2 else model["forecast_model"]


def frame_for_component(component, old_current: pd.DataFrame, raw_current: pd.DataFrame) -> pd.DataFrame:
    expected = list(getattr(component, "feature_names_in_", []))
    if not expected:
        return raw_current
    old_hits = sum(column in old_current.columns for column in expected)
    raw_hits = sum(column in raw_current.columns for column in expected)
    selected = old_current.copy() if old_hits >= raw_hits else raw_current.copy()
    for column in expected:
        if column not in selected.columns:
            selected[column] = np.nan
    return selected[expected]


def predict_future(panel: pd.DataFrame, model: dict) -> pd.DataFrame:
    horizon = int(model["horizon"])
    features = list(model["features"])
    forecasts = model_forecasts(model)
    old_current = feature_frame(panel, features)
    old_current = old_current[old_current["연도"].eq(2025)].sort_values("자치구").reset_index(drop=True)

    raw_current, _, _ = raw_feature_frame(panel, features)
    keys = panel.sort_values(["자치구", "연도"])[["연도", "자치구"]].reset_index(drop=True)
    raw_current = raw_current.reset_index(drop=True)
    raw_current.insert(1, "자치구", keys["자치구"])
    raw_current = raw_current[raw_current["연도"].eq(2025)].sort_values("자치구").reset_index(drop=True)

    history = panel[panel["연도"].le(2025)]
    output = pd.DataFrame({"연도": 2025 + horizon, "자치구": old_current["자치구"]})
    for feature in features:
        item = forecasts[feature]
        spec = item["spec"]
        component_names = [spec] if isinstance(spec, str) else [spec[0], spec[1]]
        predictions = {}
        for name in dict.fromkeys(component_names):
            component = item["components"][name]
            if isinstance(component, str):
                predictions[name] = np.asarray(
                    [
                        deterministic(
                            history[history["자치구"].eq(district)].sort_values("연도")[feature],
                            component,
                        )
                        for district in output["자치구"]
                    ],
                    dtype=float,
                )
            else:
                values = component.predict(frame_for_component(component, old_current, raw_current))
                observed = panel[feature].dropna().astype(float)
                predictions[name] = np.clip(values, observed.quantile(0.01), observed.quantile(0.99))
        if isinstance(spec, str):
            output[feature] = predictions[spec]
        else:
            output[feature] = predictions[spec[0]] * spec[2] + predictions[spec[1]] * (1 - spec[2])
    return output


def scored_service_rows(panel: pd.DataFrame, model: dict) -> pd.DataFrame:
    features = list(model["features"])
    coefficients = np.asarray([model["coefficients"][feature] for feature in features], dtype=float)
    weights = np.asarray([model["weights"][feature] for feature in features], dtype=float)
    base = panel[panel["연도"].eq(2025)][["연도", "자치구", *features]].copy()
    predicted = predict_future(panel, model)
    base_score = score_frame(base, features, coefficients, weights, "base_")
    predicted_score = score_frame(predicted, features, coefficients, weights, "predicted_")
    service = base_score[["자치구", "base_종합위험점수", "base_안전등급"]].merge(
        predicted_score[
            ["연도", "자치구", "predicted_종합위험점수", "predicted_안전등급"]
        ],
        on="자치구",
        validate="one_to_one",
    )
    service = service.rename(
        columns={
            "자치구": "district",
            "연도": "prediction_year",
            "base_종합위험점수": "base_risk_score",
            "base_안전등급": "base_safety_grade",
            "predicted_종합위험점수": "predicted_risk_score",
            "predicted_안전등급": "predicted_safety_grade",
        }
    )
    service["base_safety_score"] = 120.0 - service["base_risk_score"]
    service["predicted_safety_score"] = 120.0 - service["predicted_risk_score"]
    service["predicted_safety_rank"] = (
        service["predicted_risk_score"].rank(method="first", ascending=True).astype(int)
    )
    return service.sort_values(["prediction_year", "district"]).reset_index(drop=True)


def publish(version: int) -> Path:
    config = VERSIONS[version]
    source_dir = config["source_dir"]
    processed = config["processed"]
    service_dir = ROOT / f"models/recommender/v1/traffic_accident_v{version}"
    service_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_excel(processed, sheet_name=config["panel_sheet"])
    panel = panel.drop(columns=["데이터역할"], errors="ignore")
    panel = panel.sort_values(["연도", "자치구"]).reset_index(drop=True)

    service_frames = []
    artifacts = {}
    for horizon in (2, 3):
        source = source_model(source_dir, horizon)
        destination = service_dir / f"traffic_accident_{horizon}y.joblib"
        shutil.copy2(source, destination)
        model = joblib.load(destination)
        if int(model.get("version", -1)) != version or int(model.get("horizon", -1)) != horizon:
            raise ValueError(f"모델 메타데이터 불일치: {destination}")
        forecast = scored_service_rows(panel, model)
        forecast.insert(1, "base_year", 2025)
        forecast.insert(3, "horizon_years", horizon)
        service_frames.append(forecast)
        artifacts[str(horizon)] = {
            "file": destination.name,
            "sha256": sha256(destination),
            "bytes": destination.stat().st_size,
            "feature_count": len(model["features"]),
        }

    service = pd.concat(service_frames, ignore_index=True)
    if len(service) != 50 or service.groupby("prediction_year")["district"].nunique().to_dict() != {
        2027: 25,
        2028: 25,
    }:
        raise ValueError(f"v{version} 서비스 예측이 2027·2028년 각 25개 자치구가 아닙니다.")
    service.to_csv(service_dir / "service_forecast.csv", index=False, encoding="utf-8-sig")

    info_path = source_dir / config["info_file"]
    source_info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.is_file() else {}
    metadata = {
        "version": "v1",
        "source_model": f"교통사고 {version}차 모델",
        "temporary_model": False,
        "base_year": 2025,
        "prediction_years": [2027, 2028],
        "horizons": [2, 3],
        "grade_definition": "1등급 가장 안전, 5등급 가장 위험",
        "display_orientation": "safety",
        "internal_orientation": "safety (1=safest, 5=most risky)",
        "processed_file": str(processed.relative_to(ROOT)),
        "artifacts": artifacts,
        "source_info": source_info,
    }
    (service_dir / "model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return service_dir


def main() -> None:
    for version in (93, 94):
        output = publish(version)
        print(f"published={output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
