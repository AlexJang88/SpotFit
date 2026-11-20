from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.search_traffic_accident_v52_forward_backward import prepare_data
from ml.search_traffic_accident_v52_techniques import risk
from ml.train_traffic_accident_v51_horizon_models import grades_from_risk


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "교통사고" / "51차~60차" / "59차"
OUTPUT = ROOT / "data" / "processed" / "교통사고_59차_최종모델데이터.xlsx"
MODEL_FILES = {
    2: MODEL_DIR / "교통사고_59차_2027년_2년후_모델.joblib",
    3: MODEL_DIR / "교통사고_59차_2028년_3년후_모델.joblib",
}


def components(bundle: dict) -> list[dict]:
    if "component_a" in bundle and "component_b" in bundle:
        return [bundle["component_a"], bundle["component_b"]]
    if "component" in bundle:
        return [bundle["component"]]
    return []


def component_features(component: dict) -> list[str]:
    if "features" in component:
        return list(component["features"])
    nested = []
    for key in ("component", "component_a", "component_b"):
        if key in component:
            nested.extend(component_features(component[key]))
    return list(dict.fromkeys(nested))


def bundle_features(bundle: dict) -> list[str]:
    result = []
    for component in components(bundle):
        result.extend(component_features(component))
    return list(dict.fromkeys(result))


def predict_regression_component(component: dict, frame: pd.DataFrame) -> np.ndarray:
    features = component["features"]
    values = component["model"].predict(frame[features])
    if component.get("target_kind") == "delta":
        values = values + frame["현재연도_종합위험점수"].to_numpy(dtype=float)
    blend = float(component.get("blend", 0.0))
    if blend:
        current = frame["현재연도_종합위험점수"].to_numpy(dtype=float)
        values = (1 - blend) * values + blend * current
    return np.asarray(values, dtype=float)


def predict_bundle(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    kind = bundle.get("kind", "")
    if kind == "classifier_ensemble_rank":
        a = risk(bundle["component_a"]["model"], frame[bundle["component_a"]["features"]])
        b = risk(bundle["component_b"]["model"], frame[bundle["component_b"]["features"]])
        return float(bundle["weight_a"]) * a + (1 - float(bundle["weight_a"])) * b
    if kind == "ensemble_rank":
        a = predict_regression_component(bundle["component_a"], frame)
        b = predict_regression_component(bundle["component_b"], frame)
        return float(bundle["weight_a"]) * a + (1 - float(bundle["weight_a"])) * b
    if kind == "regression_rank":
        return predict_regression_component(bundle, frame)
    raise ValueError(f"지원하지 않는 59차 모델 형식: {kind}")


def feature_table(bundles: dict[int, dict]) -> pd.DataFrame:
    rows = []
    for horizon, bundle in bundles.items():
        for component_index, component in enumerate(components(bundle), 1):
            features = component_features(component)
            config = component.get("config") or component.get("kind") or type(component.get("model")).__name__
            for order, feature in enumerate(features, 1):
                rows.append(
                    {
                        "예측간격": horizon,
                        "예측연도": 2025 + horizon,
                        "구성요소": f"component_{component_index}",
                        "모델설정": config,
                        "앙상블가중치": bundle.get("weight_a") if component_index == 1 else 1 - bundle.get("weight_a", 1),
                        "변수순서": order,
                        "변수": feature,
                    }
                )
    return pd.DataFrame(rows)


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    bundles = {horizon: joblib.load(path) for horizon, path in MODEL_FILES.items()}
    data, _ = prepare_data()
    identifier_columns = ["연도", "자치구"]
    sheets = {}
    forecasts = []

    for horizon, bundle in bundles.items():
        features = bundle_features(bundle)
        target_columns = [
            f"{horizon}년후_종합위험점수",
            f"{horizon}년후_위험등급",
            f"{horizon}년후_목표연도",
        ]
        extra = ["현재연도_종합위험점수", "현재연도_위험등급"]
        columns = list(dict.fromkeys(identifier_columns + extra + features + target_columns))
        missing = [column for column in columns if column not in data.columns]
        if missing:
            raise KeyError(f"{horizon}년 모델 데이터 누락 컬럼: {missing}")
        sheets[horizon] = data[columns].copy()

        future = data[data["연도"] == 2025].copy().sort_values("자치구")
        predicted_risk = predict_bundle(bundle, future)
        predicted_grade = grades_from_risk(predicted_risk)
        forecast = future[["자치구", "연도"]].copy()
        forecast["예측연도"] = 2025 + horizon
        forecast["예측간격"] = horizon
        risk_rank = pd.Series(predicted_risk, index=forecast.index).rank(
            ascending=False, method="first"
        ).astype(int)
        forecast["59차_예측안전점수"] = 120.0 - predicted_risk
        forecast["59차_예측안전등급"] = 6 - predicted_grade
        forecast["59차_예측안전순위"] = 26 - risk_rank
        forecast["내부모델_예측위험점수"] = predicted_risk
        forecasts.append(forecast.sort_values("59차_예측안전순위"))

    final_validation = pd.read_excel(MODEL_DIR / "59차_최종검증결과.xlsx", sheet_name="최종성능")
    info = json.loads((MODEL_DIR / "model_info.json").read_text(encoding="utf-8"))
    info_frame = pd.DataFrame(
        [
            {"항목": "모델명", "내용": info.get("모델명")},
            {"항목": "기반", "내용": info.get("기반")},
            {"항목": "2년후 정확도", "내용": info["2년후"]["exact"]},
            {"항목": "3년후 정확도", "내용": info["3년후"]["exact"]},
            {"항목": "평균 정확도", "내용": info["평균정확등급"]},
            {"항목": "최종 등급 정의", "내용": "안전 1등급=가장 안전, 안전 5등급=가장 위험"},
            {"항목": "내부 변환", "내용": "안전등급=6-내부위험등급, 안전순위=26-내부위험순위"},
            {"항목": "2년후 고유변수수", "내용": len(bundle_features(bundles[2]))},
            {"항목": "3년후 고유변수수", "내용": len(bundle_features(bundles[3]))},
        ]
    )

    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        sheets[2].to_excel(writer, sheet_name="59차_2년모델데이터", index=False)
        sheets[3].to_excel(writer, sheet_name="59차_3년모델데이터", index=False)
        feature_table(bundles).to_excel(writer, sheet_name="59차_모델별변수", index=False)
        final_validation.to_excel(writer, sheet_name="59차_성능요약", index=False)
        pd.concat(forecasts, ignore_index=True).to_excel(
            writer, sheet_name="59차_2027_2028예측", index=False
        )
        info_frame.to_excel(writer, sheet_name="59차_모델설명", index=False)

    result = {
        "output": str(OUTPUT),
        "2년모델": {"행": len(sheets[2]), "열": len(sheets[2].columns), "고유변수": len(bundle_features(bundles[2]))},
        "3년모델": {"행": len(sheets[3]), "열": len(sheets[3].columns), "고유변수": len(bundle_features(bundles[3]))},
        "예측결과행": sum(len(frame) for frame in forecasts),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
