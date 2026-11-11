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
MODEL_DIR = ROOT / "models" / "교통사고" / "56차"
OUTPUT = ROOT / "data" / "processed" / "교통사고_56차_최종모델데이터.xlsx"
MODEL_PATHS = {
    2: MODEL_DIR / "교통사고_56차_2027년_2년후_모델.joblib",
    3: MODEL_DIR / "교통사고_56차_2028년_3년후_모델.joblib",
}


def components(bundle: dict) -> list[dict]:
    if bundle["kind"] in {"classifier_rank", "classifier_direct"}:
        return [bundle["component"]]
    return [bundle["component_a"], bundle["component_b"]]


def model_risk(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    values = [risk(component["model"], frame[component["features"]]) for component in components(bundle)]
    if len(values) == 1:
        return values[0]
    return bundle["weight_a"] * values[0] + (1 - bundle["weight_a"]) * values[1]


def feature_table(bundles: dict[int, dict]) -> pd.DataFrame:
    rows = []
    for horizon, bundle in bundles.items():
        for index, component in enumerate(components(bundle), start=1):
            for order, feature in enumerate(component["features"], start=1):
                rows.append(
                    {
                        "예측간격": horizon,
                        "예측연도": 2025 + horizon,
                        "구성요소": f"component_{index}",
                        "모델설정": component.get("config", ""),
                        "앙상블가중치": (
                            bundle.get("weight_a", 1.0)
                            if index == 1
                            else 1 - bundle.get("weight_a", 1.0)
                        ),
                        "변수순서": order,
                        "변수": feature,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    data, _ = prepare_data()
    bundles = {horizon: joblib.load(path) for horizon, path in MODEL_PATHS.items()}
    identifiers = ["연도", "자치구"]
    sheets = {}
    forecasts = []

    for horizon, bundle in bundles.items():
        target_columns = [
            f"{horizon}년후_목표연도",
            f"{horizon}년후_종합위험점수",
            f"{horizon}년후_위험등급",
        ]
        features = list(dict.fromkeys(feature for component in components(bundle) for feature in component["features"]))
        model_data = data[[*identifiers, *target_columns, *features]].copy()
        model_data.insert(2, "예측간격_년", horizon)
        model_data.insert(3, "운영예측연도", 2025 + horizon)
        model_data["데이터구분"] = np.where(
            model_data[f"{horizon}년후_위험등급"].notna(),
            "학습·순차검증 가능",
            np.where(model_data["연도"].eq(2025), f"{2025+horizon}년 운영예측 입력", "목표값 미확보"),
        )
        sheets[horizon] = model_data

        future = data[data["연도"].eq(2025)].copy()
        predicted_risk = model_risk(bundle, future)
        predicted_grade = grades_from_risk(predicted_risk)
        forecast = future[["자치구", "현재연도_종합위험점수", "현재연도_위험등급"]].copy()
        forecast.insert(0, "기준연도", 2025)
        forecast.insert(1, "예측연도", 2025 + horizon)
        forecast.insert(2, "예측간격_년", horizon)
        forecast["56차_예측위험점수"] = predicted_risk
        forecast["56차_예측위험등급"] = predicted_grade
        forecast["56차_예측위험순위"] = pd.Series(predicted_risk).rank(
            ascending=False, method="first"
        ).astype(int).to_numpy()
        forecasts.append(forecast.sort_values("56차_예측위험순위"))

    validation = pd.read_excel(MODEL_DIR / "56차_검증결과.xlsx", sheet_name="연도별검증")
    summary = validation.groupby("예측간격", as_index=False).agg(
        정확등급정확도=("정확등급정확도", "mean"),
        평균등급오차=("평균등급오차", "mean"),
        최저연도정확도=("정확등급정확도", "min"),
    )
    within = validation.groupby("예측간격")["±1등급정확도"].mean().rename("±1등급정확도")
    summary = summary.merge(within, on="예측간격")

    info = json.loads((MODEL_DIR / "model_info.json").read_text(encoding="utf-8"))
    model_info = pd.DataFrame(
        [
            ["모델명", info["모델명"]],
            ["기준연도", 2025],
            ["2년 후 예측연도", 2027],
            ["3년 후 예측연도", 2028],
            ["등급체계", "서울 25개 자치구 상대 5분위 · 1등급 고위험 · 5등급 저위험"],
            ["2년 후 제거변수", ", ".join(info.get("2년후제거", []))],
            ["3년 후 제거변수", ", ".join(info.get("3년후제거", [])) or "없음"],
            ["주의", "현재 확보 데이터의 내부 순차검증 결과이며 새 실제연도 독립검증 필요"],
        ],
        columns=["항목", "내용"],
    )

    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        sheets[2].to_excel(writer, sheet_name="56차_2년모델데이터", index=False)
        sheets[3].to_excel(writer, sheet_name="56차_3년모델데이터", index=False)
        feature_table(bundles).to_excel(writer, sheet_name="56차_모델별변수", index=False)
        summary.to_excel(writer, sheet_name="56차_성능요약", index=False)
        validation.to_excel(writer, sheet_name="56차_연도별검증", index=False)
        pd.concat(forecasts, ignore_index=True).to_excel(
            writer, sheet_name="56차_2027_2028예측", index=False
        )
        model_info.to_excel(writer, sheet_name="56차_모델설명", index=False)

    result = {
        "output": str(OUTPUT),
        "2년모델": {"행": len(sheets[2]), "컬럼": len(sheets[2].columns)},
        "3년모델": {"행": len(sheets[3]), "컬럼": len(sheets[3].columns)},
        "예측결과행": sum(len(frame) for frame in forecasts),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
