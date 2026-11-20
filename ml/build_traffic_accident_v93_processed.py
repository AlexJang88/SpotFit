from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.train_traffic_accident_v44 import prepare_panel
from ml.train_traffic_accident_v73_indicator_ensemble import label


ROOT = Path(__file__).resolve().parents[1]
TRAFFIC_MODELS = next(p for p in (ROOT / "models").iterdir() if p.is_dir() and p.name != "recommender")
MODEL_DIR = next(TRAFFIC_MODELS.glob("93*"))
OUTPUT = ROOT / "data" / "processed" / "교통사고_93차_모델사용데이터_2017_2028.xlsx"


def load_models() -> dict[int, dict]:
    files = sorted(MODEL_DIR.glob("*.joblib"))
    return {2: joblib.load(files[0]), 3: joblib.load(files[1])}


def future_specs(model: dict, horizon: int) -> dict:
    forecasts = model["forecast_models"][2] if horizon == 2 else model["forecast_model"]
    return {feature: forecasts[feature]["spec"] for feature in model["features"]}


def training_pairs(panel: pd.DataFrame, features: list[str], horizon: int) -> pd.DataFrame:
    current = panel[["연도", "자치구", *features]].copy()
    future = panel[["연도", "자치구", *features]].copy()
    future["연도"] = future["연도"] - horizon
    future = future.rename(columns={feature: f"목표_{horizon}년후__{feature}" for feature in features})
    paired = current.merge(future, on=["연도", "자치구"], how="left", validate="one_to_one")
    paired.insert(1, "목표연도", paired["연도"] + horizon)
    return paired[paired["목표연도"] <= 2025].reset_index(drop=True)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    panel = prepare_panel().sort_values(["연도", "자치구"]).reset_index(drop=True)
    models = load_models()
    feature_union = list(dict.fromkeys(models[2]["features"] + models[3]["features"]))

    identifiers = ["연도", "자치구"]
    numeric = [c for c in panel.select_dtypes(include=[np.number]).columns if c not in identifiers]
    full_input = panel[[*identifiers, *numeric]].copy()
    full_input.insert(2, "데이터역할", "미래지표 예측용 정제 패널")

    main_input = panel[["연도", "자치구", *feature_union]].copy()
    main_input.insert(2, "데이터역할", "등급 산출 메인지표")

    weights, methods = [], []
    for horizon, model in models.items():
        specs = future_specs(model, horizon)
        for feature in model["features"]:
            coefficient = float(model["coefficients"][feature])
            weights.append({
                "예측기간": f"{horizon}년 후",
                "지표": feature,
                "Poisson계수": coefficient,
                "가중치_pct": float(model["weights"][feature]) * 100,
                "위험방향": "값이 클수록 위험" if coefficient >= 0 else "값이 작을수록 위험",
            })
            methods.append({
                "예측기간": f"{horizon}년 후",
                "지표": feature,
                "미래값예측법": label(specs[feature]),
            })

    info = json.loads((MODEL_DIR / "model_info.json").read_text(encoding="utf-8"))
    summary = pd.DataFrame([
        {"항목": "모델", "내용": "교통사고 93차 미래지표 예측 최적화"},
        {"항목": "Y", "내용": "다음 연도 어린이 교통사고 사망자 수+부상자 수"},
        {"항목": "등급방향", "내용": "1=가장 안전, 5=가장 위험"},
        {"항목": "2년 메인지표 수", "내용": len(models[2]["features"])},
        {"항목": "3년 메인지표 수", "내용": len(models[3]["features"])},
        {"항목": "정제 패널", "내용": f"{len(panel)}행, {len(panel.columns)}열"},
        {"항목": "검증방식", "내용": "과거 개발연도 선택 후 2025년 독립검증"},
        {"항목": "2년 정확등급", "내용": info["결과"]["2"]["93차_2025독립"]["정확등급정확도"]},
        {"항목": "3년 정확등급", "내용": info["결과"]["3"]["93차_2025독립"]["정확등급정확도"]},
    ])

    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="93차요약", index=False)
        main_input.to_excel(writer, sheet_name="메인지표_2017_2025", index=False)
        full_input.to_excel(writer, sheet_name="전체정제패널", index=False)
        training_pairs(panel, models[2]["features"], 2).to_excel(writer, sheet_name="2년학습쌍", index=False)
        training_pairs(panel, models[3]["features"], 3).to_excel(writer, sheet_name="3년학습쌍", index=False)
        pd.DataFrame(weights).to_excel(writer, sheet_name="지표가중치", index=False)
        pd.DataFrame(methods).to_excel(writer, sheet_name="지표별예측법", index=False)
        comparison = ROOT / "data" / "processed" / "교통사고_91차_92차_93차_비교.xlsx"
        if comparison.exists():
            pd.read_excel(comparison, sheet_name="성능비교").to_excel(
                writer, sheet_name="성능비교", index=False
            )
    print(OUTPUT)
    print(f"rows={len(panel)}, numeric_inputs={len(numeric)}, main_features={len(feature_union)}")


if __name__ == "__main__":
    main()
