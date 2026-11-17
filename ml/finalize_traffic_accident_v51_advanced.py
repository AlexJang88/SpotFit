from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.improve_traffic_accident_v51_advanced import DATA, MODEL, MODEL_DIR, REPORT, add_features, grades_from_risk


def component_risk(component: dict, frame: pd.DataFrame) -> np.ndarray:
    if component["kind"] == "anchor":
        return frame[component["column"]].to_numpy()
    risk = component["model"].predict(frame[component["features"]])
    current = frame["현재연도_종합위험점수"].to_numpy()
    if component["target_kind"] == "delta":
        risk = current + risk
    return (1 - component["blend"]) * risk + component["blend"] * current


def main() -> None:
    raw = pd.read_excel(DATA, sheet_name="51차점수등급데이터")
    data, _ = add_features(raw)
    future = data[data["예측연도"] == 2026].copy()
    bundle = joblib.load(MODEL)
    if bundle["kind"] == "ensemble_rank":
        a = component_risk(bundle["component_a"], future)
        b = component_risk(bundle["component_b"], future)
        risk = bundle["weight_a"] * a + (1 - bundle["weight_a"]) * b
    else:
        risk = component_risk(bundle, future)
    future["2026_고도화_예측위험점수"] = risk
    future["2026_고도화_예측위험등급"] = grades_from_risk(risk)
    future["2026_고도화_위험순위"] = pd.Series(risk).rank(ascending=False, method="first").astype(int).to_numpy()
    forecast = future[["자치구", "현재연도_종합위험점수", "현재연도_위험등급", "2026_고도화_예측위험점수", "2026_고도화_예측위험등급", "2026_고도화_위험순위"]].sort_values("2026_고도화_위험순위")
    comparison = pd.DataFrame([
        ["기존 51차", "Linear SVM C=10 + 순위보정", 56.0, 92.8, .512],
        ["고도화 51차", "GBR 점수예측 + GBR 변화량예측 앙상블", 60.0, 90.4, .496],
    ], columns=["모델", "기법", "정확등급정확도_pct", "±1등급정확도_pct", "평균등급오차"])
    with pd.ExcelWriter(REPORT, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        comparison.to_excel(writer, sheet_name="기존대비개선", index=False)
        forecast.to_excel(writer, sheet_name="2026고도화예측", index=False)
    with pd.ExcelWriter(DATA, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        comparison.to_excel(writer, sheet_name="51차고도화성능", index=False)
        forecast.to_excel(writer, sheet_name="고도화2026등급예측", index=False)
    info_path = MODEL_DIR / "model_info.json"
    info = {
        "모델명": "교통사고 51차 가중치 5분위 점수·등급 예측모델",
        "기반모델": "49차 어린이 교통사고 피해자율 예측모델",
        "등급체계": "서울 25개 자치구 상대순위 5분위, 1등급 고위험·5등급 저위험",
        "가중치": "49차 Ridge 표준화계수 절댓값을 합계 100%로 환산",
        "점수예측모델": "Ridge alpha=10",
        "개선등급예측모델": "선형 SVM C=10 + 순위보정",
        "2021_2025_평균_정확등급정확도_pct": 56.0,
        "2021_2025_평균_1칸이내정확도_pct": 92.8,
        "평균등급오차": 0.512,
        "원천변수수": 13,
        "상태": "공식 등급예측 모델",
        "고도화모델": "ENS[GBR_d1|score|b0.4@all]+[GBR_d3|delta|b0@all_district]|w0.75",
        "고도화변수세트": "ensemble",
        "고도화_2021_2025_평균_정확등급정확도_pct": 60.0,
        "고도화_2021_2025_평균_1칸이내정확도_pct": 90.4,
        "고도화_평균등급오차": 0.496,
    }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False))
    print(forecast.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
