from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.train_best_child_traffic_accident_rate import (
    CURRENT_RATE,
    ROOT,
    TARGET,
    TRAFFIC,
    build_panel,
)


VERSION = 41
ALPHA = 0.1
OUT = ROOT / "models" / "교통사고" / "41차~49차" / f"{VERSION}차"
FEATURES = [
    CURRENT_RATE,
    "어린이피해자율_최근3년평균",
    "어린이피해자율_최근3년변동성",
    "보행어린이교통사고율_어린이1만명당",
    "보행어린이다발_지점수",
    "도로율(%)",
    "도로면적당_전체CCTV",
]


def make_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=ALPHA)),
    ])


def predict_nonnegative(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    return np.maximum(model.predict(frame[FEATURES]), 0)


def metrics(actual: pd.Series | np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual_array = np.asarray(actual, dtype=float)
    error = np.abs(actual_array - predicted)
    return {
        "MAE": float(error.mean()),
        "RMSE": float(np.sqrt(np.mean((actual_array - predicted) ** 2))),
        "R2": float(1 - np.sum((actual_array - predicted) ** 2) / np.sum((actual_array - actual_array.mean()) ** 2)),
        "WMAPE_pct": float(error.sum() / actual_array.sum() * 100),
    }


def evaluate_year(panel: pd.DataFrame, prediction_year: int) -> tuple[dict, pd.DataFrame]:
    train = panel[(panel[TARGET].notna()) & (panel["예측연도"] < prediction_year)].copy()
    test = panel[panel["예측연도"] == prediction_year].copy()
    model = make_model().fit(train[FEATURES], train[TARGET])
    prediction = predict_nonnegative(model, test)
    output = test[["연도", "예측연도", "자치구", TARGET, CURRENT_RATE, *FEATURES[1:]]].copy()
    output["모델예측"] = prediction
    output["절대오차"] = np.abs(output[TARGET] - output["모델예측"])
    output["직전연도유지예측"] = output[CURRENT_RATE]
    return metrics(test[TARGET], prediction), output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel, _, _ = build_panel()
    missing = [feature for feature in FEATURES if feature not in panel.columns]
    if missing:
        raise ValueError(f"{VERSION}차 필수 변수가 없습니다: {missing}")

    validation_2024, output_2024 = evaluate_year(panel, 2024)
    holdout_2025, output_2025 = evaluate_year(panel, 2025)
    baseline_2024 = metrics(output_2024[TARGET], output_2024[CURRENT_RATE].to_numpy())
    baseline_2025 = metrics(output_2025[TARGET], output_2025[CURRENT_RATE].to_numpy())

    labeled = panel[panel[TARGET].notna()].copy()
    future = panel[panel[TARGET].isna()].copy()
    final_model = make_model().fit(labeled[FEATURES], labeled[TARGET])
    forecast = future[["연도", "예측연도", "자치구", CURRENT_RATE, *FEATURES[1:]]].copy()
    forecast["2026년_예측_어린이1만명당_교통사고피해자수"] = predict_nonnegative(final_model, future)

    coefficients = pd.DataFrame({
        "변수": FEATURES,
        "표준화회귀계수": final_model.named_steps["model"].coef_,
    })
    coefficients["절대계수"] = coefficients["표준화회귀계수"].abs()
    coefficients = coefficients.sort_values("절대계수", ascending=False)

    payload = {
        "version": VERSION,
        "model_type": "Ridge",
        "alpha": ALPHA,
        "target": TARGET,
        "target_display_name": "다음 해 어린이 1만 명당 교통사고 피해자 수",
        "target_definition": "(다음 연도 어린이 교통사고 사망자 수 + 부상자 수) / 다음 연도 0~12세 인구 × 10,000",
        "features": FEATURES,
        "model": final_model,
        "horizon_years": 1,
        "selection_validation_year": 2024,
        "final_holdout_year": 2025,
    }
    joblib.dump(payload, OUT / f"교통사고_{VERSION}차_예측모델.joblib")

    comparison = pd.DataFrame([
        {"연도": 2024, "구분": "41차 Ridge", **validation_2024},
        {"연도": 2024, "구분": "직전연도 유지", **baseline_2024},
        {"연도": 2025, "구분": "41차 Ridge", **holdout_2025},
        {"연도": 2025, "구분": "직전연도 유지", **baseline_2025},
    ])
    with pd.ExcelWriter(OUT / "41차_모델평가_및_2026예측.xlsx", engine="openpyxl") as writer:
        comparison.to_excel(writer, sheet_name="성능비교", index=False)
        output_2024.to_excel(writer, sheet_name="2024변수선택검증", index=False)
        output_2025.to_excel(writer, sheet_name="2025독립검증", index=False)
        forecast.to_excel(writer, sheet_name="2026예측", index=False)
        coefficients.to_excel(writer, sheet_name="표준화회귀계수", index=False)
        pd.DataFrame({"번호": range(1, len(FEATURES) + 1), "변수": FEATURES}).to_excel(
            writer, sheet_name="최종7개변수", index=False
        )

    source_path = TRAFFIC / "자치구별_교통사고_통합_2017_2025_최종.csv"
    metadata = {
        "모델명": "교통사고 41차 모델",
        "Y": "다음 해 어린이 1만 명당 교통사고 피해자 수",
        "알고리즘": "Ridge",
        "alpha": ALPHA,
        "최종변수": FEATURES,
        "2024_변수선택검증": validation_2024,
        "2024_직전연도유지기준": baseline_2024,
        "2025_독립검증": holdout_2025,
        "2025_직전연도유지기준": baseline_2025,
        "학습행수": len(labeled),
        "2026_예측행수": len(forecast),
        "data_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    }
    (OUT / "model_info.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "README.md").write_text(
        f"# 교통사고 {VERSION}차 모델\n\n"
        "- Y: 다음 해 어린이 1만 명당 교통사고 피해자 수\n"
        f"- 모델: Ridge(alpha={ALPHA})\n"
        f"- 최종 변수: {len(FEATURES)}개\n"
        f"- 2024 검증 MAE: {validation_2024['MAE']:.4f}\n"
        f"- 2025 독립검증 MAE: {holdout_2025['MAE']:.4f}\n"
        f"- 2025 기준모델 MAE: {baseline_2025['MAE']:.4f}\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
