from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.train_traffic_accident_v72_indicator_forecast import build_prediction_cache
from ml.train_traffic_accident_v73_indicator_ensemble import evaluate_specs
from ml.train_traffic_accident_v74_all_available import add_expanded_cache, fit_forecast, supervised
from ml.train_traffic_accident_v83_poisson_exposure_grade import (
    POP,
    TARGET_COUNT,
    TARGET_POP,
    VICTIMS,
    fit_exposure_model,
)


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = next((ROOT / "data" / "processed").glob("*.xlsx"))
TRAFFIC_MODELS = next(
    p for p in (ROOT / "models").iterdir() if p.is_dir() and p.name != "recommender"
)
SOURCE = next((p for p in TRAFFIC_MODELS.glob("93*") if "Processed" not in p.name), None)
OUT = TRAFFIC_MODELS / "93차.Processed단독재현검증"


def parse_spec(value: str):
    value = str(value).strip()
    match = re.fullmatch(r"(.+)×([0-9.]+)\+(.+)×([0-9.]+)", value)
    if not match:
        return value
    first, first_weight, second, second_weight = match.groups()
    weight = float(first_weight)
    if not np.isclose(weight + float(second_weight), 1.0):
        raise ValueError(f"앙상블 비율 합이 1이 아닙니다: {value}")
    return first, second, weight


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 시트명 변경에도 견디도록 생성 당시 고정된 시트 순서를 사용한다.
    panel = pd.read_excel(PROCESSED, sheet_name=2)
    panel = panel.drop(columns=["데이터역할"], errors="ignore")
    # 기존 93차 학습과 동일한 순서를 유지해야 raw_feature_frame의 키 정렬과 일치한다.
    panel = panel.sort_values(["연도", "자치구"]).reset_index(drop=True)
    weights = pd.read_excel(PROCESSED, sheet_name=5)
    methods = pd.read_excel(PROCESSED, sheet_name=6)
    return panel, weights, methods


def load_original(horizon: int) -> dict:
    if SOURCE is None:
        raise FileNotFoundError("비교할 기존 93차 모델이 없습니다.")
    files = sorted(SOURCE.glob("*.joblib"))
    return joblib.load(files[horizon - 2])


def model_config(weights: pd.DataFrame, methods: pd.DataFrame, horizon: int):
    period = f"{horizon}년 후"
    w = weights[weights["예측기간"] == period].reset_index(drop=True)
    m = methods[methods["예측기간"] == period].set_index("지표")
    features = w["지표"].tolist()
    specs = {feature: parse_spec(m.loc[feature, "미래값예측법"]) for feature in features}
    recorded_coef = dict(zip(features, w["Poisson계수"].astype(float)))
    recorded_weights = dict(zip(features, w["가중치_pct"].astype(float) / 100))
    return features, specs, recorded_coef, recorded_weights


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel, weight_table, method_table = load_processed()
    # 다음 연도 정답 컬럼은 Poisson 적합에만 필요하다. 미래지표 회귀 입력에
    # 들어가면 기존 93차와 달라지고 목표 누수가 생기므로 별도 복사본에만 만든다.
    poisson_panel = panel.copy()
    poisson_group = poisson_panel.groupby("자치구", sort=False)
    poisson_panel[TARGET_COUNT] = poisson_group[VICTIMS].shift(-1)
    poisson_panel[TARGET_POP] = poisson_group[POP].shift(-1)
    labeled = poisson_panel[poisson_panel[TARGET_COUNT].notna()].copy()

    results = {}
    for horizon in (2, 3):
        features, specs, recorded_coef, recorded_weights = model_config(
            weight_table, method_table, horizon
        )
        fitted = fit_exposure_model(labeled, features, alpha=0.0)
        coefficients = np.asarray(fitted.named_steps["model"].coef_, dtype=float)
        total = np.abs(coefficients).sum()
        weights = np.abs(coefficients) / total

        expanded, predictor_cols, pool = supervised(panel, features, horizon)
        predictor_cols = [
            c
            for c in expanded.columns
            if c not in {"연도", "자치구", "목표연도"} and not c.startswith("목표__")
        ]
        from ml.train_traffic_accident_v74_all_available import candidates

        pool = candidates(len(predictor_cols))
        _, forecast_bundle = fit_forecast(
            panel, expanded, predictor_cols, pool, features, horizon, specs
        )

        rebuilt = {
            "version": 93,
            "horizon": horizon,
            "source": "processed_only",
            "processed_file": PROCESSED.name,
            "features": features,
            "poisson_model": fitted,
            "coefficients": dict(zip(features, coefficients)),
            "weights": dict(zip(features, weights)),
            "source_version": 92,
            "safety_grade_orientation": "1=가장 안전, 5=가장 위험",
        }
        if horizon == 2:
            rebuilt["forecast_models"] = {2: forecast_bundle}
        else:
            rebuilt["forecast_model"] = forecast_bundle
        output = OUT / f"교통사고_93차_{horizon}년후_Processed단독재현.joblib"
        joblib.dump(rebuilt, output)

        workbook_coef = np.array([recorded_coef[f] for f in features])
        workbook_weights = np.array([recorded_weights[f] for f in features])

        # 정확도도 Raw/Interim 없이 Processed 패널에서 독립 재계산한다.
        _, methods, cache, _, _ = build_prediction_cache(panel, features, horizon)
        _, _, _, methods, cache = add_expanded_cache(
            panel, features, horizon, cache, methods
        )
        metrics, _, _ = evaluate_specs(
            panel,
            features,
            coefficients,
            weights,
            horizon,
            [2025],
            specs,
            cache,
        )

        result = {
            "지표수": len(features),
            "재학습계수_vs_엑셀_max_abs_diff": float(np.max(np.abs(coefficients - workbook_coef))),
            "재학습가중치_vs_엑셀_max_abs_diff": float(np.max(np.abs(weights - workbook_weights))),
            "2025독립성능": metrics,
            "모델파일": output.name,
        }
        if SOURCE is not None:
            original = load_original(horizon)
            original_coef = np.array([original["coefficients"][f] for f in features])
            original_weights = np.array([original["weights"][f] for f in features])
            result.update({
                "지표순서동일": features == list(original["features"]),
                "재학습계수_vs_기존_max_abs_diff": float(np.max(np.abs(coefficients - original_coef))),
                "엑셀계수_vs_기존_max_abs_diff": float(np.max(np.abs(workbook_coef - original_coef))),
                "재학습가중치_vs_기존_max_abs_diff": float(np.max(np.abs(weights - original_weights))),
                "엑셀가중치_vs_기존_max_abs_diff": float(np.max(np.abs(workbook_weights - original_weights))),
            })
        results[str(horizon)] = result

    report = {
        "입력": str(PROCESSED.relative_to(ROOT)),
        "Raw사용": False,
        "Interim사용": False,
        "결과": results,
    }
    (OUT / "reproduction_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
