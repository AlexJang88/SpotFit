"""Rebuild both v94 models from one processed workbook only.

Required data input:
    data/processed/교통사고_94차_최종모델/교통사고_94차_데이터.xlsx

No raw/interim data or earlier joblib model is read.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings(
    "ignore",
    message="Skipping features without any observed values.*",
    category=UserWarning,
    module=r"sklearn\.impute\._base",
)

from ml.train_traffic_accident_v72_indicator_forecast import build_prediction_cache
from ml.train_traffic_accident_v73_indicator_ensemble import evaluate_specs
from ml.train_traffic_accident_v74_all_available import (
    add_expanded_cache,
    candidates,
    fit_forecast,
    supervised,
)
from ml.train_traffic_accident_v83_poisson_exposure_grade import (
    POP,
    TARGET_COUNT,
    TARGET_POP,
    VICTIMS,
    fit_exposure_model,
)
from ml.traffic_accident_pipeline_paths import MODEL_V94_REBUILT, PROCESSED_V94


PROCESSED = PROCESSED_V94
OUT = MODEL_V94_REBUILT
PANEL_SHEET = "94차전체입력패널"
CONFIG_SHEET = "94차모델설정"


def decode_spec(value: str):
    decoded = json.loads(value)
    return tuple(decoded) if isinstance(decoded, list) else decoded


def load_input(horizon: int):
    panel = pd.read_excel(PROCESSED, sheet_name=PANEL_SHEET)
    panel = panel.sort_values(["연도", "자치구"]).reset_index(drop=True)
    config = pd.read_excel(PROCESSED, sheet_name=CONFIG_SHEET)
    config = config[config["예측기간"] == f"{horizon}년 후"].sort_values("순서")
    if config.empty:
        raise ValueError(f"{horizon}년 후 모델 설정이 없습니다.")
    features = config["지표"].tolist()
    missing = sorted(set(features) - set(panel.columns))
    if missing:
        raise ValueError(f"입력 패널에 지표가 없습니다: {missing}")
    specs = dict(zip(features, config["미래값예측설정_JSON"].map(decode_spec)))
    coefficients = config["Poisson계수"].to_numpy(dtype=float)
    weights = config["가중치"].to_numpy(dtype=float)
    return panel, features, specs, coefficients, weights


def rebuild(horizon: int) -> tuple[Path, dict]:
    panel, features, specs, recorded_coef, recorded_weights = load_input(horizon)

    poisson_panel = panel.copy()
    grouped = poisson_panel.groupby("자치구", sort=False)
    poisson_panel[TARGET_COUNT] = grouped[VICTIMS].shift(-1)
    poisson_panel[TARGET_POP] = grouped[POP].shift(-1)
    labeled = poisson_panel[poisson_panel[TARGET_COUNT].notna()].copy()
    poisson_model = fit_exposure_model(labeled, features, alpha=0.0)
    fitted_coef = np.asarray(poisson_model.named_steps["model"].coef_, dtype=float)
    fitted_weights = np.abs(fitted_coef) / np.abs(fitted_coef).sum()

    coef_diff = float(np.max(np.abs(fitted_coef - recorded_coef)))
    weight_diff = float(np.max(np.abs(fitted_weights - recorded_weights)))
    if coef_diff > 1e-8 or weight_diff > 1e-8:
        raise ValueError(
            "엑셀 설정과 재학습 결과가 다릅니다: "
            f"coef_diff={coef_diff:.3g}, weight_diff={weight_diff:.3g}"
        )

    expanded, predictor_cols, _ = supervised(panel, features, horizon)
    predictor_cols = [
        column
        for column in expanded.columns
        if column not in {"연도", "자치구", "목표연도"} and not column.startswith("목표__")
    ]
    pool = candidates(len(predictor_cols))
    _, forecast_bundle = fit_forecast(
        panel, expanded, predictor_cols, pool, features, horizon, specs
    )

    model = {
        "version": 94,
        "horizon": horizon,
        "source": "processed_workbook_only",
        "processed_file": PROCESSED.name,
        "features": features,
        "poisson_model": poisson_model,
        "coefficients": dict(zip(features, fitted_coef)),
        "weights": dict(zip(features, fitted_weights)),
        "source_version": 93,
        "added_data": "TOPIS_traffic_volume_2017_2025",
        "added_features": [
            "교통량_평균일교통량",
            "교통량_평일평균일교통량",
            "교통량_주말평균일교통량",
            "교통량_등교시간비중",
            "교통량_하교시간비중",
        ],
        "safety_grade_orientation": "1=가장 안전, 5=가장 위험",
    }
    if horizon == 2:
        model["forecast_models"] = {2: forecast_bundle}
    else:
        model["forecast_model"] = forecast_bundle

    output = OUT / f"교통사고_94차_{horizon}년후_Processed단독재현.joblib"
    joblib.dump(model, output)

    _, methods, cache, _, _ = build_prediction_cache(panel, features, horizon)
    _, _, _, methods, cache = add_expanded_cache(panel, features, horizon, cache, methods)
    metrics, _, _ = evaluate_specs(
        panel, features, fitted_coef, fitted_weights, horizon, [2025], specs, cache
    )
    return output, {
        "지표수": len(features),
        "계수재현최대오차": coef_diff,
        "가중치재현최대오차": weight_diff,
        "2025독립성능": metrics,
        "모델파일": output.name,
    }


def main() -> None:
    if not PROCESSED.exists():
        raise FileNotFoundError(PROCESSED)
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for horizon in (2, 3):
        _, results[str(horizon)] = rebuild(horizon)
    report = {
        "입력파일": str(PROCESSED.relative_to(ROOT)),
        "사용한외부데이터": [],
        "Raw사용": False,
        "Interim사용": False,
        "이전모델사용": False,
        "결과": results,
    }
    (OUT / "reproduction_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
