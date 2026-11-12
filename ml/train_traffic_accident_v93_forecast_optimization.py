from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.train_traffic_accident_v44 import prepare_panel
from ml.train_traffic_accident_v72_indicator_forecast import DEV_YEARS, build_prediction_cache
from ml.train_traffic_accident_v71_fixed_rubric import objective
from ml.train_traffic_accident_v73_indicator_ensemble import candidate_specs, evaluate_specs, label
from ml.train_traffic_accident_v74_all_available import add_expanded_cache, fit_forecast


ROOT = Path(__file__).resolve().parents[1]
TRAFFIC = next(p for p in (ROOT / "models").iterdir() if p.is_dir() and p.name != "recommender")
SOURCE = next(TRAFFIC.glob("92*"))
OUT = TRAFFIC / "93차.92차미래지표예측최적화"
REPORT = ROOT / "data" / "processed" / "교통사고_91차_92차_93차_비교.xlsx"


def load_horizon(horizon: int) -> tuple[dict, dict]:
    path = sorted(SOURCE.glob("*.joblib"))[horizon - 2]
    bundle = joblib.load(path)
    forecasts = bundle["forecast_models"][2] if horizon == 2 else bundle["forecast_model"]
    specs = {feature: forecasts[feature]["spec"] for feature in bundle["features"]}
    return bundle, specs


def metric_row(version: int, horizon: int, scope: str, metrics: dict) -> dict:
    return {"차수": version, "예측기간": f"{horizon}년 후", "구분": scope, **metrics}


def optimize_specs(panel, features, coefficients, weights, horizon, methods, cache, base_specs):
    selected = dict(base_specs)
    current, _, _ = evaluate_specs(
        panel, features, coefficients, weights, horizon, DEV_YEARS, selected, cache
    )
    records = [{"단계": "92차기준", "지표": "전체", "조합": "92차", **current}]
    for feature in [features[i] for i in np.argsort(-np.abs(weights))]:
        best_spec, best_metrics = selected[feature], current
        # 기존 92차 조합은 그대로 후보로 남기고, 중첩 앙상블은 만들지 않는다.
        options = candidate_specs("Persistence", methods)
        for spec in options:
            trial = dict(selected)
            trial[feature] = spec
            metrics, _, _ = evaluate_specs(
                panel, features, coefficients, weights, horizon, DEV_YEARS, trial, cache
            )
            if objective(metrics) > objective(best_metrics):
                best_spec, best_metrics = spec, metrics
        if label(best_spec) != label(selected[feature]):
            selected[feature], current = best_spec, best_metrics
            records.append({"단계": "지표별최적화", "지표": feature,
                            "조합": label(best_spec), **current})
    return selected, current, pd.DataFrame(records)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    panel = prepare_panel().sort_values(["연도", "자치구"]).reset_index(drop=True)
    summaries, logs, strategies = [], [], []
    info: dict[str, object] = {
        "모델명": "교통사고 93차 92차 미래지표 예측 최적화 실험",
        "고정근간": "92차 변수·Poisson 계수/가중치·5분위·안전 1~5등급",
        "선택원칙": "과거 개발연도 정확등급 우선 탐색 후 2025년은 독립검증에만 사용",
        "결과": {},
    }

    for horizon in (2, 3):
        source, base_specs = load_horizon(horizon)
        features = list(source["features"])
        coefficients = np.array([source["coefficients"][f] for f in features], dtype=float)
        weights = np.array([source["weights"][f] for f in features], dtype=float)

        _, methods, cache, _, _ = build_prediction_cache(panel, features, horizon)
        expanded, predictor_cols, pool, methods, cache = add_expanded_cache(
            panel, features, horizon, cache, methods
        )
        methods = list(dict.fromkeys(methods))

        base_dev, _, _ = evaluate_specs(
            panel, features, coefficients, weights, horizon, DEV_YEARS, base_specs, cache
        )
        base_hold, _, _ = evaluate_specs(
            panel, features, coefficients, weights, horizon, [2025], base_specs, cache
        )
        candidate_specs, candidate_dev, log = optimize_specs(
            panel, features, coefficients, weights, horizon, methods, cache, base_specs
        )
        candidate_hold, _, candidate_detail = evaluate_specs(
            panel, features, coefficients, weights, horizon, [2025], candidate_specs, cache
        )

        # 2025 결과는 선택에 사용하지 않는다. 개발구간의 정확등급이 개선된 경우에만 후보를 채택한다.
        adopted = candidate_dev["정확등급정확도"] > base_dev["정확등급정확도"] + 1e-12
        final_specs = candidate_specs if adopted else base_specs
        final_dev = candidate_dev if adopted else base_dev
        final_hold = candidate_hold if adopted else base_hold

        _, fitted_forecasts = fit_forecast(
            panel, expanded, predictor_cols, pool, features, horizon, final_specs
        )
        saved = dict(source)
        saved["version"] = 93
        saved["source_version"] = 92
        saved["horizon"] = horizon
        if horizon == 2:
            saved["forecast_models"] = {2: fitted_forecasts}
        else:
            saved["forecast_model"] = fitted_forecasts
        joblib.dump(saved, OUT / f"교통사고_93차_{horizon}년후_미래지표예측최적화.joblib")

        summaries.extend([
            metric_row(92, horizon, "개발", base_dev),
            metric_row(93, horizon, "개발", final_dev),
            metric_row(92, horizon, "2025독립", base_hold),
            metric_row(93, horizon, "2025독립", final_hold),
        ])
        log.insert(0, "예측기간", f"{horizon}년 후")
        logs.append(log)
        for feature in features:
            strategies.append({
                "예측기간": f"{horizon}년 후",
                "지표": feature,
                "92차": label(base_specs[feature]),
                "93차": label(final_specs[feature]),
                "변경여부": label(base_specs[feature]) != label(final_specs[feature]),
            })
        wrong = candidate_detail[
            candidate_detail["실제_안전등급"] != candidate_detail["예측_안전등급"]
        ]
        info["결과"][str(horizon)] = {
            "채택": adopted,
            "92차_개발": base_dev,
            "93차후보_개발": candidate_dev,
            "92차_2025독립": base_hold,
            "93차_2025독립": final_hold,
            "변경지표수": sum(label(base_specs[f]) != label(final_specs[f]) for f in features),
            "2025오답자치구수": int(len(wrong)) if adopted else int(round(25 * (1-base_hold["정확등급정확도"]))),
        }

    summary_df = pd.DataFrame(summaries)
    with pd.ExcelWriter(REPORT, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="성능비교", index=False)
        pd.DataFrame(strategies).to_excel(writer, sheet_name="지표별예측법", index=False)
        pd.concat(logs, ignore_index=True).to_excel(writer, sheet_name="탐색기록", index=False)
    info["92차_평균정확등급"] = float(summary_df[(summary_df["차수"] == 92) & (summary_df["구분"] == "2025독립")]["정확등급정확도"].mean())
    info["93차_평균정확등급"] = float(summary_df[(summary_df["차수"] == 93) & (summary_df["구분"] == "2025독립")]["정확등급정확도"].mean())
    info["processed"] = str(REPORT.relative_to(ROOT))
    (OUT / "model_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
