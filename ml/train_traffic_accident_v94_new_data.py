"""Build v94 by adding official TOPIS volume features to the v93 pipeline."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.rebuild_traffic_accident_v93_from_processed import load_processed, model_config
from ml.train_traffic_accident_v71_fixed_rubric import objective
from ml.train_traffic_accident_v72_indicator_forecast import DEV_YEARS, build_prediction_cache
from ml.train_traffic_accident_v73_indicator_ensemble import candidate_specs, evaluate_specs, label
from ml.train_traffic_accident_v74_all_available import add_expanded_cache, fit_forecast


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models" / "교통사고" / "93차.어린이교통사고수_2차모델"
OUT = ROOT / "models" / "교통사고" / "94차.교통량추가모델"
TRAFFIC = ROOT / "data" / "94차_추가사용데이터" / "interim" / "자치구별_교통량_2017_2025.csv"
REPORT = ROOT / "data" / "processed" / "교통사고_94차_교통량추가모델.xlsx"
NEW_FEATURES = [
    "교통량_평균일교통량", "교통량_평일평균일교통량", "교통량_주말평균일교통량",
    "교통량_등교시간비중", "교통량_하교시간비중",
]


def load_v93(horizon: int) -> dict:
    path = SOURCE / f"교통사고_93차_{horizon}년후_미래지표예측최적화.joblib"
    return joblib.load(path)


def v93_specs(bundle: dict, horizon: int) -> dict:
    forecasts = bundle["forecast_models"][2] if horizon == 2 else bundle["forecast_model"]
    return {feature: forecasts[feature]["spec"] for feature in bundle["features"]}


def augmented_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel, _, _ = load_processed()
    volume = pd.read_csv(TRAFFIC, encoding="utf-8-sig")
    volume = volume[["연도", "자치구", *NEW_FEATURES]].copy()
    merged = panel.merge(volume, on=["연도", "자치구"], how="left", validate="one_to_one")
    # 한 자치구에 관측소가 없으므로 해당 연도의 서울 자치구 중앙값을 사용한다.
    for feature in NEW_FEATURES:
        merged[feature] = merged[feature].fillna(merged.groupby("연도")[feature].transform("median"))
    if merged[NEW_FEATURES].isna().any().any():
        raise ValueError("교통량 추가지표에 결측치가 남았습니다.")
    return merged.sort_values(["연도", "자치구"]).reset_index(drop=True), volume


def optimize(panel, features, coef, weights, horizon, methods, cache, base_specs):
    selected = dict(base_specs)
    current, _, _ = evaluate_specs(panel, features, coef, weights, horizon, DEV_YEARS, selected, cache)
    log = [{"단계": "93차기준", "지표": "전체", "조합": "93차", **current}]
    for feature in [features[i] for i in np.argsort(-np.abs(weights))]:
        best_spec, best = selected[feature], current
        for spec in candidate_specs("Persistence", methods):
            trial = dict(selected)
            trial[feature] = spec
            metrics, _, _ = evaluate_specs(panel, features, coef, weights, horizon, DEV_YEARS, trial, cache)
            if objective(metrics) > objective(best):
                best_spec, best = spec, metrics
        if label(best_spec) != label(selected[feature]):
            selected[feature], current = best_spec, best
            log.append({"단계": "교통량추가후_지표별최적화", "지표": feature,
                        "조합": label(best_spec), **current})
    return selected, current, pd.DataFrame(log)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    panel, volume = augmented_panel()
    _, weight_table, method_table = load_processed()
    summaries, strategy_rows, logs, details = [], [], [], []
    result = {
        "모델명": "교통사고 94차 TOPIS 교통량 추가모델",
        "근간": "93차 Y·Poisson 계수/가중치·5분위·안전등급 유지",
        "추가자료": "서울시 TOPIS 지점별 일자별 시간대별 교통량 2017~2025",
        "추가지표": NEW_FEATURES,
        "선택원칙": "2021~2024 개발연도 정확등급 우선; 2025는 최종 독립검증에만 사용",
        "결과": {},
    }
    for horizon in (2, 3):
        source = load_v93(horizon)
        features, _, _, _ = model_config(weight_table, method_table, horizon)
        if features != list(source["features"]):
            raise ValueError(f"{horizon}년 모델의 지표 순서가 93차 기록과 다릅니다.")
        coef = np.array([source["coefficients"][f] for f in features], dtype=float)
        weights = np.array([source["weights"][f] for f in features], dtype=float)
        base_specs = v93_specs(source, horizon)

        _, methods, cache, _, _ = build_prediction_cache(panel, features, horizon)
        expanded, predictor_cols, pool, methods, cache = add_expanded_cache(
            panel, features, horizon, cache, methods
        )
        methods = list(dict.fromkeys(methods))
        base_dev, _, _ = evaluate_specs(panel, features, coef, weights, horizon, DEV_YEARS, base_specs, cache)
        base_hold, _, base_detail = evaluate_specs(panel, features, coef, weights, horizon, [2025], base_specs, cache)
        proposed_specs, proposed_dev, log = optimize(
            panel, features, coef, weights, horizon, methods, cache, base_specs
        )
        proposed_hold, _, proposed_detail = evaluate_specs(
            panel, features, coef, weights, horizon, [2025], proposed_specs, cache
        )
        adopted = proposed_dev["정확등급정확도"] > base_dev["정확등급정확도"] + 1e-12
        final_specs = proposed_specs if adopted else base_specs
        final_dev = proposed_dev if adopted else base_dev
        final_hold = proposed_hold if adopted else base_hold

        _, forecast_bundle = fit_forecast(
            panel, expanded, predictor_cols, pool, features, horizon, final_specs
        )
        saved = dict(source)
        saved.update(version=94, source_version=93, horizon=horizon,
                     added_data="TOPIS_traffic_volume_2017_2025", added_features=NEW_FEATURES)
        if horizon == 2:
            saved["forecast_models"] = {2: forecast_bundle}
        else:
            saved["forecast_model"] = forecast_bundle
        model_path = OUT / f"교통사고_94차_{horizon}년후_교통량추가모델.joblib"
        joblib.dump(saved, model_path)

        for version, scope, metrics in [
            (93, "개발", base_dev), (94, "개발", final_dev),
            (93, "2025독립", base_hold), (94, "2025독립", final_hold),
        ]:
            summaries.append({"차수": version, "예측기간": f"{horizon}년 후", "구분": scope, **metrics})
        for feature in features:
            strategy_rows.append({"예측기간": f"{horizon}년 후", "지표": feature,
                                  "93차": label(base_specs[feature]), "94차": label(final_specs[feature]),
                                  "변경여부": label(base_specs[feature]) != label(final_specs[feature])})
        log.insert(0, "예측기간", f"{horizon}년 후")
        logs.append(log)
        base_detail.insert(0, "모델", "93차")
        proposed_detail.insert(0, "모델", "94차후보")
        base_detail.insert(1, "예측기간", f"{horizon}년 후")
        proposed_detail.insert(1, "예측기간", f"{horizon}년 후")
        details.extend([base_detail, proposed_detail])
        result["결과"][str(horizon)] = {
            "채택": adopted, "93차_개발": base_dev, "94차후보_개발": proposed_dev,
            "93차_2025독립": base_hold, "94차_2025독립": final_hold,
            "변경지표수": sum(label(base_specs[f]) != label(final_specs[f]) for f in features),
            "모델파일": model_path.name,
        }

    summary = pd.DataFrame(summaries)
    result["93차_2025평균정확등급"] = float(summary[(summary["차수"] == 93) & (summary["구분"] == "2025독립")]["정확등급정확도"].mean())
    result["94차_2025평균정확등급"] = float(summary[(summary["차수"] == 94) & (summary["구분"] == "2025독립")]["정확등급정확도"].mean())
    result["processed"] = str(REPORT.relative_to(ROOT))
    with pd.ExcelWriter(REPORT, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="성능비교", index=False)
        pd.DataFrame(strategy_rows).to_excel(writer, sheet_name="지표별예측법", index=False)
        pd.concat(logs, ignore_index=True).to_excel(writer, sheet_name="탐색기록", index=False)
        pd.concat(details, ignore_index=True).to_excel(writer, sheet_name="2025자치구별검증", index=False)
        volume.to_excel(writer, sheet_name="추가교통량지표", index=False)
        panel.to_excel(writer, sheet_name="94차전체입력패널", index=False)
    (OUT / "model_info.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
