"""Build v95 from v94 plus lag-safe annual road-link risk features."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.rebuild_traffic_accident_v93_from_processed import load_processed, model_config
from ml.train_traffic_accident_v71_fixed_rubric import objective
from ml.train_traffic_accident_v72_indicator_forecast import DEV_YEARS, build_prediction_cache
from ml.train_traffic_accident_v73_indicator_ensemble import evaluate_specs, label
from ml.train_traffic_accident_v74_all_available import add_expanded_cache, fit_forecast
from ml.train_traffic_accident_v94_new_data import augmented_panel as v94_panel, optimize


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models/교통사고/94차.11.13.개선"
OUT = ROOT / "models/교통사고/95차.도로위험도추가모델"
ROAD = ROOT / "data/95차_추가사용데이터/interim/자치구별_도로위험도_2018_2025.csv"
REPORT = ROOT / "data/processed/교통사고_95차_도로위험도추가모델.xlsx"
NEW_FEATURES = [
    "도로위험_사고링크비율", "도로위험_사고건수", "도로위험_사상자수",
    "도로위험_사망사고건수", "도로위험_중상사고건수",
    "도로위험_평균지수", "도로위험_길이가중평균지수",
]


def load_v94(horizon: int) -> dict:
    return joblib.load(SOURCE / f"교통사고_94차_{horizon}년후_교통량추가모델.joblib")


def source_specs(bundle: dict, horizon: int) -> dict:
    forecasts = bundle["forecast_models"][2] if horizon == 2 else bundle["forecast_model"]
    return {feature: forecasts[feature]["spec"] for feature in bundle["features"]}


def augmented_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel, _ = v94_panel()
    road = pd.read_csv(ROAD, encoding="utf-8-sig")
    road = road[["연도", "자치구", *NEW_FEATURES]].copy()
    merged = panel.merge(road, on=["연도", "자치구"], how="left", validate="one_to_one")
    merged["도로위험_자료가용"] = merged[NEW_FEATURES].notna().all(axis=1).astype(int)
    # 2017년은 공식 자료가 없으므로 미래값으로 역대입하지 않고 0+가용성 플래그로 명시한다.
    merged[NEW_FEATURES] = merged[NEW_FEATURES].fillna(0)
    return merged.sort_values(["연도", "자치구"]).reset_index(drop=True), road


def official_metrics(info: dict, horizon: int, scope: str) -> dict:
    result = info["결과"][str(horizon)]
    if scope == "개발":
        return result["94차후보_개발"] if result["채택"] else result["93차_개발"]
    return result["94차_2025독립"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    panel, road = augmented_panel()
    _, weight_table, method_table = load_processed()
    info94 = json.loads((SOURCE / "model_info.json").read_text(encoding="utf-8"))
    summaries, strategy_rows, logs, details = [], [], [], []
    result = {
        "모델명": "교통사고 95차 도로위험도 추가모델",
        "근간": "94차 모델 + 한국도로교통공단 도로링크별 위험도 2018~2025",
        "추가지표": NEW_FEATURES,
        "제외자료": "최신 CCTV·도로폭·과속방지턱은 2025 검증 누출 방지를 위해 학습 제외",
        "선택원칙": "2021~2024 개발연도 정확등급 우선; 2025 독립검증",
        "결과": {},
    }
    for horizon in (2, 3):
        source = load_v94(horizon)
        features, _, _, _ = model_config(weight_table, method_table, horizon)
        if features != list(source["features"]):
            raise ValueError(f"{horizon}년 모델 지표 순서 불일치")
        coef = np.array([source["coefficients"][f] for f in features], dtype=float)
        weights = np.array([source["weights"][f] for f in features], dtype=float)
        base_specs = source_specs(source, horizon)
        _, methods, cache, _, _ = build_prediction_cache(panel, features, horizon)
        expanded, predictor_cols, pool, methods, cache = add_expanded_cache(panel, features, horizon, cache, methods)
        methods = list(dict.fromkeys(methods))
        candidate_specs, candidate_dev, log = optimize(
            panel, features, coef, weights, horizon, methods, cache, base_specs
        )
        candidate_hold, _, candidate_detail = evaluate_specs(
            panel, features, coef, weights, horizon, [2025], candidate_specs, cache
        )
        base_dev = official_metrics(info94, horizon, "개발")
        base_hold = official_metrics(info94, horizon, "2025")
        adopted = candidate_dev["정확등급정확도"] > base_dev["정확등급정확도"] + 1e-12
        final_specs = candidate_specs if adopted else base_specs
        final_dev = candidate_dev if adopted else base_dev
        final_hold = candidate_hold if adopted else base_hold
        _, forecast_bundle = fit_forecast(panel, expanded, predictor_cols, pool, features, horizon, final_specs)
        saved = dict(source)
        saved.update(version=95, source_version=94, horizon=horizon,
                     added_data="KOROAD_road_risk_2018_2025", added_features=NEW_FEATURES)
        if horizon == 2:
            saved["forecast_models"] = {2: forecast_bundle}
        else:
            saved["forecast_model"] = forecast_bundle
        model_path = OUT / f"교통사고_95차_{horizon}년후_도로위험도추가모델.joblib"
        joblib.dump(saved, model_path)
        for version, scope, metrics in [
            (94, "개발", base_dev), (95, "개발", final_dev),
            (94, "2025독립", base_hold), (95, "2025독립", final_hold),
        ]:
            summaries.append({"차수": version, "예측기간": f"{horizon}년 후", "구분": scope, **metrics})
        for feature in features:
            strategy_rows.append({"예측기간": f"{horizon}년 후", "지표": feature,
                                  "94차": label(base_specs[feature]), "95차": label(final_specs[feature]),
                                  "변경여부": label(base_specs[feature]) != label(final_specs[feature])})
        log.insert(0, "예측기간", f"{horizon}년 후")
        logs.append(log)
        candidate_detail.insert(0, "모델", "95차후보")
        candidate_detail.insert(1, "예측기간", f"{horizon}년 후")
        details.append(candidate_detail)
        result["결과"][str(horizon)] = {
            "채택": adopted, "94차_개발": base_dev, "95차후보_개발": candidate_dev,
            "94차_2025독립": base_hold, "95차_2025독립": final_hold,
            "변경지표수": sum(label(base_specs[f]) != label(final_specs[f]) for f in features),
            "모델파일": model_path.name,
        }
    summary = pd.DataFrame(summaries)
    with pd.ExcelWriter(REPORT, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="성능비교", index=False)
        pd.DataFrame(strategy_rows).to_excel(writer, sheet_name="지표별예측법", index=False)
        pd.concat(logs, ignore_index=True).to_excel(writer, sheet_name="탐색기록", index=False)
        pd.concat(details, ignore_index=True).to_excel(writer, sheet_name="2025자치구별검증", index=False)
        road.to_excel(writer, sheet_name="추가도로위험지표", index=False)
        panel.to_excel(writer, sheet_name="95차전체입력패널", index=False)
    result["최종판정"] = "실험 실패: 독립검증 정확등급이 개선되지 않아 94차를 최종모델로 유지"
    result["제출권고"] = "95차는 실험기록으로만 보관하고 대표모델로 제출하지 않음"
    result["processed"] = str(REPORT.relative_to(ROOT))
    (OUT / "model_info.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
