"""Forward-select road-risk predictors one at a time on top of v94."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.rebuild_traffic_accident_v93_from_processed import load_processed, model_config
from ml.train_traffic_accident_v72_indicator_forecast import DEV_YEARS, build_prediction_cache
from ml.train_traffic_accident_v73_indicator_ensemble import evaluate_specs
from ml.train_traffic_accident_v74_all_available import add_expanded_cache, fit_forecast
from ml.train_traffic_accident_v94_new_data import augmented_panel as v94_panel


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "models/교통사고/94차.11.13.개선"
ROAD = ROOT / "data/95차_추가사용데이터/interim/자치구별_도로위험도_2018_2025.csv"
OUT = ROOT / "models/교통사고/95차.도로위험도점진추가모델"
REPORT = ROOT / "data/processed/교통사고_95차_도로위험도점진추가모델.xlsx"
CANDIDATES = [
    "도로위험_사고링크비율", "도로위험_사고건수", "도로위험_사상자수",
    "도로위험_사망사고건수", "도로위험_중상사고건수",
    "도로위험_평균지수", "도로위험_길이가중평균지수",
]


def load_v94(horizon: int) -> dict:
    return joblib.load(SOURCE / f"교통사고_94차_{horizon}년후_교통량추가모델.joblib")


def source_specs(bundle: dict, horizon: int) -> dict:
    forecasts = bundle["forecast_models"][2] if horizon == 2 else bundle["forecast_model"]
    return {feature: forecasts[feature]["spec"] for feature in bundle["features"]}


def panel_for(selected: list[str]) -> pd.DataFrame:
    panel, _ = v94_panel()
    if not selected:
        return panel
    road = pd.read_csv(ROAD, encoding="utf-8-sig")[["연도", "자치구", *selected]]
    panel = panel.merge(road, on=["연도", "자치구"], how="left", validate="one_to_one")
    # 2017년 공식 자료 부재를 미래값으로 채우지 않는다.
    panel[selected] = panel[selected].fillna(0)
    return panel.sort_values(["연도", "자치구"]).reset_index(drop=True)


def metrics_for(panel, features, coef, weights, horizon, specs, years):
    _, methods, cache, _, _ = build_prediction_cache(panel, features, horizon)
    expanded, predictor_cols, pool, methods, cache = add_expanded_cache(panel, features, horizon, cache, methods)
    metrics, _, detail = evaluate_specs(panel, features, coef, weights, horizon, years, specs, cache)
    return metrics, detail, expanded, predictor_cols, pool


def better(candidate: dict, current: dict) -> bool:
    return candidate["정확등급정확도"] > current["정확등급정확도"] + 1e-12


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "_진행중.txt").write_text("점진추가 실험 계산 중", encoding="utf-8")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    _, weight_table, method_table = load_processed()
    info94 = json.loads((SOURCE / "model_info.json").read_text(encoding="utf-8"))
    result = {
        "모델명": "교통사고 95차 도로위험도 점진추가모델",
        "선택방법": "94차 고정; 도로위험도 후보를 한 번에 하나씩 추가; 2021~2024 정확등급이 상승할 때만 채택",
        "2025사용": "지표 선택 완료 후 독립검증에만 사용",
        "결과": {},
    }
    logs, summaries, details = [], [], []
    for horizon in (2, 3):
        source = load_v94(horizon)
        features, _, _, _ = model_config(weight_table, method_table, horizon)
        coef = np.array([source["coefficients"][f] for f in features], dtype=float)
        weights = np.array([source["weights"][f] for f in features], dtype=float)
        specs = source_specs(source, horizon)
        base_key = "94차후보_개발" if info94["결과"][str(horizon)]["채택"] else "93차_개발"
        current = info94["결과"][str(horizon)][base_key]
        selected: list[str] = []
        remaining = list(CANDIDATES)
        stage = 1
        logs.append({"예측기간": f"{horizon}년 후", "단계": 0, "시험지표": "94차 기준", "채택": True,
                     "누적지표": "없음", **current})
        while remaining:
            trials = []
            for candidate in remaining:
                trial_selected = [*selected, candidate]
                trial_panel = panel_for(trial_selected)
                metrics, _, _, _, _ = metrics_for(
                    trial_panel, features, coef, weights, horizon, specs, DEV_YEARS
                )
                trials.append((candidate, metrics))
                logs.append({"예측기간": f"{horizon}년 후", "단계": stage, "시험지표": candidate,
                             "채택": False, "누적지표": " + ".join(trial_selected), **metrics})
                pd.DataFrame(logs).to_csv(OUT / "_중간탐색기록.csv", index=False, encoding="utf-8-sig")
            candidate, best = max(
                trials,
                key=lambda x: (x[1]["정확등급정확도"], -x[1]["평균등급오차"], -x[1]["종합점수_MAE"]),
            )
            if not better(best, current):
                break
            selected.append(candidate)
            remaining.remove(candidate)
            current = best
            logs.append({"예측기간": f"{horizon}년 후", "단계": stage, "시험지표": candidate,
                         "채택": True, "누적지표": " + ".join(selected), **current})
            stage += 1

        final_panel = panel_for(selected)
        final_dev, _, expanded, predictor_cols, pool = metrics_for(
            final_panel, features, coef, weights, horizon, specs, DEV_YEARS
        )
        final_hold, hold_detail, _, _, _ = metrics_for(
            final_panel, features, coef, weights, horizon, specs, [2025]
        )
        base_hold = info94["결과"][str(horizon)]["94차_2025독립"]
        adopted = bool(selected)
        if adopted:
            _, forecast_bundle = fit_forecast(final_panel, expanded, predictor_cols, pool, features, horizon, specs)
            saved = dict(source)
            saved.update(version=95, source_version=94, horizon=horizon,
                         added_data="KOROAD_road_risk_incremental", added_features=selected)
            if horizon == 2:
                saved["forecast_models"] = {2: forecast_bundle}
            else:
                saved["forecast_model"] = forecast_bundle
        else:
            saved = dict(source)
            saved.update(version=95, source_version=94, horizon=horizon,
                         added_data="none_no_incremental_gain", added_features=[])
        model_path = OUT / f"교통사고_95차_{horizon}년후_점진추가모델.joblib"
        OUT.mkdir(parents=True, exist_ok=True)
        joblib.dump(saved, model_path)
        for version, scope, metrics in [
            (94, "개발", info94["결과"][str(horizon)][base_key]),
            (95, "개발", final_dev), (94, "2025독립", base_hold), (95, "2025독립", final_hold),
        ]:
            summaries.append({"차수": version, "예측기간": f"{horizon}년 후", "구분": scope, **metrics})
        hold_detail.insert(0, "예측기간", f"{horizon}년 후")
        details.append(hold_detail)
        result["결과"][str(horizon)] = {
            "채택지표": selected, "채택": adopted, "94차_개발": info94["결과"][str(horizon)][base_key],
            "95차_개발": final_dev, "94차_2025독립": base_hold, "95차_2025독립": final_hold,
            "모델파일": model_path.name,
        }
    with pd.ExcelWriter(REPORT, engine="openpyxl") as writer:
        pd.DataFrame(summaries).to_excel(writer, sheet_name="성능비교", index=False)
        pd.DataFrame(logs).to_excel(writer, sheet_name="점진추가탐색", index=False)
        pd.concat(details, ignore_index=True).to_excel(writer, sheet_name="2025독립검증", index=False)
    result["processed"] = str(REPORT.relative_to(ROOT))
    (OUT / "model_info.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "_진행중.txt").unlink(missing_ok=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
