from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.train_traffic_accident_v44 import prepare_panel
from ml.train_traffic_accident_v71_fixed_rubric import objective, score_frame
from ml.train_traffic_accident_v72_indicator_forecast import DEV_YEARS, HOLDOUT_YEAR, build_prediction_cache
from ml.train_traffic_accident_v73_indicator_ensemble import evaluate_specs, label
from ml.train_traffic_accident_v74_all_available import add_expanded_cache, fit_forecast
from ml.train_traffic_accident_v76_promoted_indicators import add_persistence_cache, average_metrics, evaluate_feature_set, parse_spec
from ml.train_traffic_accident_v83_poisson_exposure_grade import (
    POP, TARGET_COUNT, TARGET_POP, VICTIMS, fit_exposure_model, population_forecast,
)


ROOT = Path(__file__).resolve().parents[1]
VERSION = 84
OUT = ROOT / "models" / "교통사고" / "84차.피해자수PoissonRaw전진선택"
PROCESSED = ROOT / "data" / "processed" / "교통사고_84차_피해자수PoissonRaw전진선택_2027_2028.xlsx"
HORIZONS = (2, 3)
MAX_ADDITIONS = 12


def poisson_weights(panel, features, alpha):
    labeled = panel[panel[TARGET_COUNT].notna()]
    fitted = fit_exposure_model(labeled, features, alpha)
    coef = np.asarray(fitted.named_steps["model"].coef_, float)
    total = np.abs(coef).sum()
    weights = np.abs(coef)/total if total else np.repeat(1/len(coef), len(coef))
    return fitted, coef, weights


def raw_candidates(panel, core):
    # 다음 연도 정답·식별코드만 제외한다. 현재 연도의 사고·인구 정보는 합법적인 후보다.
    excluded = {"연도", "예측연도", "행정구역코드", "자치구코드", TARGET_COUNT, TARGET_POP}
    excluded_tokens = ("다음연도_",)
    result = []
    for c in panel.select_dtypes(include=[np.number]).columns:
        if c in core or c in excluded or any(t in c for t in excluded_tokens):
            continue
        values = panel[c].to_numpy(float)
        if panel[c].nunique(dropna=True) > 1 and np.isfinite(values).all():
            result.append(c)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True); PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    panel = prepare_panel().sort_values(["자치구", "연도"]).reset_index(drop=True)
    group = panel.groupby("자치구", sort=False)
    panel[TARGET_COUNT] = group[VICTIMS].shift(-1)
    panel[TARGET_POP] = group[POP].shift(-1)

    v83_path = ROOT / "models" / "교통사고" / "83차.피해자수Poisson인구노출등급"
    v83_info = json.loads((v83_path / "model_info.json").read_text(encoding="utf-8"))
    v83_bundle = joblib.load(v83_path / "교통사고_83차_피해자수Poisson등급모델.joblib")
    core = list(v83_bundle["features"]); alpha = float(v83_info["alpha"])
    core_coef = np.array([v83_bundle["coefficients"][f] for f in core], float)
    core_weights = np.array([v83_bundle["weights"][f] for f in core], float)
    candidates = raw_candidates(panel, core)

    v74 = json.loads((ROOT / "models" / "교통사고" / "74차.보유전체데이터활용" / "model_info.json").read_text(encoding="utf-8"))
    core_specs = {h: {f: parse_spec(v) for f, v in v74["결과"][str(h)]["final_specs"].items()} for h in HORIZONS}
    caches, expanded, parts = {}, {}, {}
    for horizon in HORIZONS:
        _, methods, cache, _, _ = build_prediction_cache(panel, core, horizon)
        data, predictor_cols, pool, methods, cache = add_expanded_cache(panel, core, horizon, cache, methods)
        add_persistence_cache(panel, candidates, horizon, cache)
        caches[horizon], expanded[horizon], parts[horizon] = cache, data, (predictor_cols, pool)

    selected = list(core); specs = {h: dict(core_specs[h]) for h in HORIZONS}
    current, _ = evaluate_feature_set(panel, selected, core_coef, core_weights, caches, specs, DEV_YEARS)
    search = [{"단계": 0, "작업": "83차 시작", "추가지표": "-", "지표수": len(selected), **current}]
    all_trials = []
    for step in range(1, MAX_ADDITIONS+1):
        best = None
        for added in candidates:
            if added in selected:
                continue
            trial_features = [*selected, added]
            _, coef, weights = poisson_weights(panel, trial_features, alpha)
            trial_specs = {h: {**specs[h], added: "Persistence"} for h in HORIZONS}
            avg, _ = evaluate_feature_set(panel, trial_features, coef, weights, caches, trial_specs, DEV_YEARS)
            row = {"단계": step, "후보": added, "기준지표수": len(selected), **avg}
            all_trials.append(row)
            record = (objective(avg), added, avg, trial_features, coef, weights, trial_specs)
            if best is None or record[0] > best[0]:
                best = record
        if best is None or best[2]["정확등급정확도"] <= current["정확등급정확도"] + 1e-9:
            search.append({"단계": step, "작업": "추가중단", "추가지표": best[1] if best else "-",
                           "지표수": len(selected), **(best[2] if best else current)})
            break
        _, added, current, selected, _, _, specs = best
        search.append({"단계": step, "작업": "Raw메인지표추가", "추가지표": added,
                       "지표수": len(selected), **current})

    final_model, coef, weights = poisson_weights(panel, selected, alpha)
    dev_avg, dev_by_h = evaluate_feature_set(panel, selected, coef, weights, caches, specs, DEV_YEARS)
    hold_avg, hold_by_h = evaluate_feature_set(panel, selected, coef, weights, caches, specs, [HOLDOUT_YEAR])
    promoted = [f for f in selected if f not in core]
    weight_table = pd.DataFrame({"지표": selected,
                                 "구분": ["83차핵심13개" if f in core else "Raw추가" for f in selected],
                                 "Poisson표준화계수": coef,
                                 "위험방향": np.where(coef >= 0, "값이 클수록 위험", "값이 작을수록 위험"),
                                 "가중치_pct": weights*100}).sort_values("가중치_pct", ascending=False)

    metric_rows, detail_rows, forecasts, strategy_rows, bundles = [], [], [], [], {}
    latest = panel[panel["연도"] == 2025].set_index("자치구")
    for horizon in HORIZONS:
        for scope, result in (("개발_2021_2024", dev_by_h[horizon]), ("독립검증_2025", hold_by_h[horizon])):
            rows = result[1].copy(); rows["구분"] = scope; metric_rows.append(rows)
            detail = result[2].copy(); detail["구분"] = scope; detail_rows.append(detail)
        predictor_cols, pool = parts[horizon]
        core_future, bundle = fit_forecast(panel, expanded[horizon], predictor_cols, pool, core, horizon, core_specs[horizon])
        future = core_future.copy()
        for added in promoted:
            future[added] = future["자치구"].map(latest[added])
            bundle[added] = {"spec": "Persistence", "components": {}}
        scored = score_frame(future, selected, coef, weights, "예측_")
        output = future.merge(scored, on=["연도", "자치구"], validate="one_to_one")
        pop_future = population_forecast(panel, 2025+horizon)
        output = output.merge(pop_future, on="자치구", validate="one_to_one")
        output["예측_어린이피해자율_1만명당"] = final_model.predict(output[selected])*10000
        output["예측_어린이피해자수"] = output["예측_어린이피해자율_1만명당"]/10000*output["예측_어린이인구수"]
        output.insert(0, "예측기간", f"{horizon}년 후"); forecasts.append(output)
        bundles[horizon] = bundle
        strategy_rows.extend({"예측기간": f"{horizon}년 후", "지표": f,
                              "미래값예측법": label(specs[horizon][f])} for f in selected)

    with pd.ExcelWriter(PROCESSED, engine="openpyxl") as writer:
        weight_table.to_excel(writer, sheet_name="최종지표와가중치", index=False)
        pd.DataFrame(search).to_excel(writer, sheet_name="최고성능갱신", index=False)
        pd.DataFrame(all_trials).to_excel(writer, sheet_name="전체Raw후보평가", index=False)
        pd.DataFrame(strategy_rows).to_excel(writer, sheet_name="지표별미래예측법", index=False)
        pd.concat(metric_rows, ignore_index=True).to_excel(writer, sheet_name="성능평가", index=False)
        pd.concat(detail_rows, ignore_index=True).to_excel(writer, sheet_name="검증상세", index=False)
        pd.concat(forecasts, ignore_index=True).to_excel(writer, sheet_name="2027_2028피해자수와등급", index=False)
    integrated = hold_avg["정확등급정확도"]
    info = {"모델명": "교통사고 84차 피해자수 Poisson Raw 전진선택",
            "Y": "다음 연도 어린이 교통사고 사망자 수+부상자 수",
            "불변근간": "83차 핵심13개 고정→Raw 후보 순차추가→미래값→5분위→Poisson 가중치→안전등급",
            "Raw후보수": len(candidates), "평가조합수": len(all_trials),
            "최종지표수": len(selected), "추가지표": promoted,
            "개발통합성능": dev_avg, "2025독립통합성능": hold_avg,
            "기간별2025성능": {str(h): hold_by_h[h][0] for h in HORIZONS},
            "83차통합정확등급": .52, "74차통합정확등급": .56,
            "채택상태": "채택" if integrated > .56 + 1e-9 else "실험탈락_74차유지",
            "processed": str(PROCESSED.relative_to(ROOT))}
    joblib.dump({"version": VERSION, "features": selected, "core_features": core,
                 "promoted_features": promoted, "poisson_model": final_model,
                 "coefficients": dict(zip(selected, coef)), "weights": dict(zip(selected, weights)),
                 "forecast_models": bundles, "safety_grade_orientation": "1=가장 안전, 5=가장 위험"},
                OUT / "교통사고_84차_피해자수PoissonRaw전진선택.joblib")
    (OUT / "model_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
