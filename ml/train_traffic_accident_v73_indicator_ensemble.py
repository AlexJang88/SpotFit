from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.train_traffic_accident_v44 import prepare_panel
from ml.train_traffic_accident_v71_fixed_rubric import evaluate, load_v49, objective, score_frame
from ml.train_traffic_accident_v72_indicator_forecast import (
    DEV_YEARS, HOLDOUT_YEAR, build_prediction_cache, deterministic, feature_frame, model_candidates,
)


ROOT = Path(__file__).resolve().parents[1]
VERSION = 73
OUT = ROOT / "models" / "교통사고" / "73차.미래13지표앙상블"
PROCESSED = ROOT / "data" / "processed" / "교통사고_73차_미래13지표앙상블_2027_2028.xlsx"
HORIZONS = (2, 3)
BLEND_WEIGHTS = (.25, .5, .75)


def load_v72_maps():
    path = ROOT / "models" / "교통사고" / "72차.미래13지표개선" / "model_info.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    return {int(h): payload["final_strategies"] for h, payload in info["결과"].items()}


def label(spec) -> str:
    if isinstance(spec, str):
        return spec
    return f"{spec[0]}×{spec[2]:.2f}+{spec[1]}×{1-spec[2]:.2f}"


def cached_series(cache, year, feature, spec):
    if isinstance(spec, str):
        return cache[(year, feature, spec)]
    a, b, w = spec
    return cache[(year, feature, a)] * w + cache[(year, feature, b)] * (1 - w)


def prediction_frame(panel, features, year, specs, cache):
    districts = sorted(panel["자치구"].unique())
    out = pd.DataFrame({"연도": year, "자치구": districts})
    for feature in features:
        out[feature] = out["자치구"].map(cached_series(cache, year, feature, specs[feature]))
    return out


def evaluate_specs(panel, features, coef, weights, horizon, years, specs, cache):
    rows, details = [], []
    for year in years:
        predicted = prediction_frame(panel, features, year, specs, cache)
        actual = panel[panel["연도"] == year][["연도", "자치구", *features]]
        metrics, detail = evaluate(actual, predicted, features, coef, weights)
        rows.append({"예측기간": f"{horizon}년 후", "검증연도": year, **metrics})
        detail["검증연도"] = year; details.append(detail)
    summary = {k: float(np.mean([row[k] for row in rows])) for k in
               ("정확등급정확도", "1등급이내정확도", "평균등급오차", "종합점수_MAE")}
    return summary, pd.DataFrame(rows), pd.concat(details, ignore_index=True)


def candidate_specs(base_method, methods):
    specs = list(methods)
    for method in methods:
        if method != "Persistence":
            specs.extend(("Persistence", method, w) for w in BLEND_WEIGHTS)
        if method != base_method:
            specs.extend((base_method, method, w) for w in BLEND_WEIGHTS)
    unique = {}
    for spec in specs:
        unique[label(spec)] = spec
    return list(unique.values())


def select_ensemble(panel, features, coef, weights, horizon, methods, cache, base_map):
    base_metrics, _, _ = evaluate_specs(panel, features, coef, weights, horizon, DEV_YEARS, base_map, cache)
    selected, current = dict(base_map), base_metrics
    log = [{"단계": "72차기준", "지표": "전체", "조합": "72차", **current}]

    # 먼저 72차 전체 예측과 Persistence의 공통 혼합비를 비교한다.
    for w in (.1, .2, .3, .4, .5, .6, .7, .8, .9):
        trial = {f: (base_map[f], "Persistence", w) if base_map[f] != "Persistence" else "Persistence"
                 for f in features}
        result, _, _ = evaluate_specs(panel, features, coef, weights, horizon, DEV_YEARS, trial, cache)
        log.append({"단계": "전체혼합", "지표": "전체", "조합": f"72차 {w:.0%}+Persistence {1-w:.0%}", **result})
        if objective(result) > objective(current):
            selected, current = trial, result

    # 49차 가중치가 큰 지표부터 지표값 예측 앙상블을 한 번씩 개선한다.
    for feature in [features[i] for i in np.argsort(-np.abs(weights))]:
        best_spec, best_result = selected[feature], current
        for spec in candidate_specs(base_map[feature], methods):
            trial = dict(selected); trial[feature] = spec
            result, _, _ = evaluate_specs(panel, features, coef, weights, horizon, DEV_YEARS, trial, cache)
            if objective(result) > objective(best_result):
                best_spec, best_result = spec, result
        if label(best_spec) != label(selected[feature]):
            selected[feature], current = best_spec, best_result
            log.append({"단계": "지표별앙상블", "지표": feature, "조합": label(best_spec), **current})
    return selected, current, pd.DataFrame(log)


def fit_component(data, feature, method, numeric_all):
    if method in ("Persistence", "Mean2", "Mean3", "DampedTrend3"):
        return method
    own = [f"현재__{feature}", f"전년__{feature}", f"증감__{feature}", f"3년평균__{feature}"]
    estimator = model_candidates(numeric_all, own)[method]
    target = f"목표__{feature}"
    train = data[data[target].notna()]
    return deepcopy(estimator).fit(train, train[target])


def component_future(panel, current, feature, method, fitted):
    history = panel[panel["연도"] <= 2025]
    if isinstance(fitted, str):
        return np.asarray([deterministic(history[history["자치구"] == d].sort_values("연도")[feature], method)
                           for d in current["자치구"]], float)
    values = fitted.predict(current)
    observed = panel[feature].dropna().astype(float)
    return np.clip(values, observed.quantile(.01), observed.quantile(.99))


def fit_and_forecast(panel, data, features, horizon, specs, numeric_all):
    current = feature_frame(panel, features)
    current = current[current["연도"] == 2025].sort_values("자치구").copy()
    out = current[["자치구"]].copy(); out.insert(0, "연도", 2025 + horizon)
    bundle = {}
    for feature, spec in specs.items():
        methods = [spec] if isinstance(spec, str) else [spec[0], spec[1]]
        components, predictions = {}, {}
        for method in dict.fromkeys(methods):
            fitted = fit_component(data, feature, method, numeric_all)
            components[method] = fitted
            predictions[method] = component_future(panel, current, feature, method, fitted)
        if isinstance(spec, str):
            values = predictions[spec]
        else:
            a, b, w = spec; values = predictions[a] * w + predictions[b] * (1 - w)
        out[feature] = values
        bundle[feature] = {"spec": spec, "components": components}
    return out, bundle


def main():
    OUT.mkdir(parents=True, exist_ok=True); PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    features, coef, weights = load_v49()
    v72_maps = load_v72_maps()
    panel = prepare_panel().sort_values(["연도", "자치구"]).reset_index(drop=True)
    weight_table = pd.DataFrame({"지표": features, "49차계수": coef,
                                 "위험방향": np.where(coef >= 0, "값이 클수록 위험", "값이 작을수록 위험"),
                                 "49차고정가중치_pct": weights * 100})
    payload, logs, comparisons, detail_all, forecast_all, spec_rows = {}, [], [], [], [], []
    for horizon in HORIZONS:
        data, methods, cache, numeric_all, _ = build_prediction_cache(panel, features, horizon)
        base_map = v72_maps[horizon]
        selected, dev, log = select_ensemble(panel, features, coef, weights, horizon, methods, cache, base_map)
        base_dev, base_dev_rows, _ = evaluate_specs(panel, features, coef, weights, horizon, DEV_YEARS, base_map, cache)
        hold, hold_rows, hold_detail = evaluate_specs(panel, features, coef, weights, horizon, [HOLDOUT_YEAR], selected, cache)
        base_hold, base_hold_rows, _ = evaluate_specs(panel, features, coef, weights, horizon, [HOLDOUT_YEAR], base_map, cache)

        adopted = (dev["정확등급정확도"] > base_dev["정확등급정확도"] and
                   hold["정확등급정확도"] > base_hold["정확등급정확도"])
        final_specs = selected if adopted else base_map
        future_values, bundle = fit_and_forecast(panel, data, features, horizon, final_specs, numeric_all)
        future_score = score_frame(future_values, features, coef, weights, "예측_")
        future = future_values.merge(future_score, on=["연도", "자치구"], validate="one_to_one")
        future.insert(0, "예측기간", f"{horizon}년 후"); forecast_all.append(future)
        joblib.dump(bundle, OUT / f"교통사고_73차_{horizon}년후_지표앙상블모델.joblib")

        log.insert(0, "예측기간", f"{horizon}년 후"); logs.append(log)
        hold_rows["구분"] = "73차후보_2025독립"; base_hold_rows["구분"] = "72차_2025독립"
        base_dev_rows["구분"] = "72차_개발"; comparisons.extend([hold_rows, base_hold_rows, base_dev_rows])
        hold_detail["예측기간"] = f"{horizon}년 후"; detail_all.append(hold_detail)
        spec_rows.extend({"예측기간": f"{horizon}년 후", "지표": f, "72차예측법": label(base_map[f]),
                          "73차후보": label(selected[f]), "최종채택": label(final_specs[f])} for f in features)
        payload[str(horizon)] = {"adopted_as_v73": adopted,
                                 "final_specs": {f: label(s) for f, s in final_specs.items()},
                                 "candidate_specs": {f: label(s) for f, s in selected.items()},
                                 "candidate_dev": dev, "candidate_holdout_2025": hold,
                                 "v72_dev": base_dev, "v72_holdout_2025": base_hold,
                                 "selection_rule": "개발 평균과 2025 독립검증 정확등급 정확도가 모두 72차 초과"}

    with pd.ExcelWriter(PROCESSED, engine="openpyxl") as writer:
        weight_table.to_excel(writer, sheet_name="49차고정가중치", index=False)
        pd.DataFrame(spec_rows).to_excel(writer, sheet_name="지표별앙상블", index=False)
        pd.concat(comparisons, ignore_index=True).to_excel(writer, sheet_name="72차_73차성능비교", index=False)
        pd.concat(logs, ignore_index=True).to_excel(writer, sheet_name="앙상블선정과정", index=False)
        pd.concat(detail_all, ignore_index=True).to_excel(writer, sheet_name="2025검증상세", index=False)
        pd.concat(forecast_all, ignore_index=True).to_excel(writer, sheet_name="2027_2028예측", index=False)
    info = {"모델명": "교통사고 73차 미래 13개 지표 앙상블모델",
            "불변구조": "13개 미래값 예측→5분위→49차 고정가중치→종합위험점수→안전등급(1=안전)",
            "지표변경": False, "가중치변경": False, "방향변경": False, "등급직접예측": False,
            "앙상블적용위치": "13개 미래 지표값 예측 단계에만 적용", "결과": payload,
            "processed": str(PROCESSED.relative_to(ROOT))}
    (OUT / "model_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
