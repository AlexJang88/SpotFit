from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.train_traffic_accident_v44 import prepare_panel
from ml.train_traffic_accident_v71_fixed_rubric import load_v49, objective, risk_quintile, score_frame
from ml.train_traffic_accident_v72_indicator_forecast import DEV_YEARS, HOLDOUT_YEAR, build_prediction_cache
from ml.train_traffic_accident_v73_indicator_ensemble import cached_series, evaluate_specs, label
from ml.train_traffic_accident_v74_all_available import add_expanded_cache, fit_forecast
from ml.train_traffic_accident_v76_promoted_indicators import parse_spec


ROOT = Path(__file__).resolve().parents[1]
VERSION = 78
OUT = ROOT / "models" / "교통사고" / "78차.13지표최종공동앙상블"
PROCESSED = ROOT / "data" / "processed" / "교통사고_78차_13지표최종공동앙상블_2027_2028.xlsx"
HORIZONS = (2, 3)
BLEND_WEIGHTS = (.25, .5, .75)
BEAM_WIDTH = 24
TOP_OPTIONS = 6


def spec_key(spec):
    return label(spec)


def simple_options(base_spec, methods):
    options = [base_spec, "Persistence", *methods]
    for method in methods:
        if method != "Persistence":
            options.extend(("Persistence", method, w) for w in BLEND_WEIGHTS)
    unique = {}
    for spec in options:
        # cached_series는 단순 문자열 또는 문자열 두 개의 1단 앙상블만 허용한다.
        if isinstance(spec, tuple) and (not isinstance(spec[0], str) or not isinstance(spec[1], str)):
            continue
        unique[spec_key(spec)] = spec
    return list(unique.values())


def option_rank_score(panel, feature, direction, horizon, spec, cache):
    exact, one, rank_errors = [], [], []
    for year in DEV_YEARS:
        actual = panel[panel["연도"] == year].sort_values("자치구")
        pred = actual["자치구"].map(cached_series(cache, year, feature, spec)).astype(float)
        a = actual[feature].astype(float).reset_index(drop=True)
        p = pred.reset_index(drop=True)
        aq = risk_quintile(a, direction >= 0).to_numpy()
        pq = risk_quintile(p, direction >= 0).to_numpy()
        exact.append(np.mean(aq == pq)); one.append(np.mean(np.abs(aq - pq) <= 20))
        ar = a.rank(method="average", pct=True).to_numpy()
        pr = p.rank(method="average", pct=True).to_numpy()
        rank_errors.append(np.mean(np.abs(ar - pr)))
    return float(np.mean(exact)), float(np.mean(one)), -float(np.mean(rank_errors))


def metric_key(metrics, yearly_rows):
    yearly_exact = yearly_rows["정확등급정확도"].to_numpy(float)
    return (metrics["정확등급정확도"], float(yearly_exact.min()),
            metrics["1등급이내정확도"], -metrics["평균등급오차"], -metrics["종합점수_MAE"])


def evaluate_map(panel, features, coef, weights, horizon, specs, cache):
    metrics, rows, detail = evaluate_specs(
        panel, features, coef, weights, horizon, DEV_YEARS, specs, cache
    )
    return metric_key(metrics, rows), metrics, rows, detail


def beam_search(panel, features, coef, weights, horizon, base, methods, cache):
    option_rows, choices = [], {}
    for feature, direction in zip(features, coef):
        ranked = []
        for spec in simple_options(base[feature], methods):
            score = option_rank_score(panel, feature, direction, horizon, spec, cache)
            ranked.append((score, spec))
            option_rows.append({"예측기간": f"{horizon}년 후", "지표": feature,
                                "예측법": label(spec), "5분위정확도": score[0],
                                "5분위1칸이내": score[1], "순위오차점수": score[2]})
        ranked.sort(key=lambda x: x[0], reverse=True)
        selected = [base[feature]]
        selected.extend(spec for _, spec in ranked if label(spec) != label(base[feature]))
        choices[feature] = selected[:TOP_OPTIONS]

    base_key, base_metrics, _, _ = evaluate_map(panel, features, coef, weights, horizon, base, cache)
    beam = [(base_key, dict(base), base_metrics)]
    search_rows = [{"예측기간": f"{horizon}년 후", "단계": 0, "순서": "시작",
                    "조합": "74차", **base_metrics}]
    # 정방향과 역방향 두 번을 통과하면서 여러 지표 변경 조합을 함께 보존한다.
    step = 0
    for pass_name, order in (("정방향", features), ("역방향", list(reversed(features)))):
        for feature in order:
            step += 1; expanded = {}
            for _, mapping, _ in beam:
                for spec in choices[feature]:
                    trial = dict(mapping); trial[feature] = spec
                    signature = tuple(label(trial[f]) for f in features)
                    if signature in expanded:
                        continue
                    key, metrics, _, _ = evaluate_map(panel, features, coef, weights, horizon, trial, cache)
                    expanded[signature] = (key, trial, metrics)
            beam = sorted(expanded.values(), key=lambda x: x[0], reverse=True)[:BEAM_WIDTH]
            best = beam[0]
            search_rows.append({"예측기간": f"{horizon}년 후", "단계": step,
                                "순서": pass_name, "조합": f"{feature} 검토", **best[2]})
    best = max(beam, key=lambda x: x[0])
    return best[1], best[2], pd.DataFrame(search_rows), pd.DataFrame(option_rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True); PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    panel = prepare_panel().sort_values(["연도", "자치구"]).reset_index(drop=True)
    features, coef, weights = load_v49()
    v74 = json.loads((ROOT / "models" / "교통사고" / "74차.보유전체데이터활용" / "model_info.json").read_text(encoding="utf-8"))
    base_specs = {h: {f: parse_spec(v) for f, v in v74["결과"][str(h)]["final_specs"].items()} for h in HORIZONS}

    final_specs, results = {}, {}
    all_search, all_options, all_metrics, all_details, forecasts, strategy_rows = [], [], [], [], [], []
    bundles = {}
    for horizon in HORIZONS:
        _, methods, cache, _, _ = build_prediction_cache(panel, features, horizon)
        data, predictor_cols, pool, methods, cache = add_expanded_cache(panel, features, horizon, cache, methods)
        selected, dev_metrics, search, options = beam_search(
            panel, features, coef, weights, horizon, base_specs[horizon], methods, cache
        )
        final_specs[horizon] = selected; all_search.append(search); all_options.append(options)
        dev, dev_rows, dev_detail = evaluate_specs(panel, features, coef, weights, horizon, DEV_YEARS, selected, cache)
        hold, hold_rows, hold_detail = evaluate_specs(panel, features, coef, weights, horizon, [HOLDOUT_YEAR], selected, cache)
        base_hold, base_rows, _ = evaluate_specs(panel, features, coef, weights, horizon, [HOLDOUT_YEAR], base_specs[horizon], cache)
        dev_rows["구분"] = "78차개발"; hold_rows["구분"] = "78차2025독립"; base_rows["구분"] = "74차2025"
        all_metrics.extend([dev_rows, hold_rows, base_rows])
        dev_detail["구분"] = "78차개발"; hold_detail["구분"] = "78차2025독립"
        all_details.extend([dev_detail, hold_detail])

        future, bundle = fit_forecast(panel, data, predictor_cols, pool, features, horizon, selected)
        scores = score_frame(future, features, coef, weights, "예측_")
        output = future.merge(scores, on=["연도", "자치구"], validate="one_to_one")
        output.insert(0, "예측기간", f"{horizon}년 후"); forecasts.append(output)
        bundles[horizon] = bundle
        strategy_rows.extend({"예측기간": f"{horizon}년 후", "지표": f,
                              "74차예측법": label(base_specs[horizon][f]),
                              "78차후보예측법": label(selected[f])} for f in features)
        results[str(horizon)] = {"development": dev, "holdout_2025": hold,
                                 "v74_holdout_2025": base_hold,
                                 "changed_features": [f for f in features if label(selected[f]) != label(base_specs[horizon][f])]}

    integrated = float(np.mean([results[str(h)]["holdout_2025"]["정확등급정확도"] for h in HORIZONS]))
    adopted = integrated > .56 + 1e-9
    weight_table = pd.DataFrame({"지표": features, "49차표준화계수": coef,
                                 "위험방향": np.where(coef >= 0, "값이 클수록 위험", "값이 작을수록 위험"),
                                 "49차고정가중치_pct": weights * 100}).sort_values("49차고정가중치_pct", ascending=False)
    with pd.ExcelWriter(PROCESSED, engine="openpyxl") as writer:
        weight_table.to_excel(writer, sheet_name="고정13지표가중치", index=False)
        pd.DataFrame(strategy_rows).to_excel(writer, sheet_name="74_78예측법비교", index=False)
        pd.concat(all_options, ignore_index=True).to_excel(writer, sheet_name="지표별순위예측평가", index=False)
        pd.concat(all_search, ignore_index=True).to_excel(writer, sheet_name="공동빔서치과정", index=False)
        pd.concat(all_metrics, ignore_index=True).to_excel(writer, sheet_name="성능평가", index=False)
        pd.concat(all_details, ignore_index=True).to_excel(writer, sheet_name="검증상세", index=False)
        pd.concat(forecasts, ignore_index=True).to_excel(writer, sheet_name="2027_2028예측", index=False)
    info = {"모델명": "교통사고 78차 13지표 최종 공동 앙상블 실험",
            "채택상태": "채택" if adopted else "실험탈락_현행74차유지",
            "불변근간": "13개 미래값 예측→5분위→49차 고정가중치→종합위험점수→안전 1~5등급",
            "탐색": f"지표별 순위·5분위 상위 {TOP_OPTIONS}개 예측법, beam width {BEAM_WIDTH}, 정·역방향 공동탐색",
            "결과": results, "2025통합정확등급": integrated,
            "74차통합정확등급": .56, "processed": str(PROCESSED.relative_to(ROOT))}
    joblib.dump({"version": VERSION, "adopted": adopted, "features": features,
                 "coefficients": dict(zip(features, coef)), "weights": dict(zip(features, weights)),
                 "strategies": final_specs, "forecast_models": bundles},
                OUT / "교통사고_78차_13지표최종공동앙상블모델.joblib")
    (OUT / "model_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
