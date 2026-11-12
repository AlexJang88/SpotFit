from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.train_traffic_accident_v44 import prepare_panel
from ml.train_traffic_accident_v71_fixed_rubric import load_v49, risk_quintile, score_frame
from ml.train_traffic_accident_v72_indicator_forecast import DEV_YEARS, HOLDOUT_YEAR, build_prediction_cache
from ml.train_traffic_accident_v73_indicator_ensemble import cached_series, evaluate_specs, label
from ml.train_traffic_accident_v74_all_available import add_expanded_cache, fit_forecast
from ml.train_traffic_accident_v76_promoted_indicators import parse_spec
from ml.train_traffic_accident_v78_final_joint_ensemble import option_rank_score, simple_options


ROOT = Path(__file__).resolve().parents[1]
VERSION = 82
OUT = ROOT / "models" / "교통사고" / "82차.안정지표Persistence변화지표예측"
PROCESSED = ROOT / "data" / "processed" / "교통사고_82차_안정지표Persistence_2027_2028.xlsx"
HORIZONS = (2, 3)
STABLE_QUINTILE_THRESHOLD = .75


def stability(panel, feature, direction, horizon, cache):
    exact, one, rank_mae = [], [], []
    for year in DEV_YEARS:
        actual = panel[panel["연도"] == year].sort_values("자치구")
        pred = actual["자치구"].map(cached_series(cache, year, feature, "Persistence")).astype(float)
        a = actual[feature].astype(float).reset_index(drop=True)
        p = pred.reset_index(drop=True)
        aq = risk_quintile(a, direction >= 0).to_numpy()
        pq = risk_quintile(p, direction >= 0).to_numpy()
        exact.append(np.mean(aq == pq)); one.append(np.mean(np.abs(aq-pq) <= 20))
        rank_mae.append(np.mean(np.abs(a.rank(pct=True).to_numpy() - p.rank(pct=True).to_numpy())))
    return {"Persistence_5분위정확도": float(np.mean(exact)),
            "Persistence_5분위1칸이내": float(np.mean(one)),
            "Persistence_순위MAE": float(np.mean(rank_mae))}


def main():
    OUT.mkdir(parents=True, exist_ok=True); PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    panel = prepare_panel().sort_values(["연도", "자치구"]).reset_index(drop=True)
    features, coef, weights = load_v49()
    v74 = json.loads((ROOT / "models" / "교통사고" / "74차.보유전체데이터활용" / "model_info.json").read_text(encoding="utf-8"))
    v74_specs = {h: {f: parse_spec(v) for f, v in v74["결과"][str(h)]["final_specs"].items()} for h in HORIZONS}

    results, stability_rows, option_rows, strategy_rows = {}, [], [], []
    metrics_rows, detail_rows, forecasts, bundles = [], [], [], {}
    for horizon in HORIZONS:
        _, methods, cache, _, _ = build_prediction_cache(panel, features, horizon)
        data, predictor_cols, pool, methods, cache = add_expanded_cache(panel, features, horizon, cache, methods)
        selected = {}
        for feature, direction in zip(features, coef):
            stable = stability(panel, feature, direction, horizon, cache)
            is_stable = stable["Persistence_5분위정확도"] >= STABLE_QUINTILE_THRESHOLD
            if is_stable:
                best_spec = "Persistence"
                best_score = option_rank_score(panel, feature, direction, horizon, best_spec, cache)
            else:
                ranked = []
                for spec in simple_options(v74_specs[horizon][feature], methods):
                    score = option_rank_score(panel, feature, direction, horizon, spec, cache)
                    ranked.append((score, spec))
                    option_rows.append({"예측기간": f"{horizon}년 후", "지표": feature,
                                        "예측법": label(spec), "5분위정확도": score[0],
                                        "5분위1칸이내": score[1], "순위오차점수": score[2]})
                best_score, best_spec = max(ranked, key=lambda x: x[0])
            selected[feature] = best_spec
            stability_rows.append({"예측기간": f"{horizon}년 후", "지표": feature, **stable,
                                   "분류": "안정형_Persistence" if is_stable else "변화형_예측모델",
                                   "82차선정법": label(best_spec), "선정법_5분위정확도": best_score[0]})
            strategy_rows.append({"예측기간": f"{horizon}년 후", "지표": feature,
                                  "74차예측법": label(v74_specs[horizon][feature]),
                                  "82차예측법": label(best_spec)})

        dev, dev_rows, dev_detail = evaluate_specs(panel, features, coef, weights, horizon, DEV_YEARS, selected, cache)
        hold, hold_rows, hold_detail = evaluate_specs(panel, features, coef, weights, horizon, [HOLDOUT_YEAR], selected, cache)
        base_hold, base_rows, _ = evaluate_specs(panel, features, coef, weights, horizon, [HOLDOUT_YEAR], v74_specs[horizon], cache)
        dev_rows["구분"] = "82차개발"; hold_rows["구분"] = "82차2025독립"; base_rows["구분"] = "74차2025"
        metrics_rows.extend([dev_rows, hold_rows, base_rows])
        dev_detail["구분"] = "82차개발"; hold_detail["구분"] = "82차2025독립"
        detail_rows.extend([dev_detail, hold_detail])
        future, bundle = fit_forecast(panel, data, predictor_cols, pool, features, horizon, selected)
        scored = score_frame(future, features, coef, weights, "예측_")
        output = future.merge(scored, on=["연도", "자치구"], validate="one_to_one")
        output.insert(0, "예측기간", f"{horizon}년 후"); forecasts.append(output)
        bundles[horizon] = bundle
        results[str(horizon)] = {
            "development": dev, "holdout_2025": hold, "v74_holdout_2025": base_hold,
            "stable_features": [r["지표"] for r in stability_rows if r["예측기간"] == f"{horizon}년 후" and r["분류"].startswith("안정형")],
            "changing_features": [r["지표"] for r in stability_rows if r["예측기간"] == f"{horizon}년 후" and r["분류"].startswith("변화형")],
        }

    integrated = float(np.mean([results[str(h)]["holdout_2025"]["정확등급정확도"] for h in HORIZONS]))
    with pd.ExcelWriter(PROCESSED, engine="openpyxl") as writer:
        pd.DataFrame(stability_rows).to_excel(writer, sheet_name="지표안정성분류", index=False)
        pd.DataFrame(strategy_rows).to_excel(writer, sheet_name="74_82예측법비교", index=False)
        pd.DataFrame(option_rows).to_excel(writer, sheet_name="변화지표모델평가", index=False)
        pd.concat(metrics_rows, ignore_index=True).to_excel(writer, sheet_name="성능평가", index=False)
        pd.concat(detail_rows, ignore_index=True).to_excel(writer, sheet_name="검증상세", index=False)
        pd.concat(forecasts, ignore_index=True).to_excel(writer, sheet_name="2027_2028예측", index=False)
    info = {"모델명": "교통사고 82차 안정지표 Persistence·변화지표 예측모델",
            "불변근간": "49차 13지표·가중치·방향 고정→미래값 예측→5분위→종합점수→안전 1~5등급",
            "안정판정": f"개발구간 Persistence 5분위 정확도 {STABLE_QUINTILE_THRESHOLD:.0%} 이상",
            "결과": results, "2025통합정확등급": integrated, "74차통합정확등급": .56,
            "채택상태": "채택" if integrated > .56 + 1e-9 else "실험탈락_74차유지",
            "processed": str(PROCESSED.relative_to(ROOT))}
    joblib.dump({"version": VERSION, "features": features, "coefficients": dict(zip(features, coef)),
                 "weights": dict(zip(features, weights)), "forecast_models": bundles,
                 "safety_grade_orientation": "1=가장 안전, 5=가장 위험"},
                OUT / "교통사고_82차_안정성분기모델.joblib")
    (OUT / "model_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
