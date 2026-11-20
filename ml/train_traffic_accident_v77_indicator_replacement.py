from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from ml.train_traffic_accident_v44 import prepare_panel
from ml.train_traffic_accident_v71_fixed_rubric import load_v49, objective, risk_quintile, score_frame
from ml.train_traffic_accident_v72_indicator_forecast import DEV_YEARS, HOLDOUT_YEAR, build_prediction_cache
from ml.train_traffic_accident_v73_indicator_ensemble import cached_series, evaluate_specs, label
from ml.train_traffic_accident_v74_all_available import add_expanded_cache, fit_forecast
from ml.train_traffic_accident_v76_promoted_indicators import (
    add_persistence_cache, average_metrics, evaluate_feature_set, parse_spec,
    ridge_weights, safe_candidates,
)


ROOT = Path(__file__).resolve().parents[1]
VERSION = 77
OUT = ROOT / "models" / "교통사고" / "77차.불안정지표교체"
PROCESSED = ROOT / "data" / "processed" / "교통사고_77차_불안정지표교체_2027_2028.xlsx"
HORIZONS = (2, 3)


def indicator_diagnostics(panel, core, coef, caches, specs):
    rows = []
    for horizon in HORIZONS:
        for year in (*DEV_YEARS, HOLDOUT_YEAR):
            actual = panel[panel["연도"] == year].sort_values("자치구")
            for feature, direction in zip(core, coef):
                pred = actual["자치구"].map(cached_series(caches[horizon], year, feature, specs[horizon][feature]))
                a = actual[feature].astype(float)
                # 대부분 0인 희소 지표에서 범위가 0에 가까워지는 경우의 폭주를 막는다.
                scale = max(float(a.max() - a.min()), float(a.abs().mean()) * .1, 1.0)
                aq = risk_quintile(a.reset_index(drop=True), direction >= 0)
                pq = risk_quintile(pred.reset_index(drop=True), direction >= 0)
                rows.append({
                    "예측기간": f"{horizon}년 후", "검증연도": year, "지표": feature,
                    "MAE": mean_absolute_error(a, pred), "범위정규화_MAE": mean_absolute_error(a, pred) / scale,
                    "5분위정확도": float(np.mean(aq.to_numpy() == pq.to_numpy())),
                    "5분위1칸이내": float(np.mean(np.abs(aq.to_numpy() - pq.to_numpy()) <= 20)),
                })
    detail = pd.DataFrame(rows)
    dev = detail[detail["검증연도"].isin(DEV_YEARS)].groupby("지표", as_index=False).agg(
        개발_범위정규화_MAE=("범위정규화_MAE", "mean"),
        개발_5분위정확도=("5분위정확도", "mean"),
        개발_5분위1칸이내=("5분위1칸이내", "mean"),
    )
    hold = detail[detail["검증연도"] == HOLDOUT_YEAR].groupby("지표", as_index=False).agg(
        독립검증_범위정규화_MAE=("범위정규화_MAE", "mean"),
        독립검증_5분위정확도=("5분위정확도", "mean"),
    )
    summary = dev.merge(hold, on="지표")
    summary["불안정점수"] = summary["개발_범위정규화_MAE"] + (1 - summary["개발_5분위정확도"])
    return detail, summary.sort_values("불안정점수", ascending=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True); PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    panel = prepare_panel().sort_values(["연도", "자치구"]).reset_index(drop=True)
    core, core_coef, core_weights = load_v49()
    candidates = safe_candidates(panel, core)
    v74 = json.loads((ROOT / "models" / "교통사고" / "74차.보유전체데이터활용" / "model_info.json").read_text(encoding="utf-8"))
    core_specs = {h: {f: parse_spec(v) for f, v in v74["결과"][str(h)]["final_specs"].items()} for h in HORIZONS}

    caches, expanded_data, expanded_parts = {}, {}, {}
    for horizon in HORIZONS:
        _, methods, cache, _, _ = build_prediction_cache(panel, core, horizon)
        data, predictor_cols, pool, methods, cache = add_expanded_cache(panel, core, horizon, cache, methods)
        add_persistence_cache(panel, candidates, horizon, cache)
        caches[horizon], expanded_data[horizon], expanded_parts[horizon] = cache, data, (predictor_cols, pool)

    diag_detail, diag_summary = indicator_diagnostics(panel, core, core_coef, caches, core_specs)
    # 진단 순서는 기록하되, 교체 탐색에서는 13개 모두를 빠짐없이 검토한다.
    unstable = diag_summary["지표"].tolist()
    selected = list(core); specs = {h: dict(core_specs[h]) for h in HORIZONS}
    base_dev, _ = evaluate_feature_set(panel, selected, core_coef, core_weights, caches, specs, DEV_YEARS)
    current = base_dev
    search = [{"단계": 0, "작업": "74차 시작", "제거": "-", "추가": "-", **current}]

    # 개발 롤링검증만으로 최대 3회의 1:1 교체를 선택한다. 2025는 선택에 쓰지 않는다.
    replaced_originals = set()
    for step in range(1, 4):
        best = None
        removable = [f for f in unstable if f in selected and f not in replaced_originals]
        for removed in removable:
            for added in candidates:
                if added in selected:
                    continue
                trial_features = [f for f in selected if f != removed] + [added]
                _, coef, weights = ridge_weights(panel, trial_features)
                trial_specs = {h: {f: s for f, s in specs[h].items() if f != removed} | {added: "Persistence"}
                               for h in HORIZONS}
                avg, _ = evaluate_feature_set(panel, trial_features, coef, weights, caches, trial_specs, DEV_YEARS)
                record = (objective(avg), removed, added, avg, trial_features, coef, weights, trial_specs)
                if best is None or record[0] > best[0]:
                    best = record
        if best is None or best[3]["정확등급정확도"] <= current["정확등급정확도"] + 1e-9:
            search.append({"단계": step, "작업": "교체중단", "제거": best[1] if best else "-",
                           "추가": best[2] if best else "-", **(best[3] if best else current)})
            break
        _, removed, added, current, selected, _, _, specs = best
        replaced_originals.add(removed)
        search.append({"단계": step, "작업": "1대1교체", "제거": removed, "추가": added, **current})

    rubric_model, coef, weights = ridge_weights(panel, selected)
    dev_avg, dev_by_h = evaluate_feature_set(panel, selected, coef, weights, caches, specs, DEV_YEARS)
    hold_avg, hold_by_h = evaluate_feature_set(panel, selected, coef, weights, caches, specs, [HOLDOUT_YEAR])
    adopted = hold_avg["정확등급정확도"] > .56 + 1e-9

    weights_df = pd.DataFrame({
        "지표": selected, "표준화계수": coef,
        "위험방향": np.where(coef >= 0, "값이 클수록 위험", "값이 작을수록 위험"),
        "가중치_pct": weights * 100,
        "구분": ["49차유지" if f in core else "보조지표교체투입" for f in selected],
    }).sort_values("가중치_pct", ascending=False)
    metric_rows, detail_rows, forecasts, bundles, strategy_rows = [], [], [], {}, []
    latest = panel[panel["연도"] == 2025].set_index("자치구")
    for horizon in HORIZONS:
        for scope, result in (("개발_2021_2024", dev_by_h[horizon]), ("독립검증_2025", hold_by_h[horizon])):
            rows = result[1].copy(); rows["구분"] = scope; metric_rows.append(rows)
            det = result[2].copy(); det["구분"] = scope; detail_rows.append(det)
        predictor_cols, pool = expanded_parts[horizon]
        core_future, bundle = fit_forecast(
            panel, expanded_data[horizon], predictor_cols, pool, core, horizon, core_specs[horizon]
        )
        future = core_future[["연도", "자치구", *[f for f in selected if f in core]]].copy()
        for added in (f for f in selected if f not in core):
            future[added] = future["자치구"].map(latest[added])
            bundle[added] = {"spec": "Persistence", "components": {}}
        scored = score_frame(future, selected, coef, weights, "예측_")
        output = future.merge(scored, on=["연도", "자치구"], validate="one_to_one")
        output.insert(0, "예측기간", f"{horizon}년 후"); forecasts.append(output)
        bundles[horizon] = bundle
        strategy_rows.extend({"예측기간": f"{horizon}년 후", "지표": f,
                              "예측법": label(specs[horizon][f])} for f in selected)

    with pd.ExcelWriter(PROCESSED, engine="openpyxl") as writer:
        diag_summary.to_excel(writer, sheet_name="13지표안정성요약", index=False)
        diag_detail.to_excel(writer, sheet_name="13지표연도별오차", index=False)
        pd.DataFrame(search).to_excel(writer, sheet_name="1대1교체탐색", index=False)
        weights_df.to_excel(writer, sheet_name="후보지표가중치", index=False)
        pd.DataFrame(strategy_rows).to_excel(writer, sheet_name="지표별예측법", index=False)
        pd.concat(metric_rows, ignore_index=True).to_excel(writer, sheet_name="성능평가", index=False)
        pd.concat(detail_rows, ignore_index=True).to_excel(writer, sheet_name="검증상세", index=False)
        pd.concat(forecasts, ignore_index=True).to_excel(writer, sheet_name="2027_2028예측", index=False)

    info = {
        "모델명": "교통사고 77차 불안정 지표 1대1 교체 실험",
        "채택상태": "채택" if adopted else "실험탈락_현행74차유지",
        "불변근간": "미래 지표별 예측→5분위→사고피해자율 Ridge 계수 가중치→종합위험점수→안전 1~5등급",
        "제거지표": [f for f in core if f not in selected],
        "교체투입지표": [f for f in selected if f not in core],
        "최종지표수": len(selected), "개발통합성능": dev_avg, "2025독립통합성능": hold_avg,
        "기간별2025성능": {str(h): hold_by_h[h][0] for h in HORIZONS},
        "74차비교": {"2년정확등급": .56, "3년정확등급": .56, "통합정확등급": .56},
        "processed": str(PROCESSED.relative_to(ROOT)),
    }
    joblib.dump({"version": VERSION, "adopted": adopted, "features": selected,
                 "rubric_model": rubric_model, "weights": dict(zip(selected, weights)),
                 "coefficients": dict(zip(selected, coef)), "forecast_models": bundles},
                OUT / "교통사고_77차_불안정지표교체모델.joblib")
    (OUT / "model_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
