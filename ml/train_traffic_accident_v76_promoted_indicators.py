from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.train_best_child_traffic_accident_rate import TARGET
from ml.train_traffic_accident_v44 import prepare_panel
from ml.train_traffic_accident_v71_fixed_rubric import load_v49, objective, score_frame
from ml.train_traffic_accident_v72_indicator_forecast import DEV_YEARS, HOLDOUT_YEAR, build_prediction_cache
from ml.train_traffic_accident_v73_indicator_ensemble import evaluate_specs, fit_and_forecast, label, select_ensemble
from ml.train_traffic_accident_v74_all_available import add_expanded_cache, fit_forecast


ROOT = Path(__file__).resolve().parents[1]
VERSION = 76
OUT = ROOT / "models" / "교통사고" / "76차.보조지표메인승격"
PROCESSED = ROOT / "data" / "processed" / "교통사고_76차_보조지표메인승격_2027_2028.xlsx"
HORIZONS = (2, 3)


def parse_spec(value: str):
    """model_info에 사람이 읽는 문자열로 저장된 74차 앙상블을 복원한다."""
    if "+" not in value or "×" not in value:
        return value
    left, right = value.split("+", 1)
    left_name, left_weight = left.rsplit("×", 1)
    right_name, _ = right.rsplit("×", 1)
    return (left_name, right_name, float(left_weight))


def ridge_weights(panel: pd.DataFrame, features: list[str]):
    train = panel[panel[TARGET].notna()].copy()
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=.1)),
    ]).fit(train[features], train[TARGET])
    coef = np.asarray(model.named_steps["model"].coef_, float)
    total = np.abs(coef).sum()
    weights = np.abs(coef) / total if total else np.repeat(1 / len(coef), len(coef))
    return model, coef, weights


def safe_candidates(panel: pd.DataFrame, core: list[str]) -> list[str]:
    excluded_exact = {
        "연도", "예측연도", "행정구역코드", "자치구코드", TARGET,
        "현재연도_어린이교통사고피해자율_어린이1만명당",
        "어린이교통사고피해자수", "어린이 교통사고 사망자 수",
        "어린이 교통사고 부상자 수", "어린이부상률_어린이1만명당",
    }
    excluded_tokens = ("어린이피해자율_최근", "다음연도_")
    numeric = panel.select_dtypes(include=[np.number]).columns
    return [c for c in numeric if c not in core and c not in excluded_exact
            and not any(token in c for token in excluded_tokens)
            and panel[c].nunique(dropna=True) > 1
            # 연도·자치구별 5분위는 25개 값이 모두 있어야 동일 기준으로 계산된다.
            and panel[c].notna().all() and np.isfinite(panel[c].to_numpy(float)).all()]


def add_persistence_cache(panel, candidates, horizon, cache):
    for year in (*DEV_YEARS, HOLDOUT_YEAR):
        origin = year - horizon
        current = panel[panel["연도"] == origin].sort_values("자치구")
        for feature in candidates:
            cache[(year, feature, "Persistence")] = pd.Series(
                current[feature].to_numpy(float), index=current["자치구"].to_numpy()
            )


def average_metrics(items):
    keys = ["정확등급정확도", "1등급이내정확도", "평균등급오차", "종합점수_MAE"]
    return {k: float(np.mean([x[k] for x in items])) for k in keys}


def evaluate_feature_set(panel, features, coef, weights, caches, specs, years):
    by_horizon = {}
    for horizon in HORIZONS:
        metrics, rows, detail = evaluate_specs(
            panel, features, coef, weights, horizon, years, specs[horizon], caches[horizon]
        )
        by_horizon[horizon] = (metrics, rows, detail)
    return average_metrics([v[0] for v in by_horizon.values()]), by_horizon


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    panel = prepare_panel().sort_values(["연도", "자치구"]).reset_index(drop=True)
    core, core_coef, core_weights = load_v49()
    candidates = safe_candidates(panel, core)
    v74_info = json.loads((ROOT / "models" / "교통사고" / "74차.보유전체데이터활용" / "model_info.json").read_text(encoding="utf-8"))

    caches, methods_by_h, data_by_h, numeric_by_h = {}, {}, {}, {}
    core_specs = {
        h: {feature: parse_spec(value) for feature, value in v74_info["결과"][str(h)]["final_specs"].items()}
        for h in HORIZONS
    }
    for horizon in HORIZONS:
        data, methods, cache, numeric_all, _ = build_prediction_cache(panel, core, horizon)
        expanded_data, predictor_cols, pool, methods, cache = add_expanded_cache(panel, core, horizon, cache, methods)
        add_persistence_cache(panel, candidates, horizon, cache)
        caches[horizon], methods_by_h[horizon] = cache, methods
        data_by_h[horizon], numeric_by_h[horizon] = expanded_data, (predictor_cols, pool)

    selected = list(core)
    specs = {h: dict(core_specs[h]) for h in HORIZONS}
    base_avg, _ = evaluate_feature_set(panel, selected, core_coef, core_weights, caches, specs, DEV_YEARS)
    search_log = [{"단계": 0, "작업": "74차 시작점", "변수": "-", "지표수": len(selected), **base_avg}]
    current = base_avg

    # 49차 13개는 고정하고, 보조 후보를 하나씩 메인 지표로 승격한다.
    for step in range(1, 13):
        best = None
        for feature in candidates:
            if feature in selected:
                continue
            trial_features = [*selected, feature]
            _, coef, weights = ridge_weights(panel, trial_features)
            trial_specs = {h: {**specs[h], feature: "Persistence"} for h in HORIZONS}
            avg, _ = evaluate_feature_set(panel, trial_features, coef, weights, caches, trial_specs, DEV_YEARS)
            record = (objective(avg), feature, avg, coef, weights, trial_specs)
            if best is None or record[0] > best[0]:
                best = record
        if best is None or best[2]["정확등급정확도"] <= current["정확등급정확도"]:
            search_log.append({"단계": step, "작업": "추가중단", "변수": best[1] if best else "-",
                               "지표수": len(selected), **(best[2] if best else current)})
            break
        _, feature, current, _, _, specs = best
        selected.append(feature)
        search_log.append({"단계": step, "작업": "메인지표추가", "변수": feature,
                           "지표수": len(selected), **current})

    # 전진선택에서 검증된 조합을 그대로 확정한다. 핵심 13개는 74차 예측법,
    # 새 승격 지표는 과적합 위험이 가장 낮은 Persistence를 쓴다.
    final_model, final_coef, final_weights = ridge_weights(panel, selected)
    final_specs = specs
    selection_logs = []
    final_caches, final_data, final_numeric = caches, data_by_h, numeric_by_h
    for horizon in HORIZONS:
        selection_logs.append(pd.DataFrame([{
            "예측기간": f"{horizon}년 후", "단계": "최종확정", "지표": f,
            "조합": label(final_specs[horizon][f])
        } for f in selected]))

    dev_avg, dev_by_h = evaluate_feature_set(
        panel, selected, final_coef, final_weights, final_caches, final_specs, DEV_YEARS
    )
    hold_avg, hold_by_h = evaluate_feature_set(
        panel, selected, final_coef, final_weights, final_caches, final_specs, [HOLDOUT_YEAR]
    )

    metrics_rows, detail_rows, forecasts, strategy_rows, bundles = [], [], [], [], {}
    for horizon in HORIZONS:
        for scope, result in (("개발_2021_2024", dev_by_h[horizon]), ("독립검증_2025", hold_by_h[horizon])):
            rows = result[1].copy(); rows["구분"] = scope; metrics_rows.append(rows)
            detail = result[2].copy(); detail["구분"] = scope; detail_rows.append(detail)
        predictor_cols, pool = final_numeric[horizon]
        future, bundle = fit_forecast(
            panel, final_data[horizon], predictor_cols, pool, core, horizon, core_specs[horizon]
        )
        # 승격 지표는 전진선택 때와 동일하게 2025 현재값 유지 예측을 적용한다.
        latest = panel[panel["연도"] == 2025].set_index("자치구")
        for promoted in (f for f in selected if f not in core):
            future[promoted] = future["자치구"].map(latest[promoted])
            bundle[promoted] = {"spec": "Persistence", "components": {}}
        scored = score_frame(future, selected, final_coef, final_weights, "예측_")
        out = future.merge(scored, on=["연도", "자치구"], validate="one_to_one")
        out.insert(0, "예측기간", f"{horizon}년 후"); forecasts.append(out)
        bundles[horizon] = bundle
        joblib.dump(bundle, OUT / f"교통사고_76차_{horizon}년후_지표예측모델.joblib")
        strategy_rows.extend({"예측기간": f"{horizon}년 후", "지표": f,
                              "예측법": label(final_specs[horizon][f])} for f in selected)

    weight_table = pd.DataFrame({
        "지표": selected,
        "구분": ["49차 핵심고정" if f in core else "74차 보조→메인승격" for f in selected],
        "표준화계수": final_coef,
        "위험방향": np.where(final_coef >= 0, "값이 클수록 위험", "값이 작을수록 위험"),
        "가중치_pct": final_weights * 100,
    }).sort_values("가중치_pct", ascending=False)

    with pd.ExcelWriter(PROCESSED, engine="openpyxl") as writer:
        weight_table.to_excel(writer, sheet_name="최종메인지표와가중치", index=False)
        pd.DataFrame(search_log).to_excel(writer, sheet_name="부분집합추가과정", index=False)
        pd.DataFrame(strategy_rows).to_excel(writer, sheet_name="지표별미래예측법", index=False)
        pd.concat(metrics_rows, ignore_index=True).to_excel(writer, sheet_name="성능평가", index=False)
        pd.concat(detail_rows, ignore_index=True).to_excel(writer, sheet_name="검증상세", index=False)
        pd.concat(selection_logs, ignore_index=True).to_excel(writer, sheet_name="예측법탐색과정", index=False)
        pd.concat(forecasts, ignore_index=True).to_excel(writer, sheet_name="2027_2028예측", index=False)

    joblib.dump({"version": VERSION, "features": selected, "core_features": core,
                 "promoted_features": [f for f in selected if f not in core],
                 "rubric_model": final_model, "coefficients": dict(zip(selected, final_coef)),
                 "weights": dict(zip(selected, final_weights)), "forecast_models": bundles,
                 "safety_grade_orientation": "1=가장 안전, 5=가장 위험"},
                OUT / "교통사고_76차_보조지표메인승격모델.joblib")
    info = {
        "모델명": "교통사고 76차 보조지표 메인 승격 모델",
        "채택상태": "실험탈락_현행74차유지",
        "탈락사유": "2025 독립검증 통합 정확등급 정확도 54%로 74차 56%보다 낮음",
        "불변근간": "미래 지표별 예측→5분위→사고피해자율 Ridge 계수 기반 가중치→종합위험점수→안전 1~5등급",
        "49차핵심13개고정": True,
        "최종지표수": len(selected), "승격지표": [f for f in selected if f not in core],
        "개발통합성능": dev_avg, "2025독립통합성능": hold_avg,
        "기간별개발성능": {str(h): dev_by_h[h][0] for h in HORIZONS},
        "기간별2025성능": {str(h): hold_by_h[h][0] for h in HORIZONS},
        "processed": str(PROCESSED.relative_to(ROOT)),
    }
    (OUT / "model_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
