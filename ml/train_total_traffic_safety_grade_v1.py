from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.train_traffic_accident_v44 import prepare_panel
from ml.train_traffic_accident_v71_fixed_rubric import objective, score_frame
from ml.train_traffic_accident_v72_indicator_forecast import DEV_YEARS, HOLDOUT_YEAR, build_prediction_cache
from ml.train_traffic_accident_v73_indicator_ensemble import evaluate_specs, label, select_ensemble
from ml.train_traffic_accident_v74_all_available import add_expanded_cache, fit_forecast


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "교통사고" / "전체교통안전등급_1차"
PROCESSED = ROOT / "data" / "processed" / "전체교통안전등급_1차_2027_2028.xlsx"
Y = "다음연도_전체교통사고건수"
CURRENT_COUNT = "교통사고 건수"
HORIZONS = (2, 3)

CHILD_MANDATORY = [
    "어린이 교통사고 건수",
    "현재연도_어린이교통사고피해자율_어린이1만명당",
    "보행어린이교통사고율_어린이1만명당",
    "보호구역보행어린이교통사고율_어린이1만명당",
]


def model(alpha):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])


def metrics(y, pred):
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": float(r2_score(y, pred)),
        "WMAPE_pct": float(np.abs(np.asarray(y) - pred).sum() / np.abs(np.asarray(y)).sum() * 100),
    }


def candidate_columns(panel):
    excluded = {"연도", "예측연도", "행정구역코드", "자치구코드", Y}
    excluded_tokens = ("다음연도_",)
    return [c for c in panel.select_dtypes(include=[np.number]).columns
            if c not in excluded and not any(t in c for t in excluded_tokens)
            and panel[c].nunique(dropna=True) > 1]


def choose_features(panel, candidates):
    labeled = panel[panel[Y].notna()].copy()
    imputed = SimpleImputer(strategy="median").fit_transform(labeled[candidates])
    scores, _ = f_regression(imputed, labeled[Y])
    ranking = [candidates[i] for i in np.argsort(-np.nan_to_num(scores, nan=-np.inf))]
    mandatory = [c for c in CHILD_MANDATORY if c in candidates]
    rows, yearly_rows, configs = [], [], {}
    for k in (10, 15, 20, 25, 30, 40):
        features = list(dict.fromkeys([*mandatory, *ranking]))[:max(k, len(mandatory))]
        for alpha in (.1, 1, 10, 30, 100):
            yearly = []
            for year in (2021, 2022, 2023, 2024):
                train = panel[(panel[Y].notna()) & (panel["예측연도"] < year)]
                valid = panel[panel["예측연도"] == year]
                fitted = model(alpha).fit(train[features], train[Y])
                pred = np.maximum(fitted.predict(valid[features]), 0)
                result = metrics(valid[Y], pred); yearly.append(result)
                yearly_rows.append({"변수수": len(features), "alpha": alpha, "검증연도": year, **result})
            summary = {"변수수": len(features), "alpha": alpha,
                       "평균_MAE": np.mean([r["MAE"] for r in yearly]),
                       "평균_RMSE": np.mean([r["RMSE"] for r in yearly]),
                       "평균_R2": np.mean([r["R2"] for r in yearly]),
                       "평균_WMAPE_pct": np.mean([r["WMAPE_pct"] for r in yearly])}
            key = (len(features), alpha); configs[key] = features; rows.append(summary)
    table = pd.DataFrame(rows).sort_values(["평균_MAE", "평균_RMSE"])
    best = table.iloc[0]
    return configs[(int(best["변수수"]), float(best["alpha"]))], float(best["alpha"]), table, pd.DataFrame(yearly_rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True); PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    panel = prepare_panel().sort_values(["자치구", "연도"]).reset_index(drop=True)
    panel[Y] = panel.groupby("자치구", sort=False)[CURRENT_COUNT].shift(-1)
    candidates = candidate_columns(panel)
    features, alpha, model_table, yearly_table = choose_features(panel, candidates)

    # 2025년 전체 사고 건수 독립검증
    train = panel[(panel[Y].notna()) & (panel["예측연도"] <= 2024)]
    hold = panel[panel["예측연도"] == 2025]
    count_model = model(alpha).fit(train[features], train[Y])
    count_pred = np.maximum(count_model.predict(hold[features]), 0)
    count_hold_metrics = metrics(hold[Y], count_pred)
    hold_detail = hold[["연도", "예측연도", "자치구", Y]].copy()
    hold_detail["예측_전체교통사고건수"] = count_pred
    hold_detail["절대오차"] = np.abs(hold_detail[Y] - count_pred)

    # 전체 사고 건수 Ridge 표준화계수로 방향·가중치 결정
    full_train = panel[panel[Y].notna()]
    rubric_model = model(alpha).fit(full_train[features], full_train[Y])
    coef = np.asarray(rubric_model.named_steps["model"].coef_, float)
    weights = np.abs(coef) / np.abs(coef).sum()
    weight_table = pd.DataFrame({
        "지표": features, "표준화계수": coef,
        "위험방향": np.where(coef >= 0, "값이 클수록 위험", "값이 작을수록 위험"),
        "가중치_pct": weights * 100,
        "어린이교통사고지표": [f in CHILD_MANDATORY for f in features],
    }).sort_values("가중치_pct", ascending=False)

    grade_results, grade_metrics, grade_details, search_logs = {}, [], [], []
    forecasts, strategy_rows, bundles = [], [], {}
    for horizon in HORIZONS:
        data, methods, cache, _, _ = build_prediction_cache(panel, features, horizon)
        data, predictor_cols, pool, methods, cache = add_expanded_cache(panel, features, horizon, cache, methods)
        persistence = {f: "Persistence" for f in features}
        selected, dev, log = select_ensemble(panel, features, coef, weights, horizon, methods, cache, persistence)
        base_dev, _, _ = evaluate_specs(panel, features, coef, weights, horizon, DEV_YEARS, persistence, cache)
        hold_grade, hold_rows, hold_joined = evaluate_specs(
            panel, features, coef, weights, horizon, [HOLDOUT_YEAR], selected, cache
        )
        base_hold, base_rows, _ = evaluate_specs(
            panel, features, coef, weights, horizon, [HOLDOUT_YEAR], persistence, cache
        )
        # 개발 정확등급이 좋아진 조합만 후보로 사용한다. 2025 결과는 보고만 한다.
        final_specs = selected if objective(dev) > objective(base_dev) else persistence
        final_hold, final_rows, final_detail = evaluate_specs(
            panel, features, coef, weights, horizon, [HOLDOUT_YEAR], final_specs, cache
        )
        future, bundle = fit_forecast(panel, data, predictor_cols, pool, features, horizon, final_specs)
        scores = score_frame(future, features, coef, weights, "예측_")
        output = future.merge(scores, on=["연도", "자치구"], validate="one_to_one")
        output.insert(0, "예측기간", f"{horizon}년 후"); forecasts.append(output)
        bundles[horizon] = bundle
        log.insert(0, "예측기간", f"{horizon}년 후"); search_logs.append(log)
        final_rows["구분"] = "전체교통안전등급1차_2025"; base_rows["구분"] = "Persistence_2025"
        grade_metrics.extend([final_rows, base_rows])
        final_detail["예측기간"] = f"{horizon}년 후"; grade_details.append(final_detail)
        strategy_rows.extend({"예측기간": f"{horizon}년 후", "지표": f,
                              "선정예측법": label(final_specs[f])} for f in features)
        grade_results[str(horizon)] = {"development": dev, "holdout_2025": final_hold,
                                       "persistence_2025": base_hold}

    with pd.ExcelWriter(PROCESSED, engine="openpyxl") as writer:
        weight_table.to_excel(writer, sheet_name="전체사고등급지표가중치", index=False)
        model_table.to_excel(writer, sheet_name="건수모델선정", index=False)
        yearly_table.to_excel(writer, sheet_name="건수모델연도별검증", index=False)
        hold_detail.to_excel(writer, sheet_name="건수2025독립검증", index=False)
        pd.DataFrame(strategy_rows).to_excel(writer, sheet_name="지표별미래예측법", index=False)
        pd.concat(grade_metrics, ignore_index=True).to_excel(writer, sheet_name="등급성능", index=False)
        pd.concat(grade_details, ignore_index=True).to_excel(writer, sheet_name="등급검증상세", index=False)
        pd.concat(search_logs, ignore_index=True).to_excel(writer, sheet_name="등급모델탐색", index=False)
        pd.concat(forecasts, ignore_index=True).to_excel(writer, sheet_name="2027_2028예측", index=False)

    info = {
        "모델명": "전체 교통안전등급 1차",
        "74차와의관계": "별도 목적 모델; 74차 어린이 교통안전등급은 변경하지 않음",
        "Y": "다음 연도 자치구별 전체 교통사고 발생 건수",
        "구조": "미래 지표별 예측→자치구 5분위→전체 사고 건수 Ridge 계수 가중치→종합위험점수→안전 1~5등급",
        "안전등급방향": "1=가장 안전, 5=가장 위험",
        "변수수": len(features), "변수": features, "Ridge_alpha": alpha,
        "2025건수독립검증": count_hold_metrics, "등급결과": grade_results,
        "processed": str(PROCESSED.relative_to(ROOT)),
    }
    joblib.dump({"version": 1, "target": Y, "features": features, "count_model": rubric_model,
                 "coefficients": dict(zip(features, coef)), "weights": dict(zip(features, weights)),
                 "forecast_models": bundles, "safety_grade_orientation": "1=가장 안전, 5=가장 위험"},
                OUT / "전체교통안전등급_1차.joblib")
    (OUT / "model_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
