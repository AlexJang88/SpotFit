from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.train_traffic_accident_v44 import prepare_panel
from ml.train_traffic_accident_v71_fixed_rubric import load_v49, score_frame
from ml.train_traffic_accident_v72_indicator_forecast import DEV_YEARS, HOLDOUT_YEAR, build_prediction_cache
from ml.train_traffic_accident_v73_indicator_ensemble import evaluate_specs, label
from ml.train_traffic_accident_v74_all_available import add_expanded_cache, fit_forecast
from ml.train_traffic_accident_v76_promoted_indicators import parse_spec


ROOT = Path(__file__).resolve().parents[1]
VERSION = 83
OUT = ROOT / "models" / "교통사고" / "83차.피해자수Poisson인구노출등급"
PROCESSED = ROOT / "data" / "processed" / "교통사고_83차_피해자수Poisson인구노출등급_2027_2028.xlsx"
VICTIMS = "어린이교통사고피해자수"
POP = "어린이(13세 미만) 인구수"
TARGET_COUNT = "다음연도_어린이교통사고피해자수"
TARGET_POP = "다음연도_어린이인구수"
HORIZONS = (2, 3)
ALPHAS = (0, .001, .01, .1, 1, 10)


def estimator(alpha):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", PoissonRegressor(alpha=alpha, max_iter=3000)),
    ])


def fit_exposure_model(frame, features, alpha):
    # Poisson 빈도모형: y=피해자수/인구, sample_weight=인구는 count+exposure와 동등하다.
    frequency = frame[TARGET_COUNT].to_numpy(float) / frame[TARGET_POP].to_numpy(float)
    pipe = estimator(alpha)
    pipe.fit(frame[features], frequency, model__sample_weight=frame[TARGET_POP].to_numpy(float))
    return pipe


def count_metrics(actual, predicted):
    return {"피해자수_MAE": float(mean_absolute_error(actual, predicted)),
            "피해자수_RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
            "피해자수_R2": float(r2_score(actual, predicted)),
            "피해자수_WMAPE_pct": float(np.abs(np.asarray(actual)-predicted).sum()/np.asarray(actual).sum()*100)}


def population_forecast(panel, target_year):
    rows = []
    steps = target_year - 2025
    for district, data in panel[panel["연도"] <= 2025].groupby("자치구"):
        tail = data.sort_values("연도").tail(4)
        slope = np.polyfit(tail["연도"], tail[POP], 1)[0]
        value = max(float(tail[POP].iloc[-1] + slope*steps), 1.0)
        rows.append({"자치구": district, "예측_어린이인구수": value})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True); PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    panel = prepare_panel().sort_values(["자치구", "연도"]).reset_index(drop=True)
    group = panel.groupby("자치구", sort=False)
    panel[TARGET_COUNT] = group[VICTIMS].shift(-1)
    panel[TARGET_POP] = group[POP].shift(-1)
    features, _, _ = load_v49()

    # 알파는 2021~2024 롤링 피해자 수 MAE로만 선택하고 2025는 남겨둔다.
    alpha_rows, yearly_rows = [], []
    for alpha in ALPHAS:
        per_year = []
        for year in (2021, 2022, 2023, 2024):
            train = panel[(panel[TARGET_COUNT].notna()) & (panel["예측연도"] < year)]
            valid = panel[(panel[TARGET_COUNT].notna()) & (panel["예측연도"] == year)]
            fitted = fit_exposure_model(train, features, alpha)
            rate = fitted.predict(valid[features])
            pred_count = np.maximum(rate*valid[TARGET_POP].to_numpy(float), 0)
            m = count_metrics(valid[TARGET_COUNT], pred_count); per_year.append(m)
            yearly_rows.append({"alpha": alpha, "검증연도": year, **m})
        alpha_rows.append({"alpha": alpha, **{f"평균_{k}": np.mean([r[k] for r in per_year]) for k in per_year[0]}})
    alpha_table = pd.DataFrame(alpha_rows).sort_values(["평균_피해자수_MAE", "평균_피해자수_RMSE"])
    alpha = float(alpha_table.iloc[0]["alpha"])

    train = panel[(panel[TARGET_COUNT].notna()) & (panel["예측연도"] <= 2024)]
    hold = panel[(panel[TARGET_COUNT].notna()) & (panel["예측연도"] == 2025)]
    hold_model = fit_exposure_model(train, features, alpha)
    hold_count = np.maximum(hold_model.predict(hold[features])*hold[TARGET_POP].to_numpy(float), 0)
    hold_count_metrics = count_metrics(hold[TARGET_COUNT], hold_count)
    count_detail = hold[["연도", "예측연도", "자치구", TARGET_COUNT, TARGET_POP]].copy()
    count_detail["예측_어린이피해자수"] = hold_count
    count_detail["절대오차"] = np.abs(count_detail[TARGET_COUNT]-hold_count)

    labeled = panel[panel[TARGET_COUNT].notna()]
    final_model = fit_exposure_model(labeled, features, alpha)
    coef = np.asarray(final_model.named_steps["model"].coef_, float)
    weights = np.abs(coef)/np.abs(coef).sum()
    weight_table = pd.DataFrame({"지표": features, "Poisson표준화계수": coef,
                                 "위험방향": np.where(coef >= 0, "값이 클수록 위험", "값이 작을수록 위험"),
                                 "고정가중치_pct": weights*100}).sort_values("고정가중치_pct", ascending=False)

    v74 = json.loads((ROOT / "models" / "교통사고" / "74차.보유전체데이터활용" / "model_info.json").read_text(encoding="utf-8"))
    specs = {h: {f: parse_spec(v) for f, v in v74["결과"][str(h)]["final_specs"].items()} for h in HORIZONS}
    results, metric_rows, detail_rows, forecasts, strategy_rows, bundles = {}, [], [], [], [], {}
    for horizon in HORIZONS:
        _, methods, cache, _, _ = build_prediction_cache(panel, features, horizon)
        data, predictor_cols, pool, methods, cache = add_expanded_cache(panel, features, horizon, cache, methods)
        dev, dev_rows, dev_detail = evaluate_specs(panel, features, coef, weights, horizon, DEV_YEARS, specs[horizon], cache)
        grade_hold, hold_rows, hold_detail = evaluate_specs(panel, features, coef, weights, horizon, [HOLDOUT_YEAR], specs[horizon], cache)
        dev_rows["구분"] = "83차개발"; hold_rows["구분"] = "83차2025독립"
        metric_rows.extend([dev_rows, hold_rows]); dev_detail["구분"] = "83차개발"; hold_detail["구분"] = "83차2025독립"
        detail_rows.extend([dev_detail, hold_detail])
        future, bundle = fit_forecast(panel, data, predictor_cols, pool, features, horizon, specs[horizon])
        scored = score_frame(future, features, coef, weights, "예측_")
        output = future.merge(scored, on=["연도", "자치구"], validate="one_to_one")
        pop_future = population_forecast(panel, 2025+horizon)
        output = output.merge(pop_future, on="자치구", validate="one_to_one")
        output["예측_어린이피해자율_1만명당"] = final_model.predict(output[features])*10000
        output["예측_어린이피해자수"] = output["예측_어린이피해자율_1만명당"]/10000*output["예측_어린이인구수"]
        output.insert(0, "예측기간", f"{horizon}년 후"); forecasts.append(output)
        bundles[horizon] = bundle
        strategy_rows.extend({"예측기간": f"{horizon}년 후", "지표": f, "미래값예측법": label(specs[horizon][f])} for f in features)
        results[str(horizon)] = {"development": dev, "holdout_2025": grade_hold}

    integrated = float(np.mean([results[str(h)]["holdout_2025"]["정확등급정확도"] for h in HORIZONS]))
    with pd.ExcelWriter(PROCESSED, engine="openpyxl") as writer:
        weight_table.to_excel(writer, sheet_name="13지표Poisson가중치", index=False)
        alpha_table.to_excel(writer, sheet_name="Poisson알파선정", index=False)
        pd.DataFrame(yearly_rows).to_excel(writer, sheet_name="피해자수롤링검증", index=False)
        count_detail.to_excel(writer, sheet_name="피해자수2025독립검증", index=False)
        pd.DataFrame(strategy_rows).to_excel(writer, sheet_name="13지표미래예측법", index=False)
        pd.concat(metric_rows, ignore_index=True).to_excel(writer, sheet_name="등급성능", index=False)
        pd.concat(detail_rows, ignore_index=True).to_excel(writer, sheet_name="등급검증상세", index=False)
        pd.concat(forecasts, ignore_index=True).to_excel(writer, sheet_name="2027_2028피해자수와등급", index=False)
    info = {"모델명": "교통사고 83차 어린이 피해자 수 Poisson 인구노출 안전등급 모델",
            "Y": "다음 연도 어린이 교통사고 사망자 수+부상자 수",
            "인구처리": "어린이 인구를 Poisson exposure(sample_weight)로 반영",
            "불변근간": "49차 13지표 고정→74차 방식 미래값 예측→5분위→Poisson 계수가중치→종합점수→안전 1~5등급",
            "안전등급방향": "1=가장 안전, 5=가장 위험", "alpha": alpha,
            "피해자수2025독립검증": hold_count_metrics, "등급결과": results,
            "통합정확등급": integrated, "74차통합정확등급": .56,
            "채택상태": "채택" if integrated > .56 + 1e-9 else "실험탈락_74차유지",
            "processed": str(PROCESSED.relative_to(ROOT))}
    joblib.dump({"version": VERSION, "features": features, "poisson_model": final_model,
                 "coefficients": dict(zip(features, coef)), "weights": dict(zip(features, weights)),
                 "forecast_models": bundles, "safety_grade_orientation": "1=가장 안전, 5=가장 위험"},
                OUT / "교통사고_83차_피해자수Poisson등급모델.joblib")
    (OUT / "model_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
