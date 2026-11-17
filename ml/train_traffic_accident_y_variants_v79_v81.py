from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
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
PROCESSED = ROOT / "data" / "processed" / "교통사고_Y변경_79차_81차_비교.xlsx"
ALPHA = .1
HORIZONS = (2, 3)
CURRENT_RATE = "현재연도_어린이교통사고피해자율_어린이1만명당"
CHILD_POP = "어린이(13세 미만) 인구수"
VICTIMS = "어린이교통사고피해자수"

VARIANTS = {
    79: ("Y3년평균피해자율", "다음연도기준_최근3년평균_어린이피해자율"),
    80: ("Y베이즈보정피해자율", "다음연도_베이즈보정_어린이피해자율"),
    81: ("Y피해자수후인구환산", "다음연도_어린이교통사고피해자수"),
}


def ridge():
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()), ("model", Ridge(alpha=ALPHA))])


def target_metrics(y, pred):
    return {"MAE": float(mean_absolute_error(y, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
            "R2": float(r2_score(y, pred)),
            "WMAPE_pct": float(np.abs(np.asarray(y)-pred).sum()/np.abs(np.asarray(y)).sum()*100)}


def make_targets(panel):
    panel = panel.sort_values(["자치구", "연도"]).copy()
    group = panel.groupby("자치구", sort=False)
    rolling3 = group[CURRENT_RATE].transform(lambda s: s.rolling(3, min_periods=3).mean())
    panel[VARIANTS[79][1]] = rolling3.groupby(panel["자치구"]).shift(-1)

    # 데이터에서 산출한 전 기간 자치구 어린이 인구 중앙값을 prior exposure로 사용한다.
    prior_n = float(panel[CHILD_POP].median())
    city = panel.groupby("연도").apply(
        lambda d: d[VICTIMS].sum() / d[CHILD_POP].sum() * 10000,
        include_groups=False,
    )
    eb_current = (panel[VICTIMS] + panel["연도"].map(city) * prior_n / 10000) / (panel[CHILD_POP] + prior_n) * 10000
    panel[VARIANTS[80][1]] = eb_current.groupby(panel["자치구"]).shift(-1)
    panel[VARIANTS[81][1]] = group[VICTIMS].shift(-1)
    return panel, prior_n


def fit_weights_and_validate(panel, features, target):
    yearly = []
    for year in (2021, 2022, 2023, 2024):
        train = panel[(panel[target].notna()) & (panel["예측연도"] < year)]
        valid = panel[(panel[target].notna()) & (panel["예측연도"] == year)]
        fitted = ridge().fit(train[features], train[target])
        pred = fitted.predict(valid[features])
        yearly.append({"검증연도": year, **target_metrics(valid[target], pred)})
    train = panel[(panel[target].notna()) & (panel["예측연도"] <= 2024)]
    hold = panel[(panel[target].notna()) & (panel["예측연도"] == 2025)]
    hold_model = ridge().fit(train[features], train[target])
    hold_pred = hold_model.predict(hold[features])
    hold_metrics = target_metrics(hold[target], hold_pred)
    final = ridge().fit(panel[panel[target].notna()][features], panel.loc[panel[target].notna(), target])
    coef = np.asarray(final.named_steps["model"].coef_, float)
    weights = np.abs(coef) / np.abs(coef).sum()
    return final, coef, weights, pd.DataFrame(yearly), hold_metrics


def main():
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    panel, prior_n = make_targets(prepare_panel())
    features, v49_coef, v49_weights = load_v49()
    v74 = json.loads((ROOT / "models" / "교통사고" / "74차.보유전체데이터활용" / "model_info.json").read_text(encoding="utf-8"))
    specs = {h: {f: parse_spec(v) for f, v in v74["결과"][str(h)]["final_specs"].items()} for h in HORIZONS}

    caches, expanded, parts = {}, {}, {}
    for horizon in HORIZONS:
        _, methods, cache, _, _ = build_prediction_cache(panel, features, horizon)
        data, predictor_cols, pool, methods, cache = add_expanded_cache(panel, features, horizon, cache, methods)
        caches[horizon], expanded[horizon], parts[horizon] = cache, data, (predictor_cols, pool)

    comparison, weight_tables, target_yearly, grade_rows, detail_rows, forecasts = [], [], [], [], [], []
    infos = {}
    for version, (short_name, target) in VARIANTS.items():
        out = ROOT / "models" / "교통사고" / f"{version}차.{short_name}"
        out.mkdir(parents=True, exist_ok=True)
        target_model, coef, weights, yearly, target_hold = fit_weights_and_validate(panel, features, target)
        yearly.insert(0, "차수", version); target_yearly.append(yearly)
        weight_table = pd.DataFrame({"차수": version, "Y": target, "지표": features,
                                     "표준화계수": coef,
                                     "위험방향": np.where(coef >= 0, "값이 클수록 위험", "값이 작을수록 위험"),
                                     "가중치_pct": weights*100})
        weight_tables.append(weight_table)
        result = {}
        bundles = {}
        for horizon in HORIZONS:
            dev, dev_rows, _ = evaluate_specs(panel, features, coef, weights, horizon, DEV_YEARS, specs[horizon], caches[horizon])
            hold, hold_rows, detail = evaluate_specs(panel, features, coef, weights, horizon, [HOLDOUT_YEAR], specs[horizon], caches[horizon])
            dev_rows["차수"] = version; dev_rows["구분"] = "개발"
            hold_rows["차수"] = version; hold_rows["구분"] = "2025독립"
            grade_rows.extend([dev_rows, hold_rows]); detail["차수"] = version; detail_rows.append(detail)
            predictor_cols, pool = parts[horizon]
            future, bundle = fit_forecast(panel, expanded[horizon], predictor_cols, pool, features, horizon, specs[horizon])
            scored = score_frame(future, features, coef, weights, "예측_")
            output = future.merge(scored, on=["연도", "자치구"], validate="one_to_one")
            output.insert(0, "차수", version); output.insert(1, "예측기간", f"{horizon}년 후"); forecasts.append(output)
            bundles[horizon] = bundle
            result[str(horizon)] = {"development": dev, "holdout_2025": hold}
            comparison.append({"차수": version, "Y": target, "예측기간": f"{horizon}년 후",
                               **hold, "Y모델_2025_MAE": target_hold["MAE"], "Y모델_2025_R2": target_hold["R2"]})
        integrated = float(np.mean([result[str(h)]["holdout_2025"]["정확등급정확도"] for h in HORIZONS]))
        info = {"모델명": f"교통사고 {version}차 {short_name}", "Y": target,
                "불변근간": "49차 13지표 고정·74차 미래지표 예측법 고정→5분위→변경 Y의 Ridge 계수가중치→종합점수→안전등급",
                "안전등급방향": "1=가장 안전, 5=가장 위험", "Ridge_alpha": ALPHA,
                "베이즈보정_prior_어린이인구": prior_n if version == 80 else None,
                "Y모델2025독립검증": target_hold, "등급결과": result,
                "통합정확등급": integrated, "74차통합정확등급": .56,
                "채택상태": "후보" if integrated > .56 else "실험탈락_74차유지"}
        joblib.dump({"version": version, "target": target, "features": features,
                     "target_model": target_model, "coefficients": dict(zip(features, coef)),
                     "weights": dict(zip(features, weights)), "forecast_models": bundles},
                    out / f"교통사고_{version}차_{short_name}.joblib")
        (out / "model_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        infos[version] = info

    # 원래 74차를 같은 표에 기준으로 기록한다.
    comparison.extend([
        {"차수": 74, "Y": "다음연도 단년도 어린이 피해자율", "예측기간": "2년 후", "정확등급정확도": .56, "1등급이내정확도": .88, "평균등급오차": .56, "종합점수_MAE": 2.9904577253},
        {"차수": 74, "Y": "다음연도 단년도 어린이 피해자율", "예측기간": "3년 후", "정확등급정확도": .56, "1등급이내정확도": .60, "평균등급오차": .88, "종합점수_MAE": 3.8715277709},
    ])
    with pd.ExcelWriter(PROCESSED, engine="openpyxl") as writer:
        pd.DataFrame(comparison).to_excel(writer, sheet_name="74_79_80_81비교", index=False)
        pd.concat(weight_tables, ignore_index=True).to_excel(writer, sheet_name="Y별13지표가중치", index=False)
        pd.concat(target_yearly, ignore_index=True).to_excel(writer, sheet_name="Y모델롤링검증", index=False)
        pd.concat(grade_rows, ignore_index=True).to_excel(writer, sheet_name="등급성능", index=False)
        pd.concat(detail_rows, ignore_index=True).to_excel(writer, sheet_name="등급검증상세", index=False)
        pd.concat(forecasts, ignore_index=True).to_excel(writer, sheet_name="2027_2028예측", index=False)
    print(json.dumps({v: {"Y": i["Y"], "통합정확등급": i["통합정확등급"],
                                "등급결과": i["등급결과"], "채택상태": i["채택상태"]}
                      for v, i in infos.items()}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
