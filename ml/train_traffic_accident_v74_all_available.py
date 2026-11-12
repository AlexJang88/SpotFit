from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.train_traffic_accident_v44 import prepare_panel
from ml.train_traffic_accident_v71_fixed_rubric import load_v49, score_frame
from ml.train_traffic_accident_v72_indicator_forecast import (
    DEV_YEARS, HOLDOUT_YEAR, build_prediction_cache, feature_frame,
)
from ml.train_traffic_accident_v73_indicator_ensemble import (
    evaluate_specs, label, select_ensemble,
)


ROOT = Path(__file__).resolve().parents[1]
VERSION = 74
OUT = ROOT / "models" / "교통사고" / "74차.보유전체데이터활용"
PROCESSED = ROOT / "data" / "processed" / "교통사고_74차_보유전체데이터활용_2027_2028.xlsx"
HORIZONS = (2, 3)


def load_v72_maps():
    info = json.loads((ROOT / "models" / "교통사고" / "72차.미래13지표개선" / "model_info.json").read_text(encoding="utf-8"))
    return {int(h): p["final_strategies"] for h, p in info["결과"].items()}


def raw_feature_frame(panel, target_features):
    excluded = {"연도", "예측연도", "행정구역코드", "자치구코드",
                "다음연도_어린이교통사고피해자율_어린이1만명당"}
    raw = [c for c in panel.select_dtypes(include=[np.number]).columns if c not in excluded]
    base = panel[["연도", "자치구", *raw]].copy().sort_values(["자치구", "연도"])
    out = base.copy()
    group = base.groupby("자치구", sort=False)
    for f in target_features:
        out[f"전년__{f}"] = group[f].shift(1)
        out[f"증감__{f}"] = group[f].diff()
        out[f"3년평균__{f}"] = group[f].transform(lambda s: s.rolling(3, min_periods=1).mean())
    districts = pd.get_dummies(out["자치구"], prefix="자치구", dtype=float)
    out = pd.concat([out.drop(columns="자치구"), districts], axis=1)
    return out, raw, list(districts.columns)


def supervised(panel, target_features, horizon):
    x, raw, district_cols = raw_feature_frame(panel, target_features)
    targets = panel[["연도", "자치구", *target_features]].copy()
    targets["기준연도"] = targets["연도"] - horizon
    targets = targets.rename(columns={f: f"목표__{f}" for f in target_features}).drop(columns="연도")
    districts = pd.get_dummies(panel["자치구"], prefix="자치구", dtype=float)
    district_lookup = pd.concat([panel[["연도", "자치구"]], districts], axis=1)
    # x에 자치구 문자열을 다시 붙여 목표와 연결한다.
    keys = panel.sort_values(["자치구", "연도"])[["연도", "자치구"]].reset_index(drop=True)
    x = x.reset_index(drop=True)
    x.insert(1, "자치구", keys["자치구"])
    data = x.merge(targets, left_on=["연도", "자치구"], right_on=["기준연도", "자치구"], how="left")
    data["목표연도"] = data["연도"] + horizon
    return data.drop(columns="기준연도"), raw, district_cols


def candidates(kmax):
    def ridge(k, alpha):
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("select", SelectKBest(f_regression, k=min(k, kmax))),
                         ("scale", StandardScaler()), ("model", Ridge(alpha=alpha))])
    def tree(k, model):
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("select", SelectKBest(f_regression, k=min(k, kmax))), ("model", model)])
    return {
        "RidgeRaw_k20_a10": ridge(20, 10),
        "RidgeRaw_k40_a100": ridge(40, 100),
        "RidgeRaw_k80_a100": ridge(80, 100),
        "GBRRaw_k40": tree(40, GradientBoostingRegressor(n_estimators=160, learning_rate=.03, max_depth=1,
                                                           min_samples_leaf=5, loss="huber", random_state=74)),
        "ExtraTreesRaw_k40": tree(40, ExtraTreesRegressor(n_estimators=160, min_samples_leaf=4,
                                                            max_features=.7, random_state=74, n_jobs=1)),
        "RandomForestRaw_k40": tree(40, RandomForestRegressor(n_estimators=160, min_samples_leaf=4,
                                                                max_features=.7, random_state=74, n_jobs=1)),
    }


def add_expanded_cache(panel, features, horizon, cache, methods):
    data, raw, district_cols = supervised(panel, features, horizon)
    predictor_cols = [c for c in data.columns if c not in {"연도", "자치구", "목표연도"}
                      and not c.startswith("목표__")]
    pool = candidates(len(predictor_cols))
    fitted_final = {}
    for year in (*DEV_YEARS, HOLDOUT_YEAR):
        train = data[data["목표연도"] < year]
        valid = data[data["목표연도"] == year].sort_values("자치구")
        for feature in features:
            target = f"목표__{feature}"
            fit_train = train[train[target].notna()]
            for name, estimator in pool.items():
                model = deepcopy(estimator).fit(fit_train[predictor_cols], fit_train[target])
                pred = model.predict(valid[predictor_cols])
                observed = fit_train[target].astype(float)
                pred = np.clip(pred, observed.quantile(.01), observed.quantile(.99))
                cache[(year, feature, name)] = pd.Series(pred, index=valid["자치구"].to_numpy())
    methods.extend(pool.keys())
    return data, predictor_cols, pool, methods, cache


def fit_forecast(panel, data, predictor_cols, pool, features, horizon, specs):
    # 74차 선택 결과에는 혼합식이 올 수 있으므로 구성 모델을 각각 학습한다.
    current_raw, _, _ = raw_feature_frame(panel, features)
    keys = panel.sort_values(["자치구", "연도"])[["연도", "자치구"]].reset_index(drop=True)
    current_raw = current_raw.reset_index(drop=True); current_raw.insert(1, "자치구", keys["자치구"])
    current = current_raw[current_raw["연도"] == 2025].sort_values("자치구")
    out = current[["자치구"]].copy(); out.insert(0, "연도", 2025 + horizon)
    bundle = {}

    # 기존 72차 계열의 미래값은 72차 최종 모델 생성 함수를 재사용한다.
    from ml.train_traffic_accident_v72_indicator_forecast import fit_future_models, future_prediction, supervised as old_supervised
    old_data = old_supervised(panel, features, horizon)
    old_numeric = [c for c in old_data if c.startswith(("현재__", "전년__", "증감__", "3년평균__"))]

    for feature, spec in specs.items():
        names = [spec] if isinstance(spec, str) else [spec[0], spec[1]]
        predictions, components = {}, {}
        for name in dict.fromkeys(names):
            if name in pool:
                target = f"목표__{feature}"; train = data[data[target].notna()]
                fitted = deepcopy(pool[name]).fit(train[predictor_cols], train[target])
                values = fitted.predict(current[predictor_cols])
                observed = train[target].astype(float); values = np.clip(values, observed.quantile(.01), observed.quantile(.99))
            else:
                one = {f: "Persistence" for f in features}; one[feature] = name
                fitted_map = fit_future_models(old_data, features, one, old_numeric, 2025 + horizon)
                values = future_prediction(panel, features, horizon, one, fitted_map, old_numeric)[feature].to_numpy()
                fitted = fitted_map[feature]
            predictions[name], components[name] = np.asarray(values), fitted
        values = predictions[spec] if isinstance(spec, str) else predictions[spec[0]] * spec[2] + predictions[spec[1]] * (1-spec[2])
        out[feature] = values; bundle[feature] = {"spec": spec, "components": components}
    return out, bundle


def main():
    OUT.mkdir(parents=True, exist_ok=True); PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    features, coef, weights = load_v49(); selection_maps = load_v72_maps()
    comparison_maps = globals().get("COMPARISON_MAPS", selection_maps)
    panel = prepare_panel().sort_values(["연도", "자치구"]).reset_index(drop=True)
    results, logs, comparisons, specs_out, forecasts = {}, [], [], [], []
    weight_table = pd.DataFrame({"지표": features, "49차계수": coef,
                                 "위험방향": np.where(coef >= 0, "값이 클수록 위험", "값이 작을수록 위험"),
                                 "49차고정가중치_pct": weights * 100})
    for horizon in HORIZONS:
        _, methods, cache, _, _ = build_prediction_cache(panel, features, horizon)
        data, predictor_cols, pool, methods, cache = add_expanded_cache(panel, features, horizon, cache, methods)
        selection_base = selection_maps[horizon]
        base = comparison_maps[horizon]
        selected, dev, log = select_ensemble(panel, features, coef, weights, horizon, methods, cache, selection_base)
        base_dev, base_dev_rows, _ = evaluate_specs(panel, features, coef, weights, horizon, DEV_YEARS, base, cache)
        hold, hold_rows, detail = evaluate_specs(panel, features, coef, weights, horizon, [HOLDOUT_YEAR], selected, cache)
        base_hold, base_hold_rows, _ = evaluate_specs(panel, features, coef, weights, horizon, [HOLDOUT_YEAR], base, cache)
        adopted = dev["정확등급정확도"] > base_dev["정확등급정확도"] and hold["정확등급정확도"] > base_hold["정확등급정확도"]
        final_specs = selected if adopted else base
        values, bundle = fit_forecast(panel, data, predictor_cols, pool, features, horizon, final_specs)
        scores = score_frame(values, features, coef, weights, "예측_")
        future = values.merge(scores, on=["연도", "자치구"], validate="one_to_one")
        future.insert(0, "예측기간", f"{horizon}년 후"); forecasts.append(future)
        joblib.dump(bundle, OUT / f"교통사고_{VERSION}차_{horizon}년후_13지표예측모델.joblib")
        log.insert(0, "예측기간", f"{horizon}년 후"); logs.append(log)
        hold_rows["구분"]=f"{VERSION}차후보_2025"; base_hold_rows["구분"]="기준모델_2025"; base_dev_rows["구분"]="기준모델_개발"
        comparisons.extend([hold_rows, base_hold_rows, base_dev_rows])
        specs_out.extend({"예측기간":f"{horizon}년 후","지표":f,"기준모델":label(base[f]),
                          "74차후보":label(selected[f]),"최종채택":label(final_specs[f])} for f in features)
        results[str(horizon)]={f"adopted_as_v{VERSION}":adopted,"candidate_dev":dev,"candidate_holdout_2025":hold,
                               "baseline_dev":base_dev,"baseline_holdout_2025":base_hold,
                               "final_specs":{f:label(s) for f,s in final_specs.items()},
                               "candidate_specs":{f:label(s) for f,s in selected.items()}}
    with pd.ExcelWriter(PROCESSED,engine="openpyxl") as writer:
        weight_table.to_excel(writer,sheet_name="49차고정가중치",index=False)
        pd.DataFrame(specs_out).to_excel(writer,sheet_name="지표별예측법",index=False)
        pd.concat(comparisons,ignore_index=True).to_excel(writer,sheet_name="기준_후보성능",index=False)
        pd.concat(logs,ignore_index=True).to_excel(writer,sheet_name="탐색과정",index=False)
        pd.concat(forecasts,ignore_index=True).to_excel(writer,sheet_name="2027_2028예측",index=False)
    info={"모델명":f"교통사고 {VERSION}차 미래13지표 예측모델",
          "불변구조":"13개 미래값 예측→5분위→49차 고정가중치→종합위험점수→안전등급(1=안전)",
          "보조입력만확장":True,"지표변경":False,"가중치변경":False,"방향변경":False,"등급직접예측":False,
          "결과":results,"processed":str(PROCESSED.relative_to(ROOT))}
    (OUT/"model_info.json").write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(info,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
