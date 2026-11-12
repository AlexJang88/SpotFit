from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.train_traffic_accident_v44 import prepare_panel
from ml.train_traffic_accident_v71_fixed_rubric import evaluate, load_v49, objective, score_frame


ROOT = Path(__file__).resolve().parents[1]
VERSION = 72
OUT = ROOT / "models" / "교통사고" / "72차.미래13지표개선"
PROCESSED = ROOT / "data" / "processed" / "교통사고_72차_미래13지표예측_2027_2028.xlsx"
HORIZONS = (2, 3)
DEV_YEARS = (2021, 2022, 2023, 2024)
HOLDOUT_YEAR = 2025


def feature_frame(panel: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    base = panel[["연도", "자치구", *features]].sort_values(["자치구", "연도"]).copy()
    group = base.groupby("자치구", sort=False)
    out = base[["연도", "자치구"]].copy()
    for feature in features:
        out[f"현재__{feature}"] = base[feature]
        out[f"전년__{feature}"] = group[feature].shift(1)
        out[f"증감__{feature}"] = group[feature].diff()
        out[f"3년평균__{feature}"] = group[feature].transform(lambda s: s.rolling(3, min_periods=1).mean())
    return out


def supervised(panel: pd.DataFrame, features: list[str], horizon: int) -> pd.DataFrame:
    x = feature_frame(panel, features)
    targets = panel[["연도", "자치구", *features]].copy()
    targets["기준연도"] = targets["연도"] - horizon
    targets = targets.rename(columns={f: f"목표__{f}" for f in features}).drop(columns="연도")
    data = x.merge(targets, left_on=["연도", "자치구"], right_on=["기준연도", "자치구"], how="left")
    data["목표연도"] = data["연도"] + horizon
    return data.drop(columns="기준연도")


def model_candidates(numeric_all: list[str], numeric_own: list[str]):
    def ridge(alpha: float, cols: list[str]):
        pre = ColumnTransformer([
            ("num", Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())]), cols),
            ("district", OneHotEncoder(handle_unknown="ignore"), ["자치구"]),
        ])
        return Pipeline([("pre", pre), ("model", Ridge(alpha=alpha))])

    def tree(model, cols: list[str]):
        pre = ColumnTransformer([
            ("num", SimpleImputer(strategy="median", keep_empty_features=True), cols),
            ("district", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["자치구"]),
        ])
        return Pipeline([("pre", pre), ("model", model)])

    return {
        "RidgeOwn_a10": ridge(10, numeric_own),
        "RidgeAll_a10": ridge(10, numeric_all),
        "RidgeAll_a100": ridge(100, numeric_all),
        "GBROwn_d1": tree(GradientBoostingRegressor(n_estimators=150, learning_rate=.03, max_depth=1,
                                                      min_samples_leaf=5, loss="huber", random_state=72), numeric_own),
        "GBRAll_d1": tree(GradientBoostingRegressor(n_estimators=150, learning_rate=.03, max_depth=1,
                                                      min_samples_leaf=5, loss="huber", random_state=72), numeric_all),
        "ExtraTreesAll": tree(ExtraTreesRegressor(n_estimators=250, min_samples_leaf=4, max_features=.7,
                                                    random_state=72, n_jobs=1), numeric_all),
    }


def deterministic(history: pd.Series, method: str) -> float:
    s = history.dropna().astype(float)
    if method == "Persistence" or len(s) == 1:
        return float(s.iloc[-1])
    if method == "Mean2":
        return float(s.tail(2).mean())
    if method == "Mean3":
        return float(s.tail(3).mean())
    tail = s.tail(3)
    slope = np.polyfit(np.arange(len(tail)), tail.to_numpy(), 1)[0]
    return float(tail.iloc[-1] + .5 * slope)


def build_prediction_cache(panel, features, horizon):
    data = supervised(panel, features, horizon)
    numeric_all = [c for c in data if c.startswith(("현재__", "전년__", "증감__", "3년평균__"))]
    methods = ["Persistence", "Mean2", "Mean3", "DampedTrend3",
               "RidgeOwn_a10", "RidgeAll_a10", "RidgeAll_a100", "GBROwn_d1", "GBRAll_d1", "ExtraTreesAll"]
    cache: dict[tuple[int, str, str], pd.Series] = {}
    model_specs = {}
    for year in (*DEV_YEARS, HOLDOUT_YEAR):
        valid = data[data["목표연도"] == year].sort_values("자치구")
        train = data[data["목표연도"] < year]
        origin = year - horizon
        history = panel[panel["연도"] <= origin]
        for feature in features:
            for method in methods[:4]:
                values = []
                for district in valid["자치구"]:
                    s = history[history["자치구"] == district].sort_values("연도")[feature]
                    values.append(deterministic(s, method))
                cache[(year, feature, method)] = pd.Series(values, index=valid["자치구"].to_numpy())

            own = [f"현재__{feature}", f"전년__{feature}", f"증감__{feature}", f"3년평균__{feature}"]
            candidates = model_candidates(numeric_all, own)
            target = f"목표__{feature}"
            fit_train = train[train[target].notna()]
            for name, estimator in candidates.items():
                fitted = deepcopy(estimator).fit(fit_train, fit_train[target])
                pred = fitted.predict(valid)
                observed = fit_train[target].astype(float)
                pred = np.clip(pred, observed.quantile(.01), observed.quantile(.99))
                cache[(year, feature, name)] = pd.Series(pred, index=valid["자치구"].to_numpy())
                model_specs[(feature, name, year)] = fitted
    return data, methods, cache, numeric_all, model_specs


def prediction_frame(panel, features, year, strategy_map, cache):
    districts = sorted(panel["자치구"].unique())
    out = pd.DataFrame({"연도": year, "자치구": districts})
    for feature in features:
        out[feature] = out["자치구"].map(cache[(year, feature, strategy_map[feature])])
    return out


def evaluate_map(panel, features, coef, weights, horizon, years, strategy_map, cache):
    rows, detail = [], []
    for year in years:
        pred = prediction_frame(panel, features, year, strategy_map, cache)
        actual = panel[panel["연도"] == year][["연도", "자치구", *features]]
        metrics, joined = evaluate(actual, pred, features, coef, weights)
        rows.append({"예측기간": f"{horizon}년 후", "검증연도": year, **metrics})
        joined["검증연도"] = year
        detail.append(joined)
    summary = {k: float(np.mean([r[k] for r in rows])) for k in
               ("정확등급정확도", "1등급이내정확도", "평균등급오차", "종합점수_MAE")}
    return summary, pd.DataFrame(rows), pd.concat(detail, ignore_index=True)


def select_map(panel, features, coef, weights, horizon, methods, cache):
    persistence = {f: "Persistence" for f in features}
    base, _, _ = evaluate_map(panel, features, coef, weights, horizon, DEV_YEARS, persistence, cache)
    selected, current = dict(persistence), base
    log = [{"단계": "기준", "지표": "전체", "예측법": "Persistence", **base}]

    # 먼저 전 지표에 같은 방법을 적용한 후보를 비교한다.
    for method in methods[1:]:
        trial_map = {f: method for f in features}
        result, _, _ = evaluate_map(panel, features, coef, weights, horizon, DEV_YEARS, trial_map, cache)
        log.append({"단계": "공통방법비교", "지표": "전체", "예측법": method, **result})
        if objective(result) > objective(current):
            selected, current = trial_map, result

    # 가중치가 큰 지표부터 한 차례만 교체해 반복 최적화 과적합을 제한한다.
    for feature in [features[i] for i in np.argsort(-np.abs(weights))]:
        best_method, best_result = selected[feature], current
        for method in methods:
            trial_map = dict(selected); trial_map[feature] = method
            result, _, _ = evaluate_map(panel, features, coef, weights, horizon, DEV_YEARS, trial_map, cache)
            if objective(result) > objective(best_result):
                best_method, best_result = method, result
        if best_method != selected[feature]:
            selected[feature] = best_method; current = best_result
            log.append({"단계": "지표별개선", "지표": feature, "예측법": best_method, **current})
    return selected, current, pd.DataFrame(log)


def fit_future_models(data, features, strategy_map, numeric_all, target_year):
    origin_year = target_year - (target_year - 2025)
    # 호출부에서 만든 2025 기준 입력행에 사용할 최종 모델을 반환한다.
    result = {}
    for feature, method in strategy_map.items():
        if method in ("Persistence", "Mean2", "Mean3", "DampedTrend3"):
            result[feature] = method
            continue
        own = [f"현재__{feature}", f"전년__{feature}", f"증감__{feature}", f"3년평균__{feature}"]
        estimator = model_candidates(numeric_all, own)[method]
        target = f"목표__{feature}"
        train = data[data[target].notna()]
        result[feature] = deepcopy(estimator).fit(train, train[target])
    return result


def future_prediction(panel, features, horizon, strategy_map, fitted, numeric_all):
    target_year = 2025 + horizon
    x = feature_frame(panel, features)
    current = x[x["연도"] == 2025].sort_values("자치구").copy()
    out = current[["자치구"]].copy(); out.insert(0, "연도", target_year)
    history = panel[panel["연도"] <= 2025]
    for feature in features:
        method = strategy_map[feature]
        if isinstance(fitted[feature], str):
            vals = [deterministic(history[history["자치구"] == d].sort_values("연도")[feature], method)
                    for d in out["자치구"]]
        else:
            vals = fitted[feature].predict(current)
            observed = panel[feature].dropna().astype(float)
            vals = np.clip(vals, observed.quantile(.01), observed.quantile(.99))
        out[feature] = vals
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True); PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    features, coef, weights = load_v49()
    panel = prepare_panel().sort_values(["연도", "자치구"]).reset_index(drop=True)
    weight_table = pd.DataFrame({"지표": features, "49차계수": coef,
                                 "위험방향": np.where(coef >= 0, "값이 클수록 위험", "값이 작을수록 위험"),
                                 "49차고정가중치_pct": weights * 100})
    payload, logs, metrics_all, details, forecasts, selected_rows = {}, [], [], [], [], []
    for horizon in HORIZONS:
        data, methods, cache, numeric_all, _ = build_prediction_cache(panel, features, horizon)
        selected, dev_metrics, log = select_map(panel, features, coef, weights, horizon, methods, cache)
        hold_metrics, hold_rows, hold_detail = evaluate_map(
            panel, features, coef, weights, horizon, [HOLDOUT_YEAR], selected, cache)
        persistence = {f: "Persistence" for f in features}
        base_dev, base_dev_rows, _ = evaluate_map(panel, features, coef, weights, horizon, DEV_YEARS, persistence, cache)
        base_hold, base_hold_rows, _ = evaluate_map(panel, features, coef, weights, horizon, [HOLDOUT_YEAR], persistence, cache)

        # 72차 채택은 개발 평균과 독립연도 모두 Persistence보다 정확등급이 높을 때만 허용한다.
        adopted = (dev_metrics["정확등급정확도"] > base_dev["정확등급정확도"] and
                   hold_metrics["정확등급정확도"] > base_hold["정확등급정확도"])
        final_map = selected if adopted else persistence
        fitted = fit_future_models(data, features, final_map, numeric_all, 2025 + horizon)
        future_values = future_prediction(panel, features, horizon, final_map, fitted, numeric_all)
        future_scores = score_frame(future_values, features, coef, weights, "예측_")
        future = future_values.merge(future_scores, on=["연도", "자치구"], validate="one_to_one")
        future.insert(0, "예측기간", f"{horizon}년 후"); forecasts.append(future)

        log.insert(0, "예측기간", f"{horizon}년 후"); logs.append(log)
        hold_rows["구분"] = "72차후보_2025독립"; base_hold_rows["구분"] = "71차Persistence_2025독립"
        base_dev_rows["구분"] = "71차Persistence_개발"
        metrics_all.extend([hold_rows, base_hold_rows, base_dev_rows]); details.append(hold_detail)
        selected_rows.extend({"예측기간": f"{horizon}년 후", "지표": f, "72차후보예측법": selected[f],
                              "최종채택예측법": final_map[f]} for f in features)
        payload[str(horizon)] = {"adopted_as_v72": adopted, "final_strategies": final_map,
                                 "candidate_strategies": selected, "candidate_dev": dev_metrics,
                                 "candidate_holdout_2025": hold_metrics,
                                 "persistence_dev": base_dev, "persistence_holdout_2025": base_hold,
                                 "selection_rule": "개발 평균과 2025 독립검증 정확등급 정확도가 모두 Persistence 초과"}
        joblib.dump(fitted, OUT / f"교통사고_72차_{horizon}년후_13지표예측모델.joblib")

    with pd.ExcelWriter(PROCESSED, engine="openpyxl") as writer:
        weight_table.to_excel(writer, sheet_name="49차고정가중치", index=False)
        pd.DataFrame(selected_rows).to_excel(writer, sheet_name="지표별예측법", index=False)
        pd.concat(metrics_all, ignore_index=True).to_excel(writer, sheet_name="성능비교", index=False)
        pd.concat(logs, ignore_index=True).to_excel(writer, sheet_name="모델선정과정", index=False)
        pd.concat(details, ignore_index=True).to_excel(writer, sheet_name="2025검증상세", index=False)
        pd.concat(forecasts, ignore_index=True).to_excel(writer, sheet_name="2027_2028예측", index=False)

    info = {"모델명": "교통사고 72차 미래 13개 지표 예측 개선모델",
            "불변구조": "13개 미래값 예측→5분위→49차 고정가중치→종합위험점수→안전등급(1=안전)",
            "지표변경": False, "가중치변경": False, "방향변경": False, "등급직접예측": False,
            "결과": payload, "processed": str(PROCESSED.relative_to(ROOT))}
    (OUT / "model_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
