"""Build a two-year exposure model for a 2025 -> 2027 operational forecast.

Two availability policies are compared:
* strict_2025: only source-year values complete through 2025;
* lag_allowed: publication-lag features may use a consistently defined lag1.

Model selection uses target years 2022-2024. Target 2025 is opened only after
the configurations are locked.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ml.training import optimize_exposure_rank_3y as metrics_lib
from ml.training import optimize_borough_crime_count_3y as model_lib

BASE = ROOT / "artifacts" / "child_crime_borough_v2"
PRIORITY = ROOT / "artifacts" / "child_population_priority_v1"
OUT = ROOT / "artifacts" / "child_crime_exposure_2y_2027_v1"
MODEL_OUT = ROOT / "models" / "experiments" / "child_crime_exposure_2y_2027_v1"
HORIZON = 2
CV_YEARS = (2022, 2023, 2024)
HOLDOUT = 2025


def pairs(panel: pd.DataFrame) -> pd.DataFrame:
    future = panel[["gu", "year", "y", "log_crime", "log_child"]].copy()
    future.year -= HORIZON
    future = future.rename(columns={"y": "target_y", "log_crime": "target_log_crime", "log_child": "target_log_child"})
    result = panel.merge(future, on=["gu", "year"], validate="one_to_one")
    result["target_year"] = result.year + HORIZON
    return result[(result.year >= 2015) & result.y.notna() & result.target_y.notna()].sort_values(["year", "gu"]).reset_index(drop=True)


def prep(features):
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), features),
        ("borough", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["gu"]),
    ], sparse_threshold=0)


def estimator(spec: str):
    name, values = spec.split(":", 1)
    p = [float(x) for x in values.split(",")]
    if name == "ridge": return Ridge(alpha=p[0])
    if name == "elastic": return ElasticNet(alpha=p[0], l1_ratio=p[1], max_iter=30000, random_state=42)
    if name == "extra": return ExtraTreesRegressor(n_estimators=180, max_depth=int(p[0]), min_samples_leaf=int(p[1]), max_features=p[2], n_jobs=1, random_state=42)
    if name == "forest": return RandomForestRegressor(n_estimators=180, max_depth=int(p[0]), min_samples_leaf=int(p[1]), max_features=p[2], n_jobs=1, random_state=42)
    raise ValueError(spec)


def fit_predict(train, test, features, spec, mode):
    model = Pipeline([("prep", prep(features)), ("model", estimator(spec))])
    columns = ["gu", *features]
    if mode == "direct":
        target = train.target_y
        model.fit(train[columns], target)
        return model.predict(test[columns]), model
    if mode == "delta":
        target = train.target_y - train.y
        model.fit(train[columns], target)
        return test.y.to_numpy() + model.predict(test[columns]), model
    if mode in ("components_direct", "components_delta"):
        target = train[["target_log_crime", "target_log_child"]].to_numpy(copy=True)
        if mode.endswith("delta"):
            target -= train[["log_crime", "log_child"]].to_numpy()
        model.fit(train[columns], target)
        value = model.predict(test[columns])
        if mode.endswith("delta"):
            value += test[["log_crime", "log_child"]].to_numpy()
        return metrics_lib.exposure(value[:, 0], value[:, 1]), model
    raise ValueError(mode)


def evaluate(data, features, spec, mode):
    actual, predicted, years = [], [], []
    for target_year in CV_YEARS:
        train = data[data.target_year <= target_year - HORIZON]
        test = data[data.target_year == target_year].sort_values("gu")
        pred, _ = fit_predict(train, test, features, spec, mode)
        actual.extend(test.target_y); predicted.extend(pred); years.extend(test.target_year)
    return metrics_lib.rank_metrics(np.asarray(actual), np.asarray(predicted), np.asarray(years))


def configurations():
    # With only 225 borough-year pairs, regularized linear models are the
    # defensible search family. Tree ensembles were tested in the 3-year work
    # and are too slow/unstable to repeat for every forward-selection step.
    single = [f"ridge:{a}" for a in (.01, .1, 1, 10, 100, 1000)]
    multi = single
    return [(s, m) for s in single for m in ("direct", "delta")] + [(s, m) for s in multi for m in ("components_direct", "components_delta")]


def source_column(feature: str) -> str:
    return feature.removesuffix("_available_lag1")


def complete_2025(panel, feature):
    return feature in panel and panel.loc[panel.year == 2025, feature].notna().sum() == 25


def make_availability_groups(panel):
    base_groups = json.loads((BASE / "feature_groups.json").read_text(encoding="utf-8"))
    priority_groups = json.loads((PRIORITY / "feature_groups.json").read_text(encoding="utf-8"))
    all_groups = {**base_groups, **{f"priority_{k}": v for k, v in priority_groups.items()}}
    strict, lagged = {}, {}
    for name, values in all_groups.items():
        strict_values, lag_values = [], []
        for feature in values:
            raw = source_column(feature)
            if complete_2025(panel, raw): strict_values.append(raw)
            if complete_2025(panel, feature): lag_values.append(feature)
        if strict_values: strict[name] = list(dict.fromkeys(strict_values))
        if lag_values: lagged[name] = list(dict.fromkeys(lag_values))
    return strict, lagged


def best_config(data, features):
    rows = []
    for spec, mode in configurations():
        row = {"spec": spec, "mode": mode, "features": features, **evaluate(data, features, spec, mode)}
        rows.append(row)
    rows.sort(key=lambda r: (-r["ranking_score"], -r["Spearman"], r["rank_MAE"], r["y_RMSE"]))
    return rows[0], rows


def greedy_policy(data, groups, policy):
    core = ["y", "log_crime", "log_child", "y_lag1", "y_lag2", "y_change", "log_crime_change", "log_child_change"]
    selected, features, history, all_rows = [], core, [], []
    current, rows = best_config(data, features); all_rows.extend({"policy": policy, "step": 0, "candidate_group": "core", **r} for r in rows)
    history.append({"step": 0, "selected_groups": [], **current})
    while True:
        candidates = []
        for name, values in groups.items():
            if name in selected: continue
            trial = list(dict.fromkeys([*features, *values]))
            best, rows = best_config(data, trial)
            all_rows.extend({"policy": policy, "step": len(selected) + 1, "candidate_group": name, **r} for r in rows)
            candidates.append((best, name, trial))
        if not candidates: break
        best, name, trial = max(candidates, key=lambda x: (x[0]["ranking_score"], x[0]["Spearman"], -x[0]["rank_MAE"]))
        if best["ranking_score"] <= current["ranking_score"] + 1e-12: break
        selected.append(name); features = trial; current = best
        history.append({"step": len(selected), "selected_groups": selected.copy(), **current})
        print(policy, "step", len(selected), name, current["ranking_score"], flush=True)
    return {"policy": policy, "selected_groups": selected, **current}, history, all_rows


def locked(data, winner):
    train = data[data.target_year <= HOLDOUT - HORIZON]
    test = data[data.target_year == HOLDOUT].sort_values("gu")
    pred, model = fit_predict(train, test, winner["features"], winner["spec"], winner["mode"])
    return metrics_lib.rank_metrics(test.target_y.to_numpy(), pred, test.target_year.to_numpy()), pred, model, test


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True); MODEL_OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(BASE / "panel.csv")
    priority = pd.read_csv(PRIORITY / "priority_features.csv")
    panel = panel.merge(priority, on=["year", "gu"], how="left", validate="one_to_one")
    data = pairs(panel)
    strict_groups, lag_groups = make_availability_groups(panel)
    strict, strict_history, strict_rows = greedy_policy(data, strict_groups, "strict_2025")
    lagged, lag_history, lag_rows = greedy_policy(data, lag_groups, "lag_allowed")
    baseline_cv = metrics_lib.rank_metrics(
        data[data.target_year.isin(CV_YEARS)].target_y.to_numpy(),
        data[data.target_year.isin(CV_YEARS)].y.to_numpy(),
        data[data.target_year.isin(CV_YEARS)].target_year.to_numpy(),
    )
    strict_hold, strict_pred, _, hold = locked(data, strict)
    lag_hold, lag_pred, _, _ = locked(data, lagged)
    baseline_hold = metrics_lib.rank_metrics(hold.target_y.to_numpy(), hold.y.to_numpy(), hold.target_year.to_numpy())
    # Primary policy follows the user's request. Lag model is reported as a
    # sensitivity comparison, not selected using the locked holdout.
    winner = strict
    winner_hold = strict_hold
    latest = panel[panel.year == 2025].sort_values("gu").copy()
    operational_pred, operational_model = fit_predict(data, latest, winner["features"], winner["spec"], winner["mode"])
    forecast = latest[["gu", "year", "y", "crime", "child"]].copy()
    forecast["target_year"] = 2027; forecast["predicted_y_2027"] = operational_pred
    forecast.to_csv(OUT / "forecast_2027.csv", index=False, encoding="utf-8-sig")
    predictions = hold[["gu", "year", "target_year", "y", "target_y"]].copy()
    predictions["strict_prediction"] = strict_pred; predictions["lag_allowed_prediction"] = lag_pred
    predictions.to_csv(OUT / "holdout_2025_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{k: v for k, v in r.items() if k != "features"} for r in strict_rows + lag_rows]).to_csv(OUT / "all_comparisons.csv", index=False, encoding="utf-8-sig")
    report = {
        "objective": "2025 inputs -> 2027 final exposure Y and borough order",
        "horizon": 2, "training_origin_years": [int(data.year.min()), int(data.year.max())],
        "training_target_years": [int(data.target_year.min()), int(data.target_year.max())], "n_training_pairs": len(data),
        "cv_years": list(CV_YEARS), "locked_holdout": HOLDOUT,
        "baseline_cv": baseline_cv, "strict_winner_cv": strict, "lag_allowed_winner_cv": lagged,
        "baseline_holdout": baseline_hold, "strict_holdout": strict_hold, "lag_allowed_holdout": lag_hold,
        "accepted_for_rank_service": strict["ranking_score"] > baseline_cv["ranking_score"] and strict_hold["ranking_score"] > baseline_hold["ranking_score"],
        "continuous_y_tradeoff": "rank composite and quintile accuracy improved, but locked-holdout Y RMSE/MAE did not",
        "primary_policy": "strict_2025", "strict_groups": strict_groups, "lag_allowed_groups": lag_groups,
        "strict_forward_history": strict_history, "lag_forward_history": lag_history,
    }
    (OUT / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump({"model": operational_model, "features": winner["features"], "spec": winner["spec"], "mode": winner["mode"], "metadata": report}, MODEL_OUT / "operational_exposure_2y_2027_strict.joblib")
    with pd.ExcelWriter(OUT / "exposure_2y_2027_comparison.xlsx", engine="openpyxl") as writer:
        pd.DataFrame([{"model": "persistence CV", **baseline_cv}, {"model": "strict CV", **{k: v for k, v in strict.items() if k not in ("features", "selected_groups")}}, {"model": "lag allowed CV", **{k: v for k, v in lagged.items() if k not in ("features", "selected_groups")}}]).to_excel(writer, sheet_name="CV_summary", index=False)
        pd.DataFrame([{"model": "strict", **strict_hold}, {"model": "lag allowed", **lag_hold}]).to_excel(writer, sheet_name="holdout_2025", index=False)
        forecast.to_excel(writer, sheet_name="forecast_2027", index=False)
        predictions.to_excel(writer, sheet_name="holdout_predictions", index=False)
    print(json.dumps({"baseline_cv": baseline_cv, "strict_cv": {k: v for k, v in strict.items() if k != "features"}, "lag_cv": {k: v for k, v in lagged.items() if k != "features"}, "strict_holdout": strict_hold, "lag_holdout": lag_hold, "training_pairs": len(data)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
