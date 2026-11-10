"""Optimize a leakage-controlled three-year Seoul borough crime-count model.

Model selection uses target years 2022-2024. Target year 2025 is touched only
after the candidate is locked.  The experiment predicts the total count, not a
grade, and keeps the existing exposure panel's delayed external features.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts" / "child_crime_borough_v2"
OUT = ROOT / "artifacts" / "borough_crime_count_3y_v1"
MODEL_OUT = ROOT / "models" / "experiments" / "borough_crime_count_3y_v1"
RAW_DIR = ROOT / "data" / "raw" / "아동범죄_행정동"
CV_YEARS = (2022, 2023, 2024)
HOLDOUT_YEAR = 2025
HORIZON = 3
RANDOM_STATE = 42


def read_csv_flexible(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            pass
    raise UnicodeError(path)


def add_crime_history(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    source = read_csv_flexible(RAW_DIR / "서울시_5대범죄_2014_2025_원자료.csv")
    source["발생건수"] = pd.to_numeric(source["발생건수"], errors="coerce")
    source["체포수"] = pd.to_numeric(source["체포수"], errors="coerce")
    types = sorted(source["범죄"].dropna().unique())
    occurrence = source.pivot_table(index=["년도", "시군구"], columns="범죄", values="발생건수", aggfunc="sum")
    arrests = source.pivot_table(index=["년도", "시군구"], columns="범죄", values="체포수", aggfunc="sum")
    occurrence.columns = [f"crime_type_{i}_count" for i in range(len(types))]
    arrests.columns = [f"crime_type_{i}_arrests" for i in range(len(types))]
    history = occurrence.join(arrests).reset_index().rename(columns={"년도": "year", "시군구": "gu"})
    type_counts = list(occurrence.columns)
    type_arrests = list(arrests.columns)
    total = history[type_counts].sum(axis=1)
    for count, arrest in zip(type_counts, type_arrests):
        history[f"{count}_log"] = np.log1p(history[count])
        history[f"{count}_share"] = history[count] / total.replace(0, np.nan)
        history[f"{count}_clearance"] = history[arrest] / history[count].replace(0, np.nan)
    history = history.sort_values(["gu", "year"])
    base_cols = [f"{c}_log" for c in type_counts]
    for col in base_cols:
        history[f"{col}_lag1"] = history.groupby("gu")[col].shift(1)
        history[f"{col}_change"] = history[col] - history[f"{col}_lag1"]
    panel = panel.merge(history, on=["gu", "year"], how="left", validate="one_to_one")
    pattern = (
        base_cols
        + [f"{c}_share" for c in type_counts]
        + [f"{c}_clearance" for c in type_counts]
        + [f"{c}_log_lag1" for c in type_counts]
        + [f"{c}_log_change" for c in type_counts]
    )
    return panel, pattern, type_counts


def make_pairs(panel: pd.DataFrame, type_counts: list[str]) -> pd.DataFrame:
    targets = ["log_crime", "crime", *type_counts]
    future = panel[["gu", "year", *targets]].copy()
    future["year"] -= HORIZON
    future = future.rename(columns={c: f"target_{c}" for c in targets})
    pairs = panel.merge(future, on=["gu", "year"], validate="one_to_one")
    pairs["target_year"] = pairs["year"] + HORIZON
    return pairs[pairs["year"] >= 2015].sort_values(["year", "gu"]).reset_index(drop=True)


def preprocessing(features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), features),
            ("borough", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["gu"]),
        ],
        sparse_threshold=0,
    )


def estimator(spec: str):
    name, value = spec.split(":", 1)
    params = [float(x) for x in value.split(",")]
    if name == "ridge":
        return Ridge(alpha=params[0])
    if name == "elastic":
        return ElasticNet(alpha=params[0], l1_ratio=params[1], max_iter=30000, random_state=RANDOM_STATE)
    if name == "extra":
        return ExtraTreesRegressor(n_estimators=120, max_depth=int(params[0]), min_samples_leaf=int(params[1]), max_features=params[2], n_jobs=1, random_state=RANDOM_STATE)
    if name == "forest":
        return RandomForestRegressor(n_estimators=160, max_depth=int(params[0]), min_samples_leaf=int(params[1]), max_features=params[2], n_jobs=1, random_state=RANDOM_STATE)
    if name == "hist":
        return HistGradientBoostingRegressor(max_iter=180, max_leaf_nodes=int(params[0]), min_samples_leaf=int(params[1]), l2_regularization=params[2], learning_rate=.04, early_stopping=False, random_state=RANDOM_STATE)
    if name == "gbr":
        return GradientBoostingRegressor(n_estimators=160, max_depth=int(params[0]), min_samples_leaf=int(params[1]), learning_rate=.03, loss="huber", random_state=RANDOM_STATE)
    raise ValueError(spec)


def score(actual_log: np.ndarray, predicted_log: np.ndarray) -> dict[str, float]:
    actual = np.expm1(actual_log)
    predicted = np.maximum(0, np.expm1(np.clip(predicted_log, 0, 20)))
    return {
        "log_RMSE": float(mean_squared_error(actual_log, predicted_log) ** .5),
        "crime_RMSE": float(mean_squared_error(actual, predicted) ** .5),
        "crime_MAE": float(mean_absolute_error(actual, predicted)),
        "crime_MAPE_pct": float(np.mean(np.abs(predicted - actual) / np.maximum(actual, 1)) * 100),
        "crime_R2": float(r2_score(actual, predicted)),
        "Spearman": float(spearmanr(actual, predicted).statistic),
    }


def fit_predict(train: pd.DataFrame, valid: pd.DataFrame, features: list[str], spec: str, mode: str, type_counts: list[str]) -> tuple[np.ndarray, object]:
    columns = ["gu", *features]
    model = Pipeline([("prep", preprocessing(features)), ("model", estimator(spec))])
    if mode == "direct":
        target = train["target_log_crime"]
        model.fit(train[columns], target)
        return model.predict(valid[columns]), model
    if mode == "delta":
        target = train["target_log_crime"] - train["log_crime"]
        model.fit(train[columns], target)
        return valid["log_crime"].to_numpy() + model.predict(valid[columns]), model
    if mode == "type_sum":
        target = np.log1p(train[[f"target_{c}" for c in type_counts]].to_numpy())
        model.fit(train[columns], target)
        prediction = np.maximum(0, np.expm1(model.predict(valid[columns]))).sum(axis=1)
        return np.log1p(prediction), model
    raise ValueError(mode)


def evaluate(pairs: pd.DataFrame, features: list[str], spec: str, mode: str, type_counts: list[str]) -> dict[str, object]:
    actual, predicted, folds = [], [], []
    for target_year in CV_YEARS:
        origin = target_year - HORIZON
        train = pairs[pairs["target_year"] <= origin]
        valid = pairs[pairs["target_year"] == target_year]
        if len(valid) != 25 or train.empty:
            raise ValueError((target_year, len(train), len(valid)))
        p, _ = fit_predict(train, valid, features, spec, mode, type_counts)
        fold = score(valid["target_log_crime"].to_numpy(), p)
        folds.append({"target_year": target_year, "n_train": len(train), **fold})
        actual.extend(valid["target_log_crime"])
        predicted.extend(p)
    return {"spec": spec, "mode": mode, "n_features": len(features), "features": features, "folds": folds, **score(np.asarray(actual), np.asarray(predicted))}


def baseline_metrics(frame: pd.DataFrame) -> dict[str, float]:
    return score(frame["target_log_crime"].to_numpy(), frame["log_crime"].to_numpy())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-subsets", type=int, default=24)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(BASE / "panel.csv")
    groups = json.loads((BASE / "feature_groups.json").read_text(encoding="utf-8"))
    panel, crime_pattern, type_counts = add_crime_history(panel)
    groups = {"범죄유형·검거·추세": crime_pattern, **groups}
    pairs = make_pairs(panel, type_counts)
    group_names = list(groups)
    masks = range(1 << len(group_names))
    ridge_specs = [f"ridge:{a}" for a in (.01, .1, 1, 10, 100, 1000)]
    results: list[dict[str, object]] = []

    print(f"stage1 subsets={2 ** len(group_names)} configurations={2 ** len(group_names) * len(ridge_specs) * 2}", flush=True)
    for mask in masks:
        active = [name for i, name in enumerate(group_names) if mask & (1 << i)]
        features = list(dict.fromkeys(["log_crime", *[c for name in active for c in groups[name]]]))
        for spec in ridge_specs:
            for mode in ("direct", "delta"):
                row = evaluate(pairs, features, spec, mode, type_counts)
                results.append({"stage": "all_subsets_linear", "mask": mask, "groups": active, **row})
        if (mask + 1) % 128 == 0:
            best = min(results, key=lambda x: x["log_RMSE"])
            print(mask + 1, best["log_RMSE"], best["groups"], best["spec"], best["mode"], flush=True)

    ranked_masks = []
    for row in sorted(results, key=lambda x: (x["log_RMSE"], x["crime_MAE"], x["n_features"])):
        if row["mask"] not in ranked_masks:
            ranked_masks.append(row["mask"])
        if len(ranked_masks) == args.top_subsets:
            break
    nonlinear_specs = [
        "elastic:.001,.1", "elastic:.001,.5", "elastic:.01,.1", "elastic:.01,.5",
        "extra:4,3,.7", "extra:7,3,.8", "extra:12,2,1.0",
        "forest:5,3,.7", "forest:9,3,1.0",
        "hist:5,10,10", "hist:9,8,3", "gbr:2,5", "gbr:3,5",
    ]
    print(f"stage2 shortlisted_subsets={len(ranked_masks)}", flush=True)
    for mask in ranked_masks:
        active = [name for i, name in enumerate(group_names) if mask & (1 << i)]
        features = list(dict.fromkeys(["log_crime", *[c for name in active for c in groups[name]]]))
        for spec in nonlinear_specs:
            for mode in ("direct", "delta"):
                row = evaluate(pairs, features, spec, mode, type_counts)
                results.append({"stage": "shortlist_nonlinear", "mask": mask, "groups": active, **row})
        for spec in ("ridge:.1", "ridge:1", "ridge:10", "extra:7,3,.8", "forest:5,3,.7"):
            row = evaluate(pairs, features, spec, "type_sum", type_counts)
            results.append({"stage": "shortlist_crime_type_sum", "mask": mask, "groups": active, **row})

    cv_frame = pairs[pairs["target_year"].isin(CV_YEARS)]
    baseline_cv = baseline_metrics(cv_frame)
    results.sort(key=lambda x: (x["log_RMSE"], x["crime_MAE"], x["n_features"]))
    winner = results[0]
    holdout = pairs[pairs["target_year"] == HOLDOUT_YEAR]
    origin = HOLDOUT_YEAR - HORIZON
    train = pairs[pairs["target_year"] <= origin]
    hold_pred, locked_model = fit_predict(train, holdout, winner["features"], winner["spec"], winner["mode"], type_counts)
    hold_metrics = score(holdout["target_log_crime"].to_numpy(), hold_pred)
    baseline_holdout = baseline_metrics(holdout)
    accepted = winner["log_RMSE"] < baseline_cv["log_RMSE"] and hold_metrics["log_RMSE"] < baseline_holdout["log_RMSE"]

    serializable = []
    for row in results:
        serializable.append({k: v for k, v in row.items() if k not in ("features", "folds")})
    pd.DataFrame(serializable).to_csv(OUT / "all_model_comparisons.csv", index=False, encoding="utf-8-sig")
    by_method = pd.DataFrame(serializable).sort_values(["log_RMSE", "crime_MAE"]).drop_duplicates(["spec", "mode"])
    by_method.to_csv(OUT / "best_by_method.csv", index=False, encoding="utf-8-sig")

    predictions = holdout[["gu", "year", "target_year", "crime", "target_crime"]].copy()
    predictions["baseline_crime"] = holdout["crime"]
    predictions["predicted_crime"] = np.maximum(0, np.expm1(hold_pred))
    predictions.to_csv(OUT / "holdout_2025_predictions.csv", index=False, encoding="utf-8-sig")

    progression = [{"step": 0, "model": "Persistence", "log_RMSE": baseline_cv["log_RMSE"], "improvement_vs_baseline_pct": 0}]
    best_rmse = baseline_cv["log_RMSE"]
    for row in sorted(results, key=lambda x: (0 if x["stage"] == "all_subsets_linear" else 1, len(x["groups"]), x["mask"], x["spec"], x["mode"])):
        if row["log_RMSE"] < best_rmse - 1e-12:
            progression.append({"step": len(progression), "model": row["spec"], "mode": row["mode"], "groups": "+".join(row["groups"]), "log_RMSE": row["log_RMSE"], "crime_MAPE_pct": row["crime_MAPE_pct"], "improvement_vs_baseline_pct": (baseline_cv["log_RMSE"] - row["log_RMSE"]) / baseline_cv["log_RMSE"] * 100})
            best_rmse = row["log_RMSE"]
    pd.DataFrame(progression).to_csv(OUT / "improvement_history.csv", index=False, encoding="utf-8-sig")

    report = {
        "target": "three-year-ahead total five-major-crime count; modeled as log1p(count)",
        "selection_years": list(CV_YEARS), "untouched_holdout_year": HOLDOUT_YEAR,
        "baseline_cv": baseline_cv, "winner_cv": winner,
        "baseline_holdout": baseline_holdout, "winner_holdout": hold_metrics,
        "cv_improvement_pct": (baseline_cv["log_RMSE"] - winner["log_RMSE"]) / baseline_cv["log_RMSE"] * 100,
        "holdout_improvement_pct": (baseline_holdout["log_RMSE"] - hold_metrics["log_RMSE"]) / baseline_holdout["log_RMSE"] * 100,
        "accepted": accepted,
        "search": {"group_count": len(groups), "all_group_subsets": 2 ** len(groups), "total_configurations": len(results), "stage2_shortlisted_subsets": len(ranked_masks)},
        "leakage_rule": "for forecast origin t, training labels are restricted to target_year <= t; 2025 is excluded from selection",
        "group_dictionary": groups,
    }
    (OUT / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump({"model": locked_model, "features": winner["features"], "spec": winner["spec"], "mode": winner["mode"], "type_counts": type_counts, "metadata": report}, MODEL_OUT / "cv_best_crime_count_3y.joblib")
    print(json.dumps({k: report[k] for k in ("cv_improvement_pct", "holdout_improvement_pct", "accepted")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
