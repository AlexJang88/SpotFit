"""End-to-end optimization of 3-year exposure values and borough order.

The target is final Y directly. This avoids assuming that independently better
crime and population losses must imply a better borough ordering.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ml.training import optimize_exposure_rank_3y as rank_v1
from ml.training import optimize_borough_crime_count_3y as crime

PRIORITY = ROOT / "artifacts" / "child_population_priority_v1"
OUT = ROOT / "artifacts" / "child_crime_exposure_integrated_v2"
MODEL_OUT = ROOT / "models" / "experiments" / "child_crime_exposure_integrated_v2"


def prep(features):
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), features),
        ("borough", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["gu"]),
    ], sparse_threshold=0)


def fit_direct(train, test, features, spec, mode):
    name, value = spec.split(":", 1)
    if name == "ridge":
        estimator = Ridge(alpha=float(value))
    elif name == "elastic":
        alpha, ratio = map(float, value.split(","))
        estimator = ElasticNet(alpha=alpha, l1_ratio=ratio, max_iter=30000, random_state=42)
    else:
        estimator = crime.estimator(spec)
    model = Pipeline([("prep", prep(features)), ("model", estimator)])
    target = train.target_y - train.y if mode == "delta" else train.target_y
    model.fit(train[["gu", *features]], target)
    pred = model.predict(test[["gu", *features]])
    if mode == "delta":
        pred += test.y.to_numpy()
    return pred, model


def fit_pairwise(train, test, features, alpha):
    transformer = prep(features)
    x = transformer.fit_transform(train[["gu", *features]])
    xt = transformer.transform(test[["gu", *features]])
    diffs, targets = [], []
    for _, indices in train.groupby("target_year").groups.items():
        idx = np.asarray(list(indices))
        # DataFrame indices are not guaranteed to be positional after slicing.
        pos = train.index.get_indexer(idx)
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                diffs.append(x[pos[i]] - x[pos[j]])
                targets.append(train.loc[idx[i], "target_y"] - train.loc[idx[j], "target_y"])
    estimator = Ridge(alpha=alpha, fit_intercept=False).fit(np.asarray(diffs), np.asarray(targets))
    raw_train = x @ estimator.coef_
    calibration = np.polyfit(raw_train, train.target_y.to_numpy(), 1) if np.std(raw_train) > 0 else np.array([0., train.target_y.mean()])
    pred = calibration[0] * (xt @ estimator.coef_) + calibration[1]
    return pred, {"transformer": transformer, "estimator": estimator, "calibration": calibration}


def evaluate(pairs, features, spec, mode):
    actual, predicted, years = [], [], []
    for target_year in rank_v1.CV_YEARS:
        train = pairs[pairs.target_year <= target_year - 3].copy().reset_index(drop=True)
        test = pairs[pairs.target_year == target_year].sort_values("gu").copy()
        if mode == "pairwise":
            pred, _ = fit_pairwise(train, test, features, float(spec.split(":")[1]))
        else:
            pred, _ = fit_direct(train, test, features, spec, mode)
        actual.extend(test.target_y); predicted.extend(pred); years.extend(test.target_year)
    return rank_v1.rank_metrics(np.asarray(actual), np.asarray(predicted), np.asarray(years))


def fit_locked(train, test, row):
    if row["mode"] == "pairwise":
        return fit_pairwise(train.reset_index(drop=True), test, row["features"], float(row["spec"].split(":")[1]))
    return fit_direct(train, test, row["features"], row["spec"], row["mode"])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True); MODEL_OUT.mkdir(parents=True, exist_ok=True)
    panel, _, _, old_crime_features, _, _, _, _, new_crime_winner, old_groups = rank_v1.crime_setup()
    priority = pd.read_csv(PRIORITY / "priority_features.csv")
    priority_groups = json.loads((PRIORITY / "feature_groups.json").read_text(encoding="utf-8"))
    panel = panel.merge(priority, on=["year", "gu"], how="left", validate="one_to_one")
    pairs = rank_v1.population_pairs(panel).reset_index(drop=True)
    prior = json.loads((rank_v1.OUT / "result.json").read_text(encoding="utf-8"))["winner_cv"]

    minimal = ["y", "log_crime", "log_child", "y_lag1", "y_lag2", "y_change", "log_crime_change", "log_child_change"]
    selected_union = list(dict.fromkeys([*minimal, *prior["features"], *old_crime_features, *new_crime_winner["features"]]))
    bases = {"minimal": minimal, "selected_union": selected_union}
    direct_specs = [f"ridge:{a}" for a in (.01, .1, 1, 10, 100, 1000)] + [
        "elastic:.001,.1", "elastic:.001,.5", "elastic:.01,.1", "elastic:.01,.5",
        "extra:5,3,.7", "extra:9,2,1.0", "forest:6,3,.8", "hist:7,10,5", "gbr:2,5",
    ]
    results = []
    names = list(priority_groups)
    total = len(bases) * (1 << len(names)) * (len(direct_specs) * 2 + 6)
    done = 0
    for base_name, base in bases.items():
        for mask in range(1 << len(names)):
            active = [name for i, name in enumerate(names) if mask & (1 << i)]
            features = list(dict.fromkeys([*base, *[c for name in active for c in priority_groups[name]]]))
            for spec in direct_specs:
                for mode in ("direct", "delta"):
                    metrics = evaluate(pairs, features, spec, mode)
                    results.append({"base": base_name, "mask": mask, "priority_groups": active, "features": features, "spec": spec, "mode": mode, **metrics})
                    done += 1
            for alpha in (.01, .1, 1, 10, 100, 1000):
                spec = f"pairridge:{alpha}"
                metrics = evaluate(pairs, features, spec, "pairwise")
                results.append({"base": base_name, "mask": mask, "priority_groups": active, "features": features, "spec": spec, "mode": "pairwise", **metrics})
                done += 1
            if done % 144 == 0:
                print(f"{done}/{total}", flush=True)

    results.sort(key=lambda r: (-r["ranking_score"], -r["Spearman"], r["rank_MAE"], -r["exact_quintile_pct"], r["y_RMSE"]))
    winner = results[0]
    hold_test = pairs[pairs.target_year == rank_v1.HOLDOUT].sort_values("gu")
    hold_train = pairs[pairs.target_year <= rank_v1.HOLDOUT - 3].copy()
    hold_pred, locked_model = fit_locked(hold_train, hold_test, winner)
    hold_metrics = rank_v1.rank_metrics(hold_test.target_y.to_numpy(), hold_pred, hold_test.target_year.to_numpy())
    baseline_cv = prior
    baseline_hold = json.loads((rank_v1.OUT / "result.json").read_text(encoding="utf-8"))["winner_holdout"]
    accepted = winner["ranking_score"] > baseline_cv["ranking_score"] and hold_metrics["ranking_score"] >= baseline_hold["ranking_score"]

    latest = panel[panel.year == 2025].sort_values("gu").copy()
    operational_pred, operational_model = fit_locked(pairs.copy(), latest, winner)
    compact = pd.DataFrame([{k: v for k, v in r.items() if k != "features"} for r in results])
    compact.to_csv(OUT / "all_comparisons.csv", index=False, encoding="utf-8-sig")
    predictions = hold_test[["gu", "year", "target_year", "y", "target_y"]].copy()
    predictions["predicted_y"] = hold_pred
    predictions.to_csv(OUT / "holdout_2025_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"gu": latest.gu, "predicted_y_2028": operational_pred}).to_csv(OUT / "forecast_2028.csv", index=False, encoding="utf-8-sig")
    report = {
        "target": "direct final Y and borough ordering",
        "selection_years": list(rank_v1.CV_YEARS), "locked_holdout": rank_v1.HOLDOUT,
        "baseline_cv": baseline_cv, "winner_cv": winner,
        "baseline_holdout": baseline_hold, "winner_holdout": hold_metrics,
        "accepted": accepted, "configuration_count": len(results),
        "stacking_constraint": "Residual stacking was not used: leakage-safe base forecasts start too late to train a meta-model for every 2022-2024 selection fold.",
    }
    (OUT / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump({"model": operational_model, "features": winner["features"], "spec": winner["spec"], "mode": winner["mode"], "metadata": report}, MODEL_OUT / "integrated_exposure_3y.joblib")
    with pd.ExcelWriter(OUT / "integrated_exposure_comparison.xlsx", engine="openpyxl") as writer:
        pd.DataFrame([{"model": "baseline CV", **{k: v for k, v in baseline_cv.items() if k not in ("features", "groups")}}, {"model": "integrated winner CV", **{k: v for k, v in winner.items() if k not in ("features", "priority_groups")}}]).to_excel(writer, sheet_name="CV_summary", index=False)
        pd.DataFrame([{"model": "baseline", **baseline_hold}, {"model": "integrated winner", **hold_metrics}]).to_excel(writer, sheet_name="holdout_2025", index=False)
        predictions.to_excel(writer, sheet_name="holdout_predictions", index=False)
        compact.to_excel(writer, sheet_name="all_comparisons", index=False)
    print(json.dumps({"baseline_cv": {k: baseline_cv[k] for k in ("ranking_score", "Spearman", "rank_MAE", "exact_quintile_pct", "y_RMSE")}, "winner_cv": {k: v for k, v in winner.items() if k != "features"}, "baseline_holdout": baseline_hold, "winner_holdout": hold_metrics, "accepted": accepted}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
