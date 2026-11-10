"""Focused v2 search using cohort, school movement/count, and stress features."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ml.training import optimize_exposure_rank_3y as rank_v1

PRIORITY = ROOT / "artifacts" / "child_population_priority_v1"
OUT = ROOT / "artifacts" / "child_population_priority_v2"
MODEL_OUT = ROOT / "models" / "experiments" / "child_population_priority_v2"


def child_metrics(actual_log, predicted_log) -> dict[str, float]:
    actual, predicted = np.expm1(actual_log), np.expm1(predicted_log)
    return {
        "child_log_RMSE": float(mean_squared_error(actual_log, predicted_log) ** .5),
        "child_MAE_people": float(mean_absolute_error(actual, predicted)),
        "child_MAPE_pct": float(np.mean(np.abs(predicted - actual) / actual) * 100),
    }


def evaluate(pairs, crime_cv, features, spec, mode):
    predicted, actual_child, actual_y, years, gus = [], [], [], [], []
    for target_year in rank_v1.CV_YEARS:
        origin = target_year - rank_v1.HORIZON
        train = pairs[pairs.target_year <= origin]
        test = pairs[pairs.target_year == target_year].sort_values("gu")
        pred, _ = rank_v1.fit_population(train, test, features, spec, mode)
        predicted.extend(pred)
        actual_child.extend(test.target_log_child)
        actual_y.extend(test.target_y)
        years.extend(test.target_year)
        gus.extend(test.gu)
    predicted, actual_child = np.asarray(predicted), np.asarray(actual_child)
    actual_y, years = np.asarray(actual_y), np.asarray(years)
    cm = child_metrics(actual_child, predicted)
    aligned = crime_cv.set_index(["target_year", "gu"]).loc[list(zip(years, gus))].reset_index()
    rows = []
    for crime_name in ("current", "old", "new", "ensemble84_log", "average_log"):
        py = rank_v1.exposure(aligned[crime_name].to_numpy(), predicted)
        rows.append({"crime_candidate": crime_name, **cm, **rank_v1.rank_metrics(actual_y, py, years)})
    return rows


def locked_holdout(pairs, crime_holdout, row):
    test = pairs[pairs.target_year == rank_v1.HOLDOUT].sort_values("gu")
    train = pairs[pairs.target_year <= rank_v1.HOLDOUT - rank_v1.HORIZON]
    pred, model = rank_v1.fit_population(train, test, row["features"], row["spec"], row["mode"])
    crime = crime_holdout.sort_values("gu")
    assert crime.gu.tolist() == test.gu.tolist()
    py = rank_v1.exposure(crime[row["crime_candidate"]].to_numpy(), pred)
    return {**child_metrics(test.target_log_child.to_numpy(), pred), **rank_v1.rank_metrics(test.target_y.to_numpy(), py, test.target_year.to_numpy())}, pred, py, model, test


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    panel, crime_pairs, crime_columns, old_features, old_alpha, new_panel, new_pairs, type_counts, new_winner, _ = rank_v1.crime_setup()
    priority = pd.read_csv(PRIORITY / "priority_features.csv")
    priority_groups = json.loads((PRIORITY / "feature_groups.json").read_text(encoding="utf-8"))
    panel = panel.merge(priority, on=["year", "gu"], how="left", validate="one_to_one")
    pairs = rank_v1.population_pairs(panel)
    crime_cv = rank_v1.crime_predictions(rank_v1.CV_YEARS, crime_pairs, crime_columns, old_features, old_alpha, new_pairs, type_counts, new_winner)
    crime_holdout = rank_v1.crime_predictions((rank_v1.HOLDOUT,), crime_pairs, crime_columns, old_features, old_alpha, new_pairs, type_counts, new_winner)

    prior = json.loads((rank_v1.OUT / "result.json").read_text(encoding="utf-8"))["winner_cv"]
    base_features = prior["features"]
    specs = [f"ridge:{a}" for a in (.001, .01, .1, 1, 10, 100, 1000)] + [
        "elastic:.001,.1", "elastic:.001,.5", "elastic:.01,.1", "elastic:.01,.5",
        "extra:5,3,.7", "extra:9,2,1.0", "forest:6,3,.8", "hist:7,10,5", "gbr:2,5",
    ]
    group_names = list(priority_groups)
    results = []
    # Include the exact v1 population specification as a reproducible baseline.
    configs = [(0, prior["spec"], prior["mode"], "v1_exact")]
    configs += [(mask, spec, mode, "priority_search") for mask in range(1 << len(group_names)) for spec in specs for mode in ("direct", "delta")]
    for index, (mask, spec, mode, stage) in enumerate(configs, 1):
        active = [name for i, name in enumerate(group_names) if mask & (1 << i)]
        features = list(dict.fromkeys([*base_features, *[c for name in active for c in priority_groups[name]]]))
        for metrics in evaluate(pairs, crime_cv, features, spec, mode):
            results.append({"stage": stage, "mask": mask, "priority_groups": active, "features": features, "spec": spec, "mode": mode, **metrics})
        if index % 64 == 0:
            print(f"{index}/{len(configs)}", flush=True)

    ranking = sorted(results, key=lambda r: (-r["ranking_score"], -r["Spearman"], r["rank_MAE"], r["child_log_RMSE"]))
    population = sorted(results, key=lambda r: (r["child_log_RMSE"], r["child_MAPE_pct"], -r["ranking_score"]))
    rank_winner, child_winner = ranking[0], population[0]
    baseline = next(r for r in results if r["stage"] == "v1_exact" and r["crime_candidate"] == prior["crime_candidate"])
    rank_hold, rank_child_pred, rank_y_pred, _, hold_test = locked_holdout(pairs, crime_holdout, rank_winner)
    child_hold, _, _, _, _ = locked_holdout(pairs, crime_holdout, child_winner)

    # Refit the selected rank-oriented population component through all labels
    # available in 2025; the artifact remains experimental, not service v1.
    latest = panel[panel.year == 2025].sort_values("gu").copy()
    operational_pred, operational_model = rank_v1.fit_population(pairs, latest, rank_winner["features"], rank_winner["spec"], rank_winner["mode"])
    child_operational_pred, child_operational_model = rank_v1.fit_population(
        pairs, latest, child_winner["features"], child_winner["spec"], child_winner["mode"]
    )
    joblib.dump({
        "population_model": operational_model,
        "population_features": rank_winner["features"],
        "population_spec": rank_winner["spec"],
        "population_mode": rank_winner["mode"],
        "priority_groups": rank_winner["priority_groups"],
        "cv_winner": rank_winner,
        "locked_holdout": rank_hold,
    }, MODEL_OUT / "child_population_3y_priority.joblib")
    joblib.dump({
        "population_model": child_operational_model,
        "population_features": child_winner["features"],
        "population_spec": child_winner["spec"],
        "population_mode": child_winner["mode"],
        "priority_groups": child_winner["priority_groups"],
        "cv_winner": child_winner,
        "locked_holdout": child_hold,
        "usage": "child-population accuracy experiment; not selected for final exposure-rank service",
    }, MODEL_OUT / "child_population_3y_accuracy.joblib")

    compact = pd.DataFrame([{k: v for k, v in r.items() if k != "features"} for r in results])
    compact.to_csv(OUT / "all_comparisons.csv", index=False, encoding="utf-8-sig")
    hold_predictions = hold_test[["gu", "year", "target_year", "target_log_child", "target_y"]].copy()
    hold_predictions["predicted_log_child"] = rank_child_pred
    hold_predictions["actual_child"] = np.expm1(hold_predictions.target_log_child)
    hold_predictions["predicted_child"] = np.expm1(rank_child_pred)
    hold_predictions["predicted_y"] = rank_y_pred
    hold_predictions.to_csv(OUT / "holdout_2025_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "gu": latest.gu,
        "predicted_child_2028_rank_model": np.expm1(operational_pred),
        "predicted_child_2028_accuracy_model": np.expm1(child_operational_pred),
    }).to_csv(OUT / "predicted_child_2028.csv", index=False, encoding="utf-8-sig")

    stress_rows = [r for r in results if "stress" in r["priority_groups"]]
    no_stress_rows = [r for r in results if "stress" not in r["priority_groups"]]
    stress_best = sorted(stress_rows, key=lambda r: (-r["ranking_score"], r["child_log_RMSE"]))[0]
    no_stress_best = sorted(no_stress_rows, key=lambda r: (-r["ranking_score"], r["child_log_RMSE"]))[0]
    report = {
        "selection_policy": "CV target years 2022-2024 only; locked 2025 holdout opened once after selection",
        "baseline_cv": baseline,
        "rank_winner_cv": rank_winner,
        "child_accuracy_winner_cv": child_winner,
        "rank_winner_holdout_2025": rank_hold,
        "child_accuracy_winner_holdout_2025": child_hold,
        "stress_ablation": {"best_with_stress": stress_best, "best_without_stress": no_stress_best},
        "configuration_count": len(configs),
    }
    (OUT / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with pd.ExcelWriter(OUT / "priority_model_comparison.xlsx", engine="openpyxl") as writer:
        pd.DataFrame([
            {"model": "v1 baseline CV", **{k: v for k, v in baseline.items() if k not in ("features", "priority_groups")}},
            {"model": "v2 rank winner CV", **{k: v for k, v in rank_winner.items() if k not in ("features", "priority_groups")}},
            {"model": "v2 child winner CV", **{k: v for k, v in child_winner.items() if k not in ("features", "priority_groups")}},
        ]).to_excel(writer, sheet_name="CV_summary", index=False)
        pd.DataFrame([{"model": "rank winner", **rank_hold}, {"model": "child winner", **child_hold}]).to_excel(writer, sheet_name="holdout_2025", index=False)
        hold_predictions.to_excel(writer, sheet_name="holdout_predictions", index=False)
        compact.to_excel(writer, sheet_name="all_comparisons", index=False)
    print(json.dumps({
        "baseline_cv": {k: v for k, v in baseline.items() if k not in ("features",)},
        "rank_winner_cv": {k: v for k, v in rank_winner.items() if k not in ("features",)},
        "child_winner_cv": {k: v for k, v in child_winner.items() if k not in ("features",)},
        "rank_holdout": rank_hold,
        "stress_selected": "stress" in rank_winner["priority_groups"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
