"""Test the newly acquired cohort/school/stress data in the 3-year crime model.

Selection uses target years 2022-2024. The 2025 target is a locked holdout.
Both absolute-count accuracy and borough-order accuracy are reported.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ml.training import compare_child_crime_borough_v2 as old
from ml.training import optimize_borough_crime_count_3y as crime

PRIORITY = ROOT / "artifacts" / "child_population_priority_v1"
OUT = ROOT / "artifacts" / "borough_crime_count_priority_v2"
MODEL_OUT = ROOT / "models" / "experiments" / "borough_crime_count_priority_v2"


def old_setup():
    panel = pd.read_csv(old.OUT / "panel.csv")
    groups = json.loads((old.OUT / "feature_groups.json").read_text(encoding="utf-8"))
    columns = list(dict.fromkeys(old.CORE + [c for values in groups.values() for c in values]))
    pairs = old.make_pairs(panel, 3)
    row = pd.read_csv(old.OUT / "3y" / "best_by_method_cv.csv").query("method == 'components'").iloc[0]
    return panel, pairs, columns, ast.literal_eval(row.features), float(row.alpha)


def old_predictions(years, pairs, columns, features, alpha):
    rows = []
    for target_year in years:
        origin = target_year - 3
        test = pairs[pairs.target_year == target_year].sort_values("gu")
        train = pairs[pairs.target_year <= origin]
        context = old.context(train, test, columns, sorted(train.gu.unique()))
        model = old.fit(context, [columns.index(c) for c in features], "components", alpha)
        _, detail = old.predict(model, test, 3)
        for gu, value in zip(test.gu, np.log1p(detail["crime"])):
            rows.append({"target_year": target_year, "gu": gu, "old_log": value})
    return pd.DataFrame(rows)


def predictions(pairs, years, features, spec, mode, type_counts):
    rows, folds = [], []
    for target_year in years:
        origin = target_year - 3
        train = pairs[pairs.target_year <= origin]
        test = pairs[pairs.target_year == target_year].sort_values("gu")
        pred, _ = crime.fit_predict(train, test, features, spec, mode, type_counts)
        metrics = crime.score(test.target_log_crime.to_numpy(), pred)
        folds.append({"target_year": target_year, **metrics})
        for gu, actual, value in zip(test.gu, test.target_log_crime, pred):
            rows.append({"target_year": target_year, "gu": gu, "actual_log": actual, "new_log": value})
    frame = pd.DataFrame(rows)
    return frame, {**crime.score(frame.actual_log.to_numpy(), frame.new_log.to_numpy()), "folds": folds}


def blend_metrics(new_predictions, old_predictions_frame, weight_new):
    aligned = new_predictions.merge(old_predictions_frame, on=["target_year", "gu"], validate="one_to_one")
    value = weight_new * aligned.new_log.to_numpy() + (1 - weight_new) * aligned.old_log.to_numpy()
    return crime.score(aligned.actual_log.to_numpy(), value), value, aligned


def compact(row):
    return {k: v for k, v in row.items() if k not in ("features", "folds")}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    old_panel, old_pairs, old_columns, old_features, old_alpha = old_setup()
    base_panel = pd.read_csv(crime.BASE / "panel.csv")
    panel, _, type_counts = crime.add_crime_history(base_panel)
    priority = pd.read_csv(PRIORITY / "priority_features.csv")
    groups = json.loads((PRIORITY / "feature_groups.json").read_text(encoding="utf-8"))
    panel = panel.merge(priority, on=["year", "gu"], how="left", validate="one_to_one")
    pairs = crime.make_pairs(panel, type_counts)
    prior = json.loads((crime.OUT / "result.json").read_text(encoding="utf-8"))["winner_cv"]
    base_features = prior["features"]

    specs = [f"ridge:{a}" for a in (.01, .1, 1, 10, 100, 1000)] + [
        "elastic:.001,.1", "elastic:.001,.5", "elastic:.01,.1", "elastic:.01,.5",
        "extra:5,3,.7", "extra:9,2,1.0", "forest:6,3,.8", "hist:7,10,5", "gbr:2,5",
    ]
    names = list(groups)
    configs = [(0, prior["spec"], prior["mode"], "v1_exact")]
    configs += [(mask, spec, mode, "priority_search") for mask in range(1 << len(names)) for spec in specs for mode in ("direct", "delta")]
    multioutput_specs = [spec for spec in specs if spec.startswith(("ridge:", "extra:", "forest:"))]
    configs += [(mask, spec, "type_sum", "priority_search") for mask in range(1 << len(names)) for spec in multioutput_specs]
    results = []
    cache = {}
    for index, (mask, spec, mode, stage) in enumerate(configs, 1):
        active = [name for i, name in enumerate(names) if mask & (1 << i)]
        features = list(dict.fromkeys([*base_features, *[c for name in active for c in groups[name]]]))
        frame, metrics = predictions(pairs, crime.CV_YEARS, features, spec, mode, type_counts)
        key = f"{mask}|{spec}|{mode}|{stage}"
        cache[key] = frame
        results.append({"key": key, "stage": stage, "mask": mask, "priority_groups": active, "features": features, "spec": spec, "mode": mode, **metrics})
        if index % 64 == 0:
            print(f"{index}/{len(configs)}", flush=True)

    baseline = results[0]
    absolute_winner = min(results, key=lambda r: (r["log_RMSE"], r["crime_MAE"], -r["Spearman"]))
    rank_winner = max(results, key=lambda r: (r["Spearman"], -r["log_RMSE"], -r["crime_MAE"]))
    old_cv = old_predictions(crime.CV_YEARS, old_pairs, old_columns, old_features, old_alpha)
    ensemble_candidates = []
    # Test both the absolute and rank winner as replacement candidates for the
    # newer member of the existing old/new ensemble.
    for candidate_name, candidate in (("absolute", absolute_winner), ("rank", rank_winner)):
        frame = cache[candidate["key"]]
        for weight in np.linspace(0, 1, 101):
            metrics, _, _ = blend_metrics(frame, old_cv, float(weight))
            ensemble_candidates.append({"candidate": candidate_name, "weight_new": float(weight), **metrics})
    ensemble_absolute = min(ensemble_candidates, key=lambda r: (r["log_RMSE"], r["crime_MAE"]))
    ensemble_rank = max(ensemble_candidates, key=lambda r: (r["Spearman"], -r["log_RMSE"]))
    fixed_baseline_cv, _, _ = blend_metrics(cache[baseline["key"]], old_cv, .5)
    fixed_improved_cv, _, _ = blend_metrics(cache[absolute_winner["key"]], old_cv, .5)

    # Lock configurations and open 2025 exactly once.
    hold_rows = {}
    for name, candidate in (("baseline", baseline), ("absolute", absolute_winner), ("rank", rank_winner)):
        frame, metrics = predictions(pairs, (crime.HOLDOUT_YEAR,), candidate["features"], candidate["spec"], candidate["mode"], type_counts)
        hold_rows[name] = (frame, metrics)
    old_hold = old_predictions((crime.HOLDOUT_YEAR,), old_pairs, old_columns, old_features, old_alpha)
    ensemble_hold = {}
    for name, chosen in (("absolute", ensemble_absolute), ("rank", ensemble_rank)):
        frame = hold_rows[chosen["candidate"]][0]
        metrics, values, aligned = blend_metrics(frame, old_hold, chosen["weight_new"])
        output = aligned.copy()
        output["ensemble_log"] = values
        output["actual_crime"] = np.expm1(output.actual_log)
        output["predicted_crime"] = np.expm1(values)
        ensemble_hold[name] = {"metrics": metrics, "predictions": output}
    fixed_baseline_hold, _, _ = blend_metrics(hold_rows["baseline"][0], old_hold, .5)
    fixed_improved_hold, fixed_improved_values, fixed_improved_frame = blend_metrics(hold_rows["absolute"][0], old_hold, .5)

    # Refit both selected standalone candidates using all completed labels and
    # forecast from 2025 to 2028. They remain experiment artifacts.
    latest = panel[panel.year == 2025].sort_values("gu").copy()
    latest["target_year"] = 2028
    fitted = {}
    forecasts = latest[["gu", "year", "target_year", "crime"]].copy()
    for name, candidate in (("absolute", absolute_winner), ("rank", rank_winner)):
        pred, model = crime.fit_predict(pairs, latest, candidate["features"], candidate["spec"], candidate["mode"], type_counts)
        fitted[name] = model
        forecasts[f"{name}_predicted_crime"] = np.expm1(pred)
    latest_old = old_panel[old_panel.year == 2025].sort_values("gu").copy()
    latest_old["target_year"] = 2028
    old_context = old.context(old_pairs, latest_old, old_columns, sorted(old_pairs.gu.unique()))
    old_operational = old.fit(old_context, [old_columns.index(c) for c in old_features], "components", old_alpha)
    _, old_detail = old.predict(old_operational, latest_old, 3)
    old_log = np.log1p(old_detail["crime"])
    new_absolute_log = np.log1p(forecasts["absolute_predicted_crime"].to_numpy())
    forecasts["selected_ensemble_predicted_crime"] = np.expm1(
        ensemble_absolute["weight_new"] * new_absolute_log + (1 - ensemble_absolute["weight_new"]) * old_log
    )
    forecasts.to_csv(OUT / "forecast_2028.csv", index=False, encoding="utf-8-sig")

    stress_yes = [r for r in results if "stress" in r["priority_groups"]]
    stress_no = [r for r in results if "stress" not in r["priority_groups"]]
    report = {
        "selection_policy": "2022-2024 CV selection; 2025 locked holdout",
        "baseline_cv": baseline,
        "absolute_winner_cv": absolute_winner,
        "rank_winner_cv": rank_winner,
        "baseline_holdout_2025": hold_rows["baseline"][1],
        "absolute_winner_holdout_2025": hold_rows["absolute"][1],
        "rank_winner_holdout_2025": hold_rows["rank"][1],
        "ensemble_absolute_cv": ensemble_absolute,
        "ensemble_rank_cv": ensemble_rank,
        "ensemble_absolute_holdout_2025": ensemble_hold["absolute"]["metrics"],
        "ensemble_rank_holdout_2025": ensemble_hold["rank"]["metrics"],
        "fixed_50_50_baseline_cv": fixed_baseline_cv,
        "fixed_50_50_improved_cv": fixed_improved_cv,
        "fixed_50_50_baseline_holdout_2025": fixed_baseline_hold,
        "fixed_50_50_improved_holdout_2025": fixed_improved_hold,
        "stress_ablation": {
            "best_with_stress_absolute": min(stress_yes, key=lambda r: r["log_RMSE"]),
            "best_without_stress_absolute": min(stress_no, key=lambda r: r["log_RMSE"]),
            "best_with_stress_rank": max(stress_yes, key=lambda r: r["Spearman"]),
            "best_without_stress_rank": max(stress_no, key=lambda r: r["Spearman"]),
        },
        "configuration_count": len(configs),
    }
    (OUT / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([compact(r) for r in results]).to_csv(OUT / "all_comparisons.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ensemble_candidates).to_csv(OUT / "ensemble_weight_search.csv", index=False, encoding="utf-8-sig")
    ensemble_hold["absolute"]["predictions"].to_csv(OUT / "holdout_2025_absolute_ensemble.csv", index=False, encoding="utf-8-sig")
    ensemble_hold["rank"]["predictions"].to_csv(OUT / "holdout_2025_rank_ensemble.csv", index=False, encoding="utf-8-sig")
    fixed_output = fixed_improved_frame.copy()
    fixed_output["ensemble_log"] = fixed_improved_values
    fixed_output["actual_crime"] = np.expm1(fixed_output.actual_log)
    fixed_output["predicted_crime"] = np.expm1(fixed_improved_values)
    fixed_output.to_csv(OUT / "holdout_2025_recommended_fixed_50_50.csv", index=False, encoding="utf-8-sig")
    joblib.dump({
        "absolute_model": fitted["absolute"], "absolute_features": absolute_winner["features"], "absolute_config": absolute_winner,
        "rank_model": fitted["rank"], "rank_features": rank_winner["features"], "rank_config": rank_winner,
        "type_counts": type_counts, "metadata": report,
    }, MODEL_OUT / "crime_count_3y_priority_candidates.joblib")
    joblib.dump({
        "old_model": old_operational,
        "new_model": fitted["absolute"],
        "old_features": old_features,
        "new_features": absolute_winner["features"],
        "weight_new": ensemble_absolute["weight_new"],
        "weight_old": 1 - ensemble_absolute["weight_new"],
        "blend_space": "log",
        "type_counts": type_counts,
        "metadata": report,
        "usage": "absolute three-year crime-count experiment; not selected for exposure-rank service",
    }, MODEL_OUT / "operational_crime_count_priority_ensemble.joblib")
    joblib.dump({
        "old_model": old_operational,
        "new_model": fitted["absolute"],
        "old_features": old_features,
        "new_features": absolute_winner["features"],
        "weight_new": .5,
        "weight_old": .5,
        "blend_space": "log",
        "type_counts": type_counts,
        "metadata": report,
        "usage": "recommended fixed-policy crime-count model; exposure-rank promotion requires separate final-Y selection",
    }, MODEL_OUT / "recommended_crime_count_priority_fixed_50_50.joblib")
    with pd.ExcelWriter(OUT / "crime_priority_model_comparison.xlsx", engine="openpyxl") as writer:
        pd.DataFrame([
            {"model": "baseline CV", **compact(baseline)},
            {"model": "absolute winner CV", **compact(absolute_winner)},
            {"model": "rank winner CV", **compact(rank_winner)},
        ]).to_excel(writer, sheet_name="CV_summary", index=False)
        pd.DataFrame([
            {"model": "baseline", **hold_rows["baseline"][1]},
            {"model": "absolute winner", **hold_rows["absolute"][1]},
            {"model": "rank winner", **hold_rows["rank"][1]},
            {"model": "absolute ensemble", **ensemble_hold["absolute"]["metrics"]},
            {"model": "rank ensemble", **ensemble_hold["rank"]["metrics"]},
        ]).to_excel(writer, sheet_name="holdout_2025", index=False)
        forecasts.to_excel(writer, sheet_name="forecast_2028", index=False)
    print(json.dumps({
        "baseline_cv": compact(baseline), "absolute_winner_cv": compact(absolute_winner), "rank_winner_cv": compact(rank_winner),
        "baseline_holdout": hold_rows["baseline"][1], "absolute_holdout": hold_rows["absolute"][1], "rank_holdout": hold_rows["rank"][1],
        "ensemble_absolute_cv": ensemble_absolute, "ensemble_absolute_holdout": ensemble_hold["absolute"]["metrics"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
