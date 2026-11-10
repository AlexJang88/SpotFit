"""Evaluate leakage-controlled ensembles of the old and new crime models."""
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
from ml.training import optimize_borough_crime_count_3y as new

OUT = ROOT / "artifacts" / "borough_crime_count_ensemble_3y_v1"
MODEL_OUT = ROOT / "models" / "experiments" / "borough_crime_count_ensemble_3y_v1"


def inputs():
    old_panel = pd.read_csv(old.OUT / "panel.csv")
    old_groups = json.loads((old.OUT / "feature_groups.json").read_text(encoding="utf-8"))
    old_columns = list(dict.fromkeys(old.CORE + [c for values in old_groups.values() for c in values]))
    old_pairs = old.make_pairs(old_panel, 3)
    row = pd.read_csv(old.OUT / "3y" / "best_by_method_cv.csv").query("method == 'components'").iloc[0]
    old_features = ast.literal_eval(row.features)

    result = json.loads((new.OUT / "result.json").read_text(encoding="utf-8"))
    winner = result["winner_cv"]
    panel = pd.read_csv(new.BASE / "panel.csv")
    panel, _, type_counts = new.add_crime_history(panel)
    new_pairs = new.make_pairs(panel, type_counts)
    return old_panel, old_pairs, old_columns, old_features, float(row.alpha), panel, new_pairs, type_counts, winner


def predict_old(train, test, columns, features, alpha):
    context = old.context(train, test, columns, sorted(train.gu.unique()))
    model = old.fit(context, [columns.index(c) for c in features], "components", alpha)
    _, detail = old.predict(model, test, 3)
    return np.log1p(detail["crime"]), model


def predict_new(train, test, winner, type_counts):
    return new.fit_predict(train, test, winner["features"], winner["spec"], winner["mode"], type_counts)


def prediction_table(years, old_pairs, old_columns, old_features, old_alpha, new_pairs, type_counts, winner):
    rows = []
    for target_year in years:
        origin = target_year - 3
        old_test = old_pairs[old_pairs.target_year == target_year].sort_values("gu")
        old_train = old_pairs[old_pairs.target_year <= origin]
        new_test = new_pairs[new_pairs.target_year == target_year].sort_values("gu")
        new_train = new_pairs[new_pairs.target_year <= origin]
        assert old_test.gu.tolist() == new_test.gu.tolist()
        old_pred, _ = predict_old(old_train, old_test, old_columns, old_features, old_alpha)
        new_pred, _ = predict_new(new_train, new_test, winner, type_counts)
        for gu, actual, current, p_old, p_new in zip(
            new_test.gu, new_test.target_log_crime, new_test.log_crime, old_pred, new_pred
        ):
            rows.append({"gu": gu, "target_year": target_year, "actual_log": actual, "current_log": current, "old_log": p_old, "new_log": p_new})
    frame = pd.DataFrame(rows)
    for column in ("actual", "current", "old", "new"):
        frame[f"{column}_crime"] = np.expm1(frame[f"{column}_log"])
    return frame


def blend(frame, weight_new: float, space: str) -> np.ndarray:
    if space == "log":
        return (1 - weight_new) * frame.old_log.to_numpy() + weight_new * frame.new_log.to_numpy()
    counts = (1 - weight_new) * frame.old_crime.to_numpy() + weight_new * frame.new_crime.to_numpy()
    return np.log1p(np.maximum(0, counts))


def find_weight(frame, space: str) -> tuple[float, dict[str, float]]:
    candidates = []
    for weight in np.linspace(0, 1, 101):
        metrics = new.score(frame.actual_log.to_numpy(), blend(frame, float(weight), space))
        candidates.append((metrics["log_RMSE"], metrics["crime_MAE"], float(weight), metrics))
    _, _, weight, metrics = min(candidates)
    return weight, metrics


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    old_panel, old_pairs, old_columns, old_features, old_alpha, panel, new_pairs, type_counts, winner = inputs()
    cv = prediction_table(new.CV_YEARS, old_pairs, old_columns, old_features, old_alpha, new_pairs, type_counts, winner)
    holdout = prediction_table((new.HOLDOUT_YEAR,), old_pairs, old_columns, old_features, old_alpha, new_pairs, type_counts, winner)

    candidates = []
    for space in ("log", "count"):
        optimal_weight, optimal_cv = find_weight(cv, space)
        inverse_weight = (1 / new.score(cv.actual_log, cv.new_log)["log_RMSE"]) / (
            1 / new.score(cv.actual_log, cv.new_log)["log_RMSE"] + 1 / new.score(cv.actual_log, cv.old_log)["log_RMSE"]
        )
        for name, weight in (("simple_average", .5), ("inverse_cv_error", inverse_weight), ("cv_optimal", optimal_weight)):
            candidates.append({
                "name": name, "space": space, "weight_new": weight, "weight_old": 1 - weight,
                **{f"cv_{k}": v for k, v in new.score(cv.actual_log, blend(cv, weight, space)).items()},
                **{f"holdout_{k}": v for k, v in new.score(holdout.actual_log, blend(holdout, weight, space)).items()},
            })

    # Honest second-level estimate: each year's weight is learned from the other two CV years.
    nested_rows = []
    for space in ("log", "count"):
        actual, predicted, selected = [], [], []
        for year in new.CV_YEARS:
            tune = cv[cv.target_year != year]
            test = cv[cv.target_year == year]
            weight, _ = find_weight(tune, space)
            actual.extend(test.actual_log)
            predicted.extend(blend(test, weight, space))
            selected.append({"target_year": year, "weight_new": weight, "weight_old": 1 - weight})
        nested_rows.append({"space": space, "fold_weights": selected, **new.score(np.asarray(actual), np.asarray(predicted))})

    candidate_frame = pd.DataFrame(candidates).sort_values(["cv_log_RMSE", "cv_crime_MAE"])
    selected = candidate_frame.iloc[0].to_dict()
    accepted_vs_both = (
        selected["cv_log_RMSE"] < min(new.score(cv.actual_log, cv.old_log)["log_RMSE"], new.score(cv.actual_log, cv.new_log)["log_RMSE"])
        and selected["holdout_log_RMSE"] < min(new.score(holdout.actual_log, holdout.old_log)["log_RMSE"], new.score(holdout.actual_log, holdout.new_log)["log_RMSE"])
    )
    report = {
        "selection_rule": "minimum pooled 2022-2024 log RMSE; 2025 untouched until weight lock",
        "old_cv": new.score(cv.actual_log, cv.old_log), "new_cv": new.score(cv.actual_log, cv.new_log),
        "old_holdout": new.score(holdout.actual_log, holdout.old_log), "new_holdout": new.score(holdout.actual_log, holdout.new_log),
        "selected_ensemble": selected, "nested_weight_cv": nested_rows,
        "accepted_vs_both_models": bool(accepted_vs_both),
    }
    (OUT / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    candidate_frame.to_csv(OUT / "ensemble_comparisons.csv", index=False, encoding="utf-8-sig")
    cv.to_csv(OUT / "cv_predictions.csv", index=False, encoding="utf-8-sig")
    holdout = holdout.copy()
    holdout["ensemble_log"] = blend(holdout, selected["weight_new"], selected["space"])
    holdout["ensemble_crime"] = np.expm1(holdout.ensemble_log)
    holdout.to_csv(OUT / "holdout_2025_predictions.csv", index=False, encoding="utf-8-sig")

    # Fit both members on all labels available through 2025 and forecast from 2025 to 2028.
    latest_old = old_panel[(old_panel.year == 2025) & old_panel.y.notna()].sort_values("gu").copy()
    latest_old["target_year"] = 2028
    old_operational_pred, old_operational = predict_old(old_pairs, latest_old, old_columns, old_features, old_alpha)
    latest_new = panel[(panel.year == 2025) & panel.log_crime.notna()].sort_values("gu").copy()
    latest_new["target_year"] = 2028
    new_operational_pred, new_operational = predict_new(new_pairs, latest_new, winner, type_counts)
    assert latest_old.gu.tolist() == latest_new.gu.tolist()
    operational_frame = pd.DataFrame({"old_log": old_operational_pred, "new_log": new_operational_pred})
    final_log = blend(operational_frame.assign(old_crime=np.expm1(operational_frame.old_log), new_crime=np.expm1(operational_frame.new_log)), selected["weight_new"], selected["space"])
    forecast = latest_new[["gu", "year", "target_year", "crime"]].copy()
    forecast["old_predicted_crime"] = np.expm1(old_operational_pred)
    forecast["new_predicted_crime"] = np.expm1(new_operational_pred)
    forecast["ensemble_predicted_crime"] = np.expm1(final_log)
    forecast.to_csv(OUT / "forecast_2028.csv", index=False, encoding="utf-8-sig")
    joblib.dump({
        "old_model": old_operational, "new_model": new_operational,
        "old_features": old_features, "new_features": winner["features"],
        "weight_new": selected["weight_new"], "weight_old": selected["weight_old"],
        "blend_space": selected["space"], "metadata": report,
    }, MODEL_OUT / "operational_crime_count_ensemble_3y.joblib")

    with pd.ExcelWriter(OUT / "crime_count_ensemble_3y_comparison.xlsx", engine="openpyxl") as writer:
        candidate_frame.to_excel(writer, sheet_name="ensemble_comparison", index=False)
        pd.DataFrame(nested_rows).to_excel(writer, sheet_name="nested_weight_cv", index=False)
        cv.to_excel(writer, sheet_name="cv_predictions", index=False)
        holdout.to_excel(writer, sheet_name="holdout_2025", index=False)
        forecast.to_excel(writer, sheet_name="forecast_2028", index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
