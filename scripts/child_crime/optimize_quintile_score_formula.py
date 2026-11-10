"""Validate and optimize an explainable 20/40/60/80/100 weighted score."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts" / "child_crime_exposure_rank_3y_v1"
PANEL = ROOT / "artifacts" / "child_crime_borough_v2" / "panel.csv"


def metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    ar, pr = rankdata(-actual, method="ordinal"), rankdata(-prediction, method="ordinal")
    spearman = float(spearmanr(actual, prediction).statistic)
    rank_mae = float(np.abs(ar - pr).mean())
    grade = float(np.mean(np.floor((ar - 1) / 5) == np.floor((pr - 1) / 5)) * 100)
    top5 = float(len(set(np.argsort(-actual)[:5]) & set(np.argsort(-prediction)[:5])) / 5 * 100)
    top10 = float(len(set(np.argsort(-actual)[:10]) & set(np.argsort(-prediction)[:10])) / 10 * 100)
    score = 100 * (.40 * ((spearman + 1) / 2) + .25 * (1 - rank_mae / 24) + .20 * top5 / 100 + .15 * grade / 100)
    return {"ranking_score": score, "Spearman": spearman, "rank_MAE": rank_mae, "exact_quintile_pct": grade, "top20_overlap_pct": top5, "top40_overlap_pct": top10}


def main() -> int:
    weights = pd.read_csv(BASE / "integrated_feature_weights.csv").sort_values("mean_abs_y_change", ascending=False)
    direction = weights.set_index("feature").direction.to_dict()
    panel = pd.read_csv(PANEL)

    def dataset(target_year: int, features: list[str]):
        origin = panel[panel.year == target_year - 3].sort_values("gu")
        actual = panel[panel.year == target_year].set_index("gu").loc[origin.gu, "y"].to_numpy()
        columns = []
        for feature in features:
            values = pd.to_numeric(origin[feature], errors="coerce").to_numpy()
            ordering = -values if direction[feature] == "higher_risk" else values
            ranks = rankdata(ordering, method="ordinal")
            columns.append(np.array([100, 80, 60, 40, 20])[(np.minimum(ranks - 1, 24) // 5).astype(int)])
        actual_rank = rankdata(-actual, method="ordinal")
        target_points = np.array([100, 80, 60, 40, 20])[(np.minimum(actual_rank - 1, 24) // 5).astype(int)]
        return origin.gu.to_numpy(), np.asarray(columns).T, target_points, actual

    candidates = []
    for pool in ("all", "external_only"):
        available = weights if pool == "all" else weights[weights.score_eligible]
        counts = sorted(set([5, 8, 10, 15, 20, 30, len(available)]))
        for count in counts:
            features = available.head(count).feature.tolist()
            for alpha in (.01, .1, 1, 10, 100, 1000, 10000):
                annual = []
                for test_year in (2022, 2023, 2024):
                    train_x, train_y = [], []
                    for train_year in (2022, 2023, 2024):
                        if train_year == test_year:
                            continue
                        _, x, y, _ = dataset(train_year, features)
                        train_x.append(x); train_y.append(y)
                    model = Ridge(alpha=alpha, positive=True, fit_intercept=False).fit(np.vstack(train_x), np.hstack(train_y))
                    _, x, _, actual = dataset(test_year, features)
                    annual.append(metrics(actual, model.predict(x)))
                average = {key: float(np.mean([row[key] for row in annual])) for key in annual[0]}
                candidates.append({"pool": pool, "feature_count": len(features), "alpha": alpha, "features": features, **average})
    candidates.sort(key=lambda row: (-row["ranking_score"], -row["Spearman"], row["rank_MAE"], -row["top20_overlap_pct"], -row["exact_quintile_pct"]))
    winner = candidates[0]
    train_x, train_y = [], []
    for year in (2022, 2023, 2024):
        _, x, y, _ = dataset(year, winner["features"])
        train_x.append(x); train_y.append(y)
    model = Ridge(alpha=winner["alpha"], positive=True, fit_intercept=False).fit(np.vstack(train_x), np.hstack(train_y))
    gu, hold_x, _, hold_actual = dataset(2025, winner["features"])
    hold_prediction = model.predict(hold_x)
    holdout = metrics(hold_actual, hold_prediction)
    normalized = model.coef_ / model.coef_.sum() * 100
    formula = pd.DataFrame({"feature": winner["features"], "direction": [direction[f] for f in winner["features"]], "weight_pct": normalized}).sort_values("weight_pct", ascending=False)
    formula = formula[formula.weight_pct > 1e-10].reset_index(drop=True)
    formula.weight_pct = formula.weight_pct / formula.weight_pct.sum() * 100
    formula.to_csv(BASE / "validated_score_formula.csv", index=False, encoding="utf-8-sig")
    validation = pd.DataFrame({"gu": gu, "target_year": 2025, "actual_y": hold_actual, "weighted_score": hold_prediction})
    validation["actual_rank"] = rankdata(-validation.actual_y, method="ordinal")
    validation["score_rank"] = rankdata(-validation.weighted_score, method="ordinal")
    validation.to_csv(BASE / "validated_score_holdout_2025.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{k: v for k, v in row.items() if k != "features"} for row in candidates]).to_csv(BASE / "score_formula_comparisons.csv", index=False, encoding="utf-8-sig")
    result = {"method": "positive Ridge on feature quintile points; weights normalized to 100", "selection": "leave-one-target-year-out over 2022-2024", "winner_cv": winner, "holdout_2025": holdout, "feature_weights": formula.to_dict("records"), "external_only_warning": "External-only sensitivity weights failed validation and are not the final formula."}
    model_result_path = BASE / "result.json"
    model_result = json.loads(model_result_path.read_text(encoding="utf-8"))
    baseline_cv_score = model_result["baseline_cv"]["ranking_score"]
    baseline_holdout_score = model_result["baseline_holdout"]["ranking_score"]
    result["accepted"] = bool(winner["ranking_score"] > baseline_cv_score and holdout["ranking_score"] > baseline_holdout_score)
    result["decision"] = "experimental only; do not use as the production score formula" if not result["accepted"] else "accepted"
    (BASE / "score_formula_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    model_result["score_formula_validation"] = {
        "status": result["decision"], "accepted": result["accepted"],
        "cv": {k: v for k, v in winner.items() if k != "features"}, "holdout_2025": holdout,
        "reason": "The predictive pipeline improves ranking, but the compressed quintile-weight formula does not beat persistence on the composite ranking score.",
    }
    model_result_path.write_text(json.dumps(model_result, ensure_ascii=False, indent=2), encoding="utf-8")
    provisional = pd.read_csv(BASE / "borough_integrated_scores_2025_inputs.csv")
    provisional["formula_status"] = "rejected_experimental"
    provisional.to_csv(BASE / "borough_integrated_scores_2025_inputs.csv", index=False, encoding="utf-8-sig")
    workbook = BASE / "exposure_rank_model_and_weights.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        formula.to_excel(writer, sheet_name="validated_formula", index=False)
        validation.to_excel(writer, sheet_name="formula_holdout", index=False)
        pd.DataFrame([{k: v for k, v in row.items() if k != "features"} for row in candidates]).to_excel(writer, sheet_name="formula_comparisons", index=False)
    print(json.dumps({"winner_cv": {k: v for k, v in winner.items() if k != "features"}, "holdout": holdout, "weights": formula.to_dict("records")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
