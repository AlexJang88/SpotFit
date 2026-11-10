"""Train the one-year child regional-crime exposure rank service model."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "artifacts" / "child_crime_borough_v2" / "panel.csv"
OUT = ROOT / "models" / "recommender" / "v1" / "child_crime_exposure" / "1y"
CV_YEARS = (2022, 2023, 2024)
CORE = ["y", "log_crime", "log_child", "y_lag1", "y_lag2", "y_change", "log_crime_change", "log_child_change"]
TREND = ["log_crime_lag1", "log_crime_lag2", "log_child_lag1", "log_child_lag2"]


def make_pairs(panel: pd.DataFrame) -> pd.DataFrame:
    future = panel[["gu", "year", "y"]].copy()
    future["year"] -= 1
    result = panel.merge(future.rename(columns={"y": "target_y"}), on=["gu", "year"])
    return result.loc[result["y"].notna() & result["target_y"].notna()].copy().assign(target_year=lambda x: x.year + 1)


def make_pipeline(features: list[str], alpha: float) -> Pipeline:
    prep = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), features),
        ("borough", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["gu"]),
    ], sparse_threshold=0)
    return Pipeline([("prep", prep), ("model", Ridge(alpha=alpha))])


def rank_metrics(actual: np.ndarray, predicted: np.ndarray, years: np.ndarray) -> dict:
    rows = []
    for year in sorted(set(years)):
        mask = years == year
        a = pd.Series(actual[mask]).rank(ascending=False).to_numpy()
        p = pd.Series(predicted[mask]).rank(ascending=False).to_numpy()
        rows.append({"Spearman": float(spearmanr(actual[mask], predicted[mask]).statistic), "rank_MAE": float(np.mean(abs(a-p))), "exact_quintile_pct": float(np.mean(((a-1)//5)==((p-1)//5))*100), "top20_overlap_pct": float(len(set(np.where(a<=5)[0]) & set(np.where(p<=5)[0]))/5*100)})
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def main() -> None:
    panel = pd.read_csv(PANEL)
    pairs = make_pairs(panel)
    candidates = []
    for features in (CORE, CORE + TREND):
        for alpha in (.01, .1, 1, 10, 100):
            for mode in ("direct", "delta"):
                actual, predicted, years = [], [], []
                for target_year in CV_YEARS:
                    train, test = pairs[pairs.target_year < target_year], pairs[pairs.target_year == target_year]
                    model = make_pipeline(features, alpha)
                    target = train.target_y if mode == "direct" else train.target_y - train.y
                    model.fit(train[["gu", *features]], target)
                    pred = model.predict(test[["gu", *features]])
                    if mode == "delta": pred += test.y.to_numpy()
                    actual.extend(test.target_y); predicted.extend(pred); years.extend(test.target_year)
                candidates.append({"features": features, "alpha": alpha, "mode": mode, **rank_metrics(np.asarray(actual), np.asarray(predicted), np.asarray(years))})
    candidates.sort(key=lambda row: (-row["Spearman"], row["rank_MAE"], -row["exact_quintile_pct"]))
    winner = candidates[0]
    model = make_pipeline(winner["features"], winner["alpha"])
    target = pairs.target_y if winner["mode"] == "direct" else pairs.target_y - pairs.y
    model.fit(pairs[["gu", *winner["features"]]], target)
    latest = panel[panel.year == panel.year.max()].sort_values("gu").copy()
    prediction = model.predict(latest[["gu", *winner["features"]]])
    if winner["mode"] == "delta": prediction += latest.y.to_numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": winner["features"], "mode": winner["mode"], "horizon": 1, "metadata": winner}, OUT / "child_crime_exposure_1y.joblib")
    pd.DataFrame({"district": latest.gu, "base_year": latest.year, "prediction_year": latest.year + 1, "predicted_y": prediction}).to_csv(OUT / "raw_forecast.csv", index=False, encoding="utf-8-sig")
    (OUT / "model_metadata.json").write_text(json.dumps({"objective": "1년 뒤 아동 지역범죄 노출 순위", "selection_cv_years": CV_YEARS, "winner": winner, "candidate_count": len(candidates)}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
