import math

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scripts.child_crime import export_service_crime_forecasts as export


def test_ridge_formula_excludes_borough_and_normalizes_numeric_weights():
    frame = pd.DataFrame({
        "gu": [f"구{i:02}" for i in range(25)],
        "risk_up": np.arange(25, dtype=float),
        "risk_down": np.arange(24, -1, -1, dtype=float),
    })
    prep = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())]),
         ["risk_up", "risk_down"]),
        ("borough", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["gu"]),
    ], sparse_threshold=0)
    model = Pipeline([("prep", prep), ("model", Ridge(alpha=1))]).fit(
        frame, frame["risk_up"] - frame["risk_down"])
    formula = export.ridge_numeric_formula({
        "model": model, "features": ["risk_up", "risk_down"]})
    assert math.isclose(formula.weight_pct.sum(), 100)
    assert set(formula.feature) == {"risk_up", "risk_down"}
    assert not formula.feature.str.startswith("gu_").any()


def test_weighted_feature_score_is_continuous_sum_and_grade_comes_after_score(tmp_path, monkeypatch):
    districts = [f"구{i:02}" for i in range(25)]
    panel = pd.DataFrame({
        "gu": districts, "year": 2025,
        "risk_up": np.arange(25, dtype=float),
        "risk_down": np.roll(np.arange(25, dtype=float), 7),
    })
    panel_path = tmp_path / "panel.csv"
    panel.to_csv(panel_path, index=False, encoding="utf-8-sig")
    monkeypatch.setattr(export, "PANEL", panel_path)
    formula = pd.DataFrame({
        "feature": ["risk_up", "risk_down"],
        "direction": ["higher_risk", "lower_risk"],
        "signed_effect": [1.0, -1.0],
        "weight_pct": [63.0, 37.0],
        "weight_method": ["test", "test"],
    })
    forecasts = pd.DataFrame({
        "district": districts, "base_year": 2025, "prediction_year": 2027,
        "predicted_y": np.arange(25, dtype=float),
    })
    result = export.add_service_columns(forecasts, "predicted_y", formula, 2, tmp_path)
    assert result.safety_score.nunique() > 5
    assert result.safety_score.between(20, 100).all()
    assert result.sort_values("safety_rank").safety_score.is_monotonic_decreasing
    assert list(result.sort_values("safety_rank").safety_grade) == list("AAAAABBBBBCCCCCDDDDDEEEEE")
    details = pd.read_csv(tmp_path / "feature_score_breakdown.csv")
    recomputed = details.groupby("district").weighted_contribution.sum()
    actual = result.set_index("district").safety_score
    assert all(math.isclose(actual[district], value) for district, value in recomputed.items())


def test_rescore_existing_uses_deployed_model_without_retraining(tmp_path, monkeypatch):
    service = tmp_path / "service"
    target = service / "2y"
    target.mkdir(parents=True)
    districts = [f"구{i:02}" for i in range(25)]
    training = pd.DataFrame({
        "gu": districts, "risk": np.arange(25, dtype=float),
    })
    prep = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())]), ["risk"]),
        ("borough", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["gu"]),
    ], sparse_threshold=0)
    model = Pipeline([("prep", prep), ("model", Ridge(alpha=1))]).fit(training, training["risk"])
    joblib.dump({"model": model, "features": ["risk"]},
                target / "child_crime_exposure_2y.joblib")
    pd.DataFrame({
        "district": districts, "base_year": 2025, "prediction_year": 2027,
        "predicted_y": np.arange(25, dtype=float),
        "safety_score": [100] * 25,
    }).to_csv(target / "service_forecast.csv", index=False, encoding="utf-8-sig")
    (target / "model_metadata.json").write_text("{}", encoding="utf-8")
    panel = tmp_path / "panel.csv"
    pd.DataFrame({"gu": districts, "year": 2025, "risk": np.arange(25)}).to_csv(
        panel, index=False, encoding="utf-8-sig")
    monkeypatch.setattr(export, "SERVICE", service)
    monkeypatch.setattr(export, "PANEL", panel)
    export.rescore_existing(2)
    assert (target / "feature_score_formula.csv").is_file()
    assert (target / "feature_score_breakdown.csv").is_file()
    rescored = pd.read_csv(target / "service_forecast.csv")
    assert list(rescored.sort_values("safety_rank").safety_score.unique()) == [100, 80, 60, 40, 20]
