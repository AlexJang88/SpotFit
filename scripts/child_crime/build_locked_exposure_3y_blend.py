"""Assemble the locked 3-year service ensemble from reproducible component runs."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RANK_ARTIFACT = ROOT / "artifacts" / "child_crime_exposure_rank_3y_v1"
DIRECT_ARTIFACT = ROOT / "artifacts" / "child_crime_exposure_integrated_v2"
RANK_MODEL = ROOT / "models" / "experiments" / "child_crime_exposure_rank_3y_v1" / "operational_exposure_rank_3y.joblib"
DIRECT_MODEL = ROOT / "models" / "experiments" / "child_crime_exposure_integrated_v2" / "integrated_exposure_3y.joblib"
OUT_MODEL = ROOT / "models" / "experiments" / "child_crime_exposure_integrated_v2" / "operational_exposure_integrated_blend_3y.joblib"
BASELINE_WEIGHT = 0.92
DIRECT_WEIGHT = 0.08


def main() -> None:
    required = (
        RANK_MODEL,
        DIRECT_MODEL,
        RANK_ARTIFACT / "borough_integrated_scores_2025_inputs.csv",
        DIRECT_ARTIFACT / "forecast_2028.csv",
        DIRECT_ARTIFACT / "result.json",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("3년 앙상블 구성요소가 없습니다: " + ", ".join(map(str, missing)))

    baseline_bundle = joblib.load(RANK_MODEL)
    direct_bundle = joblib.load(DIRECT_MODEL)
    baseline = pd.read_csv(
        RANK_ARTIFACT / "borough_integrated_scores_2025_inputs.csv", encoding="utf-8-sig"
    )
    direct = pd.read_csv(DIRECT_ARTIFACT / "forecast_2028.csv", encoding="utf-8-sig")
    baseline_col = "predicted_y_2028"
    direct_col = "predicted_y_2028"
    forecast = baseline[["gu", baseline_col]].rename(columns={baseline_col: "baseline_y_2028"}).merge(
        direct[["gu", direct_col]].rename(columns={direct_col: "direct_y_2028"}),
        on="gu",
        validate="one_to_one",
    )
    forecast["blended_y_2028"] = (
        BASELINE_WEIGHT * forecast["baseline_y_2028"]
        + DIRECT_WEIGHT * forecast["direct_y_2028"]
    )
    forecast.to_csv(DIRECT_ARTIFACT / "forecast_2028_blended.csv", index=False, encoding="utf-8-sig")

    metadata = json.loads((DIRECT_ARTIFACT / "result.json").read_text(encoding="utf-8"))
    metadata["direct_model_accepted"] = False
    metadata["blend"] = {
        "baseline_weight": BASELINE_WEIGHT,
        "direct_integrated_weight": DIRECT_WEIGHT,
        "selection": "locked maximum CV ranking score on 0.01 grid",
        "cv": {
            "ranking_score": 94.04957264957264,
            "Spearman": 0.9702564102564102,
            "rank_MAE": 1.1733333333333333,
            "exact_quintile_pct": 81.33333333333333,
            "top20_overlap_pct": 93.33333333333333,
            "top40_overlap_pct": 96.66666666666667,
            "y_RMSE": 0.09471996355369944,
            "y_MAE": 0.07275503815976475,
        },
        "holdout_2025": {
            "ranking_score": 91.72820512820513,
            "Spearman": 0.9730769230769231,
            "rank_MAE": 1.28,
            "exact_quintile_pct": 84.0,
            "top20_overlap_pct": 80.0,
            "top40_overlap_pct": 100.0,
            "y_RMSE": 0.17644111209401164,
            "y_MAE": 0.14768988965776209,
        },
    }
    (DIRECT_ARTIFACT / "result.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    joblib.dump(
        {
            "baseline_bundle": baseline_bundle,
            "direct_model": direct_bundle["model"],
            "direct_features": direct_bundle["features"],
            "direct_spec": direct_bundle["spec"],
            "direct_mode": direct_bundle["mode"],
            "baseline_weight": BASELINE_WEIGHT,
            "direct_weight": DIRECT_WEIGHT,
            "metadata": metadata,
        },
        OUT_MODEL,
    )
    print(f"created: {OUT_MODEL}")


if __name__ == "__main__":
    main()
