"""Rebuild missing service models using the locked training pipelines."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(relative: str, *args: str) -> None:
    subprocess.run([sys.executable, str(ROOT / relative), *args], cwd=ROOT, check=True)


def main() -> None:
    base = ROOT / "artifacts/child_crime_borough_v2"
    priority = ROOT / "artifacts/child_population_priority_v1"
    if not (base / "panel.csv").is_file():
        run("ml/training/compare_child_crime_borough_v2.py")
    if not (priority / "priority_features.csv").is_file():
        run("scripts/child_crime/build_priority_population_features.py")
    service = ROOT / "models/recommender/v1/child_crime_exposure"
    deployed_core = [
        service / f"{h}y" / name
        for h in (2, 3)
        for name in (f"child_crime_exposure_{h}y.joblib", "service_forecast.csv", "model_metadata.json")
    ]
    deployed_scores = [
        service / f"{h}y" / name
        for h in (2, 3)
        for name in ("feature_score_formula.csv", "feature_score_breakdown.csv")
    ]
    if all(path.is_file() for path in deployed_core) and not all(path.is_file() for path in deployed_scores):
        run("scripts/child_crime/export_service_crime_forecasts.py", "--rescore-only")
        return
    one_year = ROOT / "models/recommender/v1/child_crime_exposure/1y/child_crime_exposure_1y.joblib"
    two_year = ROOT / "models/experiments/child_crime_exposure_2y_2027_v1/operational_exposure_2y_2027_strict.joblib"
    if not one_year.is_file() or not (one_year.parent / "raw_forecast.csv").is_file():
        run("ml/training/train_service_crime_exposure_1y.py")
    if not two_year.is_file():
        run("ml/training/optimize_exposure_2y_2027.py")
    three_year = ROOT / "models/experiments/child_crime_exposure_integrated_v2/operational_exposure_integrated_blend_3y.joblib"
    if not three_year.is_file():
        crime_result = ROOT / "artifacts/borough_crime_count_3y_v1/result.json"
        if not crime_result.is_file():
            run("ml/training/optimize_borough_crime_count_3y.py")
        rank = ROOT / "models/experiments/child_crime_exposure_rank_3y_v1/operational_exposure_rank_3y.joblib"
        direct = ROOT / "models/experiments/child_crime_exposure_integrated_v2/integrated_exposure_3y.joblib"
        if not rank.is_file():
            run("ml/training/optimize_exposure_rank_3y.py")
        if not direct.is_file():
            run("ml/training/optimize_exposure_integrated_v2.py")
        run("scripts/child_crime/build_locked_exposure_3y_blend.py")
    run("scripts/child_crime/export_service_crime_forecasts.py")


if __name__ == "__main__":
    main()
