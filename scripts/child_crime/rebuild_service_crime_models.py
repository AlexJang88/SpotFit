"""Restore selected crime models from crime-only raw and processed datasets."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(relative: str, *args: str) -> None:
    subprocess.run([sys.executable, str(ROOT / relative), *args], cwd=ROOT, check=True)


def main() -> None:
    data = ROOT / "data/processed/child_crime_exposure"
    if not all((data / path).is_file() for path in (
            "borough_panel/panel.csv", "population_features/priority_features.csv",
            "inference/latest_features.csv", "training/2y.csv", "training/3y.csv")):
        run("scripts/child_crime/prepare_service_data.py")
    service = ROOT / "models/recommender/v1/child_crime_exposure"
    core = [service / f"{h}y" / name for h in (2, 3) for name in
        (f"child_crime_exposure_{h}y.joblib", "service_forecast.csv", "model_metadata.json")]
    scores = [service / f"{h}y" / name for h in (2, 3) for name in
        ("feature_score_formula.csv", "feature_score_breakdown.csv")]
    if all(path.is_file() and path.stat().st_size for path in core):
        if not all(path.is_file() and path.stat().st_size for path in scores):
            run("scripts/child_crime/export_service_crime_forecasts.py", "--rescore-only")
        return
    run("scripts/child_crime/train_locked_service_models.py")


if __name__ == "__main__":
    main()
