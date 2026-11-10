"""Rebuild missing service models using the locked training pipelines."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(relative: str) -> None:
    subprocess.run([sys.executable, str(ROOT / relative)], cwd=ROOT, check=True)


def main() -> None:
    one_year = ROOT / "models/recommender/v1/child_crime_exposure/1y/child_crime_exposure_1y.joblib"
    two_year = ROOT / "models/experiments/child_crime_exposure_2y_2027_v1/operational_exposure_2y_2027_strict.joblib"
    if not one_year.is_file() or not (one_year.parent / "raw_forecast.csv").is_file():
        run("ml/training/train_service_crime_exposure_1y.py")
    if not two_year.is_file():
        run("ml/training/optimize_exposure_2y_2027.py")
    run("scripts/child_crime/export_service_crime_forecasts.py")


if __name__ == "__main__":
    main()
