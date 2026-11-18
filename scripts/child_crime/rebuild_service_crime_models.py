"""Restore selected crime models from crime-only raw and processed datasets."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(relative: str, *args: str) -> None:
    subprocess.run([sys.executable, str(ROOT / relative), *args], cwd=ROOT, check=True)


def main() -> None:
    service = ROOT / "models/recommender/v1/child_crime_exposure"
    if (service / 'current_release.json').is_file():
        sys.path.insert(0, str(ROOT / 'backend'))
        from app.services.service_model_bootstrap import missing
        if not missing('child_crime', ROOT):
            return
        from scripts.child_crime.run_refresh import process, write_json, REPORT
        result = process(force=True)
        write_json(REPORT, {'process': result})
        if result['state'] != 'promoted':
            raise RuntimeError('Crime release restoration did not pass validation; see refresh report')
        return
    data = ROOT / "data/processed/child_crime_exposure"
    if not all((data / path).is_file() for path in (
            "borough_panel/panel.csv", "population_features/priority_features.csv",
            "inference/latest_features.csv", "training/2y.csv", "training/3y.csv")):
        run("scripts/child_crime/prepare_service_data.py")
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
    sys.path.insert(0, str(ROOT))
    from scripts.child_crime.run_refresh import exclusive_run
    with exclusive_run():
        main()
