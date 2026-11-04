"""Run the saved 1y/3y child-environment models on the latest observed panel."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.training.train_child_environment_forecasts import (  # noqa: E402
    add_exact_quintile_grade, engineer_features, load_panel,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "child_environment_forecast" / "v1")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    panel, _ = load_panel()
    features, _ = engineer_features(panel)
    latest = features[features.year == features.year.max()].copy()
    rows = []
    for horizon in (1, 3):
        bundle = joblib.load(args.model_dir / f"child_environment_{horizon}y.joblib")
        columns = bundle["numeric_features"] + bundle["categorical_features"]
        pred = np.clip(bundle["pipeline"].predict(latest[columns]), *bundle["score_bounds"])
        half = bundle["interval_half_width"]
        for gu, estimate in zip(latest.gu, pred):
            rows.append({
                "base_year": int(latest.year.iloc[0]), "forecast_year": int(latest.year.iloc[0]) + horizon,
                "horizon_years": horizon, "gu": gu, "predicted_score_100": estimate,
                "lower_90": np.clip(estimate - half, *bundle["score_bounds"]),
                "upper_90": np.clip(estimate + half, *bundle["score_bounds"]),
                "model": bundle["model_name"],
            })
    result = pd.DataFrame(rows)
    result = add_exact_quintile_grade(result, "predicted_score_100", "horizon_years")
    output = args.output or args.model_dir / "forecasts_latest.csv"
    result.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(result)} predictions to {output}")


if __name__ == "__main__":
    main()
