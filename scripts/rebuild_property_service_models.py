"""Rebuild every apartment artifact used by the service and presentation models."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(relative: str, *args: str) -> None:
    subprocess.run([sys.executable, str(ROOT / relative), *args], cwd=ROOT, check=True)


def main() -> None:
    # Current-price service bundles: the recommendation API consumes these.
    sale = ROOT / "models/recommender/v1/apart_sale/model_bundle.joblib"
    rent = ROOT / "models/recommender/v1/apart_rent/model_bundle.joblib"
    if not sale.is_file() or not rent.is_file():
        run("ml/etl/property_amount/train_and_save_final_models.py")

    # 1/2/3-year sale, jeonse and contract-level wolse delivery artifacts.
    sys.path.insert(0, str(ROOT / "backend"))
    from app.services.property_model_bootstrap import ensure_models
    outcome = ensure_models(ROOT)
    failed = {name: value for name, value in outcome.items() if value.startswith("failed")}
    if failed:
        raise RuntimeError(f"부동산 미래/월세 모델 복구 실패: {failed}")

    snapshot = ROOT / "models/recommender/v1/property_service_forecasts.json"
    if not snapshot.is_file():
        run("scripts/export_property_service_catalog.py")
    print({"property": outcome, "snapshot": str(snapshot)})


if __name__ == "__main__":
    main()
