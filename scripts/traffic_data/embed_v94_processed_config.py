"""Embed the final v94 model configuration in its processed workbook.

This is a one-time packaging utility.  The resulting workbook contains every
data/configuration table needed by ``train_traffic_accident_v94_from_processed``.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.traffic_accident_pipeline_paths import PROCESSED_V94

WORKBOOK = PROCESSED_V94
MODEL_DIR = ROOT / "models/교통사고/94차.11.13.개선"


def spec_for(bundle: dict, horizon: int, feature: str):
    forecasts = bundle["forecast_models"][2] if horizon == 2 else bundle["forecast_model"]
    return forecasts[feature]["spec"]


def main() -> None:
    config_rows = []
    metadata_rows = [
        {"항목": "모델명", "값": "교통사고 94차 TOPIS 교통량 추가모델"},
        {"항목": "입력시트", "값": "94차전체입력패널"},
        {"항목": "설정시트", "값": "94차모델설정"},
        {"항목": "예측기간", "값": "2년 후, 3년 후"},
        {"항목": "안전등급방향", "값": "1=가장 안전, 5=가장 위험"},
        {"항목": "Raw/Interim 필요", "값": "없음"},
    ]
    for horizon in (2, 3):
        path = MODEL_DIR / f"교통사고_94차_{horizon}년후_교통량추가모델.joblib"
        bundle = joblib.load(path)
        for order, feature in enumerate(bundle["features"]):
            config_rows.append(
                {
                    "예측기간": f"{horizon}년 후",
                    "순서": order,
                    "지표": feature,
                    "Poisson계수": float(bundle["coefficients"][feature]),
                    "가중치": float(bundle["weights"][feature]),
                    "미래값예측설정_JSON": json.dumps(
                        spec_for(bundle, horizon, feature), ensure_ascii=False
                    ),
                }
            )

    existing = {
        sheet: pd.read_excel(WORKBOOK, sheet_name=sheet)
        for sheet in pd.ExcelFile(WORKBOOK).sheet_names
        if sheet not in {"94차모델설정", "단독재현안내"}
    }
    with pd.ExcelWriter(WORKBOOK, engine="openpyxl") as writer:
        for sheet, frame in existing.items():
            frame.to_excel(writer, sheet_name=sheet, index=False)
        pd.DataFrame(config_rows).to_excel(writer, sheet_name="94차모델설정", index=False)
        pd.DataFrame(metadata_rows).to_excel(writer, sheet_name="단독재현안내", index=False)
    print(f"packaged={WORKBOOK.relative_to(ROOT)} config_rows={len(config_rows)}")


if __name__ == "__main__":
    main()
