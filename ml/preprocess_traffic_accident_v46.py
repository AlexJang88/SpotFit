from __future__ import annotations

import json

import joblib
import pandas as pd

from ml.train_best_child_traffic_accident_rate import ROOT, TARGET
from ml.train_traffic_accident_v44 import prepare_panel


VERSION = 46
MODEL_DIR = ROOT / "models" / "교통사고" / f"{VERSION}차"
OUT_DIR = ROOT / "data" / "processed" / "안전" / "교통사고" / f"{VERSION}차"
OUT_FILE = OUT_DIR / f"교통사고_{VERSION}차_최종모델데이터.xlsx"


def split_name(row: pd.Series) -> str:
    if pd.isna(row[TARGET]):
        return "2026예측입력"
    if row["예측연도"] <= 2024:
        return "모델개발_순차검증"
    if row["예측연도"] == 2025:
        return "2025독립검증"
    return "기타"


def main() -> None:
    info = json.loads((MODEL_DIR / "model_info.json").read_text(encoding="utf-8"))
    features = info["최종변수"]
    bundle = joblib.load(MODEL_DIR / f"교통사고_{VERSION}차_예측모델.joblib")
    imputer = bundle["model"].named_steps["imputer"]

    panel = prepare_panel()
    required = ["연도", "예측연도", "자치구", TARGET, *features]
    missing = [column for column in required if column not in panel.columns]
    if missing:
        raise ValueError(f"46차 전처리 필수 컬럼이 없습니다: {missing}")

    processed = panel[required].copy()
    processed.insert(0, "서울시", "서울특별시")
    processed.insert(4, "데이터구분", processed.apply(split_name, axis=1))

    # 저장된 46차 최종모델이 학습할 때 산출한 중앙값과 정확히 동일하게 결측치를 대체한다.
    processed[features] = imputer.transform(processed[features])

    if processed.duplicated(["연도", "자치구"]).any():
        raise ValueError("연도·자치구 중복 행이 있습니다.")
    if len(processed) != 225:
        raise ValueError(f"예상 행 수 225개와 다릅니다: {len(processed)}")
    if processed[features].isna().any().any():
        columns = processed[features].columns[processed[features].isna().any()].tolist()
        raise ValueError(f"결측 대체 후에도 결측이 남았습니다: {columns}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    processed.to_excel(OUT_FILE, sheet_name="46차모델데이터", index=False, engine="openpyxl")
    print(f"파일: {OUT_FILE}")
    print(f"행: {len(processed)}, 모델변수: {len(features)}, 전체열: {len(processed.columns)}")
    print(processed["데이터구분"].value_counts().to_string())


if __name__ == "__main__":
    main()
