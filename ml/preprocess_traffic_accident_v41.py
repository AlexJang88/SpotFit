from __future__ import annotations

import pandas as pd

from ml.train_best_child_traffic_accident_rate import (
    CURRENT_RATE,
    ROOT,
    TARGET,
    build_panel,
)
from ml.train_traffic_accident_v41 import FEATURES


OUT = ROOT / "data" / "processed" / "안전" / "교통사고" / "41차"
def split_name(row: pd.Series) -> str:
    if pd.isna(row[TARGET]):
        return "2026예측입력"
    if row["예측연도"] <= 2023:
        return "학습"
    if row["예측연도"] == 2024:
        return "변수선택검증"
    if row["예측연도"] == 2025:
        return "독립검증"
    return "기타"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel, _, _ = build_panel()
    required = ["연도", "예측연도", "자치구", TARGET, *FEATURES]
    missing = [column for column in required if column not in panel.columns]
    if missing:
        raise ValueError(f"41차 전처리 필수 컬럼이 없습니다: {missing}")

    processed = panel[required].copy()
    processed.insert(0, "서울시", "서울특별시")
    processed["데이터구분"] = processed.apply(split_name, axis=1)

    # 모델 학습 시 사용하는 방식과 동일하게 각 feature의 학습기간 중앙값을 기록한다.
    training = processed[processed["데이터구분"] == "학습"]
    medians = training[FEATURES].median(numeric_only=True)
    model_ready = processed.copy()
    model_ready[FEATURES] = model_ready[FEATURES].fillna(medians)

    if processed.duplicated(["연도", "자치구"]).any():
        raise ValueError("연도·자치구 중복 행이 있습니다.")
    if len(processed) != 225:
        raise ValueError(f"예상 행 수 225개와 다릅니다: {len(processed)}")
    if model_ready[FEATURES].isna().any().any():
        columns = model_ready[FEATURES].columns[model_ready[FEATURES].isna().any()].tolist()
        raise ValueError(f"결측 대체 후에도 결측이 남았습니다: {columns}")

    output = OUT / "교통사고_41차_최종모델데이터.xlsx"
    model_ready.to_excel(output, sheet_name="41차모델데이터", index=False, engine="openpyxl")
    print(output)


if __name__ == "__main__":
    main()
