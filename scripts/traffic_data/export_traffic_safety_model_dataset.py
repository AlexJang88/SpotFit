"""교통사고 안전등급 13~24차 중 지정한 모델의 최종 데이터셋을 내보낸다."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))

from train_traffic_accident_incremental_models import add_incremental_features  # noqa: E402
from train_traffic_accident_initial_v1 import load_data  # noqa: E402
from train_traffic_safety_grade_v13_v24 import (  # noqa: E402
    TARGET_GRADE,
    TARGET_SCORE,
    add_age_vulnerability_proxy,
    add_historical_score,
    add_traffic_speed,
    internal_weights,
)


MODEL_ROOT = ROOT / "models" / "안전" / "교통사고"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, choices=range(13, 27), default=13)
    args = parser.parse_args()
    version = args.version
    group = "1~12차" if version <= 12 else "13~26차"
    metadata_path = MODEL_ROOT / group / f"{version}차" / "모델정보.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    features: list[str] = metadata["입력변수"]

    frame = add_traffic_speed(add_age_vulnerability_proxy(add_incremental_features(load_data())))
    frame["인구만명당 의료보장 사업장수"] = frame.groupby("자치구")[
        "인구만명당 의료보장 사업장수"
    ].ffill()
    frame = add_historical_score(frame, features, internal_weights(features))
    frame["데이터구분"] = ""
    frame.loc[frame["연도"].between(2017, 2023), "데이터구분"] = "학습"
    frame.loc[frame["연도"].eq(2024), "데이터구분"] = "2025년 최종검증 입력"
    frame.loc[frame["연도"].eq(2025), "데이터구분"] = "2026년 예측 입력"

    columns = [
        "연도",
        "자치구",
        "예측대상연도",
        "데이터구분",
        *features,
        "안전점수",
        "안전등급",
        TARGET_SCORE,
        TARGET_GRADE,
    ]
    output = frame[columns].copy()

    keys = ["연도", "자치구"]
    if output.duplicated(keys).any():
        raise ValueError("연도·자치구 중복 행이 있습니다.")
    counts = output.groupby("연도")["자치구"].nunique()
    if len(output) != 225 or not counts.eq(25).all():
        raise ValueError(f"연도별 25개 자치구 완전성 검증 실패: {counts.to_dict()}")
    if output[features].isna().any().any():
        raise ValueError(f"{version}차 입력변수에 결측값이 있습니다.")
    if output.loc[output["연도"].le(2024), [TARGET_SCORE, TARGET_GRADE]].isna().any().any():
        raise ValueError("학습·최종검증 정답에 결측값이 있습니다.")
    if output.loc[output["연도"].eq(2025), [TARGET_SCORE, TARGET_GRADE]].notna().any().any():
        raise ValueError("2026년 미관측 정답이 비어 있지 않습니다.")

    output_path = (
        ROOT
        / "data"
        / "processed"
        / "안전"
        / "교통사고"
        / f"{version}차"
        / f"교통사고_안전등급_{version}차_모델데이터_2017_2025.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"저장: {output_path}")
    print(f"행: {len(output)}, 열: {len(output.columns)}")
    print(f"연도별 자치구 수: {counts.to_dict()}")
    print(f"데이터 구분: {output['데이터구분'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
