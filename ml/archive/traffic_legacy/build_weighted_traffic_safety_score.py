"""Build the weighted traffic-safety score described by 안전기준표.png.

The score is deterministic, not a fitted predictor. Higher means safer.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SAFE = ROOT / "data" / "processed" / "안전"
OUT = ROOT / "artifacts" / "traffic_accident_weighted_score"
OUT.mkdir(parents=True, exist_ok=True)
KEYS = ["연도", "자치구"]

WEIGHTS = {
    "인구1만명당_교통사고사망자수": 50.00,
    "운전자안전벨트착용률": 8.99,
    "음주운전율": 8.70,
    "인구1만명당_자동차등록대수_대리지표": 2.31,
    "인구1만명당_교통단속CCTV누적_대리지표": 17.24,
}
COMMON_WEIGHT = 70.00
EXTENDED_WEIGHT = 87.24


def safety_rank(values: pd.Series, higher_is_safer: bool) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna()
    if valid.empty:
        return result
    if valid.nunique() == 1:
        result.loc[valid.index] = 50.0
        return result
    rank = valid.rank(method="average", ascending=True)
    percentile = (rank - 1) / (len(valid) - 1) * 100
    result.loc[valid.index] = percentile if higher_is_safer else 100 - percentile
    return result


def main() -> None:
    accidents = pd.read_excel(SAFE / "자치구별_교통사고_2차.xlsx", sheet_name="자치구별")
    drivers = pd.read_csv(SAFE / "자치구별_운전자안전_2차.csv", encoding="utf-8-sig")
    population = pd.read_csv(SAFE / "자치구별_인구활동량_1차.csv", encoding="utf-8-sig")
    data = accidents.merge(drivers, on=KEYS, how="inner", validate="one_to_one")
    data = data.merge(population, on=KEYS, how="inner", validate="one_to_one")
    data = data[data["연도"].between(2017, 2025)].copy()

    pop = data["생활인구_연평균"].replace(0, np.nan)
    data["인구1만명당_교통사고사망자수"] = data["사망자수"] / pop * 10_000
    data["인구1만명당_자동차등록대수_대리지표"] = data["자동차등록대수"] / pop * 10_000
    data["인구1만명당_교통단속CCTV누적_대리지표"] = data["교통단속CCTV누적"] / pop * 10_000

    direction = {
        "인구1만명당_교통사고사망자수": False,
        "운전자안전벨트착용률": True,
        "음주운전율": False,
        "인구1만명당_자동차등록대수_대리지표": False,
        "인구1만명당_교통단속CCTV누적_대리지표": True,
    }
    score_columns = []
    contribution_columns = []
    for indicator, weight in WEIGHTS.items():
        score_col = f"표준화점수_{indicator}"
        contribution_col = f"가중기여_{indicator}"
        data[score_col] = data.groupby("연도", group_keys=False)[indicator].apply(
            lambda s: safety_rank(s, direction[indicator])
        )
        data[contribution_col] = data[score_col] * weight / 100
        score_columns.append(score_col)
        contribution_columns.append(contribution_col)

    common_indicators = list(WEIGHTS)[:4]
    common_contributions = [f"가중기여_{x}" for x in common_indicators]
    cctv_contribution = "가중기여_인구1만명당_교통단속CCTV누적_대리지표"
    data["공통4지표_원가중치점수_70점"] = data[common_contributions].sum(axis=1, min_count=4)
    data["공통4지표_환산안전점수_100점"] = data["공통4지표_원가중치점수_70점"] / COMMON_WEIGHT * 100
    data["공통4지표_연도내순위"] = data.groupby("연도")["공통4지표_환산안전점수_100점"].rank(
        ascending=False, method="min").astype("Int64")

    has_cctv = data[cctv_contribution].notna()
    data["CCTV포함5지표_원가중치점수_87.24점"] = np.nan
    data.loc[has_cctv, "CCTV포함5지표_원가중치점수_87.24점"] = (
        data.loc[has_cctv, "공통4지표_원가중치점수_70점"] + data.loc[has_cctv, cctv_contribution]
    )
    data["CCTV포함5지표_환산안전점수_100점"] = (
        data["CCTV포함5지표_원가중치점수_87.24점"] / EXTENDED_WEIGHT * 100
    )
    data["CCTV포함5지표_연도내순위_14개구내"] = data.groupby("연도")[
        "CCTV포함5지표_환산안전점수_100점"
    ].rank(ascending=False, method="min").astype("Int64")

    keep = KEYS + list(WEIGHTS) + score_columns + contribution_columns + [
        "공통4지표_원가중치점수_70점", "공통4지표_환산안전점수_100점", "공통4지표_연도내순위",
        "CCTV포함5지표_원가중치점수_87.24점", "CCTV포함5지표_환산안전점수_100점",
        "CCTV포함5지표_연도내순위_14개구내",
    ]
    result = data[keep].sort_values(KEYS).reset_index(drop=True)
    result.to_csv(SAFE / "자치구별_교통사고안전_가중점수_2017_2025.csv", index=False, encoding="utf-8-sig")
    result.to_csv(OUT / "weighted_scores_2017_2025.csv", index=False, encoding="utf-8-sig")
    result[result["연도"] == 2025].sort_values("공통4지표_연도내순위").to_csv(
        OUT / "ranking_2025.csv", index=False, encoding="utf-8-sig")

    definitions = pd.DataFrame([
        ["인구1만명당 교통사고 사망자수",50.00,"낮을수록 안전","사망자수/생활인구×10000","생활인구 대리"],
        ["운전자 안전벨트 착용률",8.99,"높을수록 안전","원자료","정확 대응"],
        ["음주운전 경험률",8.70,"낮을수록 안전","음주운전율 원자료","직접 대응"],
        ["자동차등록대수당 교통법규위반건수",2.31,"낮을수록 안전","자동차등록대수/생활인구×10000","법규위반건수 부재로 차량밀도 대리"],
        ["도로면적당 교통단속 CCTV대수",17.24,"높을수록 안전","교통단속CCTV/생활인구×10000","도로면적 부재로 인구 분모 대리; 14개 구만"],
        ["인구만명당 재난약자수",7.91,"-","미반영","자료 없음"],
        ["인구만명당 의료보장 사업장수",2.09,"-","미반영","자료 없음"],
        ["지역교통 안전환경개선사업 지출액",2.76,"-","미반영","자료 없음"],
    ], columns=["기준표항목","원가중치","안전방향","적용산식","상태"])
    definitions.to_csv(OUT / "weight_definition.csv", index=False, encoding="utf-8-sig")

    config = {
        "score_direction": "higher_is_safer", "normalization": "within-year percentile rank 0-100",
        "common_score": {"indicators": 4, "original_weight_sum": COMMON_WEIGHT, "districts": 25},
        "extended_score": {"indicators": 5, "original_weight_sum": EXTENDED_WEIGHT, "districts": 14},
        "missing_weight": 12.76,
        "warnings": [
            "생활인구는 공식 주민등록인구가 아님",
            "자동차 지표는 법규위반건수 대신 차량밀도 대리",
            "CCTV 지표는 도로면적 대신 생활인구 분모 대리",
            "서로 다른 연도의 점수는 해당 연도 내 상대점수이므로 절대수준 비교에 부적합",
        ],
    }
    (OUT / "score_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result.groupby("연도").agg(rows=("자치구","size"), cctv_rows=("CCTV포함5지표_환산안전점수_100점","count")).to_string())


if __name__ == "__main__":
    main()
