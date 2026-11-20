"""Prepare externally downloaded worker-status features for the exposure model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "raw" / "아동범죄" / "external_features"
RAW = BASE / "서울시_종사상지위별_종사자수_원본.csv"
OUT = BASE / "서울시_자치구_종사상지위_피처_2011_2024.csv"
MODEL_PANEL = ROOT / "data" / "raw" / "아동범죄" / "processed" / "아동지역범죄노출_모델패널_2014_2025.csv"
MERGED_OUT = BASE / "아동지역범죄노출_외부종사자피처결합_2014_2024.csv"


def main() -> None:
    frame = pd.read_csv(RAW, encoding="cp949")
    expected = ["통계연도", "구명", "동명", "종사상지위명", "성별명", "인구수", "행정동코드"]
    if frame.columns.tolist() != expected:
        raise ValueError(f"Unexpected columns: {frame.columns.tolist()}")

    frame["통계연도"] = pd.to_numeric(frame["통계연도"], errors="raise").astype(int)
    frame["인구수"] = pd.to_numeric(frame["인구수"], errors="coerce")
    district = frame[
        frame["구명"].astype(str).str.endswith("구")
        & frame["동명"].eq("계")
        & frame["성별명"].eq("계")
    ].copy()

    pivot = district.pivot_table(
        index=["통계연도", "구명"],
        columns="종사상지위명",
        values="인구수",
        aggfunc="first",
    ).reset_index()
    pivot = pivot.rename(
        columns={
            "통계연도": "연도",
            "구명": "자치구",
            "계": "총종사자수",
            "상용근로자": "상용근로자수",
            "임시 및 일용근로자": "임시_일용근로자수",
            "자영업주 및 무급가족": "자영업주_무급가족수",
            "기타 종사자": "기타종사자수",
        }
    )

    required = [
        "총종사자수",
        "상용근로자수",
        "임시_일용근로자수",
        "자영업주_무급가족수",
        "기타종사자수",
    ]
    missing_columns = [column for column in required if column not in pivot]
    if missing_columns:
        raise ValueError(f"Missing worker categories: {missing_columns}")

    for column in required[1:]:
        pivot[f"{column}_비율"] = pivot[column] / pivot["총종사자수"]

    pivot = pivot.sort_values(["자치구", "연도"]).reset_index(drop=True)
    for column in required:
        pivot[f"{column}_전년변화율"] = pivot.groupby("자치구")[column].pct_change(fill_method=None)

    counts = pivot.groupby("연도")["자치구"].nunique()
    if pivot["연도"].min() != 2011 or pivot["연도"].max() != 2024:
        raise ValueError(f"Unexpected year range: {pivot['연도'].min()}-{pivot['연도'].max()}")
    if len(counts) != 14 or not counts.eq(25).all():
        raise ValueError(f"Incomplete district-year panel: {counts.to_dict()}")
    if pivot.duplicated(["연도", "자치구"]).any():
        raise ValueError("Duplicate district-year keys")
    if pivot[required].isna().any().any():
        raise ValueError(f"Missing core values: {pivot[required].isna().sum().to_dict()}")

    pivot.to_csv(OUT, index=False, encoding="utf-8-sig")
    model = pd.read_csv(MODEL_PANEL, encoding="utf-8-sig")
    merged = model.merge(pivot, on=["연도", "자치구"], how="inner", validate="one_to_one")
    merged_counts = merged.groupby("연도")["자치구"].nunique()
    if merged["연도"].min() != 2014 or merged["연도"].max() != 2024:
        raise ValueError("Unexpected merged year range")
    if len(merged_counts) != 11 or not merged_counts.eq(25).all() or len(merged) != 275:
        raise ValueError(f"Incomplete merged panel: {merged_counts.to_dict()}")
    merged.to_csv(MERGED_OUT, index=False, encoding="utf-8-sig")
    print(f"rows={len(pivot)} years={pivot['연도'].min()}-{pivot['연도'].max()} districts=25")
    print(f"output={OUT}")
    print(f"merged_rows={len(merged)} merged_years=2014-2024 merged_output={MERGED_OUT}")


if __name__ == "__main__":
    main()
