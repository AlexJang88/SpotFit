"""Prepare annual borough features from Seoul's quarterly elderly statistics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "raw" / "아동범죄" / "external_features"
RAW = BASE / "서울시_고령자현황_분기별_원본.csv"
OUT = BASE / "서울시_자치구_고령인구_피처_2010_2025.csv"
REGISTERED = ROOT / "data" / "raw" / "아동범죄" / "feature_sources" / "서울시_자치구_연도별_등록인구.csv"


def main() -> None:
    frame = pd.read_csv(RAW, encoding="cp949")
    expected = ["통계분기", "구명", "동명", "외국인구분", "성별명", "인구수", "행정동코드"]
    if frame.columns.tolist() != expected:
        raise ValueError(f"Unexpected columns: {frame.columns.tolist()}")

    frame["연도"] = frame["통계분기"].str[:4].astype(int)
    frame["분기"] = frame["통계분기"].str[-1:].astype(int)
    frame["인구수"] = pd.to_numeric(frame["인구수"], errors="coerce")
    district = frame[
        frame["구명"].astype(str).str.endswith("구")
        & frame["동명"].eq("계")
        & frame["외국인구분"].eq("계")
        & frame["성별명"].eq("계")
        & frame["분기"].eq(4)
        & frame["연도"].between(2010, 2025)
    ].copy()

    pivot = district[["연도", "구명", "인구수"]].rename(
        columns={"구명": "자치구", "인구수": "고령인구_내외국인"}
    )

    population = pd.read_csv(REGISTERED, encoding="cp949")
    population = population.rename(columns={"통계연도": "연도", "구명": "자치구"})
    population = population[population["자치구"].astype(str).str.endswith("구")][
        ["연도", "자치구", "인구수"]
    ].copy()
    population["인구수"] = pd.to_numeric(population["인구수"], errors="coerce")
    pivot = pivot.merge(population, on=["연도", "자치구"], how="left", validate="one_to_one")
    pivot["고령인구비율_등록인구대비_근사"] = pivot["고령인구_내외국인"] / pivot["인구수"]
    pivot = pivot.sort_values(["자치구", "연도"]).reset_index(drop=True)
    for column in ("고령인구_내외국인", "고령인구비율_등록인구대비_근사"):
        pivot[f"{column}_전년변화율"] = pivot.groupby("자치구")[column].pct_change(fill_method=None)

    counts = pivot.groupby("연도")["자치구"].nunique()
    core = ["고령인구_내외국인", "고령인구비율_등록인구대비_근사"]
    if len(counts) != 16 or not counts.eq(25).all() or len(pivot) != 400:
        raise ValueError(f"Incomplete district-year panel: {counts.to_dict()}")
    if pivot[core].isna().any().any() or pivot.duplicated(["연도", "자치구"]).any():
        raise ValueError(f"Invalid core panel: {pivot[core].isna().sum().to_dict()}")

    pivot.to_csv(OUT, index=False, encoding="utf-8-sig")
    print("rows=400 years=2010-2025 districts=25 core_missing=0")
    print(f"output={OUT}")


if __name__ == "__main__":
    main()
