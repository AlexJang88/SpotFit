"""Merge all verified external long-panel features into the model panel."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "raw" / "아동범죄"
MODEL = BASE / "processed" / "아동지역범죄노출_모델패널_2014_2025.csv"
WORKER = BASE / "external_features" / "서울시_자치구_종사상지위_피처_2011_2024.csv"
ELDERLY = BASE / "external_features" / "서울시_자치구_고령인구_피처_2010_2025.csv"
OUT = BASE / "external_features" / "아동지역범죄노출_외부장기피처결합_2014_2024.csv"
PREPARED = BASE / "external_features" / "prepared"
EXPANDED_OUT = BASE / "external_features" / "아동지역범죄노출_외부확장피처결합_2015_2024.csv"


def main() -> None:
    model = pd.read_csv(MODEL, encoding="utf-8-sig")
    worker = pd.read_csv(WORKER, encoding="utf-8-sig")
    elderly = pd.read_csv(ELDERLY, encoding="utf-8-sig")
    merged = model.merge(worker, on=["연도", "자치구"], validate="one_to_one")
    merged = merged.merge(elderly.drop(columns=["인구수"], errors="ignore"), on=["연도", "자치구"], validate="one_to_one")
    counts = merged.groupby("연도")["자치구"].nunique()
    if len(merged) != 275 or len(counts) != 11 or not counts.eq(25).all():
        raise ValueError(f"Incomplete final external panel: {counts.to_dict()}")
    merged.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"rows={len(merged)} years=2014-2024 districts=25 columns={len(merged.columns)}")
    print(f"output={OUT}")

    additions = [
        PREPARED / "서울시_자치구_가구원수별_피처.csv",
        PREPARED / "서울시_자치구_1인가구_연령피처.csv",
        PREPARED / "서울시_자치구_이혼_피처.csv",
        PREPARED / "서울시_자치구_가구증가율_피처.csv",
    ]
    expanded = merged
    for path in additions:
        feature = pd.read_csv(path, encoding="utf-8-sig")
        overlapping = [column for column in feature.columns if column in expanded.columns and column not in {"연도", "자치구"}]
        feature = feature.drop(columns=overlapping)
        expanded = expanded.merge(feature, on=["연도", "자치구"], how="inner", validate="one_to_one")
    expanded_counts = expanded.groupby("연도")["자치구"].nunique()
    if len(expanded) != 250 or len(expanded_counts) != 10 or not expanded_counts.eq(25).all():
        raise ValueError(f"Incomplete expanded panel: {expanded_counts.to_dict()}")
    expanded.to_csv(EXPANDED_OUT, index=False, encoding="utf-8-sig")
    print(f"expanded_rows={len(expanded)} years=2015-2024 districts=25 columns={len(expanded.columns)}")
    print(f"expanded_output={EXPANDED_OUT}")


if __name__ == "__main__":
    main()
