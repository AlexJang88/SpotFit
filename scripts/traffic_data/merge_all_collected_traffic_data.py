"""Merge every newly collected traffic-model dataset into one annual CSV."""
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "processed" / "안전" / "교통사고_모델보강"
OUTPUT = ROOT / "data" / "processed" / "안전" / "교통사고_모델보강_전체통합.csv"


def main() -> None:
    annual = pd.read_csv(
        SOURCE / "자치구별_연간_모델보강특성_2010_2025.csv",
        encoding="utf-8-sig",
    )
    intersections = pd.read_csv(
        SOURCE / "자치구별_교차로횡단보도_2026현황.csv",
        encoding="utf-8-sig",
    )

    # Keep the current infrastructure snapshot in its actual reference year.
    static_2026 = intersections.copy()
    static_2026.insert(0, "year", 2026)
    static_2026["시설현황_기준연도"] = 2026

    all_columns = list(annual.columns)
    for column in static_2026.columns:
        if column not in all_columns:
            all_columns.append(column)

    annual = annual.reindex(columns=all_columns)
    static_2026 = static_2026.reindex(columns=all_columns)
    combined = pd.concat([annual, static_2026], ignore_index=True)
    combined = combined.sort_values(["year", "district"]).reset_index(drop=True)

    if combined.duplicated(["year", "district"]).any():
        raise ValueError("Duplicate year-district keys found")

    combined.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"saved={OUTPUT}")
    print(f"rows={len(combined)} columns={len(combined.columns)}")
    print(f"years={combined.year.min()}-{combined.year.max()}")
    print(f"districts={combined.district.nunique()}")


if __name__ == "__main__":
    main()
