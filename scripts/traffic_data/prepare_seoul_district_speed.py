"""서울 자치구별 월별 차량통행속도를 연간 모델 변수로 전처리한다."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "raw" / "안전" / "교통사고" / "서울시_차량통행속도_월별_구별_2017_2025.csv"
OUTPUT = ROOT / "data" / "interim" / "안전" / "교통사고" / "자치구별_차량통행속도_2017_2025.csv"


def main() -> None:
    try:
        source = pd.read_csv(SOURCE, encoding="utf-8-sig", header=[0, 1])
    except UnicodeDecodeError:
        source = pd.read_csv(SOURCE, encoding="cp949", header=[0, 1])
    source = source[source.iloc[:, 0].eq("자치구별")].copy()
    districts = source.iloc[:, 1].astype(str).str.strip()

    rows = []
    for column in source.columns[2:]:
        month = pd.to_datetime(str(column[0]).replace(" ", ""), format="%Y.%m")
        values = pd.to_numeric(source[column], errors="coerce")
        rows.append(pd.DataFrame({"연도": month.year, "월": month.month, "자치구": districts, "평균통행속도_kmh": values}))
    monthly = pd.concat(rows, ignore_index=True)
    if monthly.isna().any().any():
        raise ValueError("월별 통행속도에 결측값이 있습니다.")

    annual = monthly.groupby(["연도", "자치구"], as_index=False).agg(
        차량통행속도_연평균_kmh=("평균통행속도_kmh", "mean"),
        차량통행속도_최저월_kmh=("평균통행속도_kmh", "min"),
        차량통행속도_최고월_kmh=("평균통행속도_kmh", "max"),
        차량통행속도_월변동성=("평균통행속도_kmh", "std"),
    )
    counts = annual.groupby("연도")["자치구"].nunique()
    if len(annual) != 225 or not counts.eq(25).all():
        raise ValueError(f"연도별 25개 자치구 완전성 검증 실패: {counts.to_dict()}")
    if annual.duplicated(["연도", "자치구"]).any():
        raise ValueError("연도·자치구 중복 행이 있습니다.")

    numeric = annual.columns.difference(["연도", "자치구"])
    annual[numeric] = annual[numeric].round(4)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    annual.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"저장: {OUTPUT}")
    print(f"행: {len(annual)}, 연도별 자치구 수: {counts.to_dict()}")


if __name__ == "__main__":
    main()
