"""Build annual administrative-dong transaction price history for service charts.

This is descriptive transaction data, not a model prediction. The annual value is
the mean of valid transaction per-pyeong prices; cells with fewer than five
transactions are omitted so sparse dong-years are not shown as a stable trend.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "부동산"
OUT = ROOT / "data" / "processed" / "property_price_history" / "annual_by_dong.csv"
MONTHLY_OUT = ROOT / "data" / "processed" / "property_price_history" / "annual_monthly_by_dong.csv"


def build(output: Path = OUT) -> dict:
    totals: dict[tuple[str, int, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
    sources = (
        ("sale", RAW / "아파트_매매_2010_2026.csv", "평당 금액(만원)", None),
        ("rent", RAW / "아파트_전월세_2010_2026.csv", "평당 보증금(만원)", "전세"),
    )
    for transaction, path, price_column, rent_type in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
        fields = ["행정동_코드", "계약연도", price_column]
        if rent_type:
            fields.append("전월세구분")
        for chunk in pd.read_csv(path, usecols=fields, chunksize=100_000, encoding="utf-8-sig", low_memory=False):
            if rent_type:
                chunk = chunk.loc[chunk["전월세구분"].astype(str).str.strip().eq(rent_type)]
            compact = pd.DataFrame({
                "dong_code": pd.to_numeric(chunk["행정동_코드"], errors="coerce"),
                "year": pd.to_numeric(chunk["계약연도"], errors="coerce"),
                "price": pd.to_numeric(chunk[price_column], errors="coerce"),
            })
            compact = compact.loc[compact.year.between(2010, 2025) & compact.price.between(100, 50_000)
                                  & compact.dong_code.between(10_000_000, 99_999_999)]
            grouped = compact.groupby(["dong_code", "year"]).price.agg(["sum", "count"])
            for (code, year), row in grouped.iterrows():
                cell = totals[(transaction, int(code), int(year))]
                cell[0] += float(row["sum"])
                cell[1] += float(row["count"])
    rows = [
        {"transaction": transaction, "dong_code": code, "year": year,
         "mean_per_pyeong_manwon": round(total / count, 4), "transaction_count": int(count)}
        for (transaction, code, year), (total, count) in sorted(totals.items()) if count >= 5
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["transaction", "dong_code", "year", "mean_per_pyeong_manwon",
                                "transaction_count"]).to_csv(output, index=False, encoding="utf-8-sig")
    monthly_totals: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
    rent_path = RAW / "아파트_전월세_2010_2026.csv"
    for chunk in pd.read_csv(rent_path, usecols=["행정동_코드", "계약연도", "전월세구분", "월세(만원)"],
                             chunksize=100_000, encoding="utf-8-sig", low_memory=False):
        chunk = chunk.loc[chunk["전월세구분"].astype(str).str.strip().eq("월세")]
        compact = pd.DataFrame({
            "dong_code": pd.to_numeric(chunk["행정동_코드"], errors="coerce"),
            "year": pd.to_numeric(chunk["계약연도"], errors="coerce"),
            "rent": pd.to_numeric(chunk["월세(만원)"], errors="coerce"),
        })
        compact = compact.loc[compact.year.between(2010, 2025) & compact.rent.between(1, 10_000)
                              & compact.dong_code.between(10_000_000, 99_999_999)]
        grouped = compact.groupby(["dong_code", "year"]).rent.agg(["sum", "count"])
        for (code, year), row in grouped.iterrows():
            cell = monthly_totals[(int(code), int(year))]
            cell[0] += float(row["sum"])
            cell[1] += float(row["count"])
    monthly_rows = [
        {"dong_code": code, "year": year, "mean_monthly_rent_manwon": round(total / count, 4),
         "transaction_count": int(count)}
        for (code, year), (total, count) in sorted(monthly_totals.items()) if count >= 5
    ]
    monthly_output = output.parent / MONTHLY_OUT.name
    pd.DataFrame(monthly_rows, columns=["dong_code", "year", "mean_monthly_rent_manwon",
                                        "transaction_count"]).to_csv(monthly_output, index=False, encoding="utf-8-sig")
    return {"output": str(output), "rows": len(rows), "monthly_output": str(monthly_output),
            "monthly_rows": len(monthly_rows)}


if __name__ == "__main__":
    print(build())
