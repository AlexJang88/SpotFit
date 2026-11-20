from __future__ import annotations

import csv
import html
import re
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "data"
    / "raw"
    / "5년치_시설"
    / "관광지점입장객_대체지표"
    / "2021_2025_서울_주요관광지점_입장객통계.xml"
)
OUTPUT_DIR = ROOT / "data" / "processed" / "historical_facilities" / "official_2021_2025" / "tourism_visitors"
YEARS = range(2021, 2026)


def parse_spot(raw: str) -> tuple[str, str]:
    code_match = re.search(r"detail\('([^']+)'\)", raw)
    code = code_match.group(1) if code_match else ""
    name = re.sub(r"<[^>]+>", "", raw)
    return code, html.unescape(name).strip()


def parse_count(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip().replace(",", "")
    if not value or value == "-":
        return None
    return int(value)


def main() -> None:
    root = ET.parse(SOURCE).getroot()
    long_rows: list[dict[str, object]] = []
    for row in root.findall("row"):
        spot_code, spot_name = parse_spot(row.findtext("RES_NM") or "")
        visitor_type = (row.findtext("NF_GB") or "").strip()
        for year in YEARS:
            count = parse_count(row.findtext(f"Y_{year}"))
            long_rows.append(
                {
                    "year": year,
                    "sido": (row.findtext("SIDO_NM") or "").strip(),
                    "sigungu": (row.findtext("GUNGU_NM") or "").strip(),
                    "spot_code": spot_code,
                    "spot_name": spot_name,
                    "visitor_type": visitor_type,
                    "visitors": "" if count is None else count,
                    "quality_flag": "official_major_tourist_spot_visitors; coverage_varies_by_year",
                }
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["year", "sido", "sigungu", "spot_code", "spot_name", "visitor_type", "visitors", "quality_flag"]
    with (OUTPUT_DIR / "seoul_2021_2025_long.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(long_rows)

    summary: list[dict[str, object]] = []
    for year in YEARS:
        totals = [
            row
            for row in long_rows
            if row["year"] == year and row["visitor_type"] == "합계" and row["visitors"] != ""
        ]
        summary.append(
            {
                "year": year,
                "reporting_spots": len(totals),
                "summed_visitors": sum(int(row["visitors"]) for row in totals),
                "quality_flag": "do_not_interpret_as_all_seoul_visitors; reporting_coverage_varies",
            }
        )
    with (OUTPUT_DIR / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["year", "reporting_spots", "summed_visitors", "quality_flag"])
        writer.writeheader()
        writer.writerows(summary)
    print(f"정규화 완료: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
