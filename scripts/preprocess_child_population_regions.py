"""아동인구현황의 행정구역을 자치구·행정동으로 분리하고 청년 컬럼을 제거한다."""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "interim" / "아동인구현황_2021_2025.csv"
OUTPUT_PATH = ROOT / "data" / "interim" / "아동인구현황_2021_2025_지역분리.csv"
DROP_COLUMNS = {"청년전체", "청년남자", "청년여자"}
CODE_PATTERN = re.compile(r"\s*\(\d+\)\s*$")


def split_region(value: str) -> tuple[str, str]:
    cleaned = CODE_PATTERN.sub("", value).strip()
    parts = cleaned.split()
    if len(parts) < 2:
        return "", ""
    if len(parts) == 2:
        return parts[1], ""
    return " ".join(parts[1:-1]), parts[-1]


def is_seoul(value: str) -> bool:
    cleaned = CODE_PATTERN.sub("", value).strip()
    return cleaned == "서울특별시" or cleaned.startswith("서울특별시 ")


def main() -> None:
    with INPUT_PATH.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        input_columns = reader.fieldnames or []
        value_columns = [
            column
            for column in input_columns
            if column not in DROP_COLUMNS and column != "행정구역"
        ]
        output_columns = ["연도", "자치구", "행정동"] + [
            column for column in value_columns if column != "연도"
        ]

        with OUTPUT_PATH.open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=output_columns)
            writer.writeheader()
            count = 0
            for row in reader:
                if not is_seoul(row["행정구역"]):
                    continue
                district, dong = split_region(row["행정구역"])
                processed = {
                    "연도": row["연도"],
                    "자치구": district,
                    "행정동": dong,
                }
                processed.update(
                    {
                        column: row[column]
                        for column in value_columns
                        if column != "연도"
                    }
                )
                writer.writerow(processed)
                count += 1

    print(f"전처리 완료: {count:,}행, {len(output_columns)}열")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
