"""세대당 인구수 통합본에서 서울만 선별하고 지역·컬럼 구조를 정리한다."""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "interim" / "세대당인구수_2021_2025.csv"
OUTPUT_PATH = ROOT / "data" / "interim" / "세대당인구수_2021_2025_서울_지역분리.csv"
CODE_PATTERN = re.compile(r"\s*\(\d+\)\s*$")
OUTPUT_COLUMNS = [
    "연도",
    "자치구",
    "행정동",
    "세대수",
    "세대당 인구",
    "총인구수",
    "남자 인구수",
    "여자 인구수",
    "남여 비율",
]


def clean_region(value: str) -> str:
    return CODE_PATTERN.sub("", value).strip()


def is_seoul(value: str) -> bool:
    cleaned = clean_region(value)
    return cleaned == "서울특별시" or cleaned.startswith("서울특별시 ")


def split_region(value: str) -> tuple[str, str]:
    parts = clean_region(value).split()
    if len(parts) < 2:
        return "", ""
    if len(parts) == 2:
        return parts[1], ""
    return parts[1], " ".join(parts[2:])


def main() -> None:
    with INPUT_PATH.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        with OUTPUT_PATH.open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            count = 0
            for row in reader:
                if not is_seoul(row["행정구역"]):
                    continue
                district, dong = split_region(row["행정구역"])
                writer.writerow(
                    {
                        "연도": row["연도"],
                        "자치구": district,
                        "행정동": dong,
                        "세대수": row["세대수"].strip(),
                        "세대당 인구": row["세대당 인구"].strip(),
                        "총인구수": row["총인구수"].strip(),
                        "남자 인구수": row["남자 인구수"].strip(),
                        "여자 인구수": row["여자 인구수"].strip(),
                        "남여 비율": row["남여 비율"].strip(),
                    }
                )
                count += 1

    print(f"전처리 완료: {count:,}행, {len(OUTPUT_COLUMNS)}열")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
