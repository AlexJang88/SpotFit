"""아동인구현황과 세대당 인구수 데이터를 연도·자치구·행정동으로 결합한다."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_PATH = ROOT / "data" / "interim" / "행정동별_아동인구_세대수_통합.csv"
CHILD_PATH = PROCESSED_DIR / "행정동별_아동인구현황_최종.csv"
HOUSEHOLD_PATH = PROCESSED_DIR / "행정동별_세대당 인구수_최종.csv"
KEY_COLUMNS = ["연도", "자치구", "행정동"]
OUTPUT_COLUMNS = [
    "연도",
    "자치구",
    "행정동",
    "전체",
    "남자",
    "여자",
    "세대수",
    "세대당 인구",
    "남여 비율",
    "아동전체",
    "아동남자",
    "아동여자",
    "청소년전체",
    "청소년남자",
    "청소년여자",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def key(row: dict[str, str]) -> tuple[str, str, str]:
    return tuple(row[column].strip() for column in KEY_COLUMNS)


def number(value: str) -> str:
    return value.replace(",", "").strip()


def main() -> None:
    child_rows = read_rows(CHILD_PATH)
    household_rows = read_rows(HOUSEHOLD_PATH)
    child_counts = Counter(map(key, child_rows))
    household_counts = Counter(map(key, household_rows))

    if any(count > 1 for count in child_counts.values()):
        raise ValueError("아동인구현황에 중복 결합키가 있습니다.")
    if any(count > 1 for count in household_counts.values()):
        raise ValueError("세대당 인구수에 중복 결합키가 있습니다.")
    if child_counts.keys() != household_counts.keys():
        child_only = child_counts.keys() - household_counts.keys()
        household_only = household_counts.keys() - child_counts.keys()
        raise ValueError(
            f"결합키 불일치: 아동만 {len(child_only)}건, 세대만 {len(household_only)}건"
        )

    household_by_key = {key(row): row for row in household_rows}
    comparison_columns = [
        ("전체", "총인구수"),
        ("남자", "남자 인구수"),
        ("여자", "여자 인구수"),
    ]
    mismatches = []
    for child in child_rows:
        household = household_by_key[key(child)]
        for child_column, household_column in comparison_columns:
            if number(child[child_column]) != number(household[household_column]):
                mismatches.append((key(child), child_column, household_column))
    if mismatches:
        raise ValueError(f"중복 인구 컬럼 값 불일치: {len(mismatches)}건")

    with OUTPUT_PATH.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for child in child_rows:
            household = household_by_key[key(child)]
            writer.writerow(
                {
                    "연도": child["연도"],
                    "자치구": child["자치구"],
                    "행정동": child["행정동"],
                    "전체": child["전체"],
                    "남자": child["남자"],
                    "여자": child["여자"],
                    "세대수": household["세대수"],
                    "세대당 인구": household["세대당 인구"],
                    "남여 비율": household["남여 비율"],
                    "아동전체": child["아동전체"],
                    "아동남자": child["아동남자"],
                    "아동여자": child["아동여자"],
                    "청소년전체": child["청소년전체"],
                    "청소년남자": child["청소년남자"],
                    "청소년여자": child["청소년여자"],
                }
            )

    print(f"결합 완료: {len(child_rows):,}행, {len(OUTPUT_COLUMNS)}열")
    print("결합키 매핑률: 100%")
    print("중복키: 0건, 미매핑키: 0건, 중복 인구값 불일치: 0건")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
