"""아동·세대 통합본에 0~19세 연령별 인구현황을 검증하며 결합한다."""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "interim" / "행정동별_아동인구_세대수_통합.csv"
OUTPUT_PATH = ROOT / "data" / "interim" / "행정동별_아동인구_세대수_연령별인구19_통합.csv"
KEY_COLUMNS = ["연도", "자치구", "행정동"]
AGE_COLUMNS = [f"{age}세" for age in range(20)]
CODE_PATTERN = re.compile(r"\s*\(\d+\)\s*$")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def key(row: dict[str, str]) -> tuple[str, str, str]:
    return tuple(row[column].strip() for column in KEY_COLUMNS)


def integer(value: str) -> int:
    return int(value.replace(",", "").strip())


def load_seoul_age_rows() -> list[dict[str, str]]:
    candidates = []
    for path in (ROOT / "data" / "interim").glob("*.csv"):
        with path.open(encoding="utf-8-sig", newline="") as source:
            header = next(csv.reader(source), [])
        if "행정구역" in header and "0세" in header and "19세" in header:
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(f"연령별 원본 통합본을 특정할 수 없습니다: {candidates}")

    result = []
    for row in read_rows(candidates[0]):
        region = CODE_PATTERN.sub("", row["행정구역"]).strip()
        if region != "서울특별시" and not region.startswith("서울특별시 "):
            continue
        parts = region.split()
        row["자치구"] = parts[1] if len(parts) >= 2 else ""
        row["행정동"] = " ".join(parts[2:]) if len(parts) >= 3 else ""
        result.append(row)
    return result


def main() -> None:
    base_rows = read_rows(INPUT_PATH)
    age_rows = load_seoul_age_rows()
    base_keys = Counter(map(key, base_rows))
    age_keys = Counter(map(key, age_rows))

    if any(count > 1 for count in base_keys.values()):
        raise ValueError("아동·세대 통합본에 중복키가 있습니다.")
    if any(count > 1 for count in age_keys.values()):
        raise ValueError("연령별 인구현황에 중복키가 있습니다.")

    missing_keys = base_keys.keys() - age_keys.keys()
    if missing_keys:
        raise ValueError(f"연령별 인구현황에 없는 기준키가 {len(missing_keys)}건 있습니다.")

    age_by_key = {key(row): row for row in age_rows}
    total_mismatches = []
    child_mismatches = []
    age_range_mismatches = []

    for base in base_rows:
        age = age_by_key[key(base)]
        if integer(base["전체"]) != integer(age["총인구수"]):
            total_mismatches.append(key(base))
        if integer(base["아동전체"]) != sum(integer(age[f"{year}세"]) for year in range(18)):
            child_mismatches.append(key(base))
        if integer(age["연령구간인구수"]) != sum(integer(age[column]) for column in AGE_COLUMNS):
            age_range_mismatches.append(key(base))

    if total_mismatches or child_mismatches or age_range_mismatches:
        raise ValueError(
            "교차검증 실패: "
            f"전체 {len(total_mismatches)}건, "
            f"0~17세 {len(child_mismatches)}건, "
            f"0~19세 {len(age_range_mismatches)}건"
        )

    base_columns = list(base_rows[0])
    output_columns = [*base_columns, "연령구간인구수", *AGE_COLUMNS]
    with OUTPUT_PATH.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=output_columns)
        writer.writeheader()
        for base in base_rows:
            age = age_by_key[key(base)]
            merged = dict(base)
            merged["연령구간인구수"] = age["연령구간인구수"]
            merged.update({column: age[column] for column in AGE_COLUMNS})
            writer.writerow(merged)

    print(f"결합 완료: {len(base_rows):,}행, {len(output_columns)}열")
    print(f"기준키 매핑률: 100% ({len(base_rows):,}/{len(base_rows):,})")
    print("전체 인구 불일치: 0건")
    print("아동전체와 0~17세 합계 불일치: 0건")
    print("연령구간인구수와 0~19세 합계 불일치: 0건")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
