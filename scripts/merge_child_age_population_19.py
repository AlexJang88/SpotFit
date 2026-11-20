"""2021~2025 아동 연령별 인구현황(0~19세) CSV를 하나로 결합한다."""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "raw" / "아동" / "아동 연령별 인구현황19"
OUTPUT_PATH = ROOT / "data" / "interim" / "아동_연령별인구현황19_2021_2025.csv"
FILE_PATTERN = re.compile(r"^(20\d{2})연령별인구현황19\.csv$")


def normalize_header(header: list[str], year: str) -> list[str]:
    """연도별 접두어를 없애 모든 파일의 열 이름을 동일하게 만든다."""
    prefix = f"{year}년_계_"
    normalized = ["연도"]
    for column in header:
        if column == "행정구역":
            normalized.append(column)
        elif column.startswith(prefix):
            normalized.append(column.removeprefix(prefix))
        else:
            raise ValueError(f"예상하지 못한 열 이름: {column!r} ({year}년)")
    return normalized


def main() -> None:
    source_files: list[tuple[str, Path]] = []
    for path in INPUT_DIR.glob("*.csv"):
        match = FILE_PATTERN.match(path.name)
        if match:
            source_files.append((match.group(1), path))
    source_files.sort()

    if len(source_files) != 5:
        raise RuntimeError(f"입력 파일은 5개여야 합니다: {source_files}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    expected_columns: list[str] | None = None
    total_rows = 0

    with OUTPUT_PATH.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.writer(output_file)

        for year, path in source_files:
            with path.open("r", encoding="cp949", newline="") as input_file:
                reader = csv.reader(input_file)
                source_header = next(reader)
                columns = normalize_header(source_header, year)

                if expected_columns is None:
                    expected_columns = columns
                    writer.writerow(columns)
                elif columns != expected_columns:
                    raise ValueError(f"열 구조가 일치하지 않습니다: {path}")

                for row_number, row in enumerate(reader, start=2):
                    if len(row) != len(source_header):
                        raise ValueError(
                            f"열 개수 오류: {path.name} {row_number}행 "
                            f"({len(row)}개, 예상 {len(source_header)}개)"
                        )
                    writer.writerow([year, *row])
                    total_rows += 1

    print(f"결합 완료: {len(source_files)}개 파일, {total_rows:,}행")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
