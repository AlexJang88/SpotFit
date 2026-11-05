"""2017~2025년 서울 자치구별 연령 인구 CSV를 하나로 통합한다."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "data" / "raw" / "연령별인구현황"
OUTPUT = (
    ROOT
    / "data"
    / "interim"
    / "안전"
    / "교통사고"
    / "자치구별_연령별인구현황_2017_2025.csv"
)


def parse_district(value: str) -> tuple[str, str]:
    match = re.fullmatch(r"서울특별시\s+(.+구)\s+\((\d+)\)", value.strip())
    if not match:
        return "", ""
    return match.group(1), match.group(2)


def normalized_column(column: str, year: int) -> str | None:
    prefix = f"{year}년_계_"
    if not column.startswith(prefix):
        return None
    name = column.removeprefix(prefix)
    if name == "총인구수":
        return "총인구수"
    if name == "연령구간인구수":
        return None
    return name


def main() -> None:
    frames: list[pd.DataFrame] = []
    files = sorted(SOURCE_DIR.glob("*_연령별인구현황_연간.csv"))
    expected_years = list(range(2017, 2026))
    actual_years = [int(path.name[:4]) for path in files]
    if actual_years != expected_years:
        raise ValueError(f"연도별 원본이 완전하지 않습니다: {actual_years}")

    for path, year in zip(files, actual_years):
        source = pd.read_csv(path, encoding="cp949", dtype=str)
        district_and_code = source["행정구역"].map(parse_district)
        source["자치구"] = district_and_code.map(lambda value: value[0])
        source["행정구역코드"] = district_and_code.map(lambda value: value[1])
        source = source[source["자치구"].ne("")].copy()

        rename = {
            column: normalized
            for column in source.columns
            if (normalized := normalized_column(column, year)) is not None
        }
        frame = source[["자치구", "행정구역코드", *rename]].rename(columns=rename)
        numeric_columns = [column for column in frame if column not in {"자치구", "행정구역코드"}]
        frame[numeric_columns] = frame[numeric_columns].apply(
            lambda column: pd.to_numeric(column.str.replace(",", "", regex=False))
        )
        frame = pd.concat(
            [pd.Series(year, index=frame.index, name="연도"), frame.copy()], axis=1
        )
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    key_columns = ["연도", "자치구"]
    if combined.duplicated(key_columns).any():
        raise ValueError("연도·자치구 중복 행이 있습니다.")
    counts = combined.groupby("연도")["자치구"].nunique()
    if not counts.eq(25).all() or len(combined) != 225:
        raise ValueError(f"25개 자치구 완전성 검증 실패: {counts.to_dict()}")
    age_columns = [f"{age}세" for age in range(100)] + ["100세 이상"]
    if combined[["총인구수", *age_columns]].isna().any().any():
        raise ValueError("인구수 결측값이 있습니다.")
    age_sum = combined[age_columns].sum(axis=1)
    if not age_sum.eq(combined["총인구수"]).all():
        raise ValueError("연령별 인구 합계와 총인구수가 일치하지 않습니다.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"저장: {OUTPUT}")
    print(f"행: {len(combined)}, 연도: {combined['연도'].min()}~{combined['연도'].max()}")
    print(f"연도별 자치구 수: {counts.to_dict()}")


if __name__ == "__main__":
    main()
