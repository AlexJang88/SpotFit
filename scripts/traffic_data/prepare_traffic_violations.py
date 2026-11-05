"""서울경찰청 경찰서별 교통법규 위반 자료를 자치구·연도별로 집계한다."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "raw" / "안전" / "교통사고" / "교통법규위반"
OUTPUT = (
    ROOT
    / "data"
    / "interim"
    / "안전"
    / "교통사고"
    / "자치구별_교통법규위반_2022_2024_부분.csv"
)

POLICE_TO_DISTRICT = {
    "중부": "중구", "종로": "종로구", "남대문": "중구", "서대문": "서대문구",
    "혜화": "종로구", "용산": "용산구", "성북": "성북구", "동대문": "동대문구",
    "마포": "마포구", "영등포": "영등포구", "성동": "성동구", "동작": "동작구",
    "광진": "광진구", "서부": "은평구", "강북": "강북구", "금천": "금천구",
    "중랑": "중랑구", "강남": "강남구", "관악": "관악구", "강서": "강서구",
    "강동": "강동구", "종암": "성북구", "구로": "구로구", "서초": "서초구",
    "양천": "양천구", "송파": "송파구", "노원": "노원구", "방배": "서초구",
    "은평": "은평구", "도봉": "도봉구", "수서": "강남구",
}
VIOLATION_COLUMNS = ["중앙선침범", "신호위반", "음주운전", "무면허운전", "속도위반"]


def main() -> None:
    frames = []
    for path in sorted(SOURCE.glob("*.csv")):
        match = re.search(r"(20\d{2})", path.stem)
        if not match:
            continue
        year = int(match.group(1))
        try:
            frame = pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            frame = pd.read_csv(path, encoding="cp949")
        frame = frame[frame["구분"].notna()].copy()
        frame["구분"] = frame["구분"].astype(str).str.strip()
        unknown = sorted(set(frame["구분"]) - set(POLICE_TO_DISTRICT))
        if unknown:
            raise ValueError(f"경찰서-자치구 매핑 누락: {unknown}")
        frame["자치구"] = frame["구분"].map(POLICE_TO_DISTRICT)
        frame["연도"] = year
        frame[VIOLATION_COLUMNS] = frame[VIOLATION_COLUMNS].apply(pd.to_numeric).astype(int)
        frames.append(frame[["연도", "자치구", *VIOLATION_COLUMNS]])

    combined = pd.concat(frames, ignore_index=True)
    output = combined.groupby(["연도", "자치구"], as_index=False)[VIOLATION_COLUMNS].sum()
    output["교통법규위반_합계"] = output[VIOLATION_COLUMNS].sum(axis=1)
    counts = output.groupby("연도")["자치구"].nunique()
    if list(counts.index) != [2022, 2023, 2024] or not counts.eq(25).all():
        raise ValueError(f"연도별 25개 자치구 완전성 검증 실패: {counts.to_dict()}")
    if output.duplicated(["연도", "자치구"]).any() or output.isna().any().any():
        raise ValueError("중복 또는 결측값이 있습니다.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"저장: {OUTPUT}")
    print(f"행: {len(output)}, 연도별 자치구 수: {counts.to_dict()}")


if __name__ == "__main__":
    main()
