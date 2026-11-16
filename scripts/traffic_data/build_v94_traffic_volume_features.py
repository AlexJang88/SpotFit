"""Build leakage-safe annual district traffic-volume features for v94."""
from __future__ import annotations

import re
from pathlib import Path
import sys

import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.traffic_accident_pipeline_paths import INTERIM_V94_TRAFFIC, RAW_V94_TRAFFIC

SOURCE = RAW_V94_TRAFFIC
OUT = INTERIM_V94_TRAFFIC
DISTRICT_PATTERN = re.compile(r"서울(?:특별시|시)?\s+([가-힣]+구)\b")
HOURS = [f"{hour}시" for hour in range(24)]
BOROUGH_OVERRIDES = {
    # 서울 시계 관측소는 검지기가 경기도 쪽에 있어도 서울 진입 도로의 인접 자치구로 배정한다.
    "B-04": "중랑구", "B-08": "강동구", "B-16": "서초구", "B-25": "금천구",
    "B-29": "구로구", "B-31": "강서구", "B-33": "은평구", "B-37": "광진구",
    "C-22": "강동구", "D-47": "송파구", "D-48": "송파구", "D-50": "송파구",
    "D-51": "송파구", "D-52": "송파구", "D-53": "송파구", "D-54": "송파구",
}


def station_district_map() -> pd.DataFrame:
    frames = []
    for year in range(2017, 2026):
        for path in sorted((SOURCE / str(year)).glob("*.xlsx")):
            book = openpyxl.load_workbook(path, read_only=True, data_only=True)
            if "수집지점 주소 및 좌표" in book.sheetnames:
                rows = list(book["수집지점 주소 및 좌표"].iter_rows(values_only=True))
                frames.append(pd.DataFrame(rows[1:], columns=rows[0]))
                book.close()
                break
            book.close()
    if not frames:
        raise ValueError("수집지점 주소 및 좌표 시트를 찾지 못했습니다.")
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame["지점번호"].astype(str).str.fullmatch(r"[A-Z]-\d{2}")].copy()
    address = frame["주소"].fillna(frame["도로명 주소"]).astype(str).str.replace(",", " ", regex=False)
    frame["자치구"] = address.str.extract(DISTRICT_PATTERN, expand=False)
    frame["자치구"] = frame["지점번호"].map(BOROUGH_OVERRIDES).fillna(frame["자치구"])
    if frame["자치구"].isna().any():
        missing = frame.loc[frame["자치구"].isna(), ["지점번호", "주소", "도로명 주소"]]
        raise ValueError(f"주소에서 자치구를 찾지 못한 지점:\n{missing}")
    manual = pd.DataFrame({"지점번호": list(BOROUGH_OVERRIDES), "자치구": list(BOROUGH_OVERRIDES.values())})
    for column in ("위도", "경도", "주소"):
        manual[column] = np.nan
    return pd.concat([
        frame[["지점번호", "자치구", "위도", "경도", "주소"]], manual
    ], ignore_index=True).drop_duplicates("지점번호", keep="last")


def data_sheet(path: Path):
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    for sheet in book.worksheets:
        first = next(sheet.iter_rows(min_row=1, max_row=1, values_only=True), ())
        if "일자" in first and "지점번호" in first and "0시" in first and "23시" in first:
            rows = sheet.iter_rows(values_only=True)
            columns = next(rows)
            frame = pd.DataFrame(rows, columns=columns)
            book.close()
            return frame
    book.close()
    raise ValueError(f"교통량 데이터 시트를 찾지 못했습니다: {path}")


def main() -> None:
    mapping = station_district_map()
    frames = []
    for year in range(2017, 2026):
        for path in sorted((SOURCE / str(year)).glob("*.xlsx")):
            frame = data_sheet(path)
            frame = frame[["일자", "요일", "지점번호", *HOURS]].copy()
            frame = frame[frame["지점번호"].astype(str).str.fullmatch(r"[A-Z]-\d{2}")].copy()
            parsed_year = pd.to_numeric(frame["일자"].astype(str).str[:4], errors="coerce")
            frame = frame[parsed_year.notna()].copy()
            frame["연도"] = parsed_year[parsed_year.notna()].astype(int)
            if not frame["연도"].eq(year).all():
                raise ValueError(f"파일 연도 불일치: {path}")
            for column in HOURS:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            frames.append(frame)

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.merge(mapping, on="지점번호", how="left", validate="many_to_one")
    if raw["자치구"].isna().any():
        missing = sorted(raw.loc[raw["자치구"].isna(), "지점번호"].dropna().unique())
        raise ValueError(f"2025 지점정보에 없는 과거 지점: {missing}")

    raw["일교통량"] = raw[HOURS].sum(axis=1, min_count=12)
    raw["등교시간_07_09"] = raw[["7시", "8시", "9시"]].sum(axis=1, min_count=2)
    raw["하교시간_14_18"] = raw[["14시", "15시", "16시", "17시", "18시"]].sum(axis=1, min_count=3)
    raw["평일"] = ~raw["요일"].isin(["토", "일"])

    # 방향별 행을 먼저 지점·일 단위로 합쳐 같은 지점이 두 번 집계되지 않게 한다.
    daily = raw.groupby(["연도", "자치구", "일자", "지점번호"], as_index=False).agg(
        일교통량=("일교통량", "sum"),
        등교시간_07_09=("등교시간_07_09", "sum"),
        하교시간_14_18=("하교시간_14_18", "sum"),
        평일=("평일", "first"),
    )
    daily["등교시간비중"] = daily["등교시간_07_09"] / daily["일교통량"].replace(0, np.nan)
    daily["하교시간비중"] = daily["하교시간_14_18"] / daily["일교통량"].replace(0, np.nan)

    rows = []
    for (year, district), group in daily.groupby(["연도", "자치구"], sort=True):
        weekday = group[group["평일"]]
        weekend = group[~group["평일"]]
        rows.append({
            "연도": year,
            "자치구": district,
            "교통량_관측지점수": group["지점번호"].nunique(),
            "교통량_관측지점일수": len(group),
            "교통량_평균일교통량": group["일교통량"].mean(),
            "교통량_평일평균일교통량": weekday["일교통량"].mean(),
            "교통량_주말평균일교통량": weekend["일교통량"].mean(),
            "교통량_등교시간비중": group["등교시간비중"].mean(),
            "교통량_하교시간비중": group["하교시간비중"].mean(),
        })
    result = pd.DataFrame(rows).sort_values(["연도", "자치구"]).reset_index(drop=True)
    # 관측지점이 없는 자치구는 모델 단계에서 연도별 중앙값으로 보정되므로 여기서는 원자료 집계만 보존한다.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"saved={OUT.relative_to(ROOT)} rows={len(result)} districts={result['자치구'].nunique()}")
    print(result.groupby("연도").agg(자치구수=("자치구", "nunique"), 지점수=("교통량_관측지점수", "sum")))


if __name__ == "__main__":
    main()
