"""Collect official annual registered population by Seoul district.

Source: Seoul Open Data Plaza GIS statistics API (visualization id 3).
The script keeps every response as raw JSON and writes a tidy CSV for joining.
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "안전" / "교통사고_기준표보강" / "주민등록인구"
OUTPUT = ROOT / "data" / "interim" / "안전" / "자치구별_주민등록인구_2017_2025.csv"
URL = "https://data.seoul.go.kr/bsp/wgs/gis/map.do?vslzId=3&scopCd=SGG&statsYr={year}"

DISTRICTS = {
    "11010": "종로구", "11020": "중구", "11030": "용산구", "11040": "성동구",
    "11050": "광진구", "11060": "동대문구", "11070": "중랑구", "11080": "성북구",
    "11090": "강북구", "11100": "도봉구", "11110": "노원구", "11120": "은평구",
    "11130": "서대문구", "11140": "마포구", "11150": "양천구", "11160": "강서구",
    "11170": "구로구", "11180": "금천구", "11190": "영등포구", "11200": "동작구",
    "11210": "관악구", "11220": "서초구", "11230": "강남구", "11240": "송파구",
    "11250": "강동구",
}


def fetch(year: int) -> dict:
    req = Request(URL.format(year=year), headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as response:
        return json.load(response)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for year in range(2017, 2026):
        payload = fetch(year)
        if payload.get("header", {}).get("resultCode") != "ok":
            raise RuntimeError(f"API failure for {year}: {payload.get('header')}")
        raw_path = RAW_DIR / f"seoul_registered_population_{year}.json"
        raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        for row in payload["data"]:
            code = str(row["adm_cd"])
            records.append({
                "연도": int(row["stats_yr"]),
                "자치구": DISTRICTS.get(code),
                "행정구역코드": code,
                "주민등록인구_계": row.get("popl_cnt"),
                "주민등록인구_내국인": row.get("popl_cnt_n"),
                "주민등록인구_외국인": row.get("popl_cnt_a"),
                "주민등록인구_남자": row.get("popl_cnt_m"),
                "주민등록인구_여자": row.get("popl_cnt_f"),
                "출처": "서울 열린데이터광장 주민등록인구(연도별)",
                "원자료URL": URL.format(year=year),
            })
    data = pd.DataFrame(records).sort_values(["연도", "자치구"])
    if data["자치구"].isna().any():
        raise ValueError("Unknown district code in API response")
    expected = {(year, gu) for year in range(2017, 2026) for gu in DISTRICTS.values()}
    actual = set(data[["연도", "자치구"]].itertuples(index=False, name=None))
    if actual != expected:
        raise ValueError(f"Key coverage mismatch: missing={expected-actual}, extra={actual-expected}")
    if data.duplicated(["연도", "자치구"]).any():
        raise ValueError("Duplicate year-district keys")
    data.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"saved={OUTPUT}")
    print(f"rows={len(data)}, years={data['연도'].min()}-{data['연도'].max()}, districts={data['자치구'].nunique()}")


if __name__ == "__main__":
    main()
