"""Collect the current national unmanned traffic-camera standard dataset.

Keeps raw API pages, extracts Seoul records, and aggregates the cameras that
match the 2025 Local Safety Index guidance (speed/section/multifunction,
excluding expressway cameras).
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "AI수집" / "안전" / "교통사고" / "교통사고_기준표보강" / "교통단속CCTV_최신"
OUT_DETAIL = ROOT / "data" / "interim" / "안전" / "교통사고" / "서울_교통단속CCTV_최신_상세.csv"
OUT_AGG = ROOT / "data" / "interim" / "안전" / "교통사고" / "자치구별_교통단속CCTV_최신.csv"
BASE = "https://www.data.go.kr/download/standard.json"
PK = "15028200"
TABLE = "tn_pubr_public_unmanned_traffic_camera_svc"
TOTAL = 44650
PER_PAGE = 10000

COLUMNS = [
    "MNLSS_REGLT_CAMERA_MANAGE_NO", "CTPRVN_NM", "SIGNGU_NM", "ROAD_KND",
    "ROAD_ROUTE_NO", "ROAD_ROUTE_NM", "ROAD_ROUTE_DRC", "RDNMADR", "LNMADR",
    "LATITUDE", "LONGITUDE", "ITLPC", "REGLT_SE", "LMTT_VE",
    "REGLT_SCTN_LC_SE", "OVRSPD_REGLT_SCTN_LT", "PRTCAREA_TYPE",
    "INSTALLATION_YEAR", "INSTITUTION_NM", "PHONE_NUMBER", "REFERENCE_DATE",
]
KOREAN = {
    "MNLSS_REGLT_CAMERA_MANAGE_NO": "관리번호", "CTPRVN_NM": "시도명",
    "SIGNGU_NM": "시군구명", "ROAD_KND": "도로종류", "ROAD_ROUTE_NO": "도로노선번호",
    "ROAD_ROUTE_NM": "도로노선명", "ROAD_ROUTE_DRC": "도로노선방향",
    "RDNMADR": "소재지도로명주소", "LNMADR": "소재지지번주소", "LATITUDE": "위도",
    "LONGITUDE": "경도", "ITLPC": "설치장소", "REGLT_SE": "단속구분",
    "LMTT_VE": "제한속도", "REGLT_SCTN_LC_SE": "단속구간위치구분",
    "OVRSPD_REGLT_SCTN_LT": "과속단속구간길이", "PRTCAREA_TYPE": "보호구역구분",
    "INSTALLATION_YEAR": "설치연도", "INSTITUTION_NM": "관리기관명",
    "PHONE_NUMBER": "관리기관전화번호", "REFERENCE_DATE": "데이터기준일자",
}
DISTRICTS = {
    "종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구",
    "강북구", "도봉구", "노원구", "은평구", "서대문구", "마포구", "양천구", "강서구",
    "구로구", "금천구", "영등포구", "동작구", "관악구", "서초구", "강남구", "송파구", "강동구",
}


def fetch(page: int) -> list[dict]:
    params = [
        ("publicDataPk", PK), ("svcTableNm", TABLE), ("perPage", PER_PAGE),
        ("page", page), ("totalCount", TOTAL),
    ] + [("colNmList", c) for c in COLUMNS]
    req = Request(f"{BASE}?{urlencode(params)}", headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=60) as response:
        return json.load(response)


def district_from_row(row: pd.Series) -> str | None:
    value = str(row.get("시군구명") or "").strip()
    if value in DISTRICTS:
        return value
    address = f"{row.get('소재지도로명주소', '')} {row.get('소재지지번주소', '')}"
    return next((gu for gu in DISTRICTS if gu in address), None)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    OUT_DETAIL.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    pages = (TOTAL + PER_PAGE - 1) // PER_PAGE
    for page in range(1, pages + 1):
        payload = fetch(page)
        (RAW / f"cctv_page_{page}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        rows.extend(payload)
    data = pd.DataFrame(rows).rename(columns=KOREAN)
    seoul = data[data["시도명"].astype(str).str.contains("서울", na=False)].copy()
    seoul["자치구"] = seoul.apply(district_from_row, axis=1)
    seoul["제한속도"] = pd.to_numeric(seoul["제한속도"], errors="coerce")
    road = seoul["도로종류"].fillna("").astype(str)
    seoul["고속도로제외"] = ~(road.str.contains("고속") | seoul["제한속도"].ge(100))
    category = seoul["단속구분"].fillna("").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    # Standard codes: 1 speed, 2 signal; Seoul's multifunction cameras are
    # represented in these police-managed categories.  Code 4 is parking.
    seoul["공식기능포함"] = category.isin(["1", "01", "2", "02"]) | category.str.contains(
        "과속|구간|다기능|속도|신호", regex=True
    )
    police_managed = seoul["관리기관명"].fillna("").astype(str).str.contains("경찰")
    seoul["경찰관리기관"] = police_managed
    seoul["지역안전지수_포함"] = seoul["고속도로제외"] & seoul["공식기능포함"] & police_managed
    seoul["출처"] = "공공데이터포털 전국무인교통단속카메라표준데이터"
    seoul.to_csv(OUT_DETAIL, index=False, encoding="utf-8-sig")

    eligible = seoul[seoul["지역안전지수_포함"] & seoul["자치구"].notna()]
    agg = eligible.groupby("자치구").size().rename("교통단속CCTV대수_최신").reset_index()
    agg = pd.DataFrame({"자치구": sorted(DISTRICTS)}).merge(agg, on="자치구", how="left")
    agg["교통단속CCTV대수_최신"] = agg["교통단속CCTV대수_최신"].fillna(0).astype(int)
    agg["데이터기준일_최신"] = eligible.groupby("자치구")["데이터기준일자"].max().reindex(agg["자치구"]).values
    agg["출처"] = "공공데이터포털 전국무인교통단속카메라표준데이터"
    agg["주의"] = "최신 스냅샷; 과거 연도 자료가 아님"
    agg.to_csv(OUT_AGG, index=False, encoding="utf-8-sig")
    print(f"national={len(data)}, seoul={len(seoul)}, eligible={len(eligible)}")
    print(f"districts={agg['자치구'].nunique()}, unresolved={seoul['자치구'].isna().sum()}")
    print(f"saved={OUT_DETAIL}\nsaved={OUT_AGG}")
    print("categories=", sorted(seoul["단속구분"].dropna().astype(str).unique()))


if __name__ == "__main__":
    main()
