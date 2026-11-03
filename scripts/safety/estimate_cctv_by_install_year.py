"""현재 CCTV 목록과 설치연월로 연도 말 누적 카메라 대수를 추정한다.

철거·폐기 이력이 없으므로 설치 후 계속 운영 중이라고 가정한다. 설치연월 미상
카메라는 자치구·목적별 유효 설치연도 분포로 배분하고, 표본이 없으면 서울 전체
동일 목적 분포를 사용한다.
"""
from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "raw" / "안전" / "교통사고" / "서울시_CCTV정보.csv"
OUT = ROOT / "data" / "raw" / "안전" / "교통사고" / "파생"
DISTRICTS = ("종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구",
             "강북구", "도봉구", "노원구", "은평구", "서대문구", "마포구", "양천구", "강서구",
             "구로구", "금천구", "영등포구", "동작구", "관악구", "서초구", "강남구", "송파구", "강동구")


def district_from(row: pd.Series) -> str | None:
    for column in ("소재지도로명주소", "소재지지번주소", "관리기관명"):
        text = str(row.get(column) or "")
        for district in DISTRICTS:
            if re.search(rf"(?<![가-힣]){district}(?![가-힣])", text) or district in text:
                return district
    return None


def cdf(known: pd.DataFrame, years: list[int], fallback: pd.DataFrame) -> dict[int, float]:
    source = known if known["카메라대수"].sum() > 0 else fallback
    total = source["카메라대수"].sum()
    if total <= 0:
        return {year: (1.0 if year == years[-1] else 0.0) for year in years}
    installed = source.groupby("설치연도")["카메라대수"].sum()
    return {year: float(installed[installed.index <= year].sum() / total) for year in years}


def main() -> int:
    data = pd.read_csv(SOURCE, encoding="cp949")
    data["자치구"] = data.apply(district_from, axis=1)
    data["카메라대수"] = pd.to_numeric(data["카메라대수"], errors="coerce").fillna(1).clip(lower=0)
    install = pd.to_numeric(data["설치연월"], errors="coerce")
    data["설치연도"] = (install // 100).astype("Int64")
    data.loc[~data["설치연도"].between(1980, 2026), "설치연도"] = pd.NA
    data = data[data["자치구"].isin(DISTRICTS)].copy()
    years = list(range(int(data["설치연도"].min()), 2027))
    data["목적집계"] = np.where(data["설치목적구분"].eq("교통단속"), "교통단속", "비교통단속")

    rows = []
    for district in DISTRICTS:
        district_data = data[data["자치구"].eq(district)]
        for scope in ("전체", "교통단속"):
            subset = district_data if scope == "전체" else district_data[district_data["목적집계"].eq("교통단속")]
            known = subset.dropna(subset=["설치연도"])
            if scope == "전체":
                fallback = data.dropna(subset=["설치연도"])
            else:
                fallback = data[data["목적집계"].eq("교통단속")].dropna(subset=["설치연도"])
            distribution = cdf(known, years, fallback)
            current_cameras = float(subset["카메라대수"].sum())
            known_cameras = float(known["카메라대수"].sum())
            unknown_cameras = current_cameras - known_cameras
            for year in years:
                confirmed = float(known.loc[known["설치연도"] <= year, "카메라대수"].sum())
                estimated = confirmed + unknown_cameras * distribution[year]
                rows.append({
                    "연도": year, "자치구": district, "구분": scope,
                    "설치연월확인_누적카메라수": int(round(confirmed)),
                    "설치연월미상_배분추정수": int(round(unknown_cameras * distribution[year])),
                    "폐기미반영_누적추정카메라수": int(round(estimated)),
                    "2026현재목록_카메라수": int(round(current_cameras)),
                    "설치연월확인_카메라수": int(round(known_cameras)),
                    "설치연월_카메라기준커버리지_pct": round(100 * known_cameras / current_cameras, 4) if current_cameras else 0,
                    "추정방법": "확인 누적+미상 카메라를 자치구·목적별 설치연도 CDF로 배분; 표본 없으면 서울 동일목적 CDF",
                    "폐기반영여부": "미반영",
                })
    panel = pd.DataFrame(rows)
    summary = (panel.groupby(["연도", "구분"], as_index=False)[
        ["설치연월확인_누적카메라수", "설치연월미상_배분추정수", "폐기미반영_누적추정카메라수", "2026현재목록_카메라수"]
    ].sum())
    summary["폐기반영여부"] = "미반영"
    summary["추정범위"] = "서울 25개 자치구 합계"

    OUT.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT / "서울시_CCTV_설치연도기반_자치구_연도별_추정_2003_2026.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "서울시_CCTV_설치연도기반_서울전체_연도별_추정_2003_2026.csv", index=False, encoding="utf-8-sig")
    print(f"원천 {len(data):,}행, 결과 {len(panel):,}행, 기간 {years[0]}~{years[-1]}")
    print(summary.tail(6).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
