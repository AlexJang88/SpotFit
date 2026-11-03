"""Build district-year traffic-enforcement CCTV density per road area."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ROAD = ROOT / "data/raw/안전/교통사고/서울시_도로현황_자치구_연도별.csv"
CCTV = ROOT / "data/raw/안전/교통사고/파생/서울시_CCTV_설치연도기반_자치구_연도별_추정_2003_2026.csv"
OUTPUT = ROOT / "data/interim/안전/교통사고안전지수/서울시_자치구_도로면적당_교통단속CCTV_2017_2025.csv"


def road_panel() -> pd.DataFrame:
    wide = pd.read_csv(ROAD, header=[0, 1, 2, 3, 4], encoding="utf-8-sig")
    district_column = wide.columns[1]
    rows: list[pd.DataFrame] = []
    for year in range(2017, 2026):
        candidates = [column for column in wide.columns if column[0] == str(year)]
        # The official table begins every year block with total road length and total road area.
        if len(candidates) < 2:
            raise ValueError(f"{year}년 도로현황 열을 찾지 못했습니다.")
        area_column = candidates[1]
        frame = pd.DataFrame(
            {
                "연도": year,
                "자치구": wide[district_column].astype(str).str.strip(),
                "도로면적_m2": pd.to_numeric(wide[area_column], errors="coerce"),
            }
        )
        rows.append(frame)
    result = pd.concat(rows, ignore_index=True)
    result = result[result["자치구"] != "소계"].copy()
    result["도로면적_km2"] = result["도로면적_m2"] / 1_000_000
    if len(result) != 225 or result["자치구"].nunique() != 25:
        raise ValueError("도로면적 자료가 9개년×25개 자치구 구조가 아닙니다.")
    return result


def main() -> int:
    roads = road_panel()
    cctv = pd.read_csv(CCTV, encoding="utf-8-sig")
    cctv = cctv[(cctv["구분"] == "교통단속") & cctv["연도"].between(2017, 2025)].copy()
    cctv = cctv[["연도", "자치구", "폐기미반영_누적추정카메라수", "설치연월_카메라기준커버리지_pct"]]
    result = roads.merge(cctv, on=["연도", "자치구"], how="inner", validate="one_to_one")
    result["도로면적당_교통단속CCTV대수_km2"] = (
        result["폐기미반영_누적추정카메라수"] / result["도로면적_km2"]
    )
    result["CCTV_폐기반영여부"] = "미반영"
    result["설명"] = "2026 현재목록의 설치일 기반 누적추정 교통단속 CCTV ÷ 해당연도 도로면적(km²)"
    if len(result) != 225 or result.isna().any().any():
        raise ValueError("CCTV-도로면적 결합 결과에 행 누락 또는 결측이 있습니다.")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"saved={OUTPUT} rows={len(result)} districts={result['자치구'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
