"""두 AED 원본을 정제하고 서울 행정동 경계와 공간 결합한다."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd


PUBLIC_COLUMNS = {
    "INSTL_NM": "설치장소명", "INSTL_LC": "설치위치", "RDNMADR": "도로명주소",
    "LATITUDE": "위도", "LONGITUDE": "경도", "MAKR": "제조사", "MAKR_NO": "제조번호",
    "OPER_CHARGER_NM": "운영담당자명", "OPER_PHONE_NUMBER": "운영담당자연락처",
    "INSTITUTION_NM": "관리기관명", "REFERENCE_DATE": "데이터기준일자",
}

# 안전 위치·수량 분석과 관계가 없거나 개인정보/단순 식별자/상수인 컬럼이다.
DROP_COLUMNS = [
    "제조사",
    "제조번호",
    "운영담당자명",
    "운영담당자연락처",
    "관리기관명",
    "연번",
    "관리번호",
    "역내안전설비 구분",
    "제세동기 운영방식",
    "역좌표_매칭여부",
]


def boundaries(path: Path) -> gpd.GeoDataFrame:
    area = gpd.read_file(path).to_crs("EPSG:4326")
    required = {"sggnm", "adm_nm", "geometry"}
    if missing := required - set(area.columns):
        raise ValueError(f"행정동 경계 컬럼 누락: {sorted(missing)}")
    area = area[["sggnm", "adm_nm", "geometry"]].copy()
    area["행정동"] = area["adm_nm"].str.split().str[-1]
    return area.rename(columns={"sggnm": "자치구"})[["자치구", "행정동", "geometry"]]


def add_regions(frame: pd.DataFrame, area: gpd.GeoDataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["위도"] = pd.to_numeric(result["위도"], errors="coerce")
    result["경도"] = pd.to_numeric(result["경도"], errors="coerce")
    valid = result["위도"].between(33, 39) & result["경도"].between(124, 132)
    points = gpd.GeoDataFrame(
        result.loc[valid].copy(),
        geometry=gpd.points_from_xy(result.loc[valid, "경도"], result.loc[valid, "위도"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(points, area, how="left", predicate="intersects")
    joined = joined[~joined.index.duplicated(keep="first")]
    result.insert(0, "행정동", pd.NA)
    result.insert(0, "자치구", pd.NA)
    result.loc[joined.index, "자치구"] = joined["자치구"].array
    result.loc[joined.index, "행정동"] = joined["행정동"].array
    return result


def public_aed(input_path: Path) -> pd.DataFrame:
    data = pd.read_csv(input_path, encoding="utf-8-sig")
    if missing := set(PUBLIC_COLUMNS) - set(data.columns):
        raise ValueError(f"자동심장충격기 원본 컬럼 누락: {sorted(missing)}")
    address = data["RDNMADR"].astype("string").str.strip()
    result = data.loc[address.str.startswith("서울특별시", na=False)].copy()
    result["RDNMADR"] = address.loc[result.index]
    result = result.rename(columns=PUBLIC_COLUMNS).reset_index(drop=True)
    return result


def station_name(series: pd.Series) -> pd.Series:
    normalized = (series.astype("string").str.strip()
                  .str.replace(r"\((?:\d+호선|\d+)\)$", "", regex=True)
                  .str.replace(r"역$", "", regex=True))
    return normalized.replace({
        "이수(총신대입구)": "총신대입구",
        "자양(뚝섬한강공원)": "뚝섬유원지",
    })


def subway_aed(input_path: Path, coordinate_path: Path) -> pd.DataFrame:
    aed = pd.read_excel(input_path, sheet_name=0)
    coords = pd.read_csv(coordinate_path, encoding="cp949")
    for label, data, required in (
        ("안전한동네 AED", aed, {"호선", "역명"}),
        ("역 좌표", coords, {"호선", "역명", "위도", "경도"}),
    ):
        if missing := required - set(data.columns):
            raise ValueError(f"{label} 컬럼 누락: {sorted(missing)}")
    aed["매칭호선"] = pd.to_numeric(
        aed["호선"].astype("string").str.extract(r"(\d+)", expand=False), errors="coerce"
    ).astype("Int64")
    aed["매칭역명"] = station_name(aed["역명"])
    coords["매칭호선"] = pd.to_numeric(coords["호선"], errors="coerce").astype("Int64")
    coords["매칭역명"] = station_name(coords["역명"])
    station_map = coords[["매칭호선", "매칭역명", "위도", "경도"]].drop_duplicates(
        ["매칭호선", "매칭역명"]
    )
    result = aed.merge(station_map, on=["매칭호선", "매칭역명"], how="left", validate="many_to_one")
    # 원본에 장암=6호선, 암사=7호선처럼 실제 노선과 다른 표기가 있다. 1차
    # 매칭 실패 역 중 좌표가 하나뿐인 고유 역명만 노선과 무관하게 보완한다.
    unique_name_map = coords[["매칭역명", "위도", "경도"]].drop_duplicates()
    unique_name_map = unique_name_map[
        unique_name_map.groupby("매칭역명")["매칭역명"].transform("size").eq(1)
    ].set_index("매칭역명")
    missed = result["위도"].isna()
    result.loc[missed, "위도"] = result.loc[missed, "매칭역명"].map(unique_name_map["위도"])
    result.loc[missed, "경도"] = result.loc[missed, "매칭역명"].map(unique_name_map["경도"])
    result["역좌표_매칭여부"] = result["위도"].notna()
    result = result.drop(columns=["매칭호선", "매칭역명"])
    return result


def combine_aed(public: pd.DataFrame, subway: pd.DataFrame) -> pd.DataFrame:
    public = public.copy().rename(columns={"설치위치": "설치상세위치"})
    public.insert(0, "데이터출처", "자동심장충격기")
    public["보유대수"] = 1

    subway = subway.copy().rename(columns={
        "상세위치": "설치상세위치",
        "데이터 기준일자": "데이터기준일자",
    })
    subway.insert(0, "데이터출처", "안전한동네_AED")
    subway["설치장소명"] = subway["역명"].astype("string")

    common_first = [
        "데이터출처", "자치구", "행정동", "설치장소명", "설치상세위치",
        "위도", "경도", "보유대수", "데이터기준일자",
    ]
    all_columns = common_first + [
        column for column in [*public.columns, *subway.columns]
        if column not in common_first
    ]
    # 두 원본에 같은 이름으로 존재하는 컬럼은 한 번만 둔다.
    all_columns = list(dict.fromkeys(all_columns))
    return pd.concat(
        [public.reindex(columns=all_columns), subway.reindex(columns=all_columns)],
        ignore_index=True,
    )


def run(args: argparse.Namespace) -> None:
    area = boundaries(args.boundaries)
    public_mid = public_aed(args.public_input)
    public_final = add_regions(public_mid, area)

    subway_mid = subway_aed(args.subway_input, args.station_coordinates)
    subway_final = add_regions(subway_mid, area)
    combined = combine_aed(public_final, subway_final)
    # 최종 분석 범위는 서울이다. 서울 행정동 경계에 포함되지 않는 지축·장암·
    # 하남·광명·성남 등의 역 데이터는 통합 작업본에서 제외한다.
    combined = combined[
        combined["자치구"].notna() & combined["행정동"].notna()
    ].reset_index(drop=True)
    combined = combined.drop(columns=DROP_COLUMNS, errors="ignore")
    combined = combined[["데이터기준일자", *[c for c in combined.columns if c != "데이터기준일자"]]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"자동심장충격기: {len(public_final):,}행, 행정동 매핑 {public_final['행정동'].notna().sum():,}행")
    print(f"안전한동네 AED: {len(subway_final):,}행, 역 좌표 매칭 {subway_mid['역좌표_매칭여부'].sum():,}행, "
          f"서울 행정동 매핑 {subway_final['행정동'].notna().sum():,}행")
    print(f"AED 서울 상세 통합: {len(combined):,}행 -> {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-input", type=Path, default=Path("data/raw/안전/생활안전/자동심장충격기.csv"))
    parser.add_argument("--subway-input", type=Path, default=Path("data/raw/안전/생활안전/안전한동네_AED.xlsx"))
    parser.add_argument("--station-coordinates", type=Path, default=Path("ml/etl/living_safety/station_coordinates_20250814.csv"))
    parser.add_argument("--boundaries", type=Path, default=Path("frontend/public/seoul-dong.geojson"))
    parser.add_argument("--output", type=Path, default=Path("data/interim/AED_통합_작업.csv"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
