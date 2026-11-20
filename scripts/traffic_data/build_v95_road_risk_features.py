"""Aggregate official 2018-2025 road-link accident risk data to Seoul districts."""
from __future__ import annotations

import io
from pathlib import Path
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[2]
ZIP = next((ROOT / "data/95차_추가사용데이터/raw/한국도로교통공단_도로위험도").glob("*도로위험도*.zip"))
BOUNDARY = ROOT / "frontend/public/seoul-gu.geojson"
OUT = ROOT / "data/95차_추가사용데이터/interim/자치구별_도로위험도_2018_2025.csv"
NUMERIC = [
    "사망사고건수", "중상사고건수", "경상사고건수", "부상신고사고건수",
    "사망자수", "중상자수", "경상자수", "부상신고자수",
    "세분화된 링크의 길이(m)", "도로위험도 지수 변환 값",
]


def main() -> None:
    districts = gpd.read_file(BOUNDARY).rename(columns={"name": "자치구"})[["자치구", "geometry"]]
    districts = districts.set_crs(4326)
    transformer = Transformer.from_crs(5179, 4326, always_xy=True)
    rows = []
    with ZipFile(ZIP) as archive:
        for member in archive.namelist():
            year = int(Path(member).stem[-4:])
            pieces = []
            with archive.open(member) as binary:
                stream = io.TextIOWrapper(binary, encoding="cp949", newline="")
                for chunk in pd.read_csv(stream, chunksize=100_000):
                    geometry = shapely.from_wkt(
                        chunk["세분화된 링크 값"].fillna("LINESTRING EMPTY").to_numpy(), on_invalid="ignore"
                    )
                    centers = shapely.centroid(geometry)
                    valid = ~(shapely.is_missing(centers) | shapely.is_empty(centers))
                    x = np.full(len(centers), np.nan)
                    y = np.full(len(centers), np.nan)
                    x[valid], y[valid] = shapely.get_x(centers[valid]), shapely.get_y(centers[valid])
                    lon, lat = transformer.transform(x, y)
                    # 서울 외 전국 링크 대부분을 공간조인 전에 제거한다.
                    mask = (lon >= 126.75) & (lon <= 127.20) & (lat >= 37.40) & (lat <= 37.72)
                    if not np.any(mask):
                        continue
                    part = chunk.loc[mask, ["세분화링크아이디", *NUMERIC]].copy()
                    for column in NUMERIC:
                        part[column] = pd.to_numeric(part[column], errors="coerce").fillna(0)
                    points = gpd.GeoDataFrame(
                        part, geometry=gpd.points_from_xy(np.asarray(lon)[mask], np.asarray(lat)[mask]), crs=4326
                    )
                    joined = gpd.sjoin(points, districts, how="inner", predicate="within").drop(columns="index_right")
                    if not joined.empty:
                        pieces.append(pd.DataFrame(joined.drop(columns="geometry")))
            if not pieces:
                raise ValueError(f"서울 링크를 찾지 못했습니다: {member}")
            frame = pd.concat(pieces, ignore_index=True)
            frame["사고건수합계"] = frame[["사망사고건수", "중상사고건수", "경상사고건수", "부상신고사고건수"]].sum(axis=1)
            frame["사상자수합계"] = frame[["사망자수", "중상자수", "경상자수", "부상신고자수"]].sum(axis=1)
            frame["위험도길이가중"] = frame["도로위험도 지수 변환 값"] * frame["세분화된 링크의 길이(m)"]
            grouped = frame.groupby("자치구", as_index=False).agg(
                도로위험_링크수=("세분화링크아이디", "nunique"),
                도로위험_사고링크수=("사고건수합계", lambda x: int((x > 0).sum())),
                도로위험_사고건수=("사고건수합계", "sum"),
                도로위험_사상자수=("사상자수합계", "sum"),
                도로위험_사망사고건수=("사망사고건수", "sum"),
                도로위험_중상사고건수=("중상사고건수", "sum"),
                도로위험_평균지수=("도로위험도 지수 변환 값", "mean"),
                도로위험_최대지수=("도로위험도 지수 변환 값", "max"),
                도로위험_링크길이합계=("세분화된 링크의 길이(m)", "sum"),
                도로위험_위험도길이가중합=("위험도길이가중", "sum"),
            )
            grouped["도로위험_길이가중평균지수"] = (
                grouped["도로위험_위험도길이가중합"] / grouped["도로위험_링크길이합계"].replace(0, np.nan)
            )
            grouped["도로위험_사고링크비율"] = grouped["도로위험_사고링크수"] / grouped["도로위험_링크수"].replace(0, np.nan)
            grouped.insert(0, "연도", year)
            rows.append(grouped.drop(columns="도로위험_위험도길이가중합"))
            print(year, "links", len(frame), "districts", grouped["자치구"].nunique())
    result = pd.concat(rows, ignore_index=True).sort_values(["연도", "자치구"])
    if result.groupby("연도")["자치구"].nunique().ne(25).any():
        raise ValueError(f"자치구 누락:\n{result.groupby('연도')['자치구'].nunique()}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False, encoding="utf-8-sig")
    print("saved", OUT.relative_to(ROOT), result.shape)


if __name__ == "__main__":
    main()
