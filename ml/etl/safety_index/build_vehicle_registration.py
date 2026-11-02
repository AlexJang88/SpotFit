from __future__ import annotations

import os
import re
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd


SEOUL_GU = {
    "종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구",
    "강북구", "도봉구", "노원구", "은평구", "서대문구", "마포구", "양천구", "강서구",
    "구로구", "금천구", "영등포구", "동작구", "관악구", "서초구", "강남구", "송파구", "강동구",
}


def gu_name(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    for gu in SEOUL_GU:
        if gu in text:
            return gu
    return None


def parse_legacy_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="cp949")
    df["자치구"] = df["시군구별"].map(gu_name)
    return (
        df[df["자치구"].notna()]
        .groupby(["연도", "자치구"], as_index=False)["계"]
        .sum()
        .rename(columns={"계": "자동차등록대수"})
    )


def parse_2020_2021(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, header=1)
    df["연도"] = pd.to_numeric(df["연월"], errors="coerce") // 100
    df["자치구"] = df["사용본거지시군구"].map(gu_name)
    return (
        df[df["자치구"].notna()]
        .groupby(["연도", "자치구"], as_index=False)["건수"]
        .sum()
        .rename(columns={"건수": "자동차등록대수"})
    )


def december_member(zip_path: Path, year: int) -> bytes:
    with zipfile.ZipFile(zip_path) as archive:
        candidates = []
        for name in archive.namelist():
            normalized = name.replace("\\", "/")
            if normalized.lower().endswith(".xlsx"):
                candidates.append(name)
        yy = str(year)[-2:]
        december = [
            n for n in candidates
            if re.search(rf"(?:{year}|{yy})[^0-9]?(?:12)(?:[^0-9]|$)", n)
        ]
        year_candidates = [n for n in candidates if yy in n or str(year) in n]
        chosen = (december or sorted(year_candidates or candidates)[-1:])[0]
        return archive.read(chosen)


def parse_detail_excel(source: Path | bytes, year: int) -> pd.DataFrame:
    df = pd.read_excel(BytesIO(source) if isinstance(source, bytes) else source, header=None)
    gu = df.iloc[:, 0].map(gu_name).ffill()
    values = pd.to_numeric(df.iloc[:, 15], errors="coerce")
    valid = gu.notna() & values.notna()
    out = pd.DataFrame({"연도": year, "자치구": gu[valid], "자동차등록대수": values[valid]})
    return out.groupby(["연도", "자치구"], as_index=False)["자동차등록대수"].sum()


def main() -> None:
    temp = Path(os.environ["TEMP"])
    legacy = temp / "vehicle_2005_2019_download"
    parts = [
        parse_legacy_csv(legacy),
        parse_2020_2021(temp / "vehicle_2020_2021.xlsx"),
        parse_detail_excel(december_member(temp / "vehicle_2022.zip", 2022), 2022),
        parse_detail_excel(december_member(temp / "vehicle_2023_2024Q1.zip", 2023), 2023),
        parse_detail_excel(temp / "vehicle_2024_12.xlsx", 2024),
        parse_detail_excel(temp / "vehicle_2025_12.xlsx", 2025),
    ]
    result = pd.concat(parts, ignore_index=True)
    result["연도"] = result["연도"].astype(int)
    result["자동차등록대수"] = result["자동차등록대수"].astype(int)
    result = result.query("2006 <= 연도 <= 2025").sort_values(["연도", "자치구"])
    output = Path("data/interim") / "교통안전_자치구별_자동차등록대수_2006_2025.csv"
    result.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"{output}: {len(result)} rows, {result['연도'].min()}-{result['연도'].max()}")
    print(result.groupby("연도")["자치구"].nunique().to_string())


if __name__ == "__main__":
    main()
