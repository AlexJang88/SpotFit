"""2018~2023 서울 자치구 범죄안전 대체지수 전처리."""
from __future__ import annotations

import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "안전" / "범죄"
OUT = ROOT / "data" / "interim" / "안전" / "범죄안전지수"
YEARS = list(range(2018, 2024))
DISTRICTS = ("종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구",
             "강북구", "도봉구", "노원구", "은평구", "서대문구", "마포구", "양천구", "강서구",
             "구로구", "금천구", "영등포구", "동작구", "관악구", "서초구", "강남구", "송파구", "강동구")
POLICE_MAP = {
    "중부": "중구", "남대문": "중구", "종로": "종로구", "혜화": "종로구", "용산": "용산구",
    "성동": "성동구", "광진": "광진구", "동대문": "동대문구", "중랑": "중랑구", "성북": "성북구",
    "종암": "성북구", "강북": "강북구", "도봉": "도봉구", "노원": "노원구", "은평": "은평구",
    "서부": "은평구", "서대문": "서대문구", "마포": "마포구", "양천": "양천구", "강서": "강서구",
    "구로": "구로구", "금천": "금천구", "영등포": "영등포구", "동작": "동작구", "관악": "관악구",
    "서초": "서초구", "방배": "서초구", "강남": "강남구", "수서": "강남구", "송파": "송파구", "강동": "강동구",
}
WEIGHTS = {
    # 의식 20은 가용 구분의 원래 총비율 위해:취약:경감=50:10:20으로 재배분한다.
    # 최종 구분 총가중치: 위해 62.5, 취약 12.5, 경감 25.0.
    "5대범죄_인구1만명당": ("위해", 50.0, 62.5, "위험"),
    "집객시설_인구1만명당": ("취약", 3.04, 3.80, "위험"),
    "주점업체_인구1만명당": ("취약", 5.26, 6.575, "위험"),
    "스트레스인지율": ("취약", 1.70, 2.125, "위험"),
    "경찰관서_인구1만명당": ("경감", 17.29, 25.0, "보호"),
}
EXCLUDED = [
    ("경감", "범죄예방_CCTV_인구1만명당", 2.71, "25개 자치구 6개년 완전 자료가 아님"),
    ("의식", "아동안전지킴이집_인구1만명당", 3.06, "2018~2023 자치구별 자료 미확보"),
    ("의식", "자율방범대원_인구1만명당", 16.94, "현재 자료가 2015·2023뿐이라 제외"),
]

# 공공용 연간자료는 산업 소분류까지만 제공한다. 공식 세·세세분류 지정코드를
# 포함하는 상위 소분류를 대체치로 사용한다(G4711→G471 등).
FACILITY_CODES = ("G471", "J591", "Q861", "O841", "K641", "P853")
DISTRICT_CODE_MAP = {
    "010": "종로구", "020": "중구", "030": "용산구", "040": "성동구", "050": "광진구",
    "060": "동대문구", "070": "중랑구", "080": "성북구", "090": "강북구", "100": "도봉구",
    "110": "노원구", "120": "은평구", "130": "서대문구", "140": "마포구", "150": "양천구",
    "160": "강서구", "170": "구로구", "180": "금천구", "190": "영등포구", "200": "동작구",
    "210": "관악구", "220": "서초구", "230": "강남구", "240": "송파구", "250": "강동구",
}


def csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            pass
    raise UnicodeError(path)


def population() -> pd.DataFrame:
    return csv(RAW / "서울시_자치구_연도별_등록인구.csv").rename(
        columns={"통계연도": "연도", "구명": "자치구"})[["연도", "자치구", "인구수"]]


def crime() -> pd.DataFrame:
    return csv(RAW / "범죄_2014_2025.csv").rename(
        columns={"소계_발생건": "5대범죄_발생건수"})[["연도", "자치구", "5대범죄_발생건수"]]


def bars() -> pd.DataFrame:
    return csv(RAW / "파생" / "유흥_단란주점_자치구_연도별_2010_2026.csv").rename(
        columns={"유흥_단란주점_합계": "주점업체수"})[["연도", "자치구", "주점업체수"]]


def gathering_facilities() -> pd.DataFrame:
    """전국사업체조사 항목 2·3, 10~14를 이용해 집객시설 사업체 수를 집계한다."""
    derived = RAW / "파생" / "집객시설_자치구_연도별_2018_2023.csv"
    if derived.exists():
        cached = csv(derived)
        if "집계기준" in cached and cached["집계기준"].eq("공공용_산업소분류대체").all():
            return cached
    rows = []
    for year in YEARS:
        path = next((RAW / "집객시설").glob(f"{year}_연간자료_*.csv"))
        counts = pd.Series(0, index=DISTRICTS, dtype="int64")
        for chunk in pd.read_csv(path, header=None, dtype=str, encoding="utf-8-sig",
                                 usecols=[1, 2, 9, 10, 11], chunksize=250_000,
                                 low_memory=False):
            chunk = chunk[chunk[1].eq("11")]
            if chunk.empty:
                continue
            code = chunk[9].fillna("") + chunk[10].fillna("") + chunk[11].fillna("")
            selected = chunk[code.isin(FACILITY_CODES)]
            districts = selected[2].str.zfill(3).map(DISTRICT_CODE_MAP).dropna()
            counts = counts.add(districts.value_counts(), fill_value=0).astype("int64")
        rows.extend({"연도": year, "자치구": district, "집객시설수": int(counts[district]),
                     "집계기준": "공공용_산업소분류대체"}
                    for district in DISTRICTS)
        print(f"집객시설 {year}: {int(counts.sum()):,}개")
    result = pd.DataFrame(rows)
    derived.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(derived, index=False, encoding="utf-8-sig")
    return result


def number(value: object) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", str(value or ""))
    return float(match.group()) if match else np.nan


def stress() -> pd.DataFrame:
    ws = load_workbook(RAW / "2008-2025_스트레스 인지율.xlsx", read_only=True, data_only=True).worksheets[0]
    year_cols = {int(ws.cell(12, c).value): c for c in range(1, ws.max_column + 1)
                 if isinstance(ws.cell(12, c).value, (int, float))}
    rows, province = [], None
    for r in range(14, ws.max_row + 1):
        if ws.cell(r, 3).value:
            province = str(ws.cell(r, 3).value).strip()
        district = ws.cell(r, 4).value
        if province == "서울" and district in DISTRICTS:
            for year in YEARS:
                rows.append({"연도": year, "자치구": district,
                             "스트레스인지율": number(ws.cell(r, year_cols[year] + 2).value)})
    return pd.DataFrame(rows)


def police() -> pd.DataFrame:
    frame = csv(RAW / "경찰관서" / "서울시_경찰서별_지역경찰현황_2018_2023.csv")
    frame = frame.rename(columns={"년도": "연도", "경찰수": "관서수"})
    frame["자치구"] = frame["경찰서"].map(POLICE_MAP)
    if frame["자치구"].isna().any():
        raise ValueError("경찰서-자치구 매핑 누락")
    return frame.groupby(["연도", "자치구"], as_index=False)["관서수"].sum().rename(columns={"관서수": "경찰관서수"})


def complete(frame: pd.DataFrame, value: str, name: str) -> pd.DataFrame:
    frame = frame[frame["연도"].isin(YEARS) & frame["자치구"].isin(DISTRICTS)].copy()
    frame[value] = pd.to_numeric(frame[value], errors="coerce")
    counts = frame.dropna(subset=[value]).groupby("연도")["자치구"].nunique().to_dict()
    bad = {y: counts.get(y, 0) for y in YEARS if counts.get(y, 0) != 25}
    if bad or frame.duplicated(["연도", "자치구"]).any():
        raise ValueError(f"{name} 완전성 오류: {bad}")
    return frame


def main() -> int:
    sources = [(population(), "인구수", "인구"), (crime(), "5대범죄_발생건수", "5대범죄"),
               (gathering_facilities(), "집객시설수", "집객시설"),
               (bars(), "주점업체수", "주점업"), (stress(), "스트레스인지율", "스트레스"),
               (police(), "경찰관서수", "경찰관서")]
    result = pd.MultiIndex.from_product([YEARS, DISTRICTS], names=["연도", "자치구"]).to_frame(index=False)
    for frame, value, name in sources:
        result = result.merge(complete(frame, value, name), on=["연도", "자치구"], how="left", validate="one_to_one")
    divisor = result["인구수"] / 10_000
    result["5대범죄_인구1만명당"] = result["5대범죄_발생건수"] / divisor
    result["집객시설_인구1만명당"] = result["집객시설수"] / divisor
    result["주점업체_인구1만명당"] = result["주점업체수"] / divisor
    result["경찰관서_인구1만명당"] = result["경찰관서수"] / divisor

    weight_rows = []
    for indicator, (category, original, adjusted, direction) in WEIGHTS.items():
        mean = result.groupby("연도")[indicator].transform("mean")
        std = result.groupby("연도")[indicator].transform(lambda x: x.std(ddof=0))
        safety_z = (result[indicator] - mean) / std * (-1 if direction == "위험" else 1)
        result[f"{indicator}_안전방향표준점수"] = safety_z
        result[f"{indicator}_가중기여도"] = safety_z * adjusted
        weight_rows.append((category, indicator, original, adjusted, direction))
    result["범죄안전_대체점수"] = result[[c for c in result if c.endswith("_가중기여도")]].sum(axis=1)
    result["서울25개구_순위"] = result.groupby("연도")["범죄안전_대체점수"].rank(ascending=False, method="first").astype(int)
    result["상대안전등급"] = ((result["서울25개구_순위"] - 1) // 5 + 1).astype(int)
    result["상대안전등급_문자"] = result["상대안전등급"].map(dict(enumerate("ABCDE", 1)))
    result["공식지역안전등급여부"] = "아님_대체지표기반"
    weights = pd.DataFrame(weight_rows, columns=["구분", "산출지표", "원래가중치", "조정가중치", "방향"])
    assert np.isclose(weights["조정가중치"].sum(), 100)

    OUT.mkdir(parents=True, exist_ok=True)
    result = result.sort_values(["연도", "서울25개구_순위"])
    result.to_csv(OUT / "서울시_자치구_범죄안전_대체지수_2018_2023.csv", index=False, encoding="utf-8-sig")
    weights.to_csv(OUT / "재설정_가중치_2018_2023.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(EXCLUDED, columns=["구분", "제외지표", "원래가중치", "제외사유"]).to_csv(
        OUT / "제외지표_2018_2023.csv", index=False, encoding="utf-8-sig")
    meta = {"기간": YEARS, "자치구수": 25, "행수": len(result), "조정가중치합": 100,
            "산식": "연도별 z-score→위험지표 부호 반전→조정가중치 곱→합산",
            "등급": "연도별 5개 구씩 A~E", "공식등급": False,
            "집객시설_집계기준": "공공용 산업소분류 대체(G471,J591,Q861,O841,K641,P853)",
            "집객시설_세부코드한계": "실제 CSV에 세·세세분류가 없어 지정 세부코드의 상위 소분류 전체 사용",
            "결측구분_재배분": "의식 20을 위해:취약:경감=50:10:20 비율로 재배분",
            "최종_구분가중치": {"위해": 62.5, "취약": 12.5, "경감": 25.0, "의식": 0.0}}
    (OUT / "처리메타_2018_2023.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
