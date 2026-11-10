from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "학교별_일람표" / "상반기"
OUT = ROOT / "data" / "interim" / "안전" / "자치구별_학교현황_2017_2025_최종.xlsx"
TARGET_TYPES = ["유치원", "초등학교"]
SEOUL_DISTRICTS = [
    "종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구", "강북구",
    "도봉구", "노원구", "은평구", "서대문구", "마포구", "양천구", "강서구", "구로구", "금천구",
    "영등포구", "동작구", "관악구", "서초구", "강남구", "송파구", "강동구",
]


def number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def year_from_name(path: Path) -> int:
    match = re.search(r"20[0-9]{2}", path.name)
    if not match:
        raise ValueError(f"연도를 찾을 수 없습니다: {path.name}")
    return int(match.group())


def old_format(path: Path, year: int) -> pd.DataFrame:
    layouts = {
        "유치원": {"district": 1, "name": 2, "establishment": 3, "address": 5, "classes": 7,
                "students": 8 if year == 2017 else 9},
        "초등학교": {"district": 1, "name": 2, "establishment": 3, "address": 6, "classes": 16, "students": 33},
    }
    outputs = []
    for school_type, layout in layouts.items():
        raw = pd.read_excel(path, sheet_name=school_type, header=None)
        data = pd.DataFrame({
            "연도": year,
            "서울시": "서울특별시",
            "자치구": raw.iloc[:, layout["district"]],
            "학교급": school_type,
            "학교명": raw.iloc[:, layout["name"]],
            "설립": raw.iloc[:, layout["establishment"]],
            "주소": raw.iloc[:, layout["address"]],
            "학급수": number(raw.iloc[:, layout["classes"]]),
            "학생수": number(raw.iloc[:, layout["students"]]),
        })
        data = data[data["자치구"].isin(SEOUL_DISTRICTS) & data["학교명"].notna()].copy()
        data["학교명"] = data["학교명"].astype(str).str.strip()
        outputs.append(data)
    return pd.concat(outputs, ignore_index=True)


def find_header(raw: pd.DataFrame) -> int:
    for index, value in enumerate(raw.iloc[:, 0].tolist()):
        if str(value).strip() == "조사기준일":
            return index
    raise ValueError("통합 시트에서 헤더 행을 찾지 못했습니다.")


def new_format(path: Path, year: int) -> pd.DataFrame:
    sheet = "학교별 주요통계"
    preview = pd.read_excel(path, sheet_name=sheet, header=None)
    header = find_header(preview)
    raw = pd.read_excel(path, sheet_name=sheet, header=header)
    raw.columns = [str(column).replace("\n", "").strip() for column in raw.columns]
    rename = {
        "행정구": "자치구",
        "학생수_총계_계": "학생수",
        "편성학급수_계": "학급수",
    }
    raw = raw.rename(columns=rename)
    required = ["시도", "자치구", "학교급", "학교명", "본분교", "설립", "상태", "주소", "학교수", "학급수", "학생수"]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"{path.name} 필수 컬럼 없음: {missing}")
    data = raw[
        raw["시도"].astype(str).str.contains("서울", na=False)
        & raw["자치구"].isin(SEOUL_DISTRICTS)
        & raw["학교급"].isin(TARGET_TYPES)
        & raw["본분교"].astype(str).str.contains("본교", na=False)
        & (number(raw["학교수"]) == 1)
    ].copy()
    data = data.rename(columns={"학교급": "학교급"})
    data["연도"] = year
    data["서울시"] = "서울특별시"
    data["학급수"] = number(data["학급수"])
    data["학생수"] = number(data["학생수"])
    return data[["연도", "서울시", "자치구", "학교급", "학교명", "설립", "주소", "학급수", "학생수"]]


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    grid = pd.MultiIndex.from_product([range(2017, 2026), SEOUL_DISTRICTS], names=["연도", "자치구"]).to_frame(index=False)
    rows = []
    for (year, district), part in detail.groupby(["연도", "자치구"]):
        row = {"연도": year, "자치구": district}
        for school_type, prefix in [("유치원", "유치원"), ("초등학교", "초등학교")]:
            selected = part[part["학교급"] == school_type]
            row[f"{prefix}수"] = selected["학교명"].nunique()
            row[f"{prefix}학생수"] = selected["학생수"].sum(min_count=1)
            row[f"{prefix}학급수"] = selected["학급수"].sum(min_count=1)
            row[f"{prefix}공립수"] = selected[selected["설립"].astype(str).str.contains("공립|국립", regex=True)]["학교명"].nunique()
            row[f"{prefix}사립수"] = selected[selected["설립"].astype(str).str.contains("사립", regex=False)]["학교명"].nunique()
        rows.append(row)
    summary = grid.merge(pd.DataFrame(rows), on=["연도", "자치구"], how="left", validate="one_to_one")
    summary.insert(1, "서울시", "서울특별시")
    count_columns = [c for c in summary.columns if c.endswith("수")]
    summary[count_columns] = summary[count_columns].fillna(0)
    for column in count_columns:
        summary[column] = summary[column].round().astype("int64")
    summary["어린이대상교육기관수"] = summary["유치원수"] + summary["초등학교수"]
    summary["어린이대상교육기관학생수"] = summary["유치원학생수"] + summary["초등학교학생수"]
    summary["유치원평균학생수"] = (summary["유치원학생수"] / summary["유치원수"].replace(0, np.nan)).round(2)
    summary["초등학교평균학생수"] = (summary["초등학교학생수"] / summary["초등학교수"].replace(0, np.nan)).round(2)
    return summary.sort_values(["연도", "자치구"]).reset_index(drop=True)


def main() -> None:
    files = sorted(RAW.glob("*.xlsx"), key=year_from_name)
    selected_files = [path for path in files if 2017 <= year_from_name(path) <= 2025]
    years = [year_from_name(path) for path in selected_files]
    if years != list(range(2017, 2026)):
        raise ValueError(f"2017~2025 연도 파일 구성이 완전하지 않습니다: {years}")

    parts = []
    for path in selected_files:
        year = year_from_name(path)
        parts.append(old_format(path, year) if year <= 2020 else new_format(path, year))
    detail = pd.concat(parts, ignore_index=True)
    detail = detail.sort_values(["연도", "자치구", "학교급", "학교명"]).reset_index(drop=True)
    detail["학급수"] = detail["학급수"].fillna(0).round().astype("int64")
    detail["학생수"] = detail["학생수"].fillna(0).round().astype("int64")

    duplicate = detail.duplicated(["연도", "학교급", "학교명", "주소"], keep=False)
    if duplicate.any():
        raise ValueError(f"학교별 중복 행이 있습니다: {duplicate.sum()}행")
    if not detail["자치구"].isin(SEOUL_DISTRICTS).all():
        raise ValueError("서울 25개 자치구 외 자료가 있습니다.")
    summary = summarize(detail)
    if len(summary) != 225 or summary.duplicated(["연도", "자치구"]).any():
        raise ValueError("자치구별 요약이 225개 고유 행이 아닙니다.")

    yearly = detail.groupby(["연도", "학교급"], as_index=False).agg(
        학교수=("학교명", "size"), 학생수=("학생수", "sum"), 학급수=("학급수", "sum")
    )
    quality = pd.DataFrame([
        {"검사항목": "사용 연도", "결과": f"{detail['연도'].min()}~{detail['연도'].max()}"},
        {"검사항목": "학교별 상세 행 수", "결과": len(detail)},
        {"검사항목": "자치구별 요약 행 수", "결과": len(summary)},
        {"검사항목": "연도·자치구 중복", "결과": int(summary.duplicated(["연도", "자치구"]).sum())},
        {"검사항목": "서울 외 지역 행 수", "결과": int((~detail["자치구"].isin(SEOUL_DISTRICTS)).sum())},
        {"검사항목": "포함 학교급", "결과": ", ".join(TARGET_TYPES)},
        {"검사항목": "제외 자료", "결과": "2026년, 중·고·특수·각종학교, 분교, 폐교·휴교"},
        {"검사항목": "유의사항", "결과": "학급 기준이 2020년까지 인가학급, 2021년부터 편성학급으로 변경"},
    ])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="자치구별_모델용", index=False)
        detail.to_excel(writer, sheet_name="학교별_상세", index=False)
        yearly.to_excel(writer, sheet_name="연도별_검산", index=False)
        quality.to_excel(writer, sheet_name="품질검사", index=False)
    print(OUT)
    print(f"detail={len(detail)} summary={len(summary)} years={years}")
    print(yearly.to_string(index=False))


if __name__ == "__main__":
    main()
