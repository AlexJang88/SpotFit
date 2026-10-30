"""서울시 소음·진동 민원 통계를 분석용 패널로 전처리한다."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
from openpyxl.styles import Border, Font, Side


ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "민원" / "서울시 소음·진동 민원 현황 통계_2006_2025.csv"
OUTPUT = ROOT / "data" / "interim" / "서울시_소음진동_민원현황_2006_2025_전처리.xlsx"
METRICS = ["환경관련 전체민원", "소음·진동 민원", "공장", "교통", "생활"]


def number(value: str) -> int:
    value = value.strip().replace(",", "")
    return 0 if value == "-" else int(value)


def ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.where(denominator.ne(0))).mul(100).round(2)


def add_ratios(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["환경민원 중 소음·진동 비율(%)"] = ratio(
        result["소음·진동 민원"], result["환경관련 전체민원"]
    )
    for category in ["공장", "교통", "생활"]:
        result[f"소음·진동 민원 중 {category} 비율(%)"] = ratio(
            result[category], result["소음·진동 민원"]
        )
    return result


def add_yearly_changes(data: pd.DataFrame, region_column: str) -> pd.DataFrame:
    result = data.sort_values([region_column, "연도"], kind="stable").copy()
    previous_count = result.groupby(region_column, sort=False)["소음·진동 민원"].shift(1)
    previous_share = result.groupby(region_column, sort=False)[
        "환경민원 중 소음·진동 비율(%)"
    ].shift(1)
    result["소음·진동 민원 전년 대비 증감(건)"] = result["소음·진동 민원"] - previous_count
    result["소음·진동 민원 전년 대비 증감률(%)"] = (
        result["소음·진동 민원"].sub(previous_count).div(previous_count).mul(100).round(2)
    ).where(previous_count.ne(0))
    result["환경민원 중 소음·진동 비율 전년 대비 증감(%p)"] = (
        result["환경민원 중 소음·진동 비율(%)"].sub(previous_share).round(2)
    )
    return result.sort_values(["연도", region_column], kind="stable").reset_index(drop=True)


def format_sheet(worksheet) -> None:
    thin = Side(style="thin", color="D9D9D9")
    worksheet.freeze_panes = "A2"
    for cell in worksheet[1]:
        cell.font = Font(bold=True)
    for row in worksheet.iter_rows():
        for cell in row:
            cell.border = Border(left=thin, right=thin, top=thin, bottom=thin)
    for column in worksheet.columns:
        width = min(max(len(str(cell.value or "")) for cell in column) + 2, 35)
        worksheet.column_dimensions[column[0].column_letter].width = width


def main() -> None:
    with RAW.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file))

    if len(rows[:3]) != 3 or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("원본 CSV의 헤더 또는 열 개수가 예상과 다릅니다.")

    records: list[dict[str, object]] = []
    for row in rows[3:]:
        region = row[1].strip()
        for year_index, year in enumerate(range(2006, 2026)):
            values = [number(value) for value in row[2 + year_index * 5 : 7 + year_index * 5]]
            if values[1] != sum(values[2:]):
                raise ValueError(f"소음·진동 민원 합계 불일치: {year}, {region}")
            records.append({"연도": year, "자치구": region, **dict(zip(METRICS, values))})

    data = pd.DataFrame(records)
    seoul_total = data[data["자치구"].eq("소계")].rename(columns={"자치구": "지역"}).reset_index(drop=True)
    districts = data[~data["자치구"].eq("소계")].reset_index(drop=True)

    if len(districts) != 500 or districts.duplicated(["연도", "자치구"]).any():
        raise ValueError("25개 자치구×20년 패널 구조가 아닙니다.")
    check = districts.groupby("연도", as_index=False)[METRICS].sum()
    if not check[METRICS].equals(seoul_total[METRICS]):
        raise ValueError("서울시 소계가 25개 자치구 합계와 일치하지 않습니다.")

    districts = add_ratios(districts)
    seoul_total = add_ratios(seoul_total)
    districts = add_yearly_changes(districts, "자치구")
    seoul_total = add_yearly_changes(seoul_total, "지역")

    for frame in [districts, seoul_total]:
        if (frame[METRICS] < 0).any().any():
            raise ValueError("민원 건수에 음수가 있습니다.")
        if frame["환경민원 중 소음·진동 비율(%)"].gt(100).any():
            raise ValueError("소음·진동 민원이 환경관련 전체민원보다 큰 행이 있습니다.")
        component_share = frame[
            [
                "소음·진동 민원 중 공장 비율(%)",
                "소음·진동 민원 중 교통 비율(%)",
                "소음·진동 민원 중 생활 비율(%)",
            ]
        ].sum(axis=1)
        if component_share.sub(100).abs().gt(0.02).any():
            raise ValueError("공장·교통·생활 구성비 합계가 100%와 다릅니다.")

    notes = pd.DataFrame(
        {
            "항목": ["표 구조", "자치구 자료", "서울시 전체", "원본의 '-'", "구성비", "전년 대비 변화"],
            "설명": [
                "가로형 연도 컬럼을 연도×지역의 세로형 자료로 변환",
                "25개 자치구×20년 = 500행",
                "원본 소계 20행을 별도 시트에 보존",
                "공장·교통 항목의 '-'는 합계식과 지역 합계 검증을 거쳐 0으로 변환",
                "환경민원 중 소음·진동 비율과 소음·진동 민원의 공장·교통·생활 구성비",
                "소음·진동 민원의 증감 건수·증감률 및 환경민원 중 비율의 증감(%p)",
            ],
        }
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        districts.to_excel(writer, index=False, sheet_name="자치구별")
        seoul_total.to_excel(writer, index=False, sheet_name="서울시전체")
        notes.to_excel(writer, index=False, sheet_name="전처리안내")
        for worksheet in writer.book.worksheets:
            format_sheet(worksheet)

    print(f"저장: {OUTPUT}")
    print(f"자치구별 {len(districts):,}행, 서울시전체 {len(seoul_total):,}행")


if __name__ == "__main__":
    main()
