"""서울시 소비자물가지수·물가상승률 자료를 분석하기 쉽게 정리한다."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl.styles import Border, Font, Side


ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "기타" / "소비자물가지수_물가상승률_2015_205.csv"
OUTPUT = ROOT / "data" / "interim" / "소비자물가지수_물가상승률_2015_2025_전처리.xlsx"

RENAME = {
    "소비자물가지수(2020=100)": "소비자물가지수(2020년=100)",
    "소비자물가_전년대비(%)": "소비자물가 전년 대비 상승률(%)",
    "식료품·비주류음료지수(2020=100)": "식료품·비주류음료 지수(2020년=100)",
    "식료품·비주류음료_전년대비(%)": "식료품·비주류음료 전년 대비 상승률(%)",
    "주택·수도·전기·연료지수(2020=100)": "주택·수도·전기·연료 지수(2020년=100)",
    "주택·수도·전기·연료_전년대비(%)": "주택·수도·전기·연료 전년 대비 상승률(%)",
    "교육지수(2020=100)": "교육 지수(2020년=100)",
    "교육_전년대비(%)": "교육 전년 대비 상승률(%)",
    "교통지수(2020=100)": "교통 지수(2020년=100)",
    "교통_전년대비(%)": "교통 전년 대비 상승률(%)",
}


def format_sheet(worksheet) -> None:
    thin = Side(style="thin", color="D9D9D9")
    worksheet.freeze_panes = "A2"
    for cell in worksheet[1]:
        cell.font = Font(bold=True)
    for row in worksheet.iter_rows():
        for cell in row:
            cell.border = Border(left=thin, right=thin, top=thin, bottom=thin)
    for column in worksheet.columns:
        width = min(max(len(str(cell.value or "")) for cell in column) + 2, 55)
        worksheet.column_dimensions[column[0].column_letter].width = width


def main() -> None:
    raw = pd.read_csv(RAW, encoding="utf-8-sig")
    if raw["연도"].duplicated().any():
        raise ValueError("연도 중복 행이 있습니다.")
    if sorted(raw["연도"].tolist()) != list(range(2015, 2026)):
        raise ValueError("2015~2025년 연속 자료가 아닙니다.")

    source = raw[["연도", "출처_비고", "출처_URL"]].rename(
        columns={"출처_비고": "출처 비고", "출처_URL": "출처 URL"}
    )
    data = raw.drop(columns=["출처_비고", "출처_URL"]).rename(columns=RENAME)
    data = data.sort_values("연도").reset_index(drop=True)

    notes = pd.DataFrame(
        {
            "항목": ["자료 범위", "지역 단위", "지수 기준", "2015년 결측", "출처"],
            "설명": [
                "2015~2025년 연간 자료",
                "서울특별시 전체이며 자치구별 자료가 아님",
                "2020년 연평균 물가 수준을 100으로 둔 지수",
                "2015년은 전체 소비자물가 상승률만 제공되어 나머지 값은 빈칸으로 유지",
                "연도별 원자료 비고와 URL은 '출처' 시트에 보존",
            ],
        }
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        data.to_excel(writer, index=False, sheet_name="데이터")
        source.to_excel(writer, index=False, sheet_name="출처")
        notes.to_excel(writer, index=False, sheet_name="전처리안내")
        for worksheet in writer.book.worksheets:
            format_sheet(worksheet)

    print(f"저장: {OUTPUT}")
    print(f"행 {len(data)}, 컬럼 {len(data.columns)}, 기간 {data['연도'].min()}~{data['연도'].max()}")


if __name__ == "__main__":
    main()
