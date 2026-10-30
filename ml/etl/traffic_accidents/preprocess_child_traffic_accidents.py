"""서울시 어린이 교통사고 패널의 최소 전처리본을 만든다."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl.styles import Border, Font, Side


ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "범죄&사고" / "어린이교통사고_2010_2025.xlsx"
OUTPUT = ROOT / "data" / "interim" / "어린이교통사고_2010_2025_전처리.xlsx"


def main() -> None:
    data = pd.read_excel(RAW, sheet_name="데이터", dtype={"법정동코드": str})

    # 원자료에 값이 하나도 없는 컬럼만 제거한다.
    empty_columns = data.columns[data.isna().all()].tolist()
    data = data.drop(columns=empty_columns)

    key = ["연도", "법정동코드", "사고유형"]
    if data.duplicated(key).any():
        raise ValueError("연도·지역·사고유형 중복 행이 있습니다.")

    # 원자료 안의 값만으로 정의가 명확한 지표를 추가한다.
    data["사상자수"] = data["사망자수"] + data["부상자수"]
    data = data.sort_values(["법정동코드", "사고유형", "연도"], kind="stable")
    groups = data.groupby(["법정동코드", "사고유형"], sort=False)
    previous = groups["사고건수"].shift(1)
    previous_year = groups["연도"].shift(1)
    consecutive = data["연도"].sub(previous_year).eq(1)
    data["전년 대비 사고건수 증감(건)"] = (data["사고건수"] - previous).where(consecutive)
    data["전년 대비 사고건수 증감률(%)"] = (
        (data["사고건수"] - previous).div(previous).mul(100).round(2)
    ).where(consecutive & previous.ne(0))
    data = data.sort_values(["연도", "법정동코드", "사고유형"], kind="stable").reset_index(drop=True)

    notes = pd.DataFrame(
        {
            "항목": [
                "원자료",
                "삭제 컬럼",
                "원자료 미수록",
                "사상자수",
                "전년 대비 변화",
                "지역 코드",
            ],
            "설명": [
                RAW.name,
                f"전 행이 빈칸인 {len(empty_columns)}개 컬럼: " + ", ".join(empty_columns),
                "35행은 원자료에 수록되지 않은 값이므로 0으로 바꾸지 않고 빈칸으로 유지",
                "사망자수 + 부상자수",
                "동일 지역·사고유형의 직전 연도와 비교. 전년 값이 0이면 증감률은 빈칸",
                "법정동코드라는 원본 컬럼명을 유지했으나 행정동 코드가 아니라 도로교통공단 지역 코드",
            ],
        }
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        data.to_excel(writer, index=False, sheet_name="데이터")
        notes.to_excel(writer, index=False, sheet_name="전처리안내")
        workbook = writer.book
        thin = Side(style="thin", color="D9D9D9")
        for worksheet in workbook.worksheets:
            worksheet.freeze_panes = "A2"
            for cell in worksheet[1]:
                cell.font = Font(bold=True)
            for row in worksheet.iter_rows():
                for cell in row:
                    cell.border = Border(left=thin, right=thin, top=thin, bottom=thin)
            for column in worksheet.columns:
                width = min(max(len(str(cell.value or "")) for cell in column) + 2, 45)
                worksheet.column_dimensions[column[0].column_letter].width = width

    print(f"저장: {OUTPUT}")
    print(f"행 {len(data):,}, 컬럼 {len(data.columns)}, 전체 빈 컬럼 {len(empty_columns)}개 제거")


if __name__ == "__main__":
    main()
