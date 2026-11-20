"""TAAS 서울(고속국도 제외) 기간별 엑셀을 하나의 연도별 패널로 합친다."""

from __future__ import annotations

from pathlib import Path
import warnings

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Border, Font, Side


ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw" / "안전" / "교통사고"
OUTPUT = ROOT / "data" / "interim" / "TAAS_서울_고속국도제외_2010_2025_전처리.xlsx"
METRIC_NAMES = {"사고[건]": "사고건수", "사망[명]": "사망자수", "부상[명]": "부상자수"}


def read_source(path: Path) -> list[dict[str, object]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        workbook = load_workbook(path, read_only=True, data_only=True)
    worksheet = workbook.active
    # TAAS 파일의 dimension이 A1로 잘못 기록돼 있어 실제 사용 범위를 다시 계산한다.
    worksheet.reset_dimensions()
    rows = list(worksheet.iter_rows(values_only=True))
    header = [str(value) if value is not None else "" for value in rows[0]]
    year_columns = [(index, int(value)) for index, value in enumerate(header) if value.isdigit()]

    records: list[dict[str, object]] = []
    for row in rows[1:]:
        if len(row) < 3 or row[2] not in METRIC_NAMES:
            continue
        for column, year in year_columns:
            records.append(
                {
                    "시도": row[0],
                    "자치구": row[1],
                    "연도": year,
                    "지표": METRIC_NAMES[row[2]],
                    "값": int(row[column]),
                    "원본파일": path.name,
                }
            )
    return records


def format_sheet(worksheet) -> None:
    thin = Side(style="thin", color="D9D9D9")
    worksheet.freeze_panes = "A2"
    for cell in worksheet[1]:
        cell.font = Font(bold=True)
    for row in worksheet.iter_rows():
        for cell in row:
            cell.border = Border(left=thin, right=thin, top=thin, bottom=thin)
    for column in worksheet.columns:
        width = min(max(len(str(cell.value or "")) for cell in column) + 2, 45)
        worksheet.column_dimensions[column[0].column_letter].width = width


def main() -> None:
    files = sorted(RAW_DIR.glob("TAAS_서울_고속국도제외_*.xlsx"))
    if len(files) != 4:
        raise ValueError(f"원본 파일이 4개가 아닙니다: {len(files)}개")

    records: list[dict[str, object]] = []
    for path in files:
        records.extend(read_source(path))
    long = pd.DataFrame(records)

    if long.duplicated(["시도", "자치구", "연도", "지표"]).any():
        raise ValueError("기간 파일 사이에 중복된 지역·연도·지표가 있습니다.")

    data = (
        long.pivot(index=["시도", "자치구", "연도"], columns="지표", values="값")
        .reset_index()
        .rename_axis(columns=None)
    )
    data = data[["시도", "자치구", "연도", "사고건수", "사망자수", "부상자수"]]
    data = data.sort_values(["연도", "자치구"], kind="stable").reset_index(drop=True)

    if len(data) != 416 or data["연도"].nunique() != 16 or data["자치구"].nunique() != 26:
        raise ValueError("16년×26지역 패널 구조가 아닙니다.")
    district = data[~data["자치구"].eq("합계")].copy()
    seoul = data[data["자치구"].eq("합계")].copy()
    seoul_check = seoul.set_index("연도")
    summed = district.groupby("연도")[["사고건수", "사망자수", "부상자수"]].sum()
    if not summed.equals(seoul_check[["사고건수", "사망자수", "부상자수"]]):
        raise ValueError("서울 합계가 25개 자치구 합계와 다릅니다.")

    for frame in [district, seoul]:
        frame["사상자수"] = frame["사망자수"] + frame["부상자수"]
        frame["치사율(%)"] = frame["사망자수"].div(frame["사고건수"]).mul(100).round(2)
    seoul["자치구"] = "서울 전체"

    source_map = long[["연도", "원본파일"]].drop_duplicates().sort_values("연도")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        district.to_excel(writer, index=False, sheet_name="자치구별")
        seoul.to_excel(writer, index=False, sheet_name="서울전체")
        source_map.to_excel(writer, index=False, sheet_name="원본파일")
        for worksheet in writer.book.worksheets:
            format_sheet(worksheet)

    print(f"저장: {OUTPUT}")
    print(f"자치구별 {len(district):,}행, 서울전체 {len(seoul):,}행, 기간 {data['연도'].min()}~{data['연도'].max()}")


if __name__ == "__main__":
    main()
