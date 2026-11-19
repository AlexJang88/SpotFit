"""Render the SpotFit dictionary as side-by-side per-table blocks, like PICK YOU DB구조."""
from __future__ import annotations

import json
import re
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from build_database_dictionary import ROOT, SNAPSHOT, TABLES, describe


OUTPUT = ROOT / "docs/SpotFit_DB_테이블별_정의서.xlsx"
HEADERS = ["No.", "컬럼ID", "컬럼명", "타입", "길이", "NullAble", "Default", "Key", "설명"]
BLOCK_STARTS = (2, 12)  # B:J, L:T, following the reference sheet.
FILL_TITLE = PatternFill("solid", fgColor="D8E7E2")
FILL_HEADER = PatternFill("solid", fgColor="F3F3F3")
FILL_ALT = PatternFill("solid", fgColor="F7FAF8")
THIN = Side(style="thin", color="B5C9C2")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def sql_type_parts(raw: str) -> tuple[str, str]:
    match = re.match(r"^(.*?)\((.*?)\)$", raw)
    if match:
        return match.group(1).upper(), match.group(2)
    return raw.upper(), ""


def set_cell(sheet, row: int, column: int, value, *, fill=None, bold=False, center=False):
    cell = sheet.cell(row, column, value)
    cell.font = Font(name="맑은 고딕", size=10, bold=bold, color="173E39")
    cell.alignment = Alignment(vertical="center", horizontal="center" if center else "left", wrap_text=True)
    cell.border = BORDER
    if fill:
        cell.fill = fill
    return cell


def add_table(sheet, row: int, start: int, table: dict) -> None:
    name = table["name"]
    ko, role = TABLES[name]
    sheet.merge_cells(start_row=row, start_column=start, end_row=row, end_column=start + 1)
    sheet.merge_cells(start_row=row, start_column=start + 2, end_row=row, end_column=start + 8)
    set_cell(sheet, row, start, "테이블 명", fill=FILL_TITLE, bold=True, center=True)
    set_cell(sheet, row, start + 2, "테이블 설명", fill=FILL_TITLE, bold=True, center=True)
    sheet.merge_cells(start_row=row + 1, start_column=start, end_row=row + 1, end_column=start + 1)
    sheet.merge_cells(start_row=row + 1, start_column=start + 2, end_row=row + 1, end_column=start + 8)
    set_cell(sheet, row + 1, start, name, bold=True)
    set_cell(sheet, row + 1, start + 2, f"{ko} — {role}")
    sheet.row_dimensions[row + 1].height = 32
    for idx, title in enumerate(HEADERS):
        set_cell(sheet, row + 2, start + idx, title, fill=FILL_HEADER, bold=True, center=True)
    sheet.row_dimensions[row + 2].height = 25

    constraints = table["constraints"]
    pk = {col for con in constraints if con["kind"] == "p" for col in con["columns"]}
    fk = {col: f"{con['target_table']}.{', '.join(con['target_columns'])}"
          for con in constraints if con["kind"] == "f" for col in con["columns"]}
    for index, column in enumerate(table["columns"], 1):
        colname = column["name"]
        sql_type, length = sql_type_parts(column["type"])
        key = "/".join(k for k, on in (("PK", colname in pk), ("FK", colname in fk)) if on)
        remark = f"참조: {fk[colname]}" if colname in fk else ""
        values = [index, colname, describe(name, colname), sql_type, length,
                  "YES" if column["nullable"] else "NO", column.get("default") or "", key, remark]
        r = row + 2 + index
        for offset, value in enumerate(values):
            set_cell(sheet, r, start + offset, value, fill=FILL_ALT if index % 2 == 0 else None,
                     center=offset in (0, 4, 5, 7))
        sheet.row_dimensions[r].height = 30 if len(values[2]) > 27 or len(remark) > 30 else 24


def main():
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    tables = snapshot["tables"]
    assert len(tables) == len(TABLES)
    book = Workbook()
    overview = book.active
    overview.title = "목차"
    sheet = book.create_sheet("DB구조")
    sheet.sheet_view.showGridLines = False
    overview.append(["No.", "테이블명", "한글명", "테이블 역할", "DB구조 위치"])
    for col, width in zip("ABCDE", (9, 43, 28, 75, 18)):
        overview.column_dimensions[col].width = width
    for idx, width in enumerate((8, 35, 43, 22, 13, 13, 35, 11, 40)):
        sheet.column_dimensions[get_column_letter(2 + idx)].width = width
        sheet.column_dimensions[get_column_letter(12 + idx)].width = width
    sheet.column_dimensions["A"].width = 3
    sheet.column_dimensions["K"].width = 4
    sheet.column_dimensions["U"].width = 3
    sheet.freeze_panes = "B2"
    current_row = 2
    for pair_index in range(0, len(tables), 2):
        pair = tables[pair_index:pair_index + 2]
        for side, table in enumerate(pair):
            start = BLOCK_STARTS[side]
            add_table(sheet, current_row, start, table)
            name = table["name"]
            ko, role = TABLES[name]
            anchor = f"{get_column_letter(start)}{current_row}"
            overview.append([pair_index + side + 1, name, ko, role, anchor])
            cell = overview.cell(overview.max_row, 5)
            cell.hyperlink = f"#'DB구조'!{anchor}"
            cell.style = "Hyperlink"
        current_row += max(len(t["columns"]) for t in pair) + 5
    overview.freeze_panes = "A2"
    overview.auto_filter.ref = overview.dimensions
    overview.sheet_view.showGridLines = False
    for cell in overview[1]:
        cell.fill = PatternFill("solid", fgColor="073C38")
        cell.font = Font(name="맑은 고딕", color="FFFFFF", bold=True)
    for row in overview.iter_rows(min_row=2):
        for cell in row:
            if cell.column != 5:
                cell.font = Font(name="맑은 고딕", size=10)
            cell.alignment = Alignment(vertical="center", wrap_text=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    book.save(OUTPUT)
    print(f"{OUTPUT} | tables={len(tables)} columns={sum(len(t['columns']) for t in tables)}")


if __name__ == "__main__":
    main()
