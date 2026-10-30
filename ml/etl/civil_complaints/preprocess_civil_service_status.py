"""서울시 민원사무처리현황(2006~2025)을 공통 기준으로 전처리한다."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
from openpyxl.styles import Border, Font, Side


ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw" / "민원"
OLD_RAW = RAW_DIR / "서울시 민원사무처리현황_2006_2020.csv"
NEW_RAW = RAW_DIR / "서울시 민원사무처리현황_2021_2025.csv"
OUTPUT = ROOT / "data" / "interim" / "서울시_민원사무처리현황_2006_2025_전처리.xlsx"

DISTRICTS = [
    "종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구",
    "성북구", "강북구", "도봉구", "노원구", "은평구", "서대문구", "마포구",
    "양천구", "강서구", "구로구", "금천구", "영등포구", "동작구", "관악구",
    "서초구", "강남구", "송파구", "강동구",
]

COMMON_ITEMS = {
    "인가·허가": "인가/허가",
    "특허·면허": "특허/면허",
    "승인·지정": "승인.지정",
    "신고·등록": "신고.등록",
    "시험·검사": "시험.검사",
    "확인·증명·교부": "확인증명/교부",
}


def read_csv(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.reader(file))


def numeric(value: str) -> int:
    value = value.strip().replace(",", "")
    return 0 if value == "-" else int(value)


def to_long(path: Path, header_rows: int) -> pd.DataFrame:
    rows = read_csv(path)
    headers = rows[:header_rows]
    width = len(headers[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"열 개수가 일정하지 않습니다: {path.name}")

    records: list[dict[str, object]] = []
    for row in rows[header_rows:]:
        region = row[1].strip()
        for column in range(2, width):
            year = int(headers[0][column])
            if header_rows == 2:
                category1, category2, item = "", "", headers[1][column].strip()
            else:
                category1 = headers[1][column].strip()
                category2 = headers[2][column].strip()
                item = headers[3][column].strip()
            raw_value = row[column].strip()
            records.append(
                {
                    "연도": year,
                    "지역": region,
                    "분류1": category1,
                    "분류2": category2,
                    "항목": item,
                    "값": numeric(raw_value),
                    "원본표기값": raw_value,
                    "원본파일": path.name,
                }
            )
    return pd.DataFrame(records)


def validate_source_identities(detail: pd.DataFrame) -> None:
    # 1차 검증: 2006~2020년 전체가 세부 항목 합계와 같은지 확인한다.
    old = detail[detail["연도"].le(2020)]
    old_pivot = old.pivot(index=["연도", "지역"], columns="항목", values="값")
    old_parts = ["인가/허가", "특허/면허", "승인.지정", "신고.등록", "확인증명/교부", "고충민원", "기타"]
    # 2006년에는 시험·검사 항목 자체가 없고, 2007년부터 합계에 포함된다.
    calculated = old_pivot[old_parts].sum(axis=1)
    calculated += old_pivot["시험.검사"].fillna(0)
    if not calculated.eq(old_pivot["계"]).all():
        raise ValueError("2006~2020년 전체 민원과 세부 항목 합계가 다릅니다.")

    # 2차 검증: 2021년 이후 단계별 소계가 하위 항목 합계와 같은지 확인한다.
    new = detail[detail["연도"].ge(2021)]
    keys = ["연도", "지역"]
    pick = lambda c1, c2, item: new[
        new["분류1"].eq(c1) & new["분류2"].eq(c2) & new["항목"].eq(item)
    ].set_index(keys)["값"]
    legal = pick("일반민원", "법정민원", "소계")
    legal_parts = sum((pick("일반민원", "법정민원", item) for item in [*COMMON_ITEMS.values(), "기타"]), start=legal * 0)
    general = pick("일반민원", "소계", "소계")
    query = pick("일반민원", "질의 건의 기타 민원", "소계")
    total = pick("총계", "소계", "소계")
    field = pick("현장민원", "소계", "소계")
    grievance = pick("고충민원", "소계", "소계")
    if not legal.eq(legal_parts).all():
        raise ValueError("2021년 이후 법정민원 소계가 세부 항목 합계와 다릅니다.")
    if not general.eq(legal + query).all():
        raise ValueError("2021년 이후 일반민원 소계가 하위 항목 합계와 다릅니다.")
    if not total.eq(general + field + grievance).all():
        raise ValueError("2021년 이후 총계가 일반·현장·고충민원 합계와 다릅니다.")


def validate_regional_totals(detail: pd.DataFrame) -> None:
    # 3차 검증: 원본 소계가 본청과 25개 자치구의 합과 같은지 확인한다.
    dimensions = ["연도", "분류1", "분류2", "항목"]
    total = detail[detail["지역"].eq("소계")].set_index(dimensions)["값"].sort_index()
    members = detail[~detail["지역"].eq("소계")].groupby(dimensions)["값"].sum().sort_index()
    if not total.equals(members):
        raise ValueError("원본 소계가 본청과 25개 자치구의 합과 다릅니다.")


def common_panel(detail: pd.DataFrame, region: str) -> dict[str, object]:
    year = int(detail["연도"].iloc[0])
    old_period = year <= 2020

    def value(item: str, category1: str = "", category2: str = "") -> int | None:
        selected = detail[detail["항목"].eq(item)]
        if not old_period:
            selected = selected[selected["분류1"].eq(category1) & selected["분류2"].eq(category2)]
        if selected.empty:
            return None
        if len(selected) != 1:
            raise ValueError(f"공통 항목이 한 개가 아닙니다: {year}, {region}, {item}")
        return int(selected["값"].iloc[0])

    result: dict[str, object] = {"연도": year, "지역": region}
    result["전체 민원"] = value("계") if old_period else value("소계", "총계", "소계")
    for output_name, source_name in COMMON_ITEMS.items():
        result[output_name] = value(source_name) if old_period else value(source_name, "일반민원", "법정민원")
    result["고충민원"] = value("고충민원") if old_period else value("소계", "고충민원", "소계")
    return result


def make_panel(detail: pd.DataFrame, regions: list[str], region_column: str) -> pd.DataFrame:
    records = []
    for year in range(2006, 2026):
        for region in regions:
            subset = detail[detail["연도"].eq(year) & detail["지역"].eq(region)]
            record = common_panel(subset, region)
            record[region_column] = record.pop("지역")
            records.append(record)
    panel = pd.DataFrame(records)
    panel = panel[["연도", region_column, *[column for column in panel.columns if column not in {"연도", region_column}]]]

    category_columns = [*COMMON_ITEMS.keys(), "고충민원"]
    for column in category_columns:
        panel[f"전체 민원 중 {column} 비율(%)"] = panel[column].div(panel["전체 민원"]).mul(100).round(2)

    panel = panel.sort_values([region_column, "연도"], kind="stable")
    previous = panel.groupby(region_column, sort=False)["전체 민원"].shift(1)
    panel["전체 민원 전년 대비 증감(건)"] = panel["전체 민원"] - previous
    panel["전체 민원 전년 대비 증감률(%)"] = (
        panel["전체 민원"].sub(previous).div(previous).mul(100).round(2)
    ).where(previous.ne(0))
    return panel.sort_values(["연도", region_column], kind="stable").reset_index(drop=True)


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
    detail = pd.concat([to_long(OLD_RAW, 2), to_long(NEW_RAW, 4)], ignore_index=True)
    validate_source_identities(detail)
    validate_regional_totals(detail)

    districts = make_panel(detail, DISTRICTS, "자치구")
    seoul_total = make_panel(detail, ["소계"], "지역").replace({"지역": {"소계": "서울시 전체(본청 포함)"}})
    headquarters_parts = []
    for year in range(2006, 2026):
        source_region = "서울시" if year <= 2020 else "본청"
        subset = detail[detail["연도"].eq(year) & detail["지역"].eq(source_region)]
        record = common_panel(subset, source_region)
        record["지역"] = "본청"
        headquarters_parts.append(record)
    headquarters_base = pd.DataFrame(headquarters_parts)
    # 본청도 make_panel과 같은 파생지표를 만들기 위해 임시 상세 대신 직접 계산한다.
    categories = [*COMMON_ITEMS.keys(), "고충민원"]
    for column in categories:
        headquarters_base[f"전체 민원 중 {column} 비율(%)"] = headquarters_base[column].div(headquarters_base["전체 민원"]).mul(100).round(2)
    previous = headquarters_base["전체 민원"].shift(1)
    headquarters_base["전체 민원 전년 대비 증감(건)"] = headquarters_base["전체 민원"] - previous
    headquarters_base["전체 민원 전년 대비 증감률(%)"] = headquarters_base["전체 민원"].sub(previous).div(previous).mul(100).round(2).where(previous.ne(0))

    # 최종 패널 검증: 행 수·중복·음수·비율 범위·첫해 증감 결측을 확인한다.
    if len(districts) != 500 or districts.duplicated(["연도", "자치구"]).any():
        raise ValueError("25개 자치구×20년 패널 구조가 아닙니다.")
    count_columns = ["전체 민원", *COMMON_ITEMS.keys(), "고충민원"]
    if (districts[count_columns].dropna() < 0).any().any():
        raise ValueError("공통 항목에 음수가 있습니다.")
    share_columns = [column for column in districts if column.startswith("전체 민원 중")]
    if districts[share_columns].lt(0).any().any() or districts[share_columns].gt(100).any().any():
        raise ValueError("구성비가 0~100% 범위를 벗어납니다.")
    if districts["전체 민원 전년 대비 증감(건)"].isna().sum() != 25:
        raise ValueError("전년 대비 증감의 첫 연도 결측 개수가 예상과 다릅니다.")

    notes = pd.DataFrame(
        {
            "항목": [
                "기간과 범위", "공통 항목", "분류 변경", "기타 항목", "시험·검사", "원본의 '-'",
                "서울시 전체", "본청", "전년 대비", "원본 보존",
            ],
            "설명": [
                "2006~2025년, 서울 25개 자치구",
                "전체 민원과 명칭·의미가 이어지는 6개 법정민원 항목 및 고충민원",
                "2021년부터 일반·법정·현장·고충민원 계층이 추가되어 원본분류상세 시트에 그대로 보존",
                "2021년 전후 의미가 같다고 단정할 수 없어 공통 분석표에서는 제외",
                "2006년 원자료에 항목 자체가 없어 빈칸으로 유지",
                "원본 소계와 세부 합계 검증 후 0으로 변환하고 원본표기는 상세 시트에 보존",
                "본청과 25개 자치구를 포함한 원본 소계",
                "2006~2020년 원본의 서울시 행과 2021~2025년 본청 행을 연결",
                "동일 지역의 직전 연도와 비교하며 2006년은 빈칸",
                "raw 파일은 수정하지 않고 별도의 interim 파일만 생성",
            ],
        }
    )

    detail = detail.sort_values(["연도", "지역", "분류1", "분류2", "항목"], kind="stable").reset_index(drop=True)
    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        districts.to_excel(writer, index=False, sheet_name="자치구별")
        seoul_total.to_excel(writer, index=False, sheet_name="서울시전체")
        headquarters_base.to_excel(writer, index=False, sheet_name="본청")
        detail.to_excel(writer, index=False, sheet_name="원본분류상세")
        notes.to_excel(writer, index=False, sheet_name="전처리안내")
        for worksheet in writer.book.worksheets:
            format_sheet(worksheet)

    print(f"저장: {OUTPUT}")
    print(f"자치구별 {len(districts):,}행, 원본분류상세 {len(detail):,}행")


if __name__ == "__main__":
    main()
