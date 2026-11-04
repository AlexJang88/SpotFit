"""Extract only traffic-criteria columns into a compact Excel workbook."""
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


ROOT = Path(__file__).resolve().parents[2]
SAFE = ROOT / "data" / "interim" / "안전" / "교통사고"
SOURCE = SAFE / "교통사고_안전기준표_관련데이터_통합.xlsx"
OUTPUT = SAFE / "교통사고_안전기준표_전용데이터.xlsx"


def format_sheet(ws):
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    fill = PatternFill("solid", fgColor="1F4E78")
    for cell in ws[1]:
        cell.fill = fill
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for column in ws.columns:
        values = [str(c.value) if c.value is not None else "" for c in list(column)[:200]]
        ws.column_dimensions[get_column_letter(column[0].column)].width = min(
            max(max(map(len, values), default=8) + 2, 10), 42
        )


def main():
    data = pd.read_excel(SOURCE, sheet_name="통합데이터_2017_2025")
    definitions = pd.read_excel(SOURCE, sheet_name="기준표_지표정의")
    columns = [
        "연도", "자치구",
        "사망자수", "생활인구_연평균", "인구1만명당_교통사고사망자수_생활인구대리",
        "교통단속CCTV누적", "과속신호카메라누적", "인구1만명당_교통단속CCTV누적_대리",
        "운전자안전벨트착용률", "음주운전율",
        "자동차등록대수", "인구1만명당_자동차등록대수_대리",
        "공통4지표_원가중치점수_70점", "공통4지표_환산안전점수_100점", "공통4지표_연도내순위",
        "CCTV포함5지표_원가중치점수_87.24점", "CCTV포함5지표_환산안전점수_100점",
        "CCTV포함5지표_연도내순위_14개구내",
    ]
    missing = set(columns).difference(data.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    result = data[columns].sort_values(["연도", "자치구"]).reset_index(drop=True)
    if result.duplicated(["연도", "자치구"]).any():
        raise ValueError("duplicate year-district keys")

    notes = pd.DataFrame([
        ["완전", "운전자 안전벨트 착용률", "원자료와 기준표 직접 일치"],
        ["완전", "음주운전 경험률", "음주운전율 원자료와 직접 대응"],
        ["부분", "인구만명당 교통사고 사망자수", "주민등록인구 대신 생활인구 사용"],
        ["부분", "도로면적당 교통단속 CCTV대수", "도로면적 대신 생활인구 분모 사용; 14개 구만"],
        ["부분", "자동차등록대수당 교통법규위반건수", "법규위반건수가 없어 차량밀도를 대리 사용"],
    ], columns=["확보상태", "교통사고 기준표 항목", "주의사항"])

    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="교통사고_전용데이터", index=False)
        definitions.to_excel(writer, sheet_name="전체8개지표_가중치", index=False)
        notes.to_excel(writer, sheet_name="사용시주의사항", index=False)
        for ws in writer.book.worksheets:
            format_sheet(ws)
    print(f"saved={OUTPUT}")
    print(f"rows={len(result)} columns={len(result.columns)}")


if __name__ == "__main__":
    main()
