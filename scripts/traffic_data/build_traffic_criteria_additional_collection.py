"""Package newly collected traffic safety-index inputs into one workbook."""
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[2]
SAFE = ROOT / "data" / "interim" / "안전" / "교통사고"
OUTPUT = SAFE / "교통사고_안전기준표_추가수집데이터.xlsx"


def style(ws):
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    fill = PatternFill("solid", fgColor="1F4E78")
    for cell in ws[1]:
        cell.fill = fill
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for col in ws.columns:
        values = [str(c.value or "") for c in list(col)[:300]]
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max(map(len, values)) + 2, 45)


def main():
    population = pd.read_csv(SAFE / "자치구별_주민등록인구_2017_2025.csv")
    workplaces = pd.read_csv(SAFE / "자치구별_의료보장사업장수_2024.csv")
    cctv = pd.read_csv(SAFE / "자치구별_교통단속CCTV_최신.csv")
    cctv_detail = pd.read_csv(SAFE / "서울_교통단속CCTV_최신_상세.csv")

    base_2024 = population[population["연도"] == 2024][["연도", "자치구", "주민등록인구_계"]]
    joined = base_2024.merge(
        workplaces[["연도", "자치구", "의료보장사업장수"]], on=["연도", "자치구"], validate="one_to_one"
    ).merge(cctv, on="자치구", validate="one_to_one")
    joined["인구1만명당_의료보장사업장수"] = joined["의료보장사업장수"] / joined["주민등록인구_계"] * 10_000
    joined["CCTV시점주의"] = "CCTV는 2026 최신 스냅샷이므로 2024 모델 입력에 직접 사용 금지"

    status = pd.DataFrame([
        ["주민등록인구", "확보", "2017~2025", 225, "25개 구"],
        ["의료보장 사업장수", "확보", "2024", 25, "25개 구, 건강보험통계연보"],
        ["교통단속 CCTV", "확보", "최신(기준일 최대 2026-10-20)", 25, "최신 스냅샷; 과거 모델에 시점 누출 주의"],
        ["재난약자수", "추가 결합 필요", "-", 0, "65세 이상 인구+유·초등학생"],
        ["도로면적", "추가 추출 필요", "-", 0, "KOSIS 도시계획현황 도로(기능별)"],
        ["환경개선사업 지출액", "비공개", "-", 0, "행정안전부 내부협의"],
        ["교통법규위반건수", "비공개", "-", 0, "경찰청 기관협의"],
    ], columns=["항목", "상태", "기간/시점", "행수", "메모"])

    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        joined.to_excel(writer, sheet_name="2024_결합참고", index=False)
        population.to_excel(writer, sheet_name="주민등록인구_2017_2025", index=False)
        workplaces.to_excel(writer, sheet_name="의료보장사업장_2024", index=False)
        cctv.to_excel(writer, sheet_name="교통단속CCTV_최신_집계", index=False)
        cctv_detail.to_excel(writer, sheet_name="교통단속CCTV_최신_상세", index=False)
        status.to_excel(writer, sheet_name="수집현황", index=False)
        for ws in writer.book.worksheets:
            style(ws)
    print(f"saved={OUTPUT}")
    print(f"sheets=6, joined_rows={len(joined)}, cctv_detail_rows={len(cctv_detail)}")


if __name__ == "__main__":
    main()
