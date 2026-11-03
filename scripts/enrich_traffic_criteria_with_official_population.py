"""Join official registered population to the traffic criteria workbook."""
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


ROOT = Path(__file__).resolve().parents[1]
SAFE = ROOT / "data" / "interim" / "안전"
POPULATION = SAFE / "자치구별_주민등록인구_2017_2025.csv"
OUTPUT = SAFE / "교통사고_안전기준표_전용데이터_공식인구보강.xlsx"


def format_sheet(ws):
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    fill = PatternFill("solid", fgColor="1F4E78")
    for cell in ws[1]:
        cell.fill = fill
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for column in ws.columns:
        values = [str(c.value) if c.value is not None else "" for c in list(column)[:250]]
        ws.column_dimensions[get_column_letter(column[0].column)].width = min(
            max(max(map(len, values), default=8) + 2, 10), 45
        )


def main():
    candidates = [p for p in SAFE.glob("*전용데이터.xlsx") if "공식인구보강" not in p.name]
    if not candidates:
        raise FileNotFoundError("Traffic criteria workbook not found")
    source = candidates[0]
    book = pd.ExcelFile(source)
    sheets = {name: pd.read_excel(source, sheet_name=name) for name in book.sheet_names}
    main_name = book.sheet_names[0]
    data = sheets[main_name]
    population = pd.read_csv(POPULATION)
    enriched = data.merge(population, on=["연도", "자치구"], how="left", validate="one_to_one")
    if enriched["주민등록인구_계"].isna().any():
        raise ValueError("Population join has missing rows")
    enriched["인구1만명당_교통사고사망자수_공식"] = (
        enriched["사망자수"] / enriched["주민등록인구_계"] * 10_000
    )
    enriched["인구1만명당_자동차등록대수_참고"] = (
        enriched["자동차등록대수"] / enriched["주민등록인구_계"] * 10_000
    )
    status = pd.DataFrame([
        ["주민등록인구", "확보", "2017~2025, 25개 자치구", "서울 열린데이터광장 주민등록인구(연도별)", "이번 보강파일에 결합"],
        ["재난약자수", "부분 확보 필요", "65세 이상 인구 + 유·초등학생수", "행정안전부 주민등록 인구통계 + 교육부 교육통계서비스", "두 원천을 합산해야 함"],
        ["의료보장 사업장수", "공개 원천 확인", "직장가입자(근로+공무원·교직원) 사업장수", "국민건강보험공단 건강보험 통계연보", "시군구 표 추출 필요"],
        ["도로면적", "공개 원천 확인", "도시계획현황 도로(기능별), ㎡", "KOSIS/한국국토정보공사·국토교통부", "㎡를 ㎢로 변환"],
        ["교통단속 CCTV", "공개 최신자료 가능", "과속·구간·다기능, 고속도로 제외", "공공데이터포털 전국무인교통단속카메라표준데이터", "과거 자료는 삭제될 수 있음"],
        ["지역교통안전환경개선사업 지출액", "비공개", "지방비 집행액 및 광역본청 배분", "행정안전부 내부협의 자료", "공개자료만으로 공식값 재현 불가"],
        ["교통법규위반건수", "비공개", "관할서 기준, 2024.7~2025.6", "경찰청 기관협의 자료", "복수 자치구 관할 시 인구비례 배분"],
    ], columns=["항목", "확보상태", "공식 정의/범위", "공식 원천", "처리 메모"])
    sources = pd.DataFrame([
        ["서울 주민등록인구(연도별)", "https://data.seoul.go.kr/bsp/wgs/gis/gis03.do", "공개"],
        ["2025년 지역안전지수 산출결과·지표설명", "https://www.mois.go.kr/frt/bbs/type001/commonSelectBoardArticle.do?bbsId=BBSMSTR_000000000015&nttId=123072", "공개"],
    ], columns=["자료명", "URL", "공개여부"])
    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        enriched.to_excel(writer, sheet_name="교통사고_공식인구보강", index=False)
        status.to_excel(writer, sheet_name="추가수집_현황", index=False)
        sources.to_excel(writer, sheet_name="출처", index=False)
        for name in book.sheet_names[1:]:
            sheets[name].to_excel(writer, sheet_name=name[:31], index=False)
        for ws in writer.book.worksheets:
            format_sheet(ws)
    print(f"source={source}")
    print(f"saved={OUTPUT}")
    print(f"rows={len(enriched)}, population_missing={enriched['주민등록인구_계'].isna().sum()}")


if __name__ == "__main__":
    main()
