from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim" / "안전" / "생활안전"
OUT = INTERIM / "생활안전_안전기준표_2017_2025.xlsx"


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(INTERIM / name, encoding="utf-8-sig")


pop = read_csv("자치구별_주민등록인구_2017_2025.csv")
elderly = read_csv("자치구별_65세이상인구_2017_2025.csv")
ems = read_csv("자치구별_구급구조_1차.csv")
aed = read_csv("행정동별_자동심장충겨기_1차.csv")

panel = pop[["연도", "자치구", "주민등록인구_계"]].copy()
panel = panel.merge(elderly[["연도", "자치구", "65세이상인구수"]], on=["연도", "자치구"], how="left")
panel = panel.merge(
    ems[["연도", "자치구", "구급건수", "구급_연간완전관측여부"]],
    on=["연도", "자치구"], how="left"
)
panel["인구만명당_65세이상인구수"] = panel["65세이상인구수"] / panel["주민등록인구_계"] * 10000
panel["인구만명당_구급건수_참고"] = panel["구급건수"] / panel["주민등록인구_계"] * 10000
panel["구급건수_주의"] = "공개 구급 전체 건수이며 공식 생활안전 17개 사고유형·발생지 기준과 다름"

aed_gu = (aed.assign(보유대수=pd.to_numeric(aed["보유대수"], errors="coerce").fillna(0))
          .groupby("자치구", as_index=False)["보유대수"].sum()
          .rename(columns={"보유대수": "AED_보유대수_현재자료"}))
aed_gu["주의"] = "시설별 현행 자료 집계. 2017~2025 연도별 공식 지표가 아니며 용도지역 면적 보정 전"

status = pd.DataFrame([
    ["위험", "인구만명당 생활안전 관련 119 구급건수", "부분", "2022~2024 전체 구급 참고자료", "정확 지표는 소방청 기관협약 자료(17개 사고유형·발생지 기준)"],
    ["취약", "인구만명당 제조업 종사자수", "미확보", "공식 통계 위치 확인", "전국사업체조사 제조업(C10~C34) 자치구 연도별 원자료 추출 필요"],
    ["취약", "인구만명당 재난약자수", "부분", "65세 이상 2017~2025 확보", "유치원생+초등학생 자치구 연도별 합계 추가 필요"],
    ["경감", "주상공관리지역 면적당·인구만명당 AED 설치대수", "부분", "시설별 AED 현행자료", "연도별 AED와 주거·상업·공업·관리지역 면적 필요; 공식 가중치 70:30"],
    ["경감", "공공질서 및 안전예산 지출액 비율", "미확보", "지방재정365 출처 확인", "자치구 결산액/예산현액 및 서울시 본청분 인구배분 필요"],
    ["의식", "인구만명당 생활안전 관련 안전신문고 신고건수", "미확보", "없음", "행안부 안전신문고 분석시스템 내부자료; 일반 행정민원으로 대체 불가"],
    ["의식", "기준연도 대비 응급처치 교육인원수", "미확보", "공개 연보 위치 확인", "소방청 기관협약분+국립중앙의료원 공개분을 합산해야 함"],
], columns=["분류", "공식지표", "확보상태", "현재확보", "남은작업_제약"])

sources = pd.DataFrame([
    ["65세 이상 인구", "서울 데이터 허브", "https://data.seoul.go.kr/bsp/wgs/dataView/data300View/96.do"],
    ["유치원·초등학생", "서울 열린데이터광장/서울시교육청", "https://data.seoul.go.kr/dataList/DT201004G010006/S/2/datasetView.do"],
    ["제조업 종사자", "서울 열린데이터광장/전국사업체조사", "https://data.seoul.go.kr/dataList/DT201004C010034/S/2/datasetView.do"],
    ["공공질서·안전 예산", "지방재정365", "https://www.lofin365.go.kr/portal/LF8310000.do?ntcaClsDvCd=FSL414"],
    ["용도지역 면적", "KOSIS 도시계획현황", "https://kosis.kr/serviceInfo/newContrainDataDetail.do?boardIdx=2007330&boardOrgId=460"],
], columns=["항목", "공식출처", "URL"])

with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
    panel.sort_values(["연도", "자치구"]).to_excel(writer, sheet_name="기준패널", index=False)
    elderly.to_excel(writer, sheet_name="65세이상_원자료", index=False)
    ems.to_excel(writer, sheet_name="구급_참고", index=False)
    aed_gu.to_excel(writer, sheet_name="AED_현재집계", index=False)
    status.to_excel(writer, sheet_name="수집현황", index=False)
    sources.to_excel(writer, sheet_name="공식출처", index=False)
    for ws in writer.book.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for col in ws.columns:
            width = min(max(len(str(cell.value or "")) for cell in col) + 2, 55)
            ws.column_dimensions[col[0].column_letter].width = width

print(OUT)
print(f"panel_rows={len(panel)}, years={panel['연도'].min()}-{panel['연도'].max()}, districts={panel['자치구'].nunique()}")
