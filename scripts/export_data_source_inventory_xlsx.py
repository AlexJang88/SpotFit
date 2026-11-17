from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


OUTPUT = Path.home() / "Desktop" / "SpotFit_확보데이터_출처_총정리_20261105.xlsx"

SOURCES = [
    ["부동산", "아파트·빌라 매매 실거래", "국토교통부", "실거래가 공개시스템", "서울, 주 분석자료는 2021년 이후", "data/processed/부동산/*_매매_최종.csv", "https://rt.molit.go.kr/", "높은 확률", "직접 원본과 수집 매니페스트 보완 필요"],
    ["부동산", "아파트·빌라 전월세 실거래", "서울특별시", "서울 열린데이터광장 서울시 부동산 전월세가 정보", "서울, 2021~2026", "data/processed/부동산/*_전월세_최종.csv", "https://data.seoul.go.kr/dataList/OA-21276/S/1/datasetView.do", "확인", "원천은 국토교통부 실거래 자료"],
    ["부동산", "서울시 물가상승률", "미확인", "미확인", "파일 내부 확인 필요", "data/processed/부동산/서울시_물가상승률_최종.xlsx", "", "확인 필요", "공식 통계표명과 URL 보완 필요"],
    ["교통사고", "자치구별 사고·사망·부상", "한국도로교통공단", "TAAS 교통사고분석시스템", "서울·고속국도 제외, 2010~2025", "data/raw/안전/교통사고/TAAS_서울_고속국도제외_*.xlsx", "https://taas.koroad.or.kr/", "확인", "서울 전체와 25개 자치구 제공"],
    ["교통사고", "어린이 교통사고", "한국도로교통공단 추정", "TAAS 계열 자료 추정", "2010~2025", "data/raw/안전/교통사고/어린이교통사고_2010_2025.xlsx", "https://taas.koroad.or.kr/", "확인 필요", "다운로드 조건과 원본 URL 기록 필요"],
    ["교통", "도로 현황·도로율", "통계청·서울특별시", "KOSIS 국가통계포털", "2017~2025", "data/raw/안전/교통사고/KOSIS_서울시_도로현황*.csv", "https://kosis.kr/", "확인", "도로면적·도로율"],
    ["건강보험", "의료보장 적용 사업장", "국민건강보험공단", "KOSIS 시군구별 적용인구 현황 TX_35001_A008", "2017~2024", "data/raw/안전/교통사고/KOSIS_시군구별_적용인구현황_2017_2024_한글정상.csv", "https://kosis.kr/", "확인", "인구 1만 명당 사업장수 산출에 사용"],
    ["건강보험", "건강보험 통계연보", "국민건강보험공단", "건강보험통계연보", "2024", "data/raw/안전/교통사고/교통사고_기준표보강/의료보장사업장/2024_건강보험통계연보.pdf", "https://www.nhis.or.kr/", "확인", "보조 검증자료"],
    ["교통", "차량 통행속도", "서울특별시", "서울시 통계·교통 데이터", "자치구 월별, 2017~2025", "data/raw/안전/교통사고/서울시_차량통행속도_월별_구별_2017_2025.csv", "https://data.seoul.go.kr/", "확인", "연도 단위로 집계해 사용"],
    ["교통", "교통량 조사자료", "서울특별시", "서울특별시 교통량 조사 연차보고서", "2017~2024", "data/raw/안전/교통사고/교통량_연차보고서/*.pdf", "https://data.seoul.go.kr/", "확인", "연차보고서 8개"],
    ["교통", "자동차등록대수", "서울특별시·통계청 추정", "서울시/KOSIS 자동차등록 통계 계열", "자치구, 2006~2025", "data/raw/안전/교통사고/자치구별_자동차등록대수_2006_2025.csv", "https://kosis.kr/", "높은 확률", "정확한 통계표 URL 보완 필요"],
    ["교통", "교통법규 위반 단속", "서울경찰청", "경찰서별 교통법규 위반 유형별 단속현황", "2022~2024", "data/raw/안전/교통사고/교통법규위반/*.csv", "", "확인", "경찰서 관할을 자치구로 변환한 부분 자료"],
    ["교통", "무인교통단속카메라", "공공데이터포털·각 지자체", "전국무인교통단속카메라표준데이터", "현존 목록, 기준일 2026-10-20", "data/raw/안전/교통사고/교통사고_기준표보강/교통단속CCTV_최신/*.json", "https://www.data.go.kr/download/standard.json", "확인", "설치연도로 과거 현존 수를 역산"],
    ["교통안전", "안전벨트 착용률", "질병관리청", "지역사회건강조사", "자치구, 2008~2025", "data/raw/안전/교통사고/자치구별_운전자안전벨트착용_2008_2025.csv", "https://chs.kdca.go.kr/", "확인", "운전자 안전의식 지표"],
    ["교통안전", "음주운전 경험률", "질병관리청", "지역사회건강조사", "자치구, 2008~2025", "data/raw/안전/교통사고/자치구별_음주운전_2008_2025.csv", "https://chs.kdca.go.kr/", "확인", "운전자 안전의식 지표"],
    ["인구", "주민등록인구", "서울특별시", "서울 열린데이터광장 주민등록인구", "자치구, 2017~2025", "data/raw/안전/교통사고/교통사고_기준표보강/주민등록인구/*.json", "https://data.seoul.go.kr/bsp/wgs/gis/gis03.do", "확인", "연도별 API 응답 원본 보존"],
    ["인구", "연령별 주민등록인구", "행정안전부 추정", "주민등록 인구통계", "2017~2025년 12월", "data/raw/연령별인구현황/*.csv", "https://jumin.mois.go.kr/", "높은 확률", "원 다운로드 URL·조건 기록 보완 필요"],
    ["안전지수", "지역안전지수 결과·지표설명", "행정안전부", "2025년 지역안전지수", "2025", "data/raw/안전/교통사고/교통사고_기준표보강/지역안전지수_공식/*.hwpx", "https://www.mois.go.kr/frt/bbs/type001/commonSelectBoardArticle.do?bbsId=BBSMSTR_000000000015&nttId=123072", "확인", "산출 결과와 지표 정의"],
    ["범죄", "서울 5대 범죄 발생·검거", "서울특별시·서울경찰청 계열", "범죄 통계", "2014~2025", "data/raw/안전/범죄/범죄_2014_2025.csv; 서울시_5대범죄_2014_2025.csv", "", "확인 필요", "공식 데이터셋 URL과 경찰서→자치구 규칙 보완 필요"],
    ["범죄", "지구대·파출소 주소", "경찰청", "경찰청 공공데이터", "전국, 2025 스냅샷", "data/raw/안전/범죄/경찰청_전국_지구대_파출소_주소_2025.csv", "https://www.data.go.kr/", "확인", "과거 연도 패널은 아님"],
    ["안전", "서울시 CCTV", "서울특별시", "서울 열린데이터광장 CCTV 정보 계열", "최신 목록 중심", "data/raw/안전/교통사고/서울시_CCTV정보.csv; data/raw/안전/범죄/서울시_CCTV정보.csv", "https://data.seoul.go.kr/", "높은 확률", "설치연도 누락·현존 목록 편향 확인 필요"],
    ["생활안전", "119 구급", "원출처 확인 필요", "생활안전_구급자료·안전한동네_구급", "파일 내부 확인 필요", "data/raw/안전/생활안전/01_*/*.xlsx", "", "확인 필요", "공식 17개 생활안전 사고유형 필터 필요"],
    ["생활안전", "65세 이상 인구", "서울특별시", "서울 데이터 허브 고령자 현황", "2017~2025", "data/raw/안전/생활안전/03_*/*", "https://data.seoul.go.kr/bsp/wgs/dataView/data300View/96.do", "확인", "재난약자 지표의 일부"],
    ["생활안전", "자동심장충격기(AED)", "공공데이터포털·안전한동네", "AED 시설자료", "시설 스냅샷", "data/raw/안전/생활안전/04_*/*", "https://www.data.go.kr/data/15021103/standard.do", "부분 확인", "공식 연도별 지표와 용도지역 면적 보정 부족"],
]

FILES = [
    ["processed", "부동산", "서울시_물가상승률_최종.xlsx", 0.01, "물가상승률 보조자료", "원출처 메타데이터 보완 필요"],
    ["processed", "부동산", "행정동별_아파트_매매_최종.csv", 41.51, "아파트 매매 실거래 최종본", "행정동 매핑·취소거래 제외·평당금액 계산"],
    ["processed", "부동산", "행정동별_아파트_전월세_최종.csv", 153.10, "아파트 전월세 최종본", "2021~2026 병합·행정동 매핑·금액 정제"],
    ["processed", "부동산", "행정동별_빌라_매매_최종.csv", 31.11, "빌라 매매 실거래 최종본", "행정동 매핑·취소거래 제외"],
    ["processed", "부동산", "행정동별_빌라_전월세_최종.csv", 86.20, "빌라 전월세 최종본", "2021~2026 병합·행정동 매핑·금액 정제"],
    ["raw", "교통사고", "TAAS_서울_고속국도제외_2010-2014.xlsx 외 3개", 0.03, "사고·사망·부상 원본", "2010~2025 기간별 4개 파일"],
    ["raw", "교통사고", "어린이교통사고_2010_2025.xlsx", 0.14, "어린이 교통사고", "다운로드 조건 확인 필요"],
    ["raw", "교통", "교통량_연차보고서/*.pdf", 145.10, "서울시 교통량 조사", "2017~2024 보고서 8개"],
    ["raw", "교통", "KOSIS_서울시_도로현황*.csv", None, "도로면적·도로율", "2017~2025"],
    ["raw", "교통", "서울시_차량통행속도_월별_구별_2017_2025.csv", None, "차량 통행속도", "월별 자치구 자료"],
    ["raw", "교통", "교통법규위반/*.csv", None, "경찰서별 단속현황", "2022~2024 파일 3개"],
    ["raw", "교통", "교통단속CCTV_최신/*.json", 34.81, "무인교통단속카메라 원본", "API 페이지 응답 5개"],
    ["raw", "인구", "연령별인구현황/*.csv", 0.20, "연령별 주민등록인구", "2017~2025 파일 9개"],
    ["raw", "범죄", "범죄_2014_2025.csv 외 3개", 12.47, "5대범죄·경찰시설·CCTV", "범죄 원본 4개"],
    ["raw", "생활안전", "생활안전 원본 10개", 0.95, "119 구급·고령자·AED", "일부 지표만 확보"],
    ["interim", "교통사고", "교통사고 중간가공 30개", 2.65, "자치구·연도 패널 및 안전기준표", "원천을 정규화·결합한 파생자료"],
    ["processed", "교통사고", "13차·19차·25차·26차 모델데이터", 0.12, "교통사고 모델 학습자료", "새 원천이 아닌 선정 변수 데이터셋"],
]

GAPS = [
    ["출처 메타데이터", "아파트·빌라 매매", "직접 원본·다운로드일·URL·SHA-256이 없음", "원본 매니페스트 생성", "높음"],
    ["출처 메타데이터", "서울시 물가상승률", "공식 통계표명과 URL이 없음", "파일 원출처 확인 후 기록", "높음"],
    ["출처 메타데이터", "어린이 교통사고", "TAAS 다운로드 조건이 없음", "조회 조건·다운로드일 기록", "중간"],
    ["출처 메타데이터", "범죄 통계", "공식 URL과 경찰서→자치구 규칙이 불명확", "원본 페이지와 매핑표 보존", "높음"],
    ["출처 메타데이터", "자동차등록·CCTV", "정확한 데이터셋 페이지·다운로드일 부족", "수집 매니페스트 보완", "중간"],
    ["미확보", "제조업 종사자수", "생활안전 기준표 원본 없음", "전국사업체조사 자료 수집", "높음"],
    ["부분 확보", "재난약자수", "65세 이상 인구만 확보", "유치원생·초등학생 등 결합", "높음"],
    ["미확보", "공공질서 및 안전예산", "원본 없음", "지방재정365 결산·예산 자료 수집", "중간"],
    ["미확보", "안전신문고 신고건수", "원본 없음", "행정안전부 원자료 확보", "중간"],
    ["미확보", "응급처치 교육인원", "원본 없음", "소방청·서울소방재난본부 자료 확보", "중간"],
]

URLS = [[row[1], row[2], row[3], row[6]] for row in SOURCES if row[6]]


def add_sheet(wb: Workbook, title: str, headers: list[str], rows: list[list[object]]) -> None:
    ws = wb.create_sheet(title)
    ws.append(headers)
    for row in rows:
        ws.append(row)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    ws.sheet_view.showGridLines = False

    navy = "17365D"
    light_blue = "D9EAF7"
    thin = Side(style="thin", color="D9E2F3")
    for cell in ws[1]:
        cell.fill = PatternFill("solid", fgColor=navy)
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[1].height = 32

    for row_index in range(2, ws.max_row + 1):
        if row_index % 2 == 0:
            for cell in ws[row_index]:
                cell.fill = PatternFill("solid", fgColor=light_blue)
        for cell in ws[row_index]:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = Border(bottom=thin)

    for col_index, header in enumerate(headers, 1):
        values = [str(ws.cell(r, col_index).value or "") for r in range(1, min(ws.max_row, 80) + 1)]
        width = min(max(max(map(len, values)) + 2, len(header) + 2), 48)
        ws.column_dimensions[get_column_letter(col_index)].width = width


def main() -> None:
    wb = Workbook()
    wb.remove(wb.active)
    summary = [
        ["작성 기준일", "2026-11-05"],
        ["조사 범위", "data/raw, data/interim, data/processed"],
        ["보유 파일 수", 103],
        ["전체 용량(MB)", 530.71],
        ["핵심 분야", "부동산, 교통사고·교통안전, 범죄, 생활안전, 인구"],
        ["확인 수준", "확인: 코드·파일에 출처 기록 / 높은 확률: 구조 일치 / 확인 필요: 원 URL·매니페스트 부족"],
        ["주의", "interim·processed 파일은 원천 데이터의 정제·집계·결합 결과이며 별도 외부 출처가 아님"],
        ["권장 출처 표기", "부동산 매매는 국토교통부 실거래가 공개시스템, 전월세는 서울 열린데이터광장, 교통사고는 한국도로교통공단 TAAS 자료를 사용함."],
    ]
    add_sheet(wb, "요약", ["항목", "내용"], summary)
    add_sheet(wb, "원천데이터_출처", ["분야", "확보 데이터", "제공기관", "원천 시스템·데이터명", "기간·범위", "보유 위치", "공식 URL", "확인 수준", "비고"], SOURCES)
    add_sheet(wb, "보유파일_요약", ["단계", "분야", "파일·파일군", "용량(MB)", "내용", "가공·주의사항"], FILES)
    add_sheet(wb, "공식URL", ["데이터", "제공기관", "시스템·데이터명", "URL"], URLS)
    add_sheet(wb, "보완_미확보", ["상태", "데이터", "현재 문제", "필요 조치", "우선순위"], GAPS)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
