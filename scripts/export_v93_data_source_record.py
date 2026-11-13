from __future__ import annotations

from datetime import date
import hashlib
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tmp" / "교통사고_93차_데이터출처_및_갱신주기.xlsx"
PROCESSED = ROOT / "data" / "processed" / "교통사고_93차_데이터.xlsx"
RAW_ROOT = ROOT / "data" / "93차_원천데이터백업" / "raw" / "교통사고 모델 데이터"


SOURCES = [
    {
        "분류": "교통사고 기본 통계",
        "93차 사용 내용": "전체 사고·사망·부상 건수",
        "제공기관": "한국도로교통공단",
        "원천 시스템·자료": "교통사고분석시스템(TAAS), 서울 자치구별 교통사고 통계(고속국도 제외)",
        "보유기간": "2010~2025 (93차 패널은 2017~2025)",
        "공식 갱신주기": "연 1회",
        "93차 반영 권장주기": "연 1회",
        "확인수준": "확인",
        "확인 근거·주의사항": "TAAS 기간별 원본 파일과 결합·검증 코드가 존재함. 최신 연도는 통상 다음 해 확정 통계 발표 후 반영.",
        "공식 URL": "https://taas.koroad.or.kr/",
    },
    {
        "분류": "어린이·유형별 교통사고",
        "93차 사용 내용": "어린이, 보행어린이, 어린이보호구역, 음주, 자전거 사고·피해자 지표",
        "제공기관": "한국도로교통공단",
        "원천 시스템·자료": "교통사고분석시스템(TAAS) 상세 통계",
        "보유기간": "2010~2025 또는 2017~2025",
        "공식 갱신주기": "연 1회",
        "93차 반영 권장주기": "연 1회",
        "확인수준": "확인",
        "확인 근거·주의사항": "사용자가 TAAS에서 직접 확보한 공공데이터임을 확인함. 향후 동일 조건 재수집을 위해 세부 조회 조건과 다운로드일만 추가 기록 권장.",
        "공식 URL": "https://taas.koroad.or.kr/web/shp/sbm/initTfcacdStats.do",
    },
    {
        "분류": "교통사고 다발지역",
        "93차 사용 내용": "전체·법규위반·보행어린이·보호구역어린이·자전거 다발지역 집계",
        "제공기관": "한국도로교통공단",
        "원천 시스템·자료": "교통사고분석시스템(TAAS) 교통사고 다발지역 정보",
        "보유기간": "2017~2025 패널로 가공",
        "공식 갱신주기": "연 1회",
        "93차 반영 권장주기": "연 1회",
        "확인수준": "확인",
        "확인 근거·주의사항": "사용자가 TAAS에서 직접 확보한 공공데이터임을 확인함. 원본 조회 조건, 공간 집계 반경과 다운로드일은 재현성 관리를 위해 별도 기록 권장.",
        "공식 URL": "https://taas.koroad.or.kr/",
    },
    {
        "분류": "주민등록인구",
        "93차 사용 내용": "총인구 및 영유아·어린이·아동·청소년·청년·노인 인구",
        "제공기관": "행정안전부",
        "원천 시스템·자료": "주민등록 인구통계 연령별 인구현황",
        "보유기간": "2017~2025년 12월 기준",
        "공식 갱신주기": "매월",
        "93차 반영 권장주기": "연 1회, 12월 말 기준",
        "확인수준": "확인",
        "확인 근거·주의사항": "보유 CSV의 행정구역명·10자리 행정기관코드와 0세~100세 이상 1세별 컬럼 구조가 행정안전부 행정동별 연령별 인구현황 내려받기 형식과 일치함. 주민등록인구는 외국인을 제외하며, 93차는 각 연도 12월 말 값을 사용함.",
        "공식 URL": "https://jumin.mois.go.kr/ageStatMonth.do",
    },
    {
        "분류": "자동차 등록",
        "93차 사용 내용": "자동차 합계 및 승용·승합·화물·특수·이륜·건설기계·기타 등록대수",
        "제공기관": "한국도로교통공단",
        "원천 시스템·자료": "교통사고분석시스템(TAAS) 시군구별 자동차등록대수",
        "보유기간": "2006~2025 (93차는 2017~2025)",
        "공식 갱신주기": "연 1회",
        "93차 반영 권장주기": "연 1회, 확정값 기준",
        "확인수준": "확인",
        "확인 근거·주의사항": "보유 원본의 시도·시군구·차량용도 및 승용·승합·화물·특수·이륜·건설기계·농기계 구조가 TAAS 원자료와 일치함. TAAS 안내상 농기계·건설기계는 수집 시점에 따라 전년도 값이 임시 사용되고 확정 후 변경될 수 있으므로 최신 확정값 재확인 필요.",
        "공식 URL": "https://taas.koroad.or.kr/sta/acs/exs/typical.do?menuId=WEB_KMP_OVT_UAS_ASA&patternId=7166",
    },
    {
        "분류": "도로 현황",
        "93차 사용 내용": "도로연장·도로면적·도로율",
        "제공기관": "국토교통부·서울특별시/KOSIS",
        "원천 시스템·자료": "도로현황 통계",
        "보유기간": "2017~2025",
        "공식 갱신주기": "연 1회",
        "93차 반영 권장주기": "연 1회",
        "확인수준": "확인",
        "확인 근거·주의사항": "KOSIS·서울시 도로현황 원본과 처리 기록 존재.",
        "공식 URL": "https://kosis.kr/metroindex/detailMetroIdct.do?graphTy=B&idctId=7&listId=7&page=all",
    },
    {
        "분류": "차량 통행속도",
        "93차 사용 내용": "승용차 평일·토요일·일요일·전체 평균속도",
        "제공기관": "서울특별시",
        "원천 시스템·자료": "서울시 교통정보·차량통행속도 월별 자치구 자료",
        "보유기간": "2017~2025",
        "공식 갱신주기": "월별",
        "93차 반영 권장주기": "연 1회, 월별 자료를 연평균 집계",
        "확인수준": "확인",
        "확인 근거·주의사항": "월별 자치구 자료를 연도 단위로 집계하여 사용.",
        "공식 URL": "https://data.seoul.go.kr/dataList/DT201004I040001/S/2/datasetView.do",
    },
    {
        "분류": "학교·교육",
        "93차 사용 내용": "유치원·초등·중등·고등 학교, 학생, 학급, 국공립·사립 수",
        "제공기관": "교육부·한국교육개발원",
        "원천 시스템·자료": "교육통계서비스(KESS) 학교·학생·학급 현황",
        "보유기간": "2017~2025 패널",
        "공식 갱신주기": "연 1회, 통상 4월 1일 기준",
        "93차 반영 권장주기": "연 1회",
        "확인수준": "확인",
        "확인 근거·주의사항": "보유 원본에 KEDI 작성자 정보, '유·초·중등 교육통계 학교별 주요 현황', 조사기준일 4월 1일이 명시되어 있고 학교·학생·학급·설립유형 컬럼이 KESS 학교별 데이터셋과 일치함. 2020년까지 인가학급, 2021년부터 편성학급을 사용한 점을 해석 시 유의.",
        "공식 URL": "https://kess.kedi.re.kr/",
    },
    {
        "분류": "교통안전 행동",
        "93차 사용 내용": "운전자·동승자 안전벨트, 음주운전·음주운전차량 동승, 사고·중독 경험률",
        "제공기관": "질병관리청",
        "원천 시스템·자료": "지역사회건강조사",
        "보유기간": "2008~2025 (93차는 2017~2025)",
        "공식 갱신주기": "연 1회",
        "93차 반영 권장주기": "연 1회",
        "확인수준": "확인",
        "확인 근거·주의사항": "자치구 단위 조사 결과. 조사 문항·산식 변경 여부를 갱신 때 확인해야 함.",
        "공식 URL": "https://chs.kdca.go.kr/chs/main.do",
    },
    {
        "분류": "CCTV",
        "93차 사용 내용": "방범·전체 CCTV 설치수 및 도로면적당 CCTV",
        "제공기관": "서울특별시",
        "원천 시스템·자료": "서울 열린데이터광장 「서울시 자치구 (연도별) CCTV 설치현황」 및 「서울시 자치구 (범죄예방 수사용) CCTV 설치현황」",
        "보유기간": "2017~2025 모델 패널, 2026년 6월 30일 기준 공개파일에서 연도별 값 사용",
        "공식 갱신주기": "비정기, 자료 변경 시",
        "93차 반영 권장주기": "연 1회 스냅샷",
        "확인수준": "확인",
        "확인 근거·주의사항": "보유 원본 파일명·시트명·2026.6.30 기준일·자치구별 수치 구조가 공식 공개파일과 일치함. 전체 CCTV는 연도별 설치대수, 범죄예방 CCTV는 연도별 누적 설치현황이므로 두 표의 숫자 의미를 구분해야 함.",
        "공식 URL": "https://data.seoul.go.kr/dataList/OA-2734/F/1/datasetView.do | https://data.seoul.go.kr/dataList/OA-21097/F/1/datasetView.do",
    },
    {
        "분류": "파생지표",
        "93차 사용 내용": "어린이 1만 명당 사고·피해자·자동차, 도로면적당 사고·CCTV, 최근 3년 평균·추세·변동 등",
        "제공기관": "SpotFit 내부 계산",
        "원천 시스템·자료": "위 원천자료를 자치구·연도 단위로 결합 후 산식 적용",
        "보유기간": "2017~2025",
        "공식 갱신주기": "외부 갱신주기 없음",
        "93차 반영 권장주기": "원천자료 갱신 후 재계산",
        "확인수준": "확인",
        "확인 근거·주의사항": "새 외부 출처가 아니라 내부 파생값. 분모 0, 단위, 결측치 처리와 최근 3년 계산 범위를 함께 관리해야 함.",
        "공식 URL": "해당 없음",
    },
]


DOWNLOAD_GUIDE = [
    {
        "분류": "교통사고 기본 통계·유형별 사고",
        "제공기관": "한국도로교통공단",
        "다운로드/조회 페이지": "https://taas.koroad.or.kr/",
        "직접 다운로드 여부": "조회 조건 선택 후 파일 저장",
        "재수집 조건": "서울특별시, 자치구, 해당 연도, 고속국도 제외. 전체·어린이·보행어린이·보호구역·음주·자전거 사고를 각각 조회",
        "공식 갱신주기": "연 1회",
        "93차 권장 갱신": "전년도 확정 통계 공개 후 연 1회",
    },
    {
        "분류": "교통사고 다발지역",
        "제공기관": "한국도로교통공단",
        "다운로드/조회 페이지": "https://taas.koroad.or.kr/web/shp/sbm/initGisAnals.do?menuId=WEB_KMP_GIS",
        "직접 다운로드 여부": "GIS 조건 선택 후 저장",
        "재수집 조건": "서울특별시, 전체·법규위반·보행어린이·스쿨존 어린이·자전거 다발지역",
        "공식 갱신주기": "연 1회",
        "93차 권장 갱신": "전년도 자료 공개 후 연 1회",
    },
    {
        "분류": "주민등록 연령별 인구",
        "제공기관": "행정안전부",
        "다운로드/조회 페이지": "https://jumin.mois.go.kr/ageStatMonth.do",
        "직접 다운로드 여부": "페이지에서 CSV 다운로드",
        "재수집 조건": "행정동, 서울 25개 자치구, 각 연도 12월, 계, 1세 단위",
        "공식 갱신주기": "매월 말 작성, 다음 달 1일 12시 이후 공표(휴일이면 다음 평일)",
        "93차 권장 갱신": "매년 12월 말 기준값으로 연 1회",
    },
    {
        "분류": "학교·교육",
        "제공기관": "교육부·한국교육개발원",
        "다운로드/조회 페이지": "https://kess.kedi.re.kr/",
        "직접 다운로드 여부": "KESS 학교별 주요 현황에서 다운로드",
        "재수집 조건": "유·초·중등 학교별 주요 현황, 서울, 매년 4월 1일 기준",
        "공식 갱신주기": "연 1회(4월 1일 조사 기준)",
        "93차 권장 갱신": "연 1회",
    },
    {
        "분류": "교통안전 행동",
        "제공기관": "질병관리청",
        "다운로드/조회 페이지": "https://chs.kdca.go.kr/chs/index.do",
        "직접 다운로드 여부": "원시자료 이용 신청·승인 후 다운로드",
        "재수집 조건": "서울 25개 자치구, 안전벨트·음주운전·동승·사고중독 문항, 개인가중치 적용",
        "공식 갱신주기": "연 1회 조사, 원시자료는 통상 다음 해 공개",
        "93차 권장 갱신": "새 원시자료 공개 후 연 1회",
    },
    {
        "분류": "차량 통행속도",
        "제공기관": "서울특별시",
        "다운로드/조회 페이지": "https://data.seoul.go.kr/dataList/DT201004I040001/S/2/datasetView.do",
        "직접 다운로드 여부": "파일 내려받기",
        "재수집 조건": "승용차, 자치구, 평일·토요일·일요일·전체 평균속도",
        "공식 갱신주기": "정기(매년)",
        "93차 권장 갱신": "연 1회",
    },
    {
        "분류": "전체 CCTV",
        "제공기관": "서울특별시",
        "다운로드/조회 페이지": "https://data.seoul.go.kr/dataList/OA-2734/F/1/datasetView.do",
        "직접 다운로드 여부": "파일 내려받기",
        "재수집 조건": "서울시 자치구 연도별 CCTV 설치현황, 기준일 기록",
        "공식 갱신주기": "비정기(자료 변경 시)",
        "93차 권장 갱신": "연 1회 최신 스냅샷 확인",
    },
    {
        "분류": "범죄예방 CCTV",
        "제공기관": "서울특별시",
        "다운로드/조회 페이지": "https://data.seoul.go.kr/dataList/OA-21097/F/1/datasetView.do",
        "직접 다운로드 여부": "파일 내려받기",
        "재수집 조건": "서울시 자치구 범죄예방·수사용 CCTV, 기준일과 누적값 여부 기록",
        "공식 갱신주기": "비정기(자료 변경 시)",
        "93차 권장 갱신": "연 1회 최신 스냅샷 확인",
    },
    {
        "분류": "도로 현황",
        "제공기관": "국토교통부·KOSIS",
        "다운로드/조회 페이지": "https://kosis.kr/metroindex/detailMetroIdct.do?graphTy=B&idctId=7&listId=7&page=all",
        "직접 다운로드 여부": "페이지에서 CSV 저장",
        "재수집 조건": "서울 자치구, 도로연장·도로면적·도로율, 연도별",
        "공식 갱신주기": "1년",
        "93차 권장 갱신": "연 1회",
    },
]


def raw_source(file: Path) -> tuple[str, str]:
    relative = file.relative_to(RAW_ROOT)
    top = relative.parts[0]
    name = file.name
    if top == "사고다발지역":
        return "교통사고 다발지역", "한국도로교통공단 TAAS"
    if top == "연령":
        return "주민등록 연령별 인구", "행정안전부"
    if top == "학교":
        return "학교·교육", "교육부·한국교육개발원 KESS"
    if top == "질병관리청":
        return "교통안전 행동", "질병관리청 지역사회건강조사"
    if "CCTV" in name:
        return "CCTV", "서울특별시 열린데이터광장"
    if "도로현황" in name:
        return "도로 현황", "국토교통부·서울특별시/KOSIS"
    if "통행속도" in name:
        return "차량 통행속도", "서울특별시"
    if "자동차등록" in name:
        return "자동차 등록", "한국도로교통공단 TAAS"
    return "교통사고 기본·유형별 통계", "한국도로교통공단 TAAS"


def build_raw_inventory() -> pd.DataFrame:
    rows = []
    for file in sorted(RAW_ROOT.rglob("*")):
        if not file.is_file():
            continue
        category, source = raw_source(file)
        rows.append({
            "원본 상대경로": file.relative_to(RAW_ROOT).as_posix(),
            "파일명": file.name,
            "분류": category,
            "제공기관·시스템": source,
            "파일크기(byte)": file.stat().st_size,
            "SHA-256": hashlib.sha256(file.read_bytes()).hexdigest(),
            "93차 원천백업 포함": "예",
        })
    return pd.DataFrame(rows)


def classify(column: str) -> str:
    if column in {"연도", "예측연도", "목표연도", "자치구", "데이터역할"}:
        return "식별자"
    derived_tokens = ("1만명당", "도로면적당", "도로km당", "최근3년", "인구비율", "전체교통사고건수", "전년증감")
    if any(token in column for token in derived_tokens):
        return "파생지표"
    if "다발" in column:
        return "교통사고 다발지역"
    if "교통사고" in column or "음주운전 사고" in column or "자전거 사고" in column:
        return "교통사고 기본 통계" if column.startswith("교통사고 ") else "어린이·유형별 교통사고"
    if "등록대수" in column:
        return "자동차 등록"
    if "인구수" in column or column == "총인구수":
        return "주민등록인구"
    if column in {"도로연장(km)", "도로면적(km²)", "도로율(%)"}:
        return "도로 현황"
    if "평균속도" in column:
        return "차량 통행속도"
    if "CCTV" in column:
        return "CCTV"
    if any(token in column for token in ("유치원", "초등학교", "중학교", "고등학교", "중고_")):
        return "학교·교육"
    if any(token in column for token in ("안전벨트", "음주운전경험", "음주운전차량", "사고중독")):
        return "교통안전 행동"
    return "출처 미분류"


def style_workbook(path: Path) -> None:
    from openpyxl import load_workbook

    wb = load_workbook(path)
    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    status_fills = {
        "확인": PatternFill("solid", fgColor="C6E0B4"),
        "높은 확률": PatternFill("solid", fgColor="FFF2CC"),
        "확인 필요": PatternFill("solid", fgColor="F4CCCC"),
    }
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws.row_dimensions[1].height = 32
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
                if cell.value in status_fills:
                    cell.fill = status_fills[cell.value]
                if isinstance(cell.value, str) and cell.value.startswith("https://"):
                    cell.hyperlink = cell.value
                    cell.style = "Hyperlink"
        for idx, column_cells in enumerate(ws.columns, 1):
            values = [str(c.value or "") for c in list(column_cells)[:200]]
            width = min(max(max(map(len, values), default=8) + 2, 10), 45)
            ws.column_dimensions[get_column_letter(idx)].width = width
    wb.save(path)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    processed = PROCESSED
    panel = pd.read_excel(processed, sheet_name=2)
    source_df = pd.DataFrame(SOURCES)
    lookup = source_df.set_index("분류").to_dict("index")

    column_rows = []
    for index, column in enumerate(panel.columns, 1):
        category = classify(str(column))
        meta = lookup.get(category, {})
        column_rows.append({
            "순번": index,
            "93차 전체정제패널 컬럼": column,
            "출처 분류": category,
            "제공기관": meta.get("제공기관", "해당 없음" if category == "식별자" else "미확인"),
            "확인수준": meta.get("확인수준", "확인" if category == "식별자" else "확인 필요"),
            "갱신주기": meta.get("공식 갱신주기", "해당 없음"),
            "비고": meta.get("확인 근거·주의사항", "연도·자치구·데이터 역할 식별 컬럼"),
        })

    status_counts = source_df["확인수준"].value_counts()
    overview = pd.DataFrame([
        {"항목": "문서명", "내용": "교통사고 93차 모델 데이터 출처 및 갱신주기 기록"},
        {"항목": "재점검일", "내용": date.today().isoformat()},
        {"항목": "대상 파일", "내용": processed.name},
        {"항목": "대상 범위", "내용": f"2017~2025, 서울 25개 자치구, 전체 정제패널 {len(panel)}행 × {len(panel.columns)}열"},
        {"항목": "권장 모델 갱신주기", "내용": "연 1회. 전년도 TAAS·지역사회건강조사·교육·도로통계가 확정된 뒤 일괄 갱신"},
        {"항목": "출처 확인", "내용": f"{status_counts.get('확인', 0)}개 데이터군; 원본 파일 {len(build_raw_inventory())}개 해시 점검"},
        {"항목": "출처 높은 확률", "내용": f"{status_counts.get('높은 확률', 0)}개 데이터군"},
        {"항목": "출처 확인 필요", "내용": f"{status_counts.get('확인 필요', 0)}개 데이터군"},
        {"항목": "핵심 주의", "내용": "공식 원천의 갱신주기와 93차 모델 반영주기는 다름. 월별 자료도 모델에서는 연도 단위로 집계함."},
        {"항목": "교통사고 출처 확인", "내용": "전체·어린이·보행어린이·보호구역·음주·자전거 사고와 다발지역 자료는 사용자가 TAAS에서 직접 확보한 공공데이터로 확인함."},
        {"항목": "기록 보완 항목", "내용": "출처가 틀렸다는 의미가 아니라, 향후 같은 자료를 재수집하기 위한 통계표 번호·조회 조건·다운로드일 기록이 필요한 항목을 표시함."},
    ])

    checklist = pd.DataFrame([
        {"우선순위": 1, "대상": "어린이·유형별 사고 및 다발지역", "확인할 것": "출처는 TAAS로 확인 완료. 재수집용 사고유형·연령·도로종류·지역 조건, 다운로드일, 원본 파일명 기록", "완료": "출처 확인 완료"},
        {"우선순위": 2, "대상": "자동차 등록대수", "확인할 것": "출처는 TAAS 시군구별 자동차등록대수로 확인 완료. 갱신 시 농기계·건설기계 확정 여부와 다운로드일 기록", "완료": "출처 확인 완료"},
        {"우선순위": 3, "대상": "CCTV", "확인할 것": "출처 확인 완료: 서울 열린데이터광장 OA-2734·OA-21097. 갱신 시 기준일과 연도별/누적값 의미 확인", "완료": "출처 확인 완료"},
        {"우선순위": 4, "대상": "학교·교육통계", "확인할 것": "출처 확인 완료: KEDI 교육통계서비스(KESS) 학교별 주요 현황. 갱신 시 4월 1일 기준 및 학급 정의 확인", "완료": "출처 확인 완료"},
        {"우선순위": 5, "대상": "연령별 인구", "확인할 것": "출처 확인 완료: 행정안전부 행정동별 연령별 인구현황. 갱신 시 12월 말·계·1세 단위·외국인 제외 조건 유지", "완료": "출처 확인 완료"},
        {"우선순위": 6, "대상": "공통", "확인할 것": "원본 SHA-256, 수집일, 생성 스크립트 버전, 결측치·중복 처리 규칙", "완료": "□"},
    ])

    urls = source_df[["분류", "제공기관", "원천 시스템·자료", "공식 URL", "공식 갱신주기", "93차 반영 권장주기", "확인수준"]].copy()
    raw_inventory = build_raw_inventory()
    download_guide = pd.DataFrame(DOWNLOAD_GUIDE)
    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        overview.to_excel(writer, sheet_name="00_요약", index=False)
        source_df.to_excel(writer, sheet_name="01_출처상세", index=False)
        pd.DataFrame(column_rows).to_excel(writer, sheet_name="02_컬럼별출처", index=False)
        checklist.to_excel(writer, sheet_name="03_갱신체크리스트", index=False)
        urls.to_excel(writer, sheet_name="04_공식URL", index=False)
        download_guide.to_excel(writer, sheet_name="05_다운로드안내", index=False)
        raw_inventory.to_excel(writer, sheet_name="06_RAW파일목록", index=False)
    style_workbook(OUTPUT)
    print(OUTPUT)
    print(f"source_groups={len(source_df)}, mapped_columns={len(column_rows)}")


if __name__ == "__main__":
    main()
