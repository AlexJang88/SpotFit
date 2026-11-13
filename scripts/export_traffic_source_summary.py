from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tmp" / "11.13_교통사고_다운로드경로_상세점검.xlsx"

ROWS = [
    ["교통사고", "사고·사망·부상", "한국도로교통공단 TAAS", "XLSX·CSV/API", "시군구별 통계·엑셀다운로드", "연 1회", "https://taas.koroad.or.kr/sta/acs/gus/selectSignguTfcacd.do?menuId=WEB_KMP_OVT_MVT_TAG_SGT"],
    ["", "어린이·보행어린이·보호구역 사고", "한국도로교통공단 TAAS", "XLSX·CSV/API", "어린이 관련 상세통계", "연 1회", "https://taas.koroad.or.kr/web/shp/sbm/initUnityAnalsSys.do?menuId=WEB_KMP_OVT_UAS"],
    ["", "사고 다발지역", "한국도로교통공단 TAAS", "CSV/API", "TAAS GIS 분석", "연 1회", "https://taas.koroad.or.kr/web/shp/sbm/initGisAnals.do?menuId=WEB_KMP_GIS"],
    ["", "자동차 등록대수", "한국도로교통공단 TAAS", "XLSX", "자동차등록대수 통계", "연 1회", "https://taas.koroad.or.kr/sta/acs/exs/typical.do?menuId=WEB_KMP_OVT_UAS_ASA&patternId=7166"],
    ["", "도로현황", "국토교통부·KOSIS", "CSV", "KOSIS 도로 관련 지표", "연 1회", "https://kosis.kr/metroindex/detailMetroIdct.do?graphTy=B&idctId=7&listId=7&page=all"],
    ["", "평균 통행속도", "서울특별시", "XLSX·CSV", "서울시 차량통행속도", "연 1회", "https://data.seoul.go.kr/dataList/DT201004I040001/S/2/datasetView.do"],
    ["", "주민등록 연령별 인구", "행정안전부", "CSV", "행정동별 연령별 인구현황", "월 1회", "https://jumin.mois.go.kr/ageStatMonth.do"],
    ["", "학교 현황", "교육부·한국교육개발원", "XLSX", "KESS 학교별 자료 다운로드", "연 1회", "https://kess.kedi.re.kr/"],
    ["", "안전벨트·음주운전 경험", "질병관리청", "원시자료·XLSX 정리본", "원시자료 이용 신청", "연 1회", "https://chs.kdca.go.kr/chs/rdr/rdrInfoPledgeMain.do"],
    ["", "CCTV", "서울특별시 열린데이터광장", "XLSX", "연도별 CCTV / 범죄예방 CCTV", "비정기", "https://data.seoul.go.kr/dataList/OA-2734/F/1/datasetView.do | https://data.seoul.go.kr/dataList/OA-21097/F/1/datasetView.do"],
]


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "11.13 교통사고"

    ws.append(["<11.13 교통사고>", "", "", "", "", "", ""])
    ws.append(["분류", "사용 데이터", "제공기관", "원본 형식", "다운로드·조회", "갱신주기", "URL"])
    for original in ROWS:
        row = original.copy()
        name = row[1]
        if name == "어린이·보행어린이·보호구역 사고":
            for label, endpoint, menu in [
                ("어린이 사고", "selectChildTfcacd", "CHT"),
                ("보행어린이 사고", "selectChildPdstrnTfcacd", "CPT"),
                ("어린이보호구역 사고", "selectShczTfcacd", "SZT"),
            ]:
                item = row.copy()
                item[1] = label
                item[4] = "연도 선택 → 엑셀다운로드\n전국 요약 화면. 서울 자치구 자료는 상세보기에서 조건 설정 필요"
                item[6] = f"https://taas.koroad.or.kr/sta/acs/gus/{endpoint}.do?menuId=WEB_KMP_OVT_MVT_TAS_{menu}"
                ws.append(item)
            continue
        if name == "CCTV":
            for label, ident in [("CCTV (연도별)", "OA-2734"), ("CCTV (범죄예방·수사용)", "OA-21097")]:
                item = row.copy()
                item[1] = label
                item[4] = "파일내려받기 목록 → 원하는 기준일 파일의 내려받기 버튼\n목록 확인 완료"
                item[6] = f"https://data.seoul.go.kr/dataList/{ident}/F/1/datasetView.do"
                ws.append(item)
            continue
        if name == "도로현황":
            row[2] = "서울특별시 도로계획과"
            row[4] = "도로현황(도로율) 통계\n기존 도로보급률 링크 오류 수정. 페이지에 첨부파일 없음; 통계표 내보내기 버튼은 확인 못함"
            row[6] = "https://data.seoul.go.kr/dataList/OA-688/S/1/datasetView.do"
        elif name == "평균 통행속도":
            row[4] = "자치구·시간대별 승용차 통계\n페이지에 첨부파일 없음; 통계표 내보내기 버튼은 확인 못함"
            row[6] = "https://data.seoul.go.kr/dataList/DT201004O100010/S/2/datasetView.do"
        elif name == "학교 현황":
            row[4] = "KESS 연도별 다운로드 화면: 접근 차단으로 미확인\n홈 → 유초중등 학교별 데이터셋 → 연도 선택"
            ws.append(row)
            item = row.copy()
            item[1] = "학교 현황 (2022년 대체 다운로드)"
            item[4] = "교육부 공식 게시글 → 유초중등 XLSX 옆 다운로드\n2022년 자료에 한함. 보유 원본과 동일 파일 여부는 미검증"
            item[6] = "https://www.moe.go.kr/boardCnts/viewRenew.do?boardID=351&boardSeq=92557&lev=0"
            ws.append(item)
            continue
        elif name == "자동차 등록대수":
            row[4] = "교통여건 → 사회경제지표 → 시군구별 자동차등록대수\n연도·서울 조건 조회 후 저장. 자동 장표 선택은 미확인"
        elif name == "사고 다발지역":
            row[4] = "GIS 분석 → 교통사고 다발지역 → 유형·연도·서울 선택\n조건별 저장 화면까지는 미확인"
        elif name == "사고·사망·부상":
            row[4] = "시군구별 통계 → 연도·지역 선택 → 엑셀다운로드\n다운로드 버튼 표시 확인"
        elif name == "주민등록 연령별 인구":
            row[4] = "서울·연도별 12월·계·1세 단위 선택 → 조회 → CSV 저장\n조회 화면 확인; 지정 조건 자동 적용 링크는 없음"
        elif name == "안전벨트·음주운전 경험":
            row[4] = "원시자료 요청 → 이용계획 제출·심의 → 이용 권한 후 다운로드\n바로 다운로드 불가"
        ws.append(row)

    header_fill = PatternFill("solid", fgColor="1F4E78")
    section_fill = PatternFill("solid", fgColor="D9EAF7")
    thin = Side(style="thin", color="B7B7B7")

    ws.merge_cells("A1:G1")
    ws["A1"].font = Font(bold=True, size=13)
    ws["A1"].fill = section_fill
    ws["A1"].alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[1].height = 25

    for cell in ws[2]:
        cell.fill = header_fill
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center")

    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        for cell in row:
            cell.border = Border(left=thin, right=thin, top=thin, bottom=thin)
            cell.alignment = Alignment(vertical="center", wrap_text=True)

    for row in range(3, ws.max_row + 1):
        url_cell = ws.cell(row=row, column=7)
        first_url = str(url_cell.value).split(" | ")[0]
        url_cell.hyperlink = first_url
        url_cell.style = "Hyperlink"
        url_cell.alignment = Alignment(vertical="center", wrap_text=True)
        ws.cell(row, 5).hyperlink = first_url
        ws.cell(row, 5).font = Font(color="0563C1", underline="single")
        ws.row_dimensions[row].height = 78

    widths = {"A": 13, "B": 32, "C": 28, "D": 20, "E": 65, "F": 12, "G": 85}
    for column, width in widths.items():
        ws.column_dimensions[column].width = width
    ws.freeze_panes = "A3"
    ws.auto_filter.ref = f"A2:G{ws.max_row}"
    ws.sheet_view.showGridLines = False

    wb.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
