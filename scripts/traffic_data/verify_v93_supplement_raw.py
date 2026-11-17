"""Validate downloaded originals and record coverage and remaining gaps."""
from pathlib import Path
import hashlib
import json
import zipfile
import pandas as pd
import openpyxl

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'data/93차_성능개선_추가수집'


def main():
    records=json.loads((BASE/'수집기록/download_results.json').read_text(encoding='utf-8'))
    checks=[]
    for r in records:
        p=ROOT/r['local_path']
        if not p.exists():
            if r['category'].startswith('서울시_교통량'):
                p=ROOT/'data/94차_추가사용데이터/raw/서울시_TOPIS_교통량'/str(r['year'])/r['filename']
            else:
                p=ROOT/'data/93차_미사용데이터/raw/TAAS_성능개선추가수집'/r['filename']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
        check={'분류':r['category'],'파일':r['filename'],'해시일치':True}
        if p.suffix=='.xlsx':
            with zipfile.ZipFile(p) as z: assert z.testzip() is None
            w=openpyxl.load_workbook(p,read_only=True,data_only=True)
            sheets=[]
            for ws in w:
                rows=ws.iter_rows(values_only=True)
                first=next(rows,())
                if '일자' in first and '지점명' in first and '0시' in first and '23시' in first:
                    data=next(rows,())
                    sample=str(data[first.index('일자')])
                    assert sample.startswith(str(r['year'])),(p,sample)
                    expected_month=str(r.get('month') or '').zfill(2)
                    assert sample[4:6]==expected_month,(p,sample,expected_month)
                    sheets.append(ws.title)
            assert sheets,p
            check.update(검증='ZIP 손상 없음; 일자/지점/24시간 열 및 첫 행 연월 확인',데이터시트=';'.join(sheets),연도=r['year'],월=r['month'])
            w.close()
        else:
            d=pd.read_csv(p,encoding=r['encoding'])
            check.update(검증='CSV 전체 읽기 성공',행수=len(d),열수=len(d.columns),컬럼=' / '.join(d.columns))
            if '연도' in d:
                check.update(최소연도=int(d['연도'].min()),최대연도=int(d['연도'].max()))
            else:
                check['주의']='연도 열 없음. 다발지ID와 발생연도 대응을 코드표로 확인한 후 사용'
        checks.append(check)
    monthly=[r for r in records if r['category'].startswith('서울시_교통량')]
    assert len(monthly)==108
    assert len({(r['year'],r['month']) for r in monthly})==108
    backup=ROOT/'data/93차_원천데이터백업/raw/교통사고 모델 데이터'
    existing=[]
    for p in backup.rglob('*'):
        if p.is_file() and any(x in p.name for x in ('자동차등록','어린이보호구역','사고다발','보행어린이')):
            existing.append({'기존파일':str(p.relative_to(ROOT)),'상태':'기존 raw 보유; 신규 확보로 집계하지 않음'})
    gaps=[
        {'항목':'보호구역 내 보행어린이 사고의 월별·시간대별 상세표','상태':'미확보','확인결과':'TAAS 메뉴 존재 확인. 세션 초기화 후에도 공개 조회요청 HTTP 500 / 세션 만료 반복','공식URL':'https://taas.koroad.or.kr/sta/acs/exs/typical.do?menuId=WEB_KMP_OVT_UAS_ASA&patternId=7235','대안검토':'공공데이터포털 시간대별 통계는 전국 전체사고 집계여서 자치구별 어린이 보호구역 자료를 대체하지 못함','대안URL':'https://www.data.go.kr/data/15087037/fileData.do'},
        {'항목':'자치구별 월별 자동차 등록대수','상태':'월별 신규파일 미확보; 연간 2017~2025 기존 raw 보유','확인결과':'TAAS 조회 세션 오류; 서울시 공개 상세페이지는 확인했으나 조회 응답에 원본 통계표 ID가 비어 있어 CSV 확보 실패','공식URL':'https://data.seoul.go.kr/dataList/OA-693/S/1/datasetView.do','대안검토':'기존 연간 자동차 등록 raw 활용 가능. 사고율에서 등록대수를 역산한 값을 raw로 만들지 않음'},
    ]
    notes=[
        'raw는 공식 사이트에서 받은 파일 바이트를 그대로 보존했다. 서울 필터링·집계·결측치 보정은 하지 않았다.',
        'TAAS CSV는 전국 자료이다. 서울 지역 코드 필터링과 연도 대응 확인이 필요하다.',
        '사고다발지역은 선정조건에 해당하는 지점만 포함한다. 전체 보호구역 사고건수와 동일하지 않다.',
        'TAAS 스쿨존 어린이 사고와 보행어린이 사고는 범위가 다르다. 기존 보행어린이 부상자수에 바로 대입하지 않는다.',
        '2017~2025 교통량 108개월 확보. 지점별 일자별 0~23시 자료이며 일부 최신 파일에 주소·좌표 시트가 있다.',
        '교통량 관측지점은 모든 학교 주변을 대표하지 않는다. 지점의 자치구 매핑, 관측지점 변경, 결측·특수기호를 점검한 후 사용한다.',
        '다운로드 URL 중 TOPIS는 POST 요청이다. 일반 링크 클릭만으로 재다운로드되지 않을 수 있어 출처페이지와 POST 요청값을 함께 기록했다.',
        '출처 페이지 안내상 TAAS 자료 범위는 파일명에 기재했다. 다발지역 CSV에는 명시적 연도 열이 없어 발생연도 해석은 추가 확인 대상이다.',
        'TOPIS 교통량 108개는 94차 모델에 사용했고, TAAS CSV 4개는 연도 식별 문제로 사용하지 않았다.',
        '94차 독립검증에서 정확등급 정확도는 2년·3년 후 모두 68%로 93차와 동일했다.',
        '2025년 오답을 참고한 신규 모델은 2025년을 새 독립평가로 주장하지 않는다. 예측 당시 공개되어 있었던 데이터만 사용하도록 공표일도 고려한다.',
    ]
    book=BASE/'수집기록/추가수집_출처와다운로드.xlsx'
    with pd.ExcelWriter(book,engine='openpyxl',mode='a',if_sheet_exists='replace') as writer:
        pd.DataFrame(checks).to_excel(writer,sheet_name='파일검증',index=False)
        pd.DataFrame(existing).to_excel(writer,sheet_name='기존보유자료',index=False)
        pd.DataFrame(gaps).to_excel(writer,sheet_name='미확보와대안',index=False)
        pd.DataFrame({'사용전확인사항':notes}).to_excel(writer,sheet_name='사용전확인',index=False)
        for ws in writer.book:
            ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width=min(65,max(16,max(len(str(c.value or '')) for c in col[:30])+3))
    print('Verified',len(checks),'files;',len(monthly),'unique months')


if __name__=='__main__': main()
