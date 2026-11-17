"""Download public original files; keep transformed data out of raw."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlencode
import io
import json
import re
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/93차_성능개선_추가수집'
RAW = OUT / 'raw'
META = OUT / '수집기록'
TAAS = 'https://opendata.koroad.or.kr/api/'
TOPIS = 'https://topis.seoul.go.kr'


def fetch(job):
    record = dict(job)
    path = RAW / job['category'] / job['filename']
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            content = path.read_bytes()
        else:
            r = requests.request(job.get('method','GET'),job['url'],data=job.get('params'),timeout=(15,55))
            r.raise_for_status()
            content = r.content
            if content.lstrip().lower().startswith((b'<!doctype',b'<html',b'<script')):
                raise ValueError('HTML response, not a data file')
            if path.suffix == '.xlsx' and not content.startswith(b'PK'):
                raise ValueError('Not XLSX')
            if path.suffix == '.csv':
                for encoding in ('utf-8-sig','cp949'):
                    try:
                        df = pd.read_csv(io.BytesIO(content),encoding=encoding)
                        record['encoding'] = encoding
                        record['rows'] = len(df)
                        record['columns'] = list(df.columns)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError('CSV encoding validation failed')
            path.write_bytes(content)
        record.update(status='다운로드 완료',bytes=len(content),sha256=sha256(content).hexdigest(),local_path=str(path.relative_to(ROOT)))
    except Exception as exc:
        record.update(status='실패',error=str(exc))
    print(job['filename']+': '+record['status'],flush=True)
    return record


def main():
    META.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for label, endpoint, page in [
        ('보행어린이_사고다발지역_2012_2025','childdown.jsp','selectChildDataSet.do'),
        ('어린이보호구역_사고다발지역_2012_2025','schoolzonedown.jsp','selectSchoolChildDataSet.do'),
        ('지자체별_사고다발지역_2017_2025','lgdown.jsp','selectLgDataSet.do'),
        ('지자체별_대상사고통계_2010_2025','sttdown.jsp','selectSttDataSet.do')]:
        jobs.append(dict(category='TAAS',filename=label+'.csv',url=TAAS+'down/'+endpoint,page=TAAS+page,cycle='연간'))
    s=requests.Session()
    for year in range(2017,2026):
        r=s.post(TOPIS+'/refroom/selectRefRoomListASC.do',data={'blbdDivCd':'08','bdwrDivCd':str(year),'mainBdwrRowNum':'12'},timeout=30)
        r.raise_for_status()
        rows=r.json()['rows']
        for row in rows:
            if not row.get('apndFileNm'): continue
            filename=re.sub(r'[<>:"/\\|?*]','_',row['apndFileNm'])
            params={k:row[k] for k in ('apndFileNm','apndFilePathNm','bdwrSeq','blbdDivCd')}
            jobs.append(dict(category=f'서울시_교통량/{year}',filename=filename,url=TOPIS+'/downloadFileRefRoom.do',method='POST',params=params,page=TOPIS+'/refRoom/openRefRoom_2.do',cycle='월간',year=year,month=row.get('months'),published=row.get('createDate')))
        print(f'{year}: {len(rows)} monthly entries',flush=True)
    (META/'download_jobs.json').write_text(json.dumps(jobs,ensure_ascii=False,indent=2),encoding='utf-8')
    records=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(fetch,j) for j in jobs]):
            records.append(f.result())
            (META/'download_results.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    table=[]
    for r in records:
        table.append({'분류':r['category'],'파일':r['filename'],'상태':r['status'],'원본페이지':r['page'],'다운로드URL':r['url'],'요청방식':r.get('method','GET'),'POST요청값':json.dumps(r.get('params',{}),ensure_ascii=False),'갱신주기':r['cycle'],'파일경로':r.get('local_path',''),'용량_bytes':r.get('bytes'),'SHA256':r.get('sha256',''),'확인시각':datetime.now().isoformat(timespec='seconds'),'오류':r.get('error','')})
    with pd.ExcelWriter(META/'추가수집_출처와다운로드.xlsx',engine='openpyxl') as writer:
        pd.DataFrame(table).sort_values(['분류','파일']).to_excel(writer,index=False,sheet_name='수집목록')
        ws=writer.book.active
        ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
        for cells in ws.columns:
            ws.column_dimensions[cells[0].column_letter].width=30
        for row in ws.iter_rows(min_row=2):
            for cell in (row[3],row[4]):
                cell.hyperlink=cell.value;cell.style='Hyperlink'
    print('SUMMARY',len(records),sum(r['status']=='다운로드 완료' for r in records),flush=True)


if __name__=='__main__':
    main()
