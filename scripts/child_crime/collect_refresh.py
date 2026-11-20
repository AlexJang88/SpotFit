"""Download public Seoul statistics using the portal's CSV download operation.

No login or API key; preview API is deliberately NOT used (1,000-row limit).
Downloaded snapshots never overwrite training inputs without validation.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data/raw/child_crime_exposure'
BASE = 'https://data.seoul.go.kr/bsp/wgs/dataset/'
SOURCES = {'crime': 481, 'migration': 64, 'business_density': 21337,
           'student_movement': 20799, 'household_structure': None, 'births_deaths': None}


def download_population(year: int) -> tuple[bytes, dict]:
    """MOIS public download form: year-end, single ages, all administrative dong."""
    params = {'sltOrgType': '1', 'sltOrgLvl1': 'A', 'sltOrgLvl2': '', 'gender': 'gender',
              'sum': 'sum', 'sltUndefType': '', 'searchYearStart': str(year), 'searchMonthStart': '12',
              'searchYearEnd': str(year), 'searchMonthEnd': '12', 'sltOrderType': '1',
              'sltOrderValue': 'ASC', 'sltArgTypes': '1', 'sltArgTypeA': '0', 'sltArgTypeB': '100',
              'category': 'year', 'state': '3'}
    url = 'https://jumin.mois.go.kr/downloadCsvAge.do?searchYearMonth=year&xlsStats=3'
    raw = fetch(Request(url, data=urlencode(params).encode(), headers={'Referer': 'https://jumin.mois.go.kr/ageStatYear.do'}))
    rows = list(csv.DictReader(io.StringIO(raw.decode('cp949'))))
    if not rows or f'{year}년_계_0세' not in rows[0] or f'{year}년_계_100세 이상' not in rows[0]:
        raise ValueError('MOIS year-end/single-age schema mismatch')
    seoul = [r for r in rows if r['행정구역'].startswith('서울특별시')]
    if len(seoul) < 400:
        raise ValueError('MOIS Seoul dong coverage incomplete')
    return raw, {'year': year, 'rows': len(rows), 'seoul_rows': len(seoul),
                 'source_url': 'https://jumin.mois.go.kr/ageStatYear.do',
                 'source_sha256': hashlib.sha256(raw).hexdigest()}


def fetch(request):
    for attempt in range(3):
        try:
            with urlopen(request, timeout=60) as response:
                return response.read()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def download(dataset_id: int) -> tuple[bytes, dict]:
    metadata = json.loads(fetch(BASE + f'metaTypeList.do?id={dataset_id}&currentPage=1&pagePer=100'))
    columns = metadata['metaTypeAllList']
    if not columns:
        raise ValueError('Source is an external statistical table, not a Data Hub CSV; use its source table')
    # The portal expects Korean display names, NOT database column identifiers.
    params = {'id': dataset_id, 'tdColNmArr': ','.join(c['colKorNm'] for c in columns),
              'rowFilterList': '[]'}
    raw = fetch(Request(BASE + 'dataCsvDown.do', data=urlencode(params).encode(),
                        headers={'User-Agent': 'SpotFit-data-refresh/1.0'}))
    text = raw.decode('cp949')
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows or not set(c['colKorNm'] for c in columns).issubset(rows[0]):
        raise ValueError('CSV schema/empty-response validation failed')
    detail = metadata['datasetDetail']
    return text.encode('utf-8-sig'), {
        'dataset_id': dataset_id, 'source_url': f'https://data.seoul.go.kr/bsp/wgs/dataView/data300View/{dataset_id}.do',
        'title': detail['trgtTableIndctNm'], 'cycle': detail.get('clctCycle'),
        'rows': len(rows), 'columns': list(rows[0]),
        'source_sha256': hashlib.sha256(raw).hexdigest(),
    }


def collect(resume=False) -> dict:
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    report = {'run_id': run_id, 'sources': {}, 'errors': {}}
    if resume:
        report = json.loads((RAW / 'refresh/latest.json').read_text(encoding='utf-8'))
        if report['run_id'][:8] != run_id[:8]:
            raise ValueError('Only same-day partial collection can be resumed')
        run_id = report['run_id']
    folder = RAW / 'refresh' / run_id
    folder.mkdir(parents=True, exist_ok=resume)
    for name, dataset_id in SOURCES.items():
        if name in report['sources']:
            continue
        try:
            if name in ('business_density', 'student_movement', 'household_structure', 'births_deaths'):
                from scripts.child_crime import collect_statistics
                data, info = getattr(collect_statistics, name)()
            else:
                data, info = download(dataset_id)
            (folder / f'{name}.csv').write_bytes(data)
            info['download_sha256'] = hashlib.sha256(data).hexdigest()
            report['sources'][name] = info
            report['errors'].pop(name, None)
        except Exception as error:
            report['errors'][name] = f'{type(error).__name__}: {error}'
    year = datetime.now(timezone.utc).year - 1
    try:
        if 'population' not in report['sources']:
            # Canonical raw files are immutable. Fetch every subsequent year,
            # not only the latest, so multi-year automation never leaves a gap.
            known = [int(p.name[:4]) for p in RAW.glob('*연령별인구현황_연간.csv')]
            first = min(max(known, default=year), year)
            annual = {}
            for population_year in range(first, year + 1):
                data, info = download_population(population_year)
                (folder / f'{population_year}12_{population_year}12_연령별인구현황_연간.csv').write_bytes(data)
                annual[str(population_year)] = info
            report['sources']['population'] = {'year': year, 'annual': annual}
            report['errors'].pop('population', None)
    except Exception as error:
        report['errors']['population'] = f'{type(error).__name__}: {error}'
    from scripts.child_crime.collect_licences import collect as collect_licences, KINDS
    key = os.environ.get('DATA_GO_KR_API_KEY', '')
    if not key:
        try:
            from dotenv import dotenv_values
            key = dotenv_values(ROOT / '.env').get('DATA_GO_KR_API_KEY', '')
        except ImportError:
            pass  # Airflow supplies environment variables, no dotenv dependency.
    for kind in KINDS:
        if kind in report['sources']:
            continue
        try:
            report['sources'][kind] = collect_licences(kind, folder, key)
            report['errors'].pop(kind, None)
        except Exception as error:
            report['errors'][kind] = f'{type(error).__name__}: {error}'
    (folder / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    # Publish one small pointer only after all downloads finish.
    pointer = RAW / 'refresh/latest.json'
    temporary = pointer.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(pointer)
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    print(json.dumps(collect(parser.parse_args().resume), ensure_ascii=False, indent=2))
