"""MOIS licence /info collector. Filter by the 25 Seoul licensing authorities.

Official contracts: data.go.kr/data/15154890/openapi.do (entertainment)
and data.go.kr/data/15154883/openapi.do (singing bars). Keys are never logged.
"""
from __future__ import annotations

import csv
import io
import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, unquote
from urllib.request import Request, urlopen

KINDS = {'entertainment_bars': '유흥주점', 'singing_bars': '단란주점'}
FIELDS = {'OPN_ATMY_GRP_CD': '개방자치단체코드', 'MNG_NO': '관리번호',
          'LCPMT_YMD': '인허가일자', 'CLSBIZ_YMD': '폐업일자', 'SALS_STTS_NM': '영업상태명',
          'SALS_STTS_CD': '영업상태코드', 'LOTNO_ADDR': '지번주소', 'ROAD_NM_ADDR': '도로명주소'}
SEOUL_AUTHORITIES = tuple(str(i * 1000) for i in range(3000, 3241, 10))


def fetch(kind: str, code: str, page: int, key: str) -> dict:
    params = {'serviceKey': unquote(key), 'returnType': 'json', 'pageNo': page, 'numOfRows': 100,
              'cond[OPN_ATMY_GRP_CD::EQ]': code}
    request = Request(f'https://apis.data.go.kr/1741000/{kind}/info?{urlencode(params)}')
    for attempt in range(3):
        try:
            with urlopen(request, timeout=60) as response:
                data = json.load(response)
            break
        except HTTPError as error:
            if error.code in (500, 502, 503, 504) and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            # HTTPError includes a credential-bearing URL; never stringify or chain it.
            raise RuntimeError(f'{kind}: HTTP {error.code}; check API subscription/quota') from None
        except (URLError, ValueError, TimeoutError):
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f'{kind}: network or JSON response failure') from None
    header = data.get('response', {}).get('header', {})
    if str(header.get('resultCode')) not in ('0', '00'):
        raise RuntimeError(f'{kind}: provider rejected request (check subscription/quota)')
    return data['response']['body']


def collect(kind: str, folder: Path, key: str) -> dict:
    if kind not in KINDS:
        raise ValueError('Unknown licence kind')
    if not key:
        raise RuntimeError('DATA_GO_KR_API_KEY is required')
    pages_dir = folder / f'{kind}_pages'
    pages_dir.mkdir(parents=True, exist_ok=True)
    rows, requests = [], 0
    budget = int(os.environ.get('CRIME_LICENCE_MAX_REQUESTS', '250'))
    for code in SEOUL_AUTHORITIES:
        page, received, expected = 1, 0, None
        while expected is None or received < expected:
            if requests >= budget:
                raise RuntimeError(f'{kind}: request safety limit reached; no incomplete snapshot is published')
            cached = pages_dir / f'{code}_{page}.json'
            if cached.is_file():
                body = json.loads(cached.read_text(encoding='utf-8'))
            else:
                body = fetch(kind, code, page, key)
                requests += 1
            count = int(body['totalCount'])
            if expected is not None and count != expected:
                raise ValueError('Licence count changed during pagination; retry collection')
            expected = count
            items = body.get('items', {})
            batch = items.get('item', []) if isinstance(items, dict) else items
            if isinstance(batch, dict):
                batch = [batch]
            if not isinstance(batch, list) or (not batch and received < expected):
                raise ValueError('Incomplete licence API page')
            for row in batch:
                if str(row.get('OPN_ATMY_GRP_CD')) != code or not row.get('MNG_NO'):
                    raise ValueError('API authority filter/schema mismatch')
            cached.write_text(json.dumps(body, ensure_ascii=False),encoding='utf-8')
            rows.extend(batch); received += len(batch); page += 1
            time.sleep(.1)
        if received != expected:
            raise ValueError('Licence pagination total mismatch')
    if len({r['MNG_NO'] for r in rows}) != len(rows) or not rows:
        raise ValueError('Duplicate/empty licence snapshot')
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(FIELDS.values()))
    writer.writeheader()
    for row in rows:
        writer.writerow({name: '' if row.get(key) is None else str(row[key]).strip() for key, name in FIELDS.items()})
    path = folder / f'서울시 {KINDS[kind]}영업 인허가 정보.csv'
    path.write_text(stream.getvalue(),encoding='utf-8-sig',newline='')
    return {'rows': len(rows), 'requests': requests, 'authorities': len(SEOUL_AUTHORITIES),
            'source_url': f'https://apis.data.go.kr/1741000/{kind}/info', 'file': path.name}
