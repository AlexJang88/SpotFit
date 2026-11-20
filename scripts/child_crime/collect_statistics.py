"""Public Seoul statistical-table CSV exports (no browser/API key at runtime)."""
from __future__ import annotations

import csv
import io
import json
from datetime import date
from html.parser import HTMLParser
from http.cookiejar import CookieJar
from urllib.parse import urlencode
from urllib.request import build_opener, HTTPCookieProcessor

BASE = 'https://stat.eseoul.go.kr/statHtml/'


class Form(HTMLParser):
    def __init__(self):
        super().__init__()
        self.values, self.items = {}, []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'input' and a.get('type') == 'hidden' and a.get('name'):
            self.values[a['name']] = a.get('value', '')
        if tag == 'input' and a.get('name') == 'itemChkLi':
            self.items.append(a['value'])


def download(table, dimensions, row_axis, start=2014, frequency='Y'):
    """Dimension codes come from each official table, not administrative-code guesses."""
    opener = build_opener(HTTPCookieProcessor(CookieJar()))
    url = BASE + f'statHtml.do?orgId=201&tblId={table}&conn_path=I2&mode=tab'
    with opener.open(url, timeout=60) as response:
        html = response.read().decode('utf-8')
    parsed = Form(); parsed.feed(html)
    if parsed.values.get('tblId') != table or not parsed.items:
        raise ValueError('Statistical table schema unavailable')
    periods = ([str(y) for y in range(start, date.today().year)] if frequency == 'Y' else
               [f'{y}{m:02}' for y in range(start, date.today().year) for m in range(1,13)])
    fields = [{'targetId': 'PRD', 'targetValue': '',
               'prdValue': frequency + ',' + ','.join(periods) + ',@'}]
    fields += [{'targetId':'ITM_ID','targetValue':i,'prdValue':''} for i in parsed.items]
    groups = []
    for index, (dimension, codes) in enumerate(dimensions.items(), start=1):
        fields += [{'targetId': f'OV_L{index}_ID', 'targetValue': code, 'prdValue': ''} for code in codes]
        groups.append({'objVarId': dimension, 'ovlSn': str(index)})
    form = {**parsed.values, 'fieldList': json.dumps(fields), 'colAxis': 'TIME', 'rowAxis': row_axis,
            'isFirst': 'N', 'view': 'csv', 'viewKind': '2', 'viewSubKind': '2_3',
            'tableType': 'default', 'prdSort': 'asc', 'periodCo': '99', 'enableLevelExpr': 'Y',
            'smblYn': 'Y', 'codeYn': 'N', 'classAllArr': json.dumps(groups),
            'classSet': json.dumps([{**g,'visible':'true'} for g in groups])}
    with opener.open(BASE+'downGrid.do', urlencode(form).encode(), timeout=60) as response:
        ticket = json.load(response)
    if not ticket.get('file'):
        raise ValueError('Statistical CSV export failed')
    form['file'] = ticket['file']
    with opener.open(BASE+'downNormal.do', urlencode(form).encode(), timeout=60) as response:
        data = response.read()
    rows = list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))
    if not rows or '항목' not in rows[0]:
        raise ValueError('Invalid statistical CSV response')
    return data, {'source_url': url, 'table': table, 'rows': len(rows), 'columns': list(rows[0])}


def business_density():
    return download('DT_201004_C010014', {'A': [f'001{i:03}' for i in range(1,26)]}, 'A,ITEM')


def student_movement():
    return download('DT_201004_G010004', {'B': ['001001','001002','002001','002002'],
                    'A': [f'001{i:03}' for i in range(1,26)]}, 'A,ITEM,B', start=2012)


def births_deaths():
    return download('DT_201004_A050018', {'A': [f'001{i:03}' for i in range(1,26)],
                    'B': ['001']}, 'A,B,ITEM', frequency='M')


def household_structure():
    return download('DT_201004_B010006', {'A': [f'001{i:03}' for i in range(1,26)],
                    'B': ['001','002','003','004','003001','003002'],
                    'C': ['001','002','002001','002002']}, 'A,B,C,ITEM', start=2015)
