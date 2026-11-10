from pathlib import Path
import re
from html import unescape
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data/raw/아동범죄_행정동'
def spreadsheet(path):
    text = path.read_text(encoding='cp949')
    rows=[]
    for match in re.finditer(r'<Row\b[^>]*>(.*?)</Row>', text, re.S):
        row=[]
        for cell in re.finditer(r'<Cell\b([^>]*?)(?:/>|>(.*?)</Cell>)',match[1],re.S):
            index=re.search(r'ss:Index="(\d+)"',cell[1])
            if index:
                row.extend(['']*max(0,int(index[1])-1-len(row)))
            data=re.search(r'<Data\b[^>]*>(.*?)</Data>',cell[2] or '',re.S)
            row.append(unescape(data[1]) if data else '')
        rows.append(row)
    return rows

def csvread(path, **kwargs):
    for enc in ('utf-8-sig','cp949'):
        try: return pd.read_csv(path,encoding=enc,**kwargs)
        except UnicodeDecodeError: pass
    raise ValueError(path)

if __name__=='__main__':
    for p in sorted(RAW.rglob('*.xls')):
        rows=spreadsheet(p)
        print('\nFILE',p.name,'ROWS',len(rows))
        print('HEADER',rows[1])
        gu=[r for r in rows[2:] if r and r[0] in ('종로구','강남구','강동구')]
        print('GU',gu[:5])
        print('SAMPLE',rows[25:27])
    for p in sorted((RAW/'집객시설').glob('*.csv')):
        print('MDIS',p.name,csvread(p,header=None,nrows=1).shape,csvread(p,header=None,nrows=1).values.tolist())
    p=next((RAW/'집객시설').glob('*.xlsx'))
    book=pd.ExcelFile(p)
    print('CODEBOOK SHEETS',book.sheet_names)
    for sheet in book.sheet_names:
        print(sheet,pd.read_excel(book,sheet_name=sheet,header=None).head(18).to_string(index=False,header=False))
