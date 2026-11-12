"""Check the target's geographic unit before fitting dong-level forecasts."""
from pathlib import Path
import json
import re
from html import unescape
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data/raw/아동범죄_행정동'
OUT = ROOT / 'artifacts/child_crime_dong_audit'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = RAW / '서울시_5대범죄_2014_2025_원자료.csv'
    crime = pd.read_csv(source, encoding='utf-8-sig')
    coverage = crime.groupby('년도').agg(
        rows=('시군구', 'size'), regions=('시군구', 'nunique'), types=('범죄', 'nunique'))
    report = {
        'target_source': str(source),
        'columns': crime.columns.tolist(),
        'regions': sorted(crime['시군구'].unique().tolist()),
        'coverage': coverage.reset_index().to_dict('records'),
        'duplicate_keys': int(crime.duplicated(['년도', '시군구', '범죄']).sum()),
        'target_geography': '자치구',
        'dong_target_available': False,
        'status': '행정동 학습 정답 없음. 자치구 집계 학습 전환 또는 별도 행정동 범죄 자료 필요.',
        'files': [{'path': str(p.relative_to(RAW)), 'bytes': p.stat().st_size}
                  for p in sorted(RAW.rglob('*')) if p.is_file()],
    }
    (OUT / 'target_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    coverage.to_csv(OUT / 'crime_year_coverage.csv', encoding='utf-8-sig')
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}, ensure_ascii=False, indent=2))
    for folder in sorted(p for p in RAW.iterdir() if p.is_dir()):
        path = next((p for p in sorted(folder.iterdir()) if p.suffix in ('.csv', '.xls')), None)
        if path is None:
            continue
        print('\nSOURCE:', path.relative_to(RAW))
        if path.suffix == '.xls':
            # Preview only: some exports contain XML-invalid unescaped text.
            rows = re.findall(r'<Row\b[^>]*>(.*?)</Row>', path.read_text(encoding='cp949'), re.S)
            print('Spreadsheet row count:', len(rows))
            for row in rows[:5]:
                print([unescape(cell) for cell in re.findall(r'<Data\b[^>]*>(.*?)</Data>', row, re.S)][:16])
            continue
        for encoding in ('utf-8-sig', 'cp949'):
            try:
                with path.open(encoding=encoding) as stream:
                    sample = [next(stream, '') for _ in range(4)]
                print(''.join(sample)[:1000])
                break
            except UnicodeError:
                continue


if __name__ == '__main__':
    main()
