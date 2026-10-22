# 1인가구 월별 엑셀 → 연도별 CSV 하나로 합치기 (값은 그대로, 파일 이름의 월 정보만 '기준_년월' 컬럼으로 추가)
import glob, os, re
import pandas as pd

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'raw', '1인가구')

for year_dir in sorted(glob.glob(os.path.join(DIR, '[0-9]' * 4))):
    year = os.path.basename(year_dir)
    months = []
    for f in glob.glob(os.path.join(year_dir, '*.xlsx')):
        month = int(re.search(r'\.(\d+)월', os.path.basename(f)).group(1))   # '2022.1월_...' → 1
        df = pd.read_excel(f)
        df.insert(0, '기준_년월', int(year) * 100 + month)                    # 202201
        months.append(df)

    merged = pd.concat(months).sort_values(['기준_년월', '행정동코드', '성별', '연령대'])
    out = os.path.join(DIR, f'1인가구_{year}년.csv')
    merged.to_csv(out, index=False, encoding='cp949')
    print(f'{year}년: {len(months)}개월, {len(merged)}행 → {os.path.basename(out)}')

# 연도별 CSV 4개 → 1인가구_전체.csv 하나로 (파일 내용을 그대로 이어 붙임: 헤더는 한 번만, 값은 변경 없음)
years = sorted(glob.glob(os.path.join(DIR, '1인가구_[0-9][0-9][0-9][0-9]년.csv')))
all_lines = []
for i, f in enumerate(years):
    with open(f, 'rb') as fp:
        lines = fp.read().splitlines(keepends=True)
    if i == 0:
        header = lines[0]
    assert lines[0] == header, f'{os.path.basename(f)} 헤더가 다름'    # 컬럼이 다르면 중단
    all_lines += lines[1:]

out = os.path.join(DIR, '1인가구_전체.csv')
with open(out, 'wb') as fp:
    fp.write(header + b''.join(all_lines))
print(f'전체: {len(years)}개 파일, {len(all_lines)}행 → {os.path.basename(out)}')
