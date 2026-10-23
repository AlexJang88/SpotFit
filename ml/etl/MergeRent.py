# 서울특별시 전월세가 연도별 CSV 6개(2021~2026) → 서울시_전월세가_전체.csv 하나로 합치기
# 값은 그대로 두고, ① 2026 파일의 컬럼 이름을 다른 해와 맞추고 ② 정확히 같은 행(파일 안·파일 사이 중복)만 제거함. 원본 파일은 그대로 둠
import glob, os
import numpy as np, pandas as pd

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'raw', '부동산')
OUT = os.path.join(RAW, '..', '..', 'processed', '부동산', '서울시_전월세가_전체.csv')

def read(f):            # 파일마다 글자 인코딩이 달라서(UTF-8 / CP949) 둘 다 시도. 모든 값을 글자로 읽어 '0001' 같은 앞자리 0을 지키지 않게 숫자로 바뀌는 것 방지
    for enc in ['utf-8-sig', 'cp949']:
        try: return pd.read_csv(f, encoding=enc, dtype=str)
        except UnicodeDecodeError: pass

files = sorted(glob.glob(os.path.join(RAW, '서울특별시_전월세가_*.csv')))
cols = read(files[0]).columns                       # 2021 파일의 컬럼 이름을 기준으로 맞춤 (2026 파일만 이름이 다르고 순서·내용은 같음)
seen = np.empty(0, dtype=np.uint64)                 # 앞선 파일에서 이미 나온 행의 지문(해시): 파일 사이 중복 확인용
os.makedirs(os.path.dirname(OUT), exist_ok=True)
total = 0
for i, f in enumerate(files):
    d = read(f)
    assert len(d.columns) == len(cols), f'{os.path.basename(f)} 컬럼 수가 다름'
    d.columns = cols
    n = len(d)
    h = pd.util.hash_pandas_object(d.fillna(''), index=False).values
    keep = ~pd.Series(h).duplicated().values        # 같은 파일 안의 완전히 같은 행: 첫 번째만 남김
    same = int((~keep).sum())
    keep &= ~np.isin(h, seen)                       # 앞선 파일에 이미 있는 행: 뒤 파일의 것을 뺌
    seen = np.concatenate([seen, h[keep]])
    d = d[keep].assign(원본파일=os.path.basename(f))
    d.to_csv(OUT, mode='w' if i == 0 else 'a', header=(i == 0), index=False, encoding='utf-8-sig')
    total += len(d)
    print(f'{os.path.basename(f)}: {n:>9,}행 → 파일 안 중복 {same:>7,} / 앞 파일과 겹침 {n - same - len(d):>6,} 제외 → {len(d):>9,}행')
print(f'합계 {total:,}행 → {os.path.abspath(OUT)}')
