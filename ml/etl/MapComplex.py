# K-apt 서울 단지(ExtractComplexSeoul.py 결과)에 행정동 붙이기 → data/interim/기타/단지_행정동.csv
#  순서: ① 도로명주소(도로명+건물번호) → ② 법정동주소의 지번(동+번지) → ③ 번지 보완(같은 본번, 법정동에 행정동 하나뿐, 앞뒤 번지) (연결방식 컬럼에 남김)
# 필요 파일: data/interim/기타/단지_기본정보_서울.csv, data/raw/기타/rnaddrkor_seoul.txt
import os
import numpy as np, pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data') + os.sep
OLD = {11230515: 11230536, 11230533: 11230536, 11740525: 11740520, 11740526: 11740520, 11680675: 11680740}    # 주소 DB 행정동_코드 → 상권분석 코드

# 1) 주소 DB
a = pd.read_csv(DATA + 'raw/기타/rnaddrkor_seoul.txt', sep='|', encoding='cp949', header=None, dtype=str, usecols=[3, 4, 6, 7, 8, 10, 11, 12, 13, 14])
a.columns = ['구', '동', '산', '본', '부', '도로', '지하', '건본', '건부', '행정동코드']
for c in ['본', '부', '건본', '건부']: a[c] = a[c].astype(int)
a['행정동_코드'] = a['행정동코드'].str[:8].astype(int).replace(OLD)
road = a[a['지하'] == '0'].drop_duplicates(['구', '도로', '건본', '건부'])[['구', '도로', '건본', '건부', '행정동_코드']].rename(columns={'행정동_코드': '도로명동'})
lot = a.assign(산=np.where(a['산'] == '1', '산', '')).drop_duplicates(['구', '동', '산', '본', '부'])[['구', '동', '산', '본', '부', '행정동_코드']].rename(columns={'행정동_코드': '지번동'})
main = lot.groupby(['구', '동', '산', '본'])['지번동'].agg(['nunique', 'first']).reset_index()
main = main[main['nunique'] == 1].rename(columns={'first': '본번동'})[['구', '동', '산', '본', '본번동']]
one = lot.groupby(['구', '동'])['지번동'].agg(['nunique', 'first']).reset_index()
one = one[one['nunique'] == 1].rename(columns={'first': '하나뿐동'})[['구', '동', '하나뿐동']]
near = lot[lot['산'] == ''].drop_duplicates(['구', '동', '본']).sort_values('본')

# 2) 단지 주소 읽기
c = pd.read_csv(DATA + 'interim/기타/단지_기본정보_서울.csv', encoding='utf-8-sig', low_memory=False).reset_index(drop=True)
gu = c['시군구'].astype(str).str.strip()
RD = r'([가-힣]+(?:대로|로)(?:\s*\d+[가-힣]?번?길)?|[가-힣]+\d*[가-힣]?번?길)\s*(?:지하\s*)?(\d+)(?:-(\d+))?'          # 사직로8길 34, 고척로21다길 34-1
clean = lambda s: s.fillna('').str.replace(r'\(.*?\)', ' ', regex=True).str.replace(r'서울특별시|서울시|[가-힣]+구(?![가-힣])', ' ', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()

m = clean(c['도로명주소']).str.extract(RD)
rk = pd.DataFrame({'구': gu, '도로': m[0].str.replace(r'\s', '', regex=True), '건본': pd.to_numeric(m[1]), '건부': pd.to_numeric(m[2]).fillna(0)})
c['도로명동'] = rk.merge(road, on=['구', '도로', '건본', '건부'], how='left')['도로명동'].values

dong = c['동리'].astype(str).str.strip()                                        # 법정동 이름(동리 컬럼)과 번지(법정동주소의 동 이름 바로 뒤 숫자)
bj = pd.Series([pd.Series([a_]).str.extract(r'(?:^|\s)' + d_ + r'\s*(산?)\s*(\d+)(?:-(\d+))?')[[0, 1, 2]].iloc[0].tolist() for a_, d_ in zip(c['법정동주소'].astype(str), dong)])
k = pd.DataFrame({'구': gu, '동': dong, '산': [x[0] if isinstance(x[0], str) else '' for x in bj], '본': pd.to_numeric(pd.Series([x[1] for x in bj])), '부': pd.to_numeric(pd.Series([x[2] for x in bj])).fillna(0)})
c['지번동'] = k.merge(lot, on=['구', '동', '산', '본', '부'], how='left')['지번동'].values

r = k.merge(main, on=['구', '동', '산', '본'], how='left').merge(one, on=['구', '동'], how='left')           # 번지 보완
code = r['본번동'].fillna(r['하나뿐동']).values.copy()
miss = pd.isna(code) & k['본'].notna().values & (k['산'] == '').values
q = k[miss].drop_duplicates(['구', '동', '본']).sort_values('본').astype({'본': 'int64'})
if len(q):
    kw = dict(on='본', by=['구', '동'])
    pv = pd.merge_asof(q, near[['구', '동', '본', '지번동']], direction='backward', **kw)['지번동'].values
    nx = pd.merge_asof(q, near[['구', '동', '본', '지번동']], direction='forward', **kw)['지번동'].values
    adj = q.assign(보완=np.where(pv == nx, pv, np.nan)).dropna(subset=['보완'])[['구', '동', '본', '보완']]
    code = np.where(pd.isna(code) & miss, k.merge(adj, on=['구', '동', '본'], how='left')['보완'].values, code)
c['보완동'] = pd.Series(code, dtype='float')

c['행정동_코드'] = c['도로명동'].fillna(c['지번동']).fillna(c['보완동'])
c['연결방식'] = np.where(c['도로명동'].notna(), '도로명주소', np.where(c['지번동'].notna(), '지번주소', np.where(c['보완동'].notna(), '번지 보완(본번·인접)', '연결안됨')))
print(f'단지 {len(c):,}개 | ' + ', '.join(f'{k_} {v}' for k_, v in c['연결방식'].value_counts().items()))
both = c['도로명동'].notna() & c['지번동'].notna()
print(f'도로명·지번 둘 다 찾은 {both.sum():,}개 중 서로 다른 행정동 {(c.loc[both, "도로명동"] != c.loc[both, "지번동"]).sum()}개 → 도로명주소를 따름')
c['행정동_코드'] = c['행정동_코드'].astype('Int64')
print('연결 안 된 단지:\n' + c[c['연결방식'] == '연결안됨'][['시군구', '단지명', '도로명주소', '법정동주소']].to_string(index=False))
c.drop(columns=['도로명동', '지번동', '보완동']).to_csv(DATA + 'interim/기타/단지_행정동.csv', index=False, encoding='utf-8-sig')
print('행정동', c['행정동_코드'].nunique(), '곳 | 저장:', os.path.abspath(DATA + 'interim/기타/단지_행정동.csv'))
