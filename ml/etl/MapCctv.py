# 서울 CCTV 정보에 행정동 붙이기 + 행정동별 CCTV 수 집계
#  → data/interim/기타/CCTV_행정동.csv (CCTV 한 줄마다 행정동), data/processed/행정동별_CCTV.csv (행정동별 카메라 대수 합)
#  행정동 붙이는 순서: ① 도로명주소(도로명+건물번호) → ② 지번주소(동+번지) → ③ 번지 보완(같은 본번, 법정동에 행정동 하나뿐, 앞뒤 번지) → ④ 위경도로 가장 가까운 행정동 중심점 (연결방식 컬럼에 남김)
#  이전 방식(AptHypothesis.cctv_count)은 모든 CCTV를 위경도로 가장 가까운 중심점에만 배정했음
# 생활방범 CCTV만 설치연도별로 따로 저장 → data/interim/기타/CCTV_생활방범_연도별/CCTV_생활방범_연도.csv, 행정동별 연도별 설치 대수 → data/processed/행정동별_생활방범CCTV_연도별.csv
# 필요 파일: data/raw/기타/서울시_CCTV정보.csv, data/raw/기타/rnaddrkor_seoul.txt, data/raw/서울시 상권분석서비스(영역-행정동).csv
import os
import numpy as np, pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data') + os.sep
RAW = DATA + 'raw/'
OLD = {11230515: 11230536, 11230533: 11230536, 11740525: 11740520, 11740526: 11740520, 11680675: 11680740}    # 주소 DB 행정동_코드 → 상권분석 코드

# 1) 주소 DB → 도로명(구·도로명·건물본번·부번)·지번(구·동·산·본번·부번)별 행정동
a = pd.read_csv(RAW + '기타/rnaddrkor_seoul.txt', sep='|', encoding='cp949', header=None, dtype=str, usecols=[3, 4, 6, 7, 8, 10, 11, 12, 13, 14])
a.columns = ['구', '동', '산', '본', '부', '도로', '지하', '건본', '건부', '행정동코드']
for c in ['본', '부', '건본', '건부']: a[c] = a[c].astype(int)
a['행정동_코드'] = a['행정동코드'].str[:8].astype(int).replace(OLD)
road = a[a['지하'] == '0'].drop_duplicates(['구', '도로', '건본', '건부'])[['구', '도로', '건본', '건부', '행정동_코드']].rename(columns={'행정동_코드': '도로명동'})
lot = a.assign(산=np.where(a['산'] == '1', '산', '')).drop_duplicates(['구', '동', '산', '본', '부'])[['구', '동', '산', '본', '부', '행정동_코드']].rename(columns={'행정동_코드': '지번동'})

main = lot.groupby(['구', '동', '산', '본'])['지번동'].agg(['nunique', 'first']).reset_index()
main = main[main['nunique'] == 1].rename(columns={'first': '본번동'})[['구', '동', '산', '본', '본번동']]       # 같은 본번이 한 행정동뿐
one = lot.groupby(['구', '동'])['지번동'].agg(['nunique', 'first']).reset_index()
one = one[one['nunique'] == 1].rename(columns={'first': '하나뿐동'})[['구', '동', '하나뿐동']]                # 법정동에 행정동이 하나뿐
near = lot[lot['산'] == ''].drop_duplicates(['구', '동', '본']).sort_values('본')

# 2) CCTV: 주소 읽기
c = pd.read_csv(RAW + '기타/서울시_CCTV정보.csv', encoding='cp949', low_memory=False)
n0 = len(c)
c['카메라대수'] = pd.to_numeric(c['카메라대수'], errors='coerce')
print(f'CCTV {n0:,}건 | 카메라대수 빈 값 {c["카메라대수"].isna().sum()}건, 0 이하 {(c["카메라대수"] <= 0).sum()}건, 합 {c["카메라대수"].sum():,.0f}대')
c['카메라대수'] = c['카메라대수'].fillna(1).clip(lower=1)                    # 설치된 CCTV 한 줄은 최소 1대 (빈 값·0은 1대로 봄)
c = c.reset_index(drop=True)

# 구: 주소에 적힌 '○○구'가 있으면 그것, 없으면 관리기관명(서울특별시 종로구청)에서
txt = c['소재지도로명주소'].fillna('') + ' | ' + c['소재지지번주소'].fillna('')
gu = txt.str.extract(r'서울특별시\s+([가-힣]+구)(?![가-힣])')[0].fillna(c['관리기관명'].astype(str).str.extract(r'([가-힣]+구)청')[0])
print('구를 알 수 없는 CCTV', gu.isna().sum(), '건')

def clean(col):          # 괄호 안 설명, '서울특별시', 구 이름을 지우고 공백 정리
    return c[col].fillna('').str.replace(r'\(.*?\)', ' ', regex=True).str.replace(r'서울특별시|[가-힣]+구(?![가-힣])', ' ', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()
RD = r'([가-힣]+(?:대로|로)(?:\s*\d+[가-힣]?번?길)?|[가-힣]+\d*[가-힣]?번?길)\s*(?:지하\s*)?(\d+)(?:-(\d+))?'          # 북촌로 134-1, 해등로32길 8, 고척로21다길 34-1, 개봉로 18길 29-13, 숭인동2길 30
JB = r'([가-힣]+\d*[동가])\s*(산?)\s*(\d+)(?:-(\d+))?'                                               # 중계동 435-2, 금호동2가 322-1, 홍은1동 8-368

def find_road(col):
    m = clean(col).str.extract(RD)
    return pd.DataFrame({'구': gu, '도로': m[0].str.replace(r'\s', '', regex=True), '건본': pd.to_numeric(m[1]), '건부': pd.to_numeric(m[2]).fillna(0)}).merge(road, on=['구', '도로', '건본', '건부'], how='left')['도로명동'].values

def find_lot(col):          # 동 이름에 숫자가 있으면(홍은1동) 법정동(홍은동)으로도 한 번 더 찾음
    m = clean(col).str.extract(JB)
    k = pd.DataFrame({'구': gu, '동': m[0], '산': m[1].fillna(''), '본': pd.to_numeric(m[2]), '부': pd.to_numeric(m[3]).fillna(0)})
    r1 = k.merge(lot, on=['구', '동', '산', '본', '부'], how='left')['지번동']
    k['동'] = k['동'].str.replace(r'\d+(?=[동가])', '', regex=True)
    r2 = k.merge(lot, on=['구', '동', '산', '본', '부'], how='left')['지번동']
    return r1.fillna(r2).values

def fallback(col):          # 주소 DB에 그 번지가 없을 때(공원·도로·산 등): ① 같은 본번 ② 법정동에 행정동이 하나뿐 ③ 앞·뒤 번지가 같은 행정동
    m = clean(col).str.extract(JB)
    k = pd.DataFrame({'구': gu, '동': m[0].str.replace(r'\d+(?=[동가])', '', regex=True), '산': m[1].fillna(''), '본': pd.to_numeric(m[2])})
    k['i'] = np.arange(len(k))
    r = k.merge(main, on=['구', '동', '산', '본'], how='left').merge(one, on=['구', '동'], how='left')
    code = r['본번동'].fillna(r['하나뿐동']).values.copy()
    miss = pd.isna(code) & k['본'].notna().values & (k['산'] == '').values
    q = k[miss].drop_duplicates(['구', '동', '본']).sort_values('본').astype({'본': 'int64'})
    if len(q):
        kw = dict(on='본', by=['구', '동'])
        pv = pd.merge_asof(q, near[['구', '동', '본', '지번동']], direction='backward', **kw)['지번동'].values
        nx = pd.merge_asof(q, near[['구', '동', '본', '지번동']], direction='forward', **kw)['지번동'].values
        adj = q.assign(보완=np.where(pv == nx, pv, np.nan)).dropna(subset=['보완'])[['구', '동', '본', '보완']]
        r2 = k.merge(adj, on=['구', '동', '본'], how='left')['보완'].values
        code = np.where(pd.isna(code) & miss, r2, code)
    return pd.Series(code, dtype='float')

c['도로명동'] = pd.Series(find_road('소재지도로명주소')).fillna(pd.Series(find_road('소재지지번주소')))
c['지번동'] = pd.Series(find_lot('소재지지번주소')).fillna(pd.Series(find_lot('소재지도로명주소')))
c['보완동'] = fallback('소재지지번주소').fillna(fallback('소재지도로명주소'))

# 3) 위경도: 가장 가까운 행정동 중심점 (주소로 못 찾은 것 보완 + 주소 방식과 비교)
area = pd.read_csv(RAW + '서울시 상권분석서비스(영역-행정동).csv', encoding='cp949').drop_duplicates('행정동_코드').set_index('행정동_코드')
tf = Transformer.from_crs('EPSG:4326', 'EPSG:5181', always_xy=True)
ok = c['WGS84위도'].between(37.4, 37.75) & c['WGS84경도'].between(126.7, 127.2)          # 서울 밖 좌표는 쓰지 않음
x, y = tf.transform(c.loc[ok, 'WGS84경도'].values, c.loc[ok, 'WGS84위도'].values)
dist, i = cKDTree(area[['엑스좌표_값', '와이좌표_값']].values).query(np.c_[x, y])
c['중심점동'] = np.nan
c.loc[ok, '중심점동'] = area.index.values[i]
c.loc[ok, '중심점거리'] = dist

c['행정동_코드'] = c['도로명동'].fillna(c['지번동']).fillna(c['보완동']).fillna(c['중심점동'])
c['연결방식'] = np.where(c['도로명동'].notna(), '도로명주소', np.where(c['지번동'].notna(), '지번주소', np.where(c['보완동'].notna(), '번지 보완(본번·인접)', np.where(c['중심점동'].notna(), '좌표(가까운 중심점)', '연결안됨'))))
print('\n연결방식:\n' + c['연결방식'].value_counts().to_string())
both = c['도로명동'].notna() & c['중심점동'].notna()
print(f'\n도로명주소로 찾은 {c["도로명동"].notna().sum():,}건 중 가까운 중심점 방식과 같은 행정동 {(c.loc[both, "도로명동"] == c.loc[both, "중심점동"]).mean() * 100:.1f}% (다른 건 {(c.loc[both, "도로명동"] != c.loc[both, "중심점동"]).sum():,}건)')
bad = c['도로명동'].notna() & c['지번동'].notna() & (c['도로명동'] != c['지번동'])
print(f'도로명·지번 둘 다 찾았는데 서로 다른 행정동 {bad.sum()}건 → 도로명주소를 따름')

c['행정동_코드'] = c['행정동_코드'].astype('Int64')
os.makedirs(DATA + 'interim/기타/', exist_ok=True)
c[['관리번호', '관리기관명', '소재지도로명주소', '소재지지번주소', '설치목적구분', '카메라대수', '설치연월', 'WGS84위도', 'WGS84경도', '행정동_코드', '연결방식']].to_csv(DATA + 'interim/기타/CCTV_행정동.csv', index=False, encoding='utf-8-sig')

# 4) 행정동별 집계 (카메라 대수 합, 설치 장소 수)
g = c.dropna(subset=['행정동_코드']).groupby('행정동_코드')
t = pd.DataFrame({'CCTV 수': g['카메라대수'].sum(), 'CCTV 설치장소 수': g.size()}).reset_index()
os.makedirs(DATA + 'processed/', exist_ok=True)
t.to_csv(DATA + 'processed/행정동별_CCTV.csv', index=False, encoding='utf-8-sig')
print(f'\n행정동 {len(t)}곳 | 카메라 {t["CCTV 수"].sum():,.0f}대 (원본 합 {c["카메라대수"].sum():,.0f}대) | 저장: {os.path.abspath(DATA + "processed/행정동별_CCTV.csv")}')

# 5) 생활방범만 설치연도별로 따로 저장 (설치연월 YYYYMM → 연도)
b = c[c['설치목적구분'] == '생활방범'].copy()
ym = pd.to_numeric(b['설치연월'], errors='coerce')
b['설치연도'] = (ym // 100).astype('Int64')
print('\n설치목적구분:\n' + c['설치목적구분'].value_counts(dropna=False).to_string())
print(f'\n생활방범 {len(b):,}건 | 설치연월 빈 값 {b["설치연도"].isna().sum()}건 | 행정동 연결 안 됨 {b["행정동_코드"].isna().sum()}건')
out = DATA + 'interim/기타/CCTV_생활방범_연도별/'
os.makedirs(out, exist_ok=True)
for yr, grp in b.groupby(b['설치연도'].astype('float')):
    grp.drop(columns=['설치연도']).to_csv(out + f'CCTV_생활방범_{int(yr)}.csv', index=False, encoding='utf-8-sig')
b[b['설치연도'].isna()].drop(columns=['설치연도']).to_csv(out + 'CCTV_생활방범_연도모름.csv', index=False, encoding='utf-8-sig')
w = b.dropna(subset=['행정동_코드', '설치연도']).pivot_table(index='행정동_코드', columns='설치연도', values='카메라대수', aggfunc='sum', fill_value=0)
w.columns = [f'{int(y)}년 설치 카메라 수' for y in w.columns]
w.reset_index().to_csv(DATA + 'processed/행정동별_생활방범CCTV_연도별.csv', index=False, encoding='utf-8-sig')
print('연도별 (건수 / 카메라 대수):\n' + b.groupby('설치연도').agg(건수=('카메라대수', 'size'), 카메라=('카메라대수', 'sum')).to_string())
