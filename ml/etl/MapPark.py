# 전국 도시공원 정보 → 서울만 추려 전처리 + 행정동 붙이기 + 행정동별 공원 수·면적 집계
#  → data/interim/기타/공원_행정동.csv (서울 공원 한 줄마다 행정동), data/processed/행정동별_공원.csv (행정동별 공원 수·면적 합)
#  행정동 붙이는 순서(MapCctv.py와 같음): ① 도로명주소(도로명+건물번호) → ② 지번주소(동+번지) → ③ 번지 보완(같은 본번, 법정동에 행정동 하나뿐, 앞뒤 번지) → ④ 위경도로 가장 가까운 행정동 중심점 (연결방식 컬럼에 남김)
#  전처리: 서울이 아닌 공원 제외, 관리번호·(공원명+지번주소) 중복 제거, 공원면적 숫자로 바꿈(0 이하·빈 값은 빈 값으로 두고 채우지 않음), 서울 밖 좌표는 빈 값 처리
#  큰 공원 = 어린이공원·소공원을 뺀 나머지 유형(근린·문화·역사·수변·체육 등)
# 필요 파일: data/raw/기타/전국도시공원정보표준데이터.csv, data/raw/기타/rnaddrkor_seoul.txt, data/raw/서울시 상권분석서비스(영역-행정동).csv
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

# 2) 공원: 서울만 추리고 전처리
c = pd.read_csv(RAW + '기타/전국도시공원정보표준데이터.csv', encoding='cp949', low_memory=False)
n_all = len(c)
txt = c['소재지도로명주소'].fillna('') + ' | ' + c['소재지지번주소'].fillna('')
is_seoul = txt.str.contains(r'(?<![가-힣])서울') | c['제공기관명'].fillna('').str.startswith('서울')
c = c[is_seoul].reset_index(drop=True)
print(f'전국 공원 {n_all:,}건 → 서울 {len(c):,}건 (주소에 서울이 있거나 제공기관이 서울인 공원)')

n1 = len(c)
c = c.drop_duplicates('관리번호').drop_duplicates(['공원명', '소재지지번주소']).reset_index(drop=True)
print(f'중복 제거(관리번호, 공원명+지번주소): {n1 - len(c)}건 제거 → {len(c):,}건')

c['공원면적'] = pd.to_numeric(c['공원면적'], errors='coerce')
print(f'공원면적: 빈 값 {c["공원면적"].isna().sum()}건, 0 이하 {(c["공원면적"] <= 0).sum()}건 → 이 값들은 빈 값으로 두고 채우지 않음 | 중앙값 {c["공원면적"].median():,.0f}㎡, 최대 {c["공원면적"].max():,.0f}㎡')
c.loc[c['공원면적'] <= 0, '공원면적'] = np.nan
print('공원구분:', c['공원구분'].value_counts().to_dict())
c['공원구분'] = c['공원구분'].fillna('기타')
c['큰 공원'] = ~c['공원구분'].isin(['어린이공원', '소공원'])
c['지정연도'] = pd.to_numeric(c['지정고시일'].astype(str).str[:4], errors='coerce')

# 구: 주소에 적힌 '○○구'가 있으면 그것, 없으면 제공기관명(서울특별시 ○○구)에서
txt = c['소재지도로명주소'].fillna('') + ' | ' + c['소재지지번주소'].fillna('')
gu = txt.str.extract(r'서울(?:특별시|시)?\s+([가-힣]+구)(?![가-힣])')[0].fillna(c['제공기관명'].astype(str).str.extract(r'([가-힣]+구)')[0]).fillna(c['관리기관명'].astype(str).str.extract(r'([가-힣]+구)')[0])
print('구를 알 수 없는 공원', gu.isna().sum(), '건')

def clean(col):          # 괄호 안 설명, '서울특별시', 구 이름을 지우고 공백 정리
    return c[col].fillna('').str.replace(r'\(.*?\)', ' ', regex=True).str.replace(r'서울특별시|서울시|(?<![가-힣])서울(?![가-힣])|[가-힣]+구(?![가-힣])', ' ', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()
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
c['위도'] = pd.to_numeric(c['위도'], errors='coerce'); c['경도'] = pd.to_numeric(c['경도'], errors='coerce')
ok = c['위도'].between(37.4, 37.75) & c['경도'].between(126.7, 127.2)          # 서울 밖 좌표는 쓰지 않음
print(f'좌표: 서울 범위 안 {ok.sum():,}건, 빈 값 또는 서울 밖 {(~ok).sum()}건 → 그 좌표는 빈 값으로 두고 채우지 않음')
c.loc[~ok, ['위도', '경도']] = np.nan
x, y = tf.transform(c.loc[ok, '경도'].values, c.loc[ok, '위도'].values)
dist, i = cKDTree(area[['엑스좌표_값', '와이좌표_값']].values).query(np.c_[x, y])
c['중심점동'] = np.nan
c.loc[ok, '중심점동'] = area.index.values[i]

c['행정동_코드'] = c['도로명동'].fillna(c['지번동']).fillna(c['보완동']).fillna(c['중심점동'])
c['연결방식'] = np.where(c['도로명동'].notna(), '도로명주소', np.where(c['지번동'].notna(), '지번주소', np.where(c['보완동'].notna(), '번지 보완(본번·인접)', np.where(c['중심점동'].notna(), '좌표(가까운 중심점)', '연결안됨'))))
print('\n연결방식:\n' + c['연결방식'].value_counts().to_string())
both = c['도로명동'].notna() & c['중심점동'].notna()
print(f'\n도로명주소로 찾은 {c["도로명동"].notna().sum():,}건 중 가까운 중심점 방식과 같은 행정동 {(c.loc[both, "도로명동"] == c.loc[both, "중심점동"]).mean() * 100:.1f}% (다른 건 {(c.loc[both, "도로명동"] != c.loc[both, "중심점동"]).sum():,}건)')
both = c['지번동'].notna() & c['중심점동'].notna() & c['도로명동'].isna()
print(f'지번주소로 찾은 {c["지번동"].notna().sum():,}건 중 가까운 중심점 방식과 같은 행정동 {(c.loc[c["지번동"].notna() & c["중심점동"].notna(), "지번동"] == c.loc[c["지번동"].notna() & c["중심점동"].notna(), "중심점동"]).mean() * 100:.1f}%')
both = c['보완동'].notna() & c['중심점동'].notna() & c['도로명동'].isna() & c['지번동'].isna()
if both.any(): print(f'번지 보완으로 찾은 {both.sum():,}건 중 가까운 중심점 방식과 같은 행정동 {(c.loc[both, "보완동"] == c.loc[both, "중심점동"]).mean() * 100:.1f}%')
bad = c['도로명동'].notna() & c['지번동'].notna() & (c['도로명동'] != c['지번동'])
print(f'도로명·지번 둘 다 찾았는데 서로 다른 행정동 {bad.sum()}건 → 도로명주소를 따름')
print('상권분석 코드에 없는 행정동 코드', int((~c['행정동_코드'].dropna().astype(int).isin(area.index)).sum()), '건')

c['행정동_코드'] = c['행정동_코드'].astype('Int64')
os.makedirs(DATA + 'interim/기타/', exist_ok=True)
c[['관리번호', '공원명', '공원구분', '큰 공원', '공원면적', '소재지도로명주소', '소재지지번주소', '위도', '경도', '지정연도', '관리기관명', '행정동_코드', '연결방식']].to_csv(DATA + 'interim/기타/공원_행정동.csv', index=False, encoding='utf-8-sig')

# 4) 행정동별 집계 (공원 수, 유형별 수, 면적 합). 공원이 없는 행정동은 이 표에 나오지 않음(모델 쪽에서 0으로 볼지는 따로 정함)
d = c.dropna(subset=['행정동_코드'])
g = d.groupby('행정동_코드')
t = pd.DataFrame({'공원 수': g.size(),
                  '어린이공원 수': g['공원구분'].apply(lambda s: (s == '어린이공원').sum()),
                  '소공원 수': g['공원구분'].apply(lambda s: (s == '소공원').sum()),
                  '큰 공원 수': g['큰 공원'].sum(),
                  '공원 면적 합(㎡)': g['공원면적'].sum(min_count=1),
                  '큰 공원 면적 합(㎡)': d[d['큰 공원']].groupby('행정동_코드')['공원면적'].sum(min_count=1)}).reset_index()
t['큰 공원 수'] = t['큰 공원 수'].astype(int)
os.makedirs(DATA + 'processed/', exist_ok=True)
t.to_csv(DATA + 'processed/행정동별_공원.csv', index=False, encoding='utf-8-sig')
print(f'\n행정동 {len(t)}곳(공원이 있는 곳) | 공원 {t["공원 수"].sum():,}개 (서울 공원 {len(c):,}개 중 행정동 연결 {len(d):,}개) | 저장: {os.path.abspath(DATA + "processed/행정동별_공원.csv")}')
print(t.drop(columns='행정동_코드').describe().round(0).T[['count', 'min', '50%', 'max']].to_string())
