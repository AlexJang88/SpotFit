# 부동산 실거래의 법정동+지번 → 행정동 변환 (도로명주소 DB의 지번별 행정동 사용) → *_juso.csv 로 저장
#  - 아파트·빌라 매매: apt_sale / villa_sale
#  - 전월세(아파트·연립다세대(빌라)·단독다가구·오피스텔): 서울시_전월세가_전체.csv (MergeRent.py 로 합친 파일)
# 필요 파일: data/raw/기타/rnaddrkor_seoul.txt (juso.go.kr 도로명주소 한글 전체분, 서울)
import os
import numpy as np, pandas as pd

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'raw') + os.sep
PROC = os.path.join(RAW, '..', 'processed', '부동산') + os.sep                  # 고친 지번 파일(*_with_hdong.csv), 합친 전월세 파일이 있는 곳
JUSO = os.path.join(RAW, '..', 'processed', '부동산(도로명주소)') + os.sep    # 결과(*_juso.csv)를 저장하는 곳
os.makedirs(JUSO, exist_ok=True)

# 1) 주소 DB → 지번별 행정동 (건물마다 한 줄이라 지번 기준으로 하나만 남김)
a = pd.read_csv(RAW + '기타/rnaddrkor_seoul.txt', sep='|', encoding='cp949', header=None, dtype=str, usecols=[1, 4, 6, 7, 8, 14, 15])
a.columns = ['법정동코드', '동', '산', '본번', '부번', '행정동코드', '행정동명']
a['구'] = a['법정동코드'].str[:5]
a['jibun'] = np.where(a['산'] == '1', '산', '') + a['본번'].astype(int).astype(str) + np.where(a['부번'].astype(int) > 0, '-' + a['부번'].astype(int).astype(str), '')   # 거래 데이터의 지번 형식(316, 316-1, 산12)
lot = a.drop_duplicates(['구', '동', 'jibun'])[['구', '동', 'jibun', '행정동코드', '행정동명']]
lot = lot.assign(행정동_코드=lot['행정동코드'].str[:8].astype(int))       # 상권분석 데이터와 같은 앞 8자리
lot['본번'] = lot['jibun'].str.extract(r'^(\d+)')[0].astype(float)          # 번호 앞부분 (316-1 → 316), 산·가-12 등은 NaN
one = lot.groupby(['구', '동'])['행정동_코드'].agg(['nunique', 'first'])        # 법정동 안의 행정동이 하나뿐인 곳
near = lot.dropna(subset=['본번']).drop_duplicates(['구', '동', '본번']).sort_values('본번')

def to_dong(d):         # d: sggCd(구 코드), umdNm(법정동), jibun(지번) 컬럼이 있는 거래표 → 행정동_코드·행정동명·연결방식 컬럼을 붙여서 돌려줌
    n = len(d)
    d['구'], d['jibun'] = d['sggCd'].astype(str), d['jibun'].astype(str).str.strip()
    d = d.merge(lot[['구', '동', 'jibun', '행정동_코드', '행정동명']], left_on=['구', 'umdNm', 'jibun'], right_on=['구', '동', 'jibun'], how='left').drop(columns='동')
    assert len(d) == n, '지번이 겹쳐 행이 늘어남'
    d['연결방식'] = np.where(d['행정동_코드'].notna(), '지번 일치', '')

    # 주소 DB에 없는 지번(합필·철거 등)이나 지번이 없는 거래는 보완: ① 법정동에 행정동이 하나뿐이면 그 동 ② 번호가 가까운 앞·뒤 지번이 같은 행정동이면 그 동 (검증: 98% 일치)
    miss = d['행정동_코드'].isna()
    key = pd.MultiIndex.from_frame(d.loc[miss, ['구', 'umdNm']])
    single = one.reindex(key)
    d.loc[miss, '행정동_코드'] = np.where(single['nunique'].eq(1), single['first'], np.nan)
    d.loc[miss & d['행정동_코드'].notna(), '연결방식'] = '법정동 하나뿐'
    miss = d['행정동_코드'].isna()
    q = d.loc[miss, ['구', 'umdNm', 'jibun']].drop_duplicates().assign(본번=lambda x: x['jibun'].str.extract(r'^(\d+)')[0].astype(float)).dropna(subset=['본번']).sort_values('본번')
    kw = dict(left_on='본번', right_on='본번', left_by=['구', 'umdNm'], right_by=['구', '동'])
    pv = pd.merge_asof(q, near[['구', '동', '본번', '행정동_코드']], direction='backward', **kw)
    nx = pd.merge_asof(q, near[['구', '동', '본번', '행정동_코드']], direction='forward', **kw)
    q = q.assign(앞=pv['행정동_코드'].values, 뒤=nx['행정동_코드'].values)
    q = q[q['앞'].notna() & (q['앞'] == q['뒤'])].assign(보완=lambda x: x['앞'])
    d = d.merge(q[['구', 'umdNm', 'jibun', '보완']], on=['구', 'umdNm', 'jibun'], how='left')
    fill = d['행정동_코드'].isna() & d['보완'].notna()
    d.loc[fill, '행정동_코드'], d.loc[fill, '연결방식'] = d.loc[fill, '보완'], '인접 지번'
    d = d.drop(columns=['구', '보완'])
    d['행정동_코드'] = d['행정동_코드'].astype('Int64')                    # 소수(11110530.0)가 아니라 정수로 저장
    return d

def summary(d, by=None):     # 행정동 연결 결과 요약
    f = lambda t: ', '.join(f'{k} {v:.1f}%' for k, v in t['연결방식'].replace('', '연결안됨').value_counts(normalize=True).mul(100).items())
    return f(d) if by is None else '\n'.join(f'   {k}: {len(t):,}건 | ' + f(t) for k, t in d.groupby(by))

# 2) 아파트·빌라 매매 (이미 변환해 둔 파일을 그대로 쓰므로 기본은 건너뜀. 다시 변환하려면 RUN_SALE = True)
RUN_SALE = False
for name in (['apt_sale', 'villa_sale'] if RUN_SALE else []):
    d = pd.read_csv(PROC + f'{name}_with_hdong.csv', encoding='utf-8-sig', low_memory=False).drop(columns=['행정동코드', '행정동명'], errors='ignore')   # 파일에 있는 법정동 단위 행정동은 쓰지 않음. 지번은 엑셀 날짜 변환(48-2 → Feb-48)이 고쳐진 with_hdong 파일 사용
    d = to_dong(d)
    d.to_csv(JUSO + f'{name}_juso.csv', index=False)
    print(f'{name}: {len(d):,}건 → {name}_juso.csv | 행정동 연결 {d["행정동_코드"].notna().mean() * 100:.1f}% | ' + summary(d))

# 3) 전월세(아파트·연립다세대·단독다가구·오피스텔). 단독다가구는 지번(본번)이 거의 비어 있어서 지번 매핑이 안 됨 → 법정동이 행정동 하나뿐인 거래만 행정동이 붙음
cols = ['자치구코드', '법정동명', '지번구분코드', '본번', '부번', '층', '계약일', '전월세구분', '임대면적', '보증금(만원)', '임대료(만원)', '건축년도', '건물용도', '신규계약구분']
d = pd.read_csv(PROC + '서울시_전월세가_전체.csv', encoding='utf-8-sig', dtype=str, usecols=cols)
d = d[d['건물용도'].isin(['아파트', '연립다세대', '단독다가구', '오피스텔'])].rename(columns={'자치구코드': 'sggCd', '법정동명': 'umdNm'}).reset_index(drop=True)
bon, bu = pd.to_numeric(d['본번'], errors='coerce'), pd.to_numeric(d['부번'], errors='coerce').fillna(0)
ok = (bon > 0) & (d['지번구분코드'] != '3')                                  # 본번이 있고 '블럭' 지번이 아닌 것만 지번으로 사용 (블럭 번호는 대지 번호와 달라서 잘못 연결될 수 있음)
d['jibun'] = pd.Series(np.where(d['지번구분코드'] == '2', '산', '') + bon.fillna(0).astype(int).astype(str) + np.where(bu > 0, '-' + bu.astype(int).astype(str), ''), index=d.index).where(ok)
d = to_dong(d)
d.to_csv(JUSO + '서울시_전월세가_juso.csv', index=False)
print(f'서울시_전월세가: {len(d):,}건 → 서울시_전월세가_juso.csv | 건물용도별 행정동 연결 결과\n' + summary(d, by='건물용도'))
