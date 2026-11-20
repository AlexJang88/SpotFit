# 서울시 도시계획사업(서울플랜+) 정비사업·소규모정비사업 현황의 구역에 행정동 붙이기 → data/interim/부동산/도시계획사업_행정동.csv
#  - 구역 위치(사업위치: '잠원동 50-5', '사당동 252-15 일대')의 동+번지를 도로명주소 DB(rnaddrkor_seoul.txt)의 지번과 맞춰 행정동_코드(앞 8자리)를 붙임
#  - 순서: ① 지번 일치 → ② 같은 본번이 한 행정동뿐이면 그 동 → ③ 동 이름이 행정동 이름이면 그 동 → ④ 법정동에 행정동이 하나뿐이면 그 동 → ⑤ 앞·뒤 번호가 같은 행정동이면 그 동
#  - 번지가 여러 개(1618,1630)면 첫 번째 번지로 찾음. 연결방식 컬럼에 어떤 방법으로 붙었는지 남김
# 필요 파일: data/raw/기타/rnaddrkor_seoul.txt, data/raw/부동산/서울시 도시계획사업(서울플랜+) 정비사업 현황.csv, ...소규모정비사업 현황.csv
import os
import numpy as np, pandas as pd

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'raw') + os.sep
OUT = os.path.join(RAW, '..', 'interim', '부동산') + os.sep
os.makedirs(OUT, exist_ok=True)

# 1) 주소 DB: 지번(구·동·산·본번·부번) → 행정동
a = pd.read_csv(RAW + '기타/rnaddrkor_seoul.txt', sep='|', encoding='cp949', header=None, dtype=str, usecols=[3, 4, 6, 7, 8, 14, 15])
a.columns = ['구', '동', '산', '본', '부', '행정동코드', '행정동명']
for c in ['본', '부']: a[c] = a[c].astype(int)
a['산'] = np.where(a['산'] == '1', '산', '')
a['행정동_코드'] = a['행정동코드'].str[:8].astype(int)
lot = a.drop_duplicates(['구', '동', '산', '본', '부'])[['구', '동', '산', '본', '부', '행정동_코드']]
main = lot.groupby(['구', '동', '산', '본'])['행정동_코드'].agg(['nunique', 'first']).reset_index()
main = main[main['nunique'] == 1].rename(columns={'first': '본번동'})[['구', '동', '산', '본', '본번동']]       # 본번이 같으면 행정동도 같은 곳
hname = a.assign(행정동명=a['행정동명'].str.replace(r'제(?=\d)', '', regex=True)).drop_duplicates(['구', '행정동명'])[['구', '행정동명', '행정동_코드']].rename(columns={'행정동명': '동', '행정동_코드': '이름동'})   # 면목7동 같은 행정동 이름 (DB에는 '면목제7동'으로 있어서 '제' 제거)
one = a.groupby(['구', '동'])['행정동_코드'].agg(['nunique', 'first']).reset_index()
one = one[one['nunique'] == 1].rename(columns={'first': '하나뿐동'})[['구', '동', '하나뿐동']]
near = lot[lot['산'] == ''].drop_duplicates(['구', '동', '본']).sort_values('본')

TYPO = {'시당동': '사당동', '싱계2동': '상계2동', '앙재동': '양재동', '면목3?8동': '면목3.8동', '장위 50-1': '장위동 50-1', '서소문 58-7': '서소문동 58-7'}   # 위치 입력 오타·누락 (원본 파일은 그대로 두고 여기서만 고침)

def parse(loc, gu):         # '  상암동 1618,1630,1631' → 동 '상암동', 산 '', 본 1618, 부 0
    s = loc.astype(str).str.strip()
    for k, v in TYPO.items(): s = s.str.replace(k, v, regex=False)
    s = pd.Series([x[len(g):].strip() if x.startswith(g) else x for x, g in zip(s, gu)], index=s.index)      # '구로구 구로동 313' → '구로동 313' (구 이름이 앞에 붙은 경우)
    s = s.str.replace(r'^[가-힣]+동\s+(?=[가-힣]+[\d.]*동\s)', '', regex=True).str.replace(r'(\d),\d동', r'동', regex=True)   # '시흥동 시흥4동 1-1' → '시흥4동 1-1', '상계3,4동' → '상계3동'
    s = s.str.replace(r'\(.*?\)|번지|(?<=[\d\s])(?:일대|일원)(?![가-힣])|외\s*\d*필지', '', regex=True).str.strip()
    p = s.str.extract(r'^(\D+?\d*가|\D+?[\d.]*동|\D+?)\s*(산?)\s*(\d+)(?:-(\d+))?')       # 종로6가·자양4동처럼 동 이름 안의 숫자는 번지로 읽지 않음
    return pd.DataFrame({'동': p[0].str.replace(r'\s', '', regex=True), '산': p[1], '본': pd.to_numeric(p[2]), '부': pd.to_numeric(p[3]).fillna(0)})

def to_dong(d):
    k = pd.concat([d[['지차체 코드명']].rename(columns={'지차체 코드명': '구'}).reset_index(drop=True), parse(d['사업위치'], d['지차체 코드명'].str.strip()).reset_index(drop=True)], axis=1)
    k['구'] = k['구'].str.strip()
    nb = k['본'].isna()                                                    # 사업위치에 번지가 없으면('갈현동') 사업명칭('갈현동 510 일대')에서 읽음
    if nb.any():
        kn = parse(d['사업명칭'], d['지차체 코드명'].str.strip()).reset_index(drop=True)
        same = pd.Series([isinstance(x, str) and x in str(loc).replace(' ', '') for x, loc in zip(kn['동'], d['사업위치'])], index=k.index)      # 명칭에서 읽은 동이 위치에 적힌 동과 같을 때만
        use = nb & kn['본'].notna() & same
        k.loc[use, ['동', '산', '본', '부']] = kn.loc[use, ['동', '산', '본', '부']]
    k['동2'] = k['동'].str.replace(r'[\d.]+', '', regex=True)                # 상계4동 → 상계동 (행정동식 이름으로 적힌 번지를 법정동 지번으로 다시 찾기 위함)
    r = k.merge(lot, on=['구', '동', '산', '본', '부'], how='left').merge(main, on=['구', '동', '산', '본'], how='left') \
         .merge(hname, on=['구', '동'], how='left').merge(one, on=['구', '동'], how='left')          .merge(lot.rename(columns={'동': '동2', '행정동_코드': '지번2동'}), on=['구', '동2', '산', '본', '부'], how='left')          .merge(main.rename(columns={'동': '동2', '본번동': '본번2동'}), on=['구', '동2', '산', '본'], how='left')
    assert len(r) == len(d), '같은 지번이 겹쳐 행이 늘어남'
    code, how = r['행정동_코드'].copy(), pd.Series(np.where(r['행정동_코드'].notna(), '지번 일치', ''), index=r.index)
    for col, name in [('본번동', '본번 일치'), ('이름동', '행정동 이름'), ('지번2동', '지번 일치(행정동식 이름 보정)'), ('본번2동', '본번 일치(행정동식 이름 보정)'), ('하나뿐동', '법정동 하나뿐')]:
        m = code.isna() & r[col].notna(); code[m], how[m] = r.loc[m, col], name
    miss = code.isna() & r['본'].notna()                                   # 앞·뒤 번호가 가까운 지번이 같은 행정동이면 그 동
    q = r.loc[miss, ['구', '동', '본']].drop_duplicates().sort_values('본').astype({'본': 'int64'})
    kw = dict(on='본', by=['구', '동'])
    pv = pd.merge_asof(q, near[['구', '동', '본', '행정동_코드']], direction='backward', **kw)['행정동_코드'].values
    nx = pd.merge_asof(q, near[['구', '동', '본', '행정동_코드']], direction='forward', **kw)['행정동_코드'].values
    q = q.assign(보완=np.where(pv == nx, pv, np.nan)).dropna(subset=['보완'])
    r2 = r[['구', '동', '본']].merge(q, on=['구', '동', '본'], how='left')
    m = miss & r2['보완'].notna().values; code[m], how[m] = r2.loc[m, '보완'], '인접 지번'
    d = d.reset_index(drop=True).copy()
    d['행정동_코드'], d['연결방식'] = code.astype('Int64'), how.replace('', '연결안됨')
    return d

parts = []
for 구분, f in [('정비사업', '서울시 도시계획사업(서울플랜+) 정비사업 현황.csv'), ('소규모정비사업', '서울시 도시계획사업(서울플랜+) 소규모정비사업 현황.csv')]:
    d = to_dong(pd.read_csv(RAW + '부동산/' + f, encoding='cp949'))
    d.insert(0, '구분', 구분)
    print(f'{구분}: 구역 {len(d)}개 | ' + ', '.join(f'{m} {n}' for m, n in d['연결방식'].value_counts().items()))
    parts.append(d)
r = pd.concat(parts, ignore_index=True)

# 중복 구역 표시: 같은 구역이 재건축·재개발 줄과 신속통합기획 줄로 두 번 올라온 경우 등 → 같은 구+위치(번지) 또는 같은 구+사업명이면 한 줄만 남김
#  남기는 순서: 정비사업(신속통합기획 제외) > 소규모정비사업(모아타운 제외) > 모아타운 > 신속통합기획 (모아타운·신속통합기획은 같은 자리 사업의 사전 단계라서, 실제 추진단계·공급세대가 있는 줄을 남기려는 것)
r['순위'] = np.where(r['소분류 코드명'] == '신속통합기획', 3, np.where(r['소분류 코드명'] == '모아타운', 2, np.where(r['구분'] == '정비사업', 0, 1)))
r['위치키'] = r['지차체 코드명'].str.strip() + '|' + r['사업위치'].astype(str).str.replace(r'번지|(?<=[\d\s])(?:일대|일원)(?![가-힣])', '', regex=True).str.replace(r'\s', '', regex=True)
r['명칭키'] = r['지차체 코드명'].str.strip() + '|' + r['사업명칭'].astype(str).str.replace(r'\s', '', regex=True)
order = r.sort_values('순위', kind='stable').index
dup1 = r.loc[order].duplicated('위치키').reindex(r.index)
dup2 = r.loc[order][~dup1.reindex(order).values].duplicated('명칭키').reindex(r.index).fillna(False)
r['중복'] = dup1 | dup2
r['중복이유'] = np.where(dup1, '같은 구+위치', np.where(dup2, '같은 구+사업명', ''))
print(f'중복으로 표시 {r["중복"].sum()}개 (같은 구+위치 {dup1.sum()}, 같은 구+사업명 {dup2.sum()}) → 남는 구역 {(~r["중복"]).sum()}개')
print(r[r['중복']].groupby(['구분', '소분류 코드명']).size().to_string())
r = r.drop(columns=['순위', '위치키', '명칭키'])
r.to_csv(OUT + '도시계획사업_행정동.csv', index=False, encoding='utf-8-sig')
print('저장:', os.path.abspath(OUT + '도시계획사업_행정동.csv'))
print('못 붙은 구역의 단계 분포 (전체 대비):')
print(pd.concat([r[r['연결방식'] == '연결안됨']['추진단계 코드명'].value_counts().rename('못 붙음'), r['추진단계 코드명'].value_counts().rename('전체')], axis=1).fillna(0).astype(int).to_string())
