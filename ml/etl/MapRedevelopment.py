# 서울시 정비사업 추진현황(재개발·재건축) 구역에 행정동 붙이기 → data/interim/부동산/정비사업_행정동.csv
#  - 구역 위치(지번·도로명)를 도로명주소 DB(rnaddrkor_seoul.txt)에서 찾아 행정동_코드(앞 8자리)를 붙임
#  - ① 지번으로 먼저 찾고 ② 못 찾은 구역만 도로명주소로 찾음. 지번과 도로명이 서로 다른 행정동이면 지번을 따르고 '도로명과 다름'에 표시
#  - 26년 3월·6월 기준 두 파일을 모두 처리 (기준 컬럼으로 구분)
# 필요 파일: data/raw/기타/rnaddrkor_seoul.txt, data/raw/부동산/★(26년 3월기준|6월기준) 서울시 정비사업 추진현황.xlsx
import os
import numpy as np, pandas as pd

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'raw') + os.sep
OUT = os.path.join(RAW, '..', 'interim', '부동산') + os.sep
os.makedirs(OUT, exist_ok=True)

# 1) 주소 DB: 지번(구·동·산·본번·부번) → 행정동, 도로명(구·도로명·건물본번·부번) → 행정동
a = pd.read_csv(RAW + '기타/rnaddrkor_seoul.txt', sep='|', encoding='cp949', header=None, dtype=str, usecols=[3, 4, 6, 7, 8, 10, 11, 12, 13, 14])
a.columns = ['구', '동', '산', '본', '부', '도로', '지하', '건본', '건부', '행정동코드']
for c in ['본', '부', '건본', '건부']: a[c] = a[c].astype(int)
a['행정동_코드'] = a['행정동코드'].str[:8].astype(int)
lot = a.drop_duplicates(['구', '동', '산', '본', '부']).assign(산=lambda x: np.where(x['산'] == '1', '산', ''))[['구', '동', '산', '본', '부', '행정동_코드']]
road = a[a['지하'] == '0'].drop_duplicates(['구', '도로', '건본', '건부'])[['구', '도로', '건본', '건부', '행정동_코드']]

def clean(s):          # 공백·괄호·'번지'·'일대'·'일원' 제거, '양천구신정동1200'의 구 이름 제거
    return s.astype(str).str.replace(r'\s|\(.*?\)|번지|일대|일원', '', regex=True).str.replace(r'^[가-힣]+구(?=[가-힣]+(동|가|로|길)\d)', '', regex=True)

def load(path, 기준):
    d = pd.read_excel(path, header=None, skiprows=4)         # 위 4줄은 이중 머리글
    d = d[d[2].notna() & (d[2] != '자치구')].rename(columns={0: '코드', 2: '자치구', 3: '구역명', 4: '지번주소', 5: '도로명주소', 6: '공공민간', 7: '일반재촉', 8: '사업유형', 9: '추진단계', 10: '기존가구수', 23: '건립세대수', 24: '분양', 25: '임대'})
    d = d[['코드', '자치구', '구역명', '지번주소', '도로명주소', '공공민간', '일반재촉', '사업유형', '추진단계', '기존가구수', '건립세대수', '분양', '임대']].reset_index(drop=True)
    d.insert(0, '기준', 기준)
    for c in ['기존가구수', '건립세대수', '분양', '임대']: d[c] = pd.to_numeric(d[c], errors='coerce')
    p = clean(d['지번주소']).str.extract(r'^(\D+?\d*가|\D+?)(산?)(\d+)(?:-(\d+))?')
    r = clean(d['도로명주소']).str.extract(r'^(.+?)(지하)?(\d+)(?:-(\d+))?')
    k = pd.DataFrame({'구': d['자치구'], '동': p[0], '산': p[1], '본': pd.to_numeric(p[2]), '부': pd.to_numeric(p[3]).fillna(0)})
    h1 = k.merge(lot, on=['구', '동', '산', '본', '부'], how='left')['행정동_코드']
    k = pd.DataFrame({'구': d['자치구'], '도로': r[0], '건본': pd.to_numeric(r[2]), '건부': pd.to_numeric(r[3]).fillna(0)})
    h2 = k.merge(road, on=['구', '도로', '건본', '건부'], how='left')['행정동_코드']
    d['행정동_코드'] = h1.fillna(h2).astype('Int64')
    d['연결방식'] = np.where(h1.notna(), '지번', np.where(h2.notna(), '도로명', '연결안됨'))
    d['도로명과 다름'] = h1.notna() & h2.notna() & (h1 != h2)
    return d

r = pd.concat([load(RAW + f'부동산/★(26년 {m}월기준) 서울시 정비사업 추진현황.xlsx', f'26년 {m}월') for m in (3, 6)], ignore_index=True)
for k, t in r.groupby('기준'):
    print(f'{k}: 구역 {len(t)}개 | ' + ', '.join(f'{m} {n}' for m, n in t['연결방식'].value_counts().items()) + f' | 지번·도로명 행정동이 다른 구역 {t["도로명과 다름"].sum()}개')
r.to_csv(OUT + '정비사업_행정동.csv', index=False, encoding='utf-8-sig')
print('저장:', os.path.abspath(OUT + '정비사업_행정동.csv'))
