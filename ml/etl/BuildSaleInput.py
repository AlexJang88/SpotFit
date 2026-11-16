# 매매 최종 모델(AptSaleForecastV64)이 쓰는 데이터 컬럼만 모아 한 파일로 → data/processed/매매_V64.csv
#   한 행 = 거래 1건. 매매 거래와 전세 거래를 '구분' 열로 나눠 위아래로 쌓음 (전세가율·전세 12개월 상승률에 전세 거래가 필요)
#   입력: raw/부동산/아파트_매매_2010_2026.csv, 아파트_전월세_2010_2026.csv, 서울시 상권분석서비스(영역-행정동).csv, 한강공원 위치현황.csv, 예금은행_대출금리_신규취급액_기준__*.csv
#   전처리 원칙: 원본에 있는 행을 지우지 않음 (날짜 자르기·이상값 제거·중복 제거 없음), 행 순서도 원본 그대로. 결측은 추정해서 채우지 않고 비워 둠.
#   하는 일: 컬럼 이름 통일(평당 가격, 면적) → 동 좌표 붙이기 → 한강공원·업무지구 거리 계산(V64와 같은 방식) → 계약 월의 주택담보대출 금리 붙이기
#   업무지구 좌표는 V64 코드의 상수와 같음 (아파트_매매_업무지구.csv의 거리값에서 역산한 값)
# 실행: python BuildSaleInput.py
import glob, os
import numpy as np, pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data') + os.sep
RAW, OUT = BASE + 'raw/부동산/', BASE + 'processed/'
BIZ = {'강남(GBD)': (127.0276, 37.4979), '광화문(CBD)': (126.9770, 37.5657), '여의도(YBD)': (126.9242, 37.5216)}

# ---------- 1. 거래: 매매 + 전세 쌓기 ----------
sale = pd.read_csv(RAW + '아파트_매매_2010_2026.csv', low_memory=False,
                   usecols=['행정동_코드(상권분석)', '자치구', '행정동', '계약일', '평당 금액(만원)', '층', '건축년도', '전용면적(㎡)'])
sale = sale.rename(columns={'평당 금액(만원)': '평당 가격(만원)', '전용면적(㎡)': '면적(㎡)'}); sale.insert(0, '구분', '매매')
rent = pd.read_csv(RAW + '아파트_전월세_2010_2026.csv', low_memory=False,
                   usecols=['행정동_코드(상권분석)', '자치구', '행정동', '계약일', '전월세구분', '평당 보증금(만원)', '층', '건축년도', '임대면적(㎡)'])
rent = rent[rent['전월세구분'] == '전세'].drop(columns='전월세구분')
rent = rent.rename(columns={'평당 보증금(만원)': '평당 가격(만원)', '임대면적(㎡)': '면적(㎡)'}); rent.insert(0, '구분', '전세')
print(f'매매 {len(sale):,}건, 전세 {len(rent):,}건')
d = pd.concat([sale, rent], ignore_index=True)
d['계약일'] = pd.to_datetime(d['계약일'], errors='coerce')
assert d['계약일'].notna().all() and d['행정동_코드(상권분석)'].notna().all()
d['행정동_코드(상권분석)'] = d['행정동_코드(상권분석)'].astype(int)
                                                                              # 행 순서는 원본 파일 순서 그대로 (매매 → 전세). 정렬하면 코드 하나에 행정동명이 둘인 동의 이름이 바뀜

# ---------- 2. 동 좌표 + 거리 (동마다 값 하나) ----------
area = pd.read_csv(RAW + '서울시 상권분석서비스(영역-행정동).csv', encoding='cp949').drop_duplicates('행정동_코드').set_index('행정동_코드')
dongs = pd.Index(sorted(d['행정동_코드(상권분석)'].unique()), name='행정동_코드(상권분석)')
a = area.reindex(dongs)
assert a[['엑스좌표_값', '와이좌표_값']].notna().all().all(), '영역 파일에 없는 동 있음'
hp = pd.read_csv(RAW + '한강공원 위치현황.csv', encoding='cp949').iloc[1:]
hx, hy = Transformer.from_crs('EPSG:4326', 'EPSG:5181', always_xy=True).transform(hp['위도'].astype(float).values, hp['경도'].astype(float).values)
st = pd.DataFrame(index=dongs)
st['엑스좌표_값'], st['와이좌표_값'] = a['엑스좌표_값'].values, a['와이좌표_값'].values
st['가장 가까운 한강공원까지 거리(km)'] = cKDTree(np.c_[hx, hy]).query(a[['엑스좌표_값', '와이좌표_값']].values)[0] / 1000
lon, lat = Transformer.from_crs('EPSG:5181', 'EPSG:4326', always_xy=True).transform(a['엑스좌표_값'].values, a['와이좌표_값'].values)
def haversine_km(lon0, lat0, lon, lat):
    p1, p2 = np.radians(lat0), np.radians(lat)
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon - lon0) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(h))
for k, (lo, la) in BIZ.items(): st[f'{k}까지 거리(km)'] = haversine_km(lo, la, lon, lat)
d = d.merge(st.reset_index(), on='행정동_코드(상권분석)', how='left')

# ---------- 3. 주택담보대출 금리 (계약 월 기준, 금리 파일에 없는 달은 빈 칸) ----------
rt = pd.read_csv(glob.glob(RAW + '예금은행_대출금리_신규취급액_기준__*.csv')[0], encoding='cp949')
rt = pd.to_numeric(rt[rt['계정항목별'].str.strip() == '주택담보대출 (연리%)'].iloc[0, 1:], errors='coerce')
rate = {c: v for c, v in rt.items()}                                          # '2010.01' → 금리
d['주택담보대출 금리(연%)'] = (d['계약일'].dt.year.astype(str) + '.' + d['계약일'].dt.month.astype(str).str.zfill(2)).map(rate)

# ---------- 4. 저장 ----------
cols = ['구분', '행정동_코드(상권분석)', '자치구', '행정동', '계약일', '평당 가격(만원)', '층', '건축년도', '면적(㎡)', '엑스좌표_값', '와이좌표_값',
        '가장 가까운 한강공원까지 거리(km)', '강남(GBD)까지 거리(km)', '광화문(CBD)까지 거리(km)', '여의도(YBD)까지 거리(km)', '주택담보대출 금리(연%)']
d = d[cols]
for c in ['가장 가까운 한강공원까지 거리(km)', '강남(GBD)까지 거리(km)', '광화문(CBD)까지 거리(km)', '여의도(YBD)까지 거리(km)']: d[c] = d[c].round(6)
d['계약일'] = d['계약일'].dt.strftime('%Y-%m-%d')
path = OUT + '매매_V64.csv'
d.to_csv(path, index=False, encoding='utf-8-sig')
print(f'저장: {os.path.abspath(path)} | {len(d):,}행 {len(cols)}열 | 동 {d["행정동_코드(상권분석)"].nunique()}곳')
print(d.groupby('구분').agg(건수=('계약일', 'size'), 시작=('계약일', 'min'), 끝=('계약일', 'max')).to_string())
print('\n빈 칸 현황:\n' + d.isna().groupby(d['구분']).sum().T.loc[lambda t: t.sum(axis=1) > 0].to_string())
