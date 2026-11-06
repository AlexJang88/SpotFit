# 모델 버전 44: v41(10개 항목)에 서울 주요 업무지구 접근성(거리) 3개를 더하면 좋아지는지 확인 (v41 = 10개 vs v44 = 13개, 둘 다 XGBoost)
#   업무지구 거리 = 동네 중심점(서울시 상권분석서비스 영역-행정동)에서 3대 업무지구까지 직선거리(km): GBD 강남역, CBD 광화문·시청 일대, YBD 여의도역
#     기준점은 data/processed/부동산(전처리)/아파트_매매_업무지구.csv의 거리값에서 역산함 (GBD 127.0276,37.4979 / CBD 126.9770,37.5657 / YBD 126.9242,37.5216, 파일 거리와 0m 오차로 일치, 145,188건 검증)
#     파일은 거래의 46%에 좌표가 없어 동네 값이 비는 곳이 많아, 파일의 동네 값을 그대로 쓰지 않고 같은 기준점으로 409개 동네 전부에 직접 계산함
#   X (v41 10개): 층수, 거래일, 완공연도, 직장인구 수, 유동인구 수, 평균 전용면적, 단지 평균 세대수, 지역난방 단지 비율, 100세대당 경비 인원, 가장 가까운 한강공원까지 거리(km)  →  v44는 여기에 업무지구 거리 3개를 더한 13개
#   K-apt 단지가 없는 동네(22곳)는 값을 채우지 않고 빈 값으로 두며, XGBoost가 빈 값을 스스로 처리함 (compare의 allow_missing=True)
#   XGBoost 설정은 v3와 같음. 같은 동네 409곳, 같은 분할(5분할 × 20번 반복)
#   주의: 업무지구 거리는 동네 위치를 대신 잡는 값이라 가격의 원인으로 말하면 안 됨. 강남 거리는 집값이 비싼 지역과 가까워 효과가 크게 나올 수 있음(한 곳을 고른 것이 아니라 3대 업무지구를 함께 사용)
#   판단: 차이가 반복마다 흔들리는 폭(v41 R² 표준편차) 안이면 "더해도 성능 차이 없음"
# 실행: python AptSaleXGB_v44.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python AptSaleXGB_v44.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록)
import os
import numpy as np, pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from apt_sale_data import X_COLS, load
from apt_sale_compare import AGE_COLS, AREA, add_age, add_area, compare

V23 = '단지 평균 세대수'
V24 = '지역난방 단지 비율(%)'
V25 = '100세대당 경비 인원'
HG = '가장 가까운 한강공원까지 거리(km)'
BIZ = {'강남(GBD)': (127.0276, 37.4979), '광화문(CBD)': (126.9770, 37.5657), '여의도(YBD)': (126.9242, 37.5216)}       # 업무지구 기준점(경도, 위도)
d = add_age(add_area(load()))

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data') + os.sep
area = pd.read_csv(DATA + 'raw/부동산/서울시 상권분석서비스(영역-행정동).csv', encoding='cp949').drop_duplicates('행정동_코드').set_index('행정동_코드')
ctr = area.reindex(d.index)[['엑스좌표_값', '와이좌표_값']]
assert ctr.notna().all().all(), '중심점이 없는 동네가 있음'
hp = pd.read_csv(DATA + 'raw/기타/한강공원 위치현황.csv', encoding='cp949').iloc[1:]                      # 첫 줄은 영문 컬럼명 중복이라 제외
lat, lon = hp['경도'].astype(float).values, hp['위도'].astype(float).values                              # 칸 이름이 바뀌어 있어 반대로 읽음
assert 37.4 < lat.min() and lat.max() < 37.7 and 126.7 < lon.min() and lon.max() < 127.2, '한강공원 좌표가 서울 범위 밖'
hx, hy = Transformer.from_crs('EPSG:4326', 'EPSG:5181', always_xy=True).transform(lon, lat)
d[HG] = cKDTree(np.c_[hx, hy]).query(ctr.values)[0] / 1000
cx = pd.read_csv(DATA + 'processed/행정동별_단지정보.csv', encoding='utf-8-sig').set_index('행정동_코드')
d[V23] = cx[V23].reindex(d.index)
d[V24] = cx[V24].reindex(d.index)
d[V25] = cx[V25].reindex(d.index)
print(f'K-apt 단지 정보가 없어 빈 값으로 두는 동네 {d[V25].isna().sum()}곳 (전체 {len(d)}곳)')

def haversine_km(lon0, lat0, lon, lat):
    p1, p2 = np.radians(lat0), np.radians(lat)
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon - lon0) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(h))
dlon, dlat = Transformer.from_crs('EPSG:5181', 'EPSG:4326', always_xy=True).transform(ctr['엑스좌표_값'].values, ctr['와이좌표_값'].values)
BIZ_COLS = []
for k, (lo, la) in BIZ.items():
    c = f'{k}까지 거리(km)'; d[c] = haversine_km(lo, la, dlon, dlat); BIZ_COLS.append(c)
print('업무지구 거리(km) 범위:', {c: (round(d[c].min(), 1), round(d[c].max(), 1)) for c in BIZ_COLS}, '| 빈 값', int(d[BIZ_COLS].isna().sum().sum()))
X14 = X_COLS + [AREA] + AGE_COLS                                                                    # v13의 14개
X17 = [c for c in X14 if c not in ('CCTV 수', '50대 주민 비율(%)')]                                # v17의 12개
X33 = X17 + [V23, V24, V25, HG]                                                                      # v33의 16개
X40 = [c for c in X33 if c not in AGE_COLS]                                                          # v40: 연령대 비율(50대는 이미 빠져 있음)을 모두 뺀 11개
X41 = [c for c in X40 if c != '1인가구 비율(%)']                                                                  # v41: 거기서 1인가구 비율만 뺀 10개
X44 = X41 + BIZ_COLS
compare(d, old=('v41', X41), new=('v44', X44), name='XGBoost v44 (v41에 업무지구 3곳 거리 추가)', run='apt_sale_xgboost_v44_add_business_district_distance',
        why='v41 XGBoost(10개)에 서울 3대 업무지구(강남 GBD, 광화문 CBD, 여의도 YBD)까지 거리를 추가. 기준점은 아파트_매매_업무지구.csv의 거리값에서 역산하고 409개 동네 전부에 직접 계산. 빈 값은 채우지 않고 XGBoost가 처리.',
        old_alg='xgboost', new_alg='xgboost', allow_missing=True)
