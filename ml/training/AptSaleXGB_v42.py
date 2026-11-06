# 모델 버전 42: v41(v40에서 1인가구 비율을 뺀 10개 항목)에서 '유동인구 수'를 빼도 되는지 확인 (v41 = 10개 vs v42 = 9개, 둘 다 XGBoost)
#   이유: 유동인구 수는 v41 항목 중요도가 0.007로 가장 낮고, 평당 금액과 순위상관이 -0.06으로 거의 없음. 매매 v26(v24에서 제외), v35(v34에서 제외), v36(v33에서 제외)에서도 빼도 성능이 같았음
#   X (v41 10개): 층수, 거래일, 완공연도, 직장인구 수, 유동인구 수, 평균 전용면적, 단지 평균 세대수, 지역난방 단지 비율, 100세대당 경비 인원, 가장 가까운 한강공원까지 거리(km)
#     →  v42는 여기서 유동인구 수만 뺀 9개. (v40은 v33의 연령대 비율 5개를 뺀 것, v41은 v40에서 1인가구 비율을 뺀 것)
#   한강공원 거리 = 동네 중심점(영역-행정동의 엑스·와이좌표, EPSG:5181 m)에서 한강공원 11곳(한강공원 위치현황.csv) 중 가장 가까운 곳까지의 직선거리(km). 경도·위도 칸 이름이 바뀌어 있어 반대로 읽음 (v33과 같음)
#   K-apt 단지가 없는 동네(22곳)는 값을 채우지 않고 빈 값으로 두며, XGBoost가 빈 값을 스스로 처리함 (compare의 allow_missing=True)
#   XGBoost 설정은 v3와 같음. 같은 동네 409곳, 같은 분할(5분할 × 20번 반복)
#   판단: 차이가 반복마다 흔들리는 폭(v41 R² 표준편차) 안이면 "빼도 성능 차이 없음"
# 실행: python AptSaleXGB_v42.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python AptSaleXGB_v42.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록)
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
DROP = '유동인구 수'
d = add_age(add_area(load()))

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data') + os.sep
area = pd.read_csv(DATA + 'raw/서울시 상권분석서비스(영역-행정동).csv', encoding='cp949').drop_duplicates('행정동_코드').set_index('행정동_코드')
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
X14 = X_COLS + [AREA] + AGE_COLS                                                                    # v13의 14개
X17 = [c for c in X14 if c not in ('CCTV 수', '50대 주민 비율(%)')]                                # v17의 12개
X33 = X17 + [V23, V24, V25, HG]                                                                      # v33의 16개
X40 = [c for c in X33 if c not in AGE_COLS]                                                          # v40: 연령대 비율(50대는 이미 빠져 있음)을 모두 뺀 11개
X41 = [c for c in X40 if c != '1인가구 비율(%)']                                                      # v41: 거기서 1인가구 비율만 뺀 10개
X42 = [c for c in X41 if c != DROP]                                                                  # v42: 거기서 유동인구 수만 뺀 9개
compare(d, old=('v41', X41), new=('v42', X42), name='XGBoost v42 (v41에서 유동인구 수 제외)', run='apt_sale_xgboost_v42_drop_flow_from_v41',
        why='v41 XGBoost(연령대 비율 5개·1인가구 비율 제외, 한강공원 거리 포함 10개)에서 유동인구 수를 제외. 항목 중요도가 0.007로 가장 낮고 매매 v26·v35·v36에서도 빼도 성능이 같았음. 빈 값은 채우지 않고 XGBoost가 처리.',
        old_alg='xgboost', new_alg='xgboost', allow_missing=True)
