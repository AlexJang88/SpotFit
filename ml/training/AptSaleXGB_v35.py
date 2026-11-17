# 모델 버전 35: v34(v32 + 한강공원 거리, 18개 항목)에서 '유동인구 수'를 빼도 되는지 확인 (v34 = 18개 vs v35 = 17개, 둘 다 XGBoost)
#   이유: v34 항목 중요도에서 유동인구 수가 0.002로 가장 낮음. 아파트 매매 v26(v24에서 유동인구 제외, -0.0014)에서도 빼도 성능이 같았음. 빼도 성능이 같으면 항목을 줄일 수 있음
#   X (v34 18개): 층수, 거래일, 완공연도, 직장인구 수, 1인가구 비율, 유동인구 수, 평균 전용면적, 연령대 비율 5개(10·20·30·40·60대 이상), 단지 평균 세대수, 지역난방 단지 비율, 100세대당 경비 인원,
#     동네 중심 좌표 x·y, 가장 가까운 한강공원까지 거리(km)  →  v35는 여기서 유동인구 수만 뺀 17개
#   한강공원 거리 = 동네 중심점(영역-행정동의 엑스·와이좌표, EPSG:5181 m)에서 한강공원 11곳(한강공원 위치현황.csv) 중 가장 가까운 곳까지의 직선거리(km). 경도·위도 칸 이름이 바뀌어 있어 반대로 읽음 (v33·v34와 같음)
#   K-apt 단지가 없는 동네(22곳)는 값을 채우지 않고 빈 값으로 두며, XGBoost가 빈 값을 스스로 처리함 (compare의 allow_missing=True)
#   XGBoost 설정은 v3와 같음. 같은 동네 409곳, 같은 분할(5분할 × 20번 반복)
#   주의: 좌표는 새로운 권역을 통째로 빼고 평가하면 오히려 떨어지므로(내 PC 간이 시험 R² 0.313 → 0.082) 이 결과는 "이웃 동네를 아는 상황"의 비교임
#   판단: 차이가 반복마다 흔들리는 폭(v34 R² 표준편차) 안이면 "빼도 성능 차이 없음"
# 실행: python AptSaleXGB_v35.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python AptSaleXGB_v35.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록)
import os
import numpy as np, pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from apt_sale_data import X_COLS, load
from apt_sale_compare import AGE_COLS, AREA, add_age, add_area, compare

V23 = '단지 평균 세대수'
V24 = '지역난방 단지 비율(%)'
V25 = '100세대당 경비 인원'
XY = ['중심 x좌표', '중심 y좌표']
HG = '가장 가까운 한강공원까지 거리(km)'
DROP = '유동인구 수'
d = add_age(add_area(load()))

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data') + os.sep
area = pd.read_csv(DATA + 'raw/부동산/서울시 상권분석서비스(영역-행정동).csv', encoding='cp949').drop_duplicates('행정동_코드').set_index('행정동_코드')
ctr = area.reindex(d.index)[['엑스좌표_값', '와이좌표_값']]
assert ctr.notna().all().all(), '중심점이 없는 동네가 있음'
d[XY[0]], d[XY[1]] = ctr['엑스좌표_값'].values, ctr['와이좌표_값'].values
hp = pd.read_csv(DATA + 'raw/부동산/한강공원 위치현황.csv', encoding='cp949').iloc[1:]                      # 첫 줄은 영문 컬럼명 중복이라 제외
lat, lon = hp['경도'].astype(float).values, hp['위도'].astype(float).values                              # 칸 이름이 바뀌어 있어 반대로 읽음
assert 37.4 < lat.min() and lat.max() < 37.7 and 126.7 < lon.min() and lon.max() < 127.2, '한강공원 좌표가 서울 범위 밖'
hx, hy = Transformer.from_crs('EPSG:4326', 'EPSG:5181', always_xy=True).transform(lon, lat)
d[HG] = cKDTree(np.c_[hx, hy]).query(ctr.values)[0] / 1000
cx = pd.read_csv(DATA + 'raw/부동산/행정동별_단지정보.csv', encoding='utf-8-sig').set_index('행정동_코드')
d[V23] = cx[V23].reindex(d.index)
d[V24] = cx[V24].reindex(d.index)
d[V25] = cx[V25].reindex(d.index)
print(f'K-apt 단지 정보가 없어 빈 값으로 두는 동네 {d[V25].isna().sum()}곳 (전체 {len(d)}곳) | 유동인구 수와 평당 금액 순위상관 {d[[DROP, "평당 금액(만원)"]].corr(method="spearman").iloc[0, 1]:+.2f}')
X14 = X_COLS + [AREA] + AGE_COLS                                                                    # v13의 14개
X17 = [c for c in X14 if c not in ('CCTV 수', '50대 주민 비율(%)')]                                # v17의 12개
X34 = X17 + [V23, V24, V25] + XY + [HG]                                                              # v34의 18개
X35 = [c for c in X34 if c != DROP]
compare(d, old=('v34', X34), new=('v35', X35), name='XGBoost v35 (v34에서 유동인구 수 제외)', run='apt_sale_xgboost_v35_drop_flow_from_v34',
        why='v34 XGBoost(좌표 2개·한강공원 거리 포함 18개)에서 유동인구 수를 제외. v34 항목 중요도에서 유동인구가 0.002로 가장 낮았고 매매 v26에서도 빼도 성능이 같았음. 빈 값은 채우지 않고 XGBoost가 처리.',
        old_alg='xgboost', new_alg='xgboost', allow_missing=True)
