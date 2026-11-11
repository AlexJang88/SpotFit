# 모델 버전 33: v25(15개 항목)에 '가장 가까운 한강공원까지 거리(km)'를 더하면 좋아지는지 확인 (v25 = 15개 vs v33 = 16개, 둘 다 XGBoost)
#   이유: 한강 접근성(수변 공원)은 집값에 영향을 줄 수 있고, 특정 장소를 고르지 않는 환경 요인임. v25는 한강과 관련된 항목이 없고, 가장 크게 틀리는 동네(반포·성수·보광 등)가 한강변임
#   한강공원 거리 = 동네 중심점(서울시 상권분석서비스 영역-행정동의 엑스·와이좌표, EPSG:5181 m)에서 한강공원 11곳(한강공원 위치현황.csv: 광나루·잠실·뚝섬·잠원·반포·이촌·여의도·양화·난지·망원·강서) 중
#     가장 가까운 곳까지의 직선거리(km). 파일의 경도·위도 칸 이름이 서로 바뀌어 있어(경도 칸에 37.5, 위도 칸에 127.1) 반대로 읽음. 공원은 점 1개씩이라 실제 강변까지 거리와 1~2km 어긋날 수 있음
#   K-apt 단지가 없는 동네(22곳)는 값을 채우지 않고 빈 값으로 두며, XGBoost가 빈 값을 스스로 처리함 (compare의 allow_missing=True)
#   XGBoost 설정은 v3와 같음. 같은 동네 409곳, 같은 분할(5분할 × 20번 반복)
#   주의: 한강공원 거리는 동네 위치를 대신 잡는 값이라 가격의 원인으로 말하면 안 됨. 좌표(v32)와 달리 새로운 권역을 통째로 빼고 평가해도 효과가 유지됨(내 PC 간이 시험 R² 0.313 → 0.385)
#   판단: 차이가 반복마다 흔들리는 폭(v25 R² 표준편차) 안이면 "더해도 성능 차이 없음"
# 실행: python AptSaleXGB_v33.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python AptSaleXGB_v33.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록)
import os
import numpy as np, pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from apt_sale_data import X_COLS, load
from apt_sale_compare import AGE_COLS, AREA, add_age, add_area, compare

V23 = '단지 평균 세대수'
V24 = '지역난방 단지 비율(%)'
V25 = '100세대당 경비 인원'
NEW = '가장 가까운 한강공원까지 거리(km)'
d = add_age(add_area(load()))

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data') + os.sep
area = pd.read_csv(DATA + 'raw/부동산/서울시 상권분석서비스(영역-행정동).csv', encoding='cp949').drop_duplicates('행정동_코드').set_index('행정동_코드')
ctr = area.reindex(d.index)[['엑스좌표_값', '와이좌표_값']]
assert ctr.notna().all().all(), '중심점이 없는 동네가 있음'
hp = pd.read_csv(DATA + 'raw/부동산/한강공원 위치현황.csv', encoding='cp949').iloc[1:]                      # 첫 줄은 영문 컬럼명 중복이라 제외
lat, lon = hp['경도'].astype(float).values, hp['위도'].astype(float).values                              # 칸 이름이 바뀌어 있어 반대로 읽음
assert 37.4 < lat.min() and lat.max() < 37.7 and 126.7 < lon.min() and lon.max() < 127.2, '한강공원 좌표가 서울 범위 밖'
hx, hy = Transformer.from_crs('EPSG:4326', 'EPSG:5181', always_xy=True).transform(lon, lat)
d[NEW] = cKDTree(np.c_[hx, hy]).query(ctr.values)[0] / 1000
print(f'{NEW}: 한강공원 {len(hp)}곳, 거리 범위 {d[NEW].min():.2f}~{d[NEW].max():.2f}km, 중앙 {d[NEW].median():.2f}km, 3km 이내 동네 {int((d[NEW] <= 3).sum())}곳 (전체 {len(d)}곳, 빈 값 없음)')
cx = pd.read_csv(DATA + 'raw/부동산/행정동별_단지정보.csv', encoding='utf-8-sig').set_index('행정동_코드')
d[V23] = cx[V23].reindex(d.index)
d[V24] = cx[V24].reindex(d.index)
d[V25] = cx[V25].reindex(d.index)
print(f'K-apt 단지 정보가 없어 빈 값으로 두는 동네 {d[V25].isna().sum()}곳 (전체 {len(d)}곳)')
X14 = X_COLS + [AREA] + AGE_COLS                                                                    # v13의 14개
X17 = [c for c in X14 if c not in ('CCTV 수', '50대 주민 비율(%)')]                                # v17의 12개
compare(d, old=('v25', X17 + [V23, V24, V25]), new=('v33', X17 + [V23, V24, V25, NEW]), name='XGBoost v33 (v25에 가장 가까운 한강공원까지 거리 추가)', run='apt_sale_xgboost_v33_add_hangang_park_distance',
        why='v25 XGBoost(15개)에 동네 중심점에서 가장 가까운 한강공원까지 직선거리(km, 한강공원 11곳 점 좌표)를 추가. 한강 접근성이라는 환경 요인이며 특정 장소를 고르지 않음. 빈 값은 채우지 않고 XGBoost가 처리.',
        old_alg='xgboost', new_alg='xgboost', allow_missing=True)
