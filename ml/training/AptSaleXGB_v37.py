# 모델 버전 37: v33(v25 + 한강공원 거리, 16개 항목)에 '항목 가중치'를 주면 좋아지는지 확인 (v33 = 기본 XGBoost vs v37 = 같은 16개 항목, 항목 가중치를 준 XGBoost)
#   이유: 중요도가 높은 항목을 나무가 더 자주 쓰게 하면 성능이 달라지는지 확인. XGBoost는 학습하면서 중요한 항목을 스스로 정하므로 효과가 작을 수 있음 (한 번 확인)
#   항목 가중치 = 각 학습 구간(교차검증 폴드의 학습 동네) 안에서 먼저 기본 XGBoost를 학습해 구한 항목별 total_gain 비율. 가중치가 큰 항목이 나무를 만들 때 더 자주 선택됨
#     (XGBoost fit의 feature_weights, 항목 80%만 쓰는 colsample_bytree=0.8 설정에서 선택 확률을 조절). 평가용 동네는 가중치 계산에 쓰지 않음 → 정답이 새지 않음
#     기여도가 0에 가까운 항목도 완전히 빼지 않도록 최소 가중치 0.01을 줌
#   X (v33 16개): 층수, 거래일, 완공연도, 직장인구 수, 1인가구 비율, 유동인구 수, 평균 전용면적, 연령대 비율 5개(10·20·30·40·60대 이상), 단지 평균 세대수, 지역난방 단지 비율, 100세대당 경비 인원, 가장 가까운 한강공원까지 거리(km)
#   한강공원 거리 = 동네 중심점(영역-행정동의 엑스·와이좌표, EPSG:5181 m)에서 한강공원 11곳(한강공원 위치현황.csv) 중 가장 가까운 곳까지의 직선거리(km). 경도·위도 칸 이름이 바뀌어 있어 반대로 읽음 (v33과 같음)
#   K-apt 단지가 없는 동네(22곳)는 값을 채우지 않고 빈 값으로 두며, XGBoost가 빈 값을 스스로 처리함 (compare의 allow_missing=True)
#   나머지 XGBoost 설정은 v3와 같음. 같은 동네 409곳, 같은 분할(5분할 × 20번 반복)
#   판단: 차이가 반복마다 흔들리는 폭(v33 R² 표준편차) 안이면 "항목 가중치를 줘도 성능 차이 없음"
# 실행: python AptSaleXGB_v37.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python AptSaleXGB_v37.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록)
import os
import numpy as np, pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.base import BaseEstimator, RegressorMixin
from apt_sale_data import X_COLS, load
from apt_sale_compare import AGE_COLS, AREA, MODELS, add_age, add_area, compare, xgb

V23 = '단지 평균 세대수'
V24 = '지역난방 단지 비율(%)'
V25 = '100세대당 경비 인원'
HG = '가장 가까운 한강공원까지 거리(km)'
MIN_W = 0.01                                                                                         # 최소 항목 가중치

class WeightedXGB(BaseEstimator, RegressorMixin):
    """학습 데이터 안에서 기본 XGBoost의 항목별 total_gain 비율을 구해, 그 비율을 항목 가중치(feature_weights)로 다시 학습"""
    def fit(self, X, y):
        gain = xgb().fit(X, y).get_booster().get_score(importance_type='total_gain')
        share = np.array([gain.get(c, 0.0) for c in X.columns], dtype=float); share = share / share.sum()
        self.weights_ = np.maximum(share, MIN_W)
        self.model_ = xgb().fit(X, y, feature_weights=self.weights_)
        self.feature_names_ = list(X.columns)
        return self
    def predict(self, X): return self.model_.predict(X)
MODELS['xgboost_fw'] = ('XGBoost(항목 가중치)', 'XGBfw', WeightedXGB)

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
compare(d, old=('v33', X33), new=('v37', X33), name='XGBoost v37 (v33에 항목 가중치 추가)', run='apt_sale_xgboost_v37_feature_weights_on_v33',
        why='v33 XGBoost(한강공원 거리 포함 16개, 항목은 그대로)에 항목 가중치를 추가. 각 학습 구간 안에서 기본 XGBoost의 total_gain 비율을 구해 feature_weights로 사용(최소 0.01). 평가 동네는 가중치 계산에 쓰지 않음. 빈 값은 채우지 않고 XGBoost가 처리.',
        old_alg='xgboost', new_alg='xgboost_fw', allow_missing=True)
