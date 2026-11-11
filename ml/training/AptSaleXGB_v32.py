# 모델 버전 32: v25(15개 항목)에 동네 중심 좌표 2개(x, y)를 더하면 좋아지는지 확인 (v25 = 15개 vs v32 = 17개, 둘 다 XGBoost)
#   이유: v25는 입지 정보가 없어 강남·서초·성동·용산 같은 비싼 동네를 평균 1,100만원/평 낮게 예측함(약점확인.csv). 위치 자체를 알려 주는 항목을 더해 봄
#   좌표 = 서울시 상권분석서비스(영역-행정동)의 행정동 중심점 엑스좌표_값, 와이좌표_값 (EPSG:5181, 단위 m). 새로 계산한 값이 아니라 파일 값 그대로
#   K-apt 단지가 없는 동네(22곳)는 값을 채우지 않고 빈 값으로 두며, XGBoost가 빈 값을 스스로 처리함 (compare의 allow_missing=True)
#   XGBoost 설정은 v3와 같음. 같은 동네 409곳, 같은 분할(5분할 × 20번 반복)
#   주의: 좌표는 "서울 지도 위 어디인가"를 알려 줄 뿐 가격이 오르는 이유가 아니며, 구를 통째로 빼고 평가하면 과하게 맞춘 정도를 따로 볼 수 있음
#   판단: 차이가 반복마다 흔들리는 폭(v25 R² 표준편차) 안이면 "더해도 성능 차이 없음"
# 실행: python AptSaleXGB_v32.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python AptSaleXGB_v32.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록)
import os
import pandas as pd
from apt_sale_data import X_COLS, load
from apt_sale_compare import AGE_COLS, AREA, add_age, add_area, compare

V23 = '단지 평균 세대수'
V24 = '지역난방 단지 비율(%)'
V25 = '100세대당 경비 인원'
NEW = ['중심 x좌표', '중심 y좌표']
d = add_age(add_area(load()))

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data') + os.sep
area = pd.read_csv(DATA + 'raw/부동산/서울시 상권분석서비스(영역-행정동).csv', encoding='cp949').drop_duplicates('행정동_코드').set_index('행정동_코드')
d[NEW[0]] = area['엑스좌표_값'].reindex(d.index)
d[NEW[1]] = area['와이좌표_값'].reindex(d.index)
assert d[NEW].notna().all().all(), '좌표가 없는 동네가 있음'
print(f'좌표: x {d[NEW[0]].min():,.0f}~{d[NEW[0]].max():,.0f}, y {d[NEW[1]].min():,.0f}~{d[NEW[1]].max():,.0f} (전체 {len(d)}곳, 빈 값 없음)')
cx = pd.read_csv(DATA + 'processed/행정동별_단지정보.csv', encoding='utf-8-sig').set_index('행정동_코드')
d[V23] = cx[V23].reindex(d.index)
d[V24] = cx[V24].reindex(d.index)
d[V25] = cx[V25].reindex(d.index)
print(f'K-apt 단지 정보가 없어 빈 값으로 두는 동네 {d[V25].isna().sum()}곳 (전체 {len(d)}곳)')
X14 = X_COLS + [AREA] + AGE_COLS                                                                    # v13의 14개
X17 = [c for c in X14 if c not in ('CCTV 수', '50대 주민 비율(%)')]                                # v17의 12개
compare(d, old=('v25', X17 + [V23, V24, V25]), new=('v32', X17 + [V23, V24, V25] + NEW), name='XGBoost v32 (v25에 동네 중심 좌표 x, y 추가)', run='apt_sale_xgboost_v32_add_coords',
        why='v25 XGBoost(15개)에 행정동 중심 좌표(x, y, 영역-행정동 파일 값 그대로)를 추가. 비싼 동네를 낮게 예측하는 약점을 위치 정보로 보완하는지 확인. 빈 값은 채우지 않고 XGBoost가 처리.',
        old_alg='xgboost', new_alg='xgboost', allow_missing=True)
