# 모델 버전 31: v25(15개 항목)에 '생활방범 CCTV 수'를 더하면 좋아지는지 확인 (v25 = 15개 vs v31 = 16개, 둘 다 XGBoost)
#   이유: CCTV는 평당 금액과 상관이 있어 보여서(설치연도를 아는 13개 구 222곳 기준 순위 상관 -0.34, 내 PC 간이 검사) 더해 봄
#   생활방범 CCTV 수 = 서울시_CCTV정보.csv에서 설치목적구분이 '생활방범'인 CCTV의 카메라 대수를 행정동별로 합한 값 (MapCctv.py로 행정동 연결, 설치연도는 쓰지 않음)
#   CCTV가 하나도 없는 동네는 0 (CCTV 자료는 서울 전체를 담고 있음)
#   K-apt 단지가 없는 동네(22곳)는 값을 채우지 않고 빈 값으로 두며, XGBoost가 빈 값을 스스로 처리함 (compare의 allow_missing=True)
#   XGBoost 설정은 v3와 같음. 같은 동네 409곳, 같은 분할(5분할 × 20번 반복)
#   판단: 차이가 반복마다 흔들리는 폭(v25 R² 표준편차) 안이면 "더해도 성능 차이 없음"
# 실행: python AptSaleXGB_v31.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python AptSaleXGB_v31.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록)
import os
import pandas as pd
from apt_sale_data import X_COLS, load
from apt_sale_compare import AGE_COLS, AREA, add_age, add_area, compare

V23 = '단지 평균 세대수'
V24 = '지역난방 단지 비율(%)'
V25 = '100세대당 경비 인원'
NEW = '생활방범 CCTV 수'
d = add_age(add_area(load()))

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data') + os.sep
cc = pd.read_csv(DATA + 'interim/기타/CCTV_행정동.csv', encoding='utf-8-sig', low_memory=False)
cc = cc[cc['설치목적구분'] == '생활방범']
d[NEW] = cc.groupby('행정동_코드')['카메라대수'].sum().reindex(d.index).fillna(0)
print(f'생활방범 CCTV 수: CCTV가 없어 0인 동네 {int((d[NEW] == 0).sum())}곳 (전체 {len(d)}곳)')
cx = pd.read_csv(DATA + 'processed/행정동별_단지정보.csv', encoding='utf-8-sig').set_index('행정동_코드')
d[V23] = cx[V23].reindex(d.index)
d[V24] = cx[V24].reindex(d.index)
d[V25] = cx[V25].reindex(d.index)
print(f'K-apt 단지 정보가 없어 빈 값으로 두는 동네 {d[V25].isna().sum()}곳 (전체 {len(d)}곳)')
X14 = X_COLS + [AREA] + AGE_COLS                                                                    # v13의 14개
X17 = [c for c in X14 if c not in ('CCTV 수', '50대 주민 비율(%)')]                                # v17의 12개
compare(d, old=('v25', X17 + [V23, V24, V25]), new=('v31', X17 + [V23, V24, V25, NEW]), name='XGBoost v31 (v25에 생활방범 CCTV 수 추가)', run='apt_sale_xgboost_v31_add_cctv',
        why='v25 XGBoost(15개)에 동네별 생활방범 CCTV 카메라 수(설치연도 무관)를 추가. 빈 값은 채우지 않고 XGBoost가 처리.',
        old_alg='xgboost', new_alg='xgboost', allow_missing=True)
