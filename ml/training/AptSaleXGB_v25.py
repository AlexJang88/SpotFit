# 모델 버전 25: v24(14개 항목)에 '100세대당 경비 인원'을 더하면 좋아지는지 확인 (v24 = 14개 vs v25 = 15개, 둘 다 XGBoost)
#   이유: v24에 K-apt 단지 정보 항목 21개를 하나씩 더해 본 결과 100세대당 경비 인원이 가장 컸음 (내 PC 시험: R² +0.017, 반복 20번 중 20번 앞섬)
#   100세대당 경비 인원 = K-apt 서울 단지 중 아파트·주상복합의 경비관리 인원을 세대수로 나눈 값의 행정동별 평균 (ml/etl/AptComplexDong.py)
#   K-apt 단지가 없는 동네(22곳)는 값을 채우지 않고 빈 값으로 두며, XGBoost가 빈 값을 스스로 처리함 (compare의 allow_missing=True)
#   XGBoost 설정은 v3와 같음. 같은 동네 409곳, 같은 분할(5분할 × 20번 반복)
#   판단: 차이가 반복마다 흔들리는 폭(v24 R² 표준편차) 안이면 "더해도 성능 차이 없음"
# 실행: python AptSaleXGB_v25.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python AptSaleXGB_v25.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록)
import os
import pandas as pd
from apt_sale_data import X_COLS, load
from apt_sale_compare import AGE_COLS, AREA, add_age, add_area, compare

V23 = '단지 평균 세대수'
V24 = '지역난방 단지 비율(%)'
NEW = '100세대당 경비 인원'
d = add_age(add_area(load()))
cx = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed', '행정동별_단지정보.csv'), encoding='utf-8-sig').set_index('행정동_코드')
d[V23] = cx[V23].reindex(d.index)
d[V24] = cx[V24].reindex(d.index)
d[NEW] = cx[NEW].reindex(d.index)
print(f'K-apt 단지 정보가 없어 빈 값으로 두는 동네 {d[NEW].isna().sum()}곳 (전체 {len(d)}곳)')
X14 = X_COLS + [AREA] + AGE_COLS                                                                    # v13의 14개
X17 = [c for c in X14 if c not in ('CCTV 수', '50대 주민 비율(%)')]                                # v17의 12개
compare(d, old=('v24', X17 + [V23, V24]), new=('v25', X17 + [V23, V24, NEW]), name='XGBoost v25 (v24에 100세대당 경비 인원 추가)', run='apt_sale_xgboost_v25_add_guard_per_100',
        why='v24 XGBoost(14개)에 K-apt 단지 정보의 행정동별 100세대당 경비 인원(아파트·주상복합)을 추가. 빈 값은 채우지 않고 XGBoost가 처리.',
        old_alg='xgboost', new_alg='xgboost', allow_missing=True)
