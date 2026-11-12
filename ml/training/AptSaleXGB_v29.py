# 모델 버전 29: v25(15개 항목)에 '거래 수'를 더하면 좋아지는지 확인 (v25 = 15개 vs v29 = 16개, 둘 다 XGBoost)
#   이유: 단독으로는 흔들리는 폭 안이었지만 반복 20번 중 17번 앞선 거래 수(v27, R² +0.0055)를 가장 높은 v25 위에 더해 합쳐도 좋아지는지 확인
#   거래 수 = 아파트_매매.csv에서 동네별로 센 거래 건수 (apt_sale_data.load()의 '거래 수', 가격 정보 없음)
#   100세대당 경비 인원 = K-apt 서울 단지 중 아파트·주상복합의 경비관리 인원을 세대수로 나눈 값의 행정동별 평균 (v25) (ml/etl/AptComplexDong.py)
#   K-apt 단지가 없는 동네(22곳)는 값을 채우지 않고 빈 값으로 두며, XGBoost가 빈 값을 스스로 처리함 (compare의 allow_missing=True)
#   XGBoost 설정은 v3와 같음. 같은 동네 409곳, 같은 분할(5분할 × 20번 반복)
#   판단: 차이가 반복마다 흔들리는 폭(v25 R² 표준편차) 안이면 "더해도 성능 차이 없음"
# 실행: python AptSaleXGB_v29.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python AptSaleXGB_v29.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록)
import os
import pandas as pd
from apt_sale_data import X_COLS, load
from apt_sale_compare import AGE_COLS, AREA, add_age, add_area, compare

V23 = '단지 평균 세대수'
V24 = '지역난방 단지 비율(%)'
V25 = '100세대당 경비 인원'
NEW = '거래 수'
d = add_age(add_area(load()))
cx = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'raw', '부동산', '행정동별_단지정보.csv'), encoding='utf-8-sig').set_index('행정동_코드')
d[V23] = cx[V23].reindex(d.index)
d[V24] = cx[V24].reindex(d.index)
d[V25] = cx[V25].reindex(d.index)
print(f'K-apt 단지 정보가 없어 빈 값으로 두는 동네 {d[V25].isna().sum()}곳 (전체 {len(d)}곳)')
X14 = X_COLS + [AREA] + AGE_COLS                                                                    # v13의 14개
X17 = [c for c in X14 if c not in ('CCTV 수', '50대 주민 비율(%)')]                                # v17의 12개
compare(d, old=('v25', X17 + [V23, V24, V25]), new=('v29', X17 + [V23, V24, V25, NEW]), name='XGBoost v29 (v25에 거래 수 추가)', run='apt_sale_xgboost_v29_add_trade_count_on_v25',
        why='v25 XGBoost(15개)에 동네별 아파트 거래 건수를 추가. 빈 값은 채우지 않고 XGBoost가 처리.',
        old_alg='xgboost', new_alg='xgboost', allow_missing=True)
