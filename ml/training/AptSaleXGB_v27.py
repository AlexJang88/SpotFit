# 모델 버전 27: v24(14개 항목)에 '거래 수'를 더하면 좋아지는지 확인 (v24 = 14개 vs v27 = 15개, 둘 다 XGBoost)
#   이유: 동네별 아파트 거래 건수(거래가 활발한 동네인지)가 평당 금액과 관련 있는지 확인. 거래 수는 아파트_매매.csv에서 동네별로 센 값이라 가격 정보가 들어 있지 않음
#   K-apt 단지가 없는 동네(16곳)는 값을 채우지 않고 빈 값으로 두며, XGBoost가 빈 값을 스스로 처리함 (compare의 allow_missing=True)
#   XGBoost 설정은 v3와 같음. 같은 동네 409곳, 같은 분할(5분할 × 20번 반복)
#   판단: 차이가 반복마다 흔들리는 폭(v24 R² 표준편차) 안이면 "더해도 성능 차이 없음"
# 실행: python AptSaleXGB_v27.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python AptSaleXGB_v27.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록)
import os
import pandas as pd
from apt_sale_data import X_COLS, load
from apt_sale_compare import AGE_COLS, AREA, add_age, add_area, compare

V23 = '단지 평균 세대수'
V24 = '지역난방 단지 비율(%)'
NEW = '거래 수'
d = add_age(add_area(load()))
cx = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed', '행정동별_단지정보.csv'), encoding='utf-8-sig').set_index('행정동_코드')
d[V23] = cx[V23].reindex(d.index)
d[V24] = cx[V24].reindex(d.index)
print(f'K-apt 단지 정보가 없어 빈 값으로 두는 동네 {d[V24].isna().sum()}곳 (전체 {len(d)}곳)')
X14 = X_COLS + [AREA] + AGE_COLS                                                                    # v13의 14개
X17 = [c for c in X14 if c not in ('CCTV 수', '50대 주민 비율(%)')]                                # v17의 12개
X24 = X17 + [V23, V24]
compare(d, old=('v24', X24), new=('v27', X24 + [NEW]), name='XGBoost v27 (v24에 거래 수 추가)', run='apt_sale_xgboost_v27_add_trade_count',
        why='v24 XGBoost(14개)에 동네별 아파트 거래 건수를 추가. 빈 값은 채우지 않고 XGBoost가 처리.',
        old_alg='xgboost', new_alg='xgboost', allow_missing=True)
