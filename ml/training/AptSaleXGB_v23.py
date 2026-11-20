# 모델 버전 23: v22와 같은 항목(v17 12개 + 단지 평균 세대수)을 K-apt 단지가 없는 동네까지 포함한 409곳 전체로 확인 (v17 vs v23, 둘 다 XGBoost)
#   이유: v22는 K-apt 단지가 없는 16곳을 빼고 393곳으로 비교했음 → 동네를 줄이지 않고 같은 항목이 좋아지는지 확인
#   그 16곳의 단지 평균 세대수는 값을 채우지 않고 빈 값으로 두며, XGBoost가 빈 값을 스스로 처리함 (compare의 allow_missing=True)
#   XGBoost 설정은 v3와 같음. 같은 동네 409곳, 같은 분할(5분할 × 20번 반복)
#   판단: 차이가 반복마다 흔들리는 폭(v17 R² 표준편차) 안이면 "더해도 성능 차이 없음"
# 실행: python AptSaleXGB_v23.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python AptSaleXGB_v23.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록)
import os
import pandas as pd
from apt_sale_data import X_COLS, load
from apt_sale_compare import AGE_COLS, AREA, add_age, add_area, compare

NEW = '단지 평균 세대수'
d = add_age(add_area(load()))
cx = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'raw', '부동산', '행정동별_단지정보.csv'), encoding='utf-8-sig').set_index('행정동_코드')
d[NEW] = cx[NEW].reindex(d.index)
print(f'K-apt 단지 정보가 없어 빈 값으로 두는 동네 {d[NEW].isna().sum()}곳 (전체 {len(d)}곳)')
X14 = X_COLS + [AREA] + AGE_COLS                                                                    # v13의 14개
X17 = [c for c in X14 if c not in ('CCTV 수', '50대 주민 비율(%)')]                                # v17의 12개
compare(d, old=('v17', X17), new=('v23', X17 + [NEW]), name='XGBoost v23 (v17에 단지 평균 세대수 추가, 409곳 전체)', run='apt_sale_xgboost_v23_add_complex_households_all_dongs',
        why='v22와 같은 항목(v17 + K-apt 행정동별 평균 세대수)을 K-apt 단지가 없는 동네를 빼지 않고 409곳 전체로 확인. 빈 값은 채우지 않고 XGBoost가 처리.',
        old_alg='xgboost', new_alg='xgboost', allow_missing=True)
