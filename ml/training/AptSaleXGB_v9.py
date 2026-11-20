# 모델 버전 9: v8과 같은 10개 항목을 XGBoost로 학습 → v8(랜덤포레스트)과 비교
#   이유: 모델 종류는 처음 7개 항목(v1~v3)으로만 비교했으므로, 바뀐 항목(전용면적·연령대 포함)에서도 랜덤포레스트가 맞는지 확인
#   XGBoost 설정은 v3와 같음. 같은 동네 408곳, 같은 분할(5분할 × 20번 반복)
# 실행: python AptSaleXGB_v9.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python AptSaleXGB_v9.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록)
from apt_sale_data import X_COLS, load
from apt_sale_compare import AGE_COLS, AREA, add_age, add_area, compare

d = add_age(add_area(load()))
DROP = ['50대 주민 비율(%)', 'CCTV 수', '1인가구 비율(%)', '유동인구 수']
X8 = [c for c in X_COLS + [AREA] + AGE_COLS if c not in DROP]                                       # v8의 10개 (순서도 v8과 같음)
compare(d, old=('v8', X8), new=('v9', X8), name='XGBoost v9 (v8과 같은 10개 항목)', run='apt_sale_xgboost_v9_v8_features',
        why='v8과 같은 10개 항목을 XGBoost(v3와 같은 설정)로 학습해 랜덤포레스트 v8과 비교. 모델 종류를 바뀐 항목에서 다시 확인.',
        old_alg='random_forest', new_alg='xgboost')
