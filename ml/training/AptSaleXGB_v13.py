# 모델 버전 13: v7과 같은 실험을 XGBoost로 → 상주인구 연령대 6개를 더하면 좋아지는지 확인 (v12 = 8개 항목 vs v13 = 8개 + 연령대 6개 = 14개, 둘 다 XGBoost)
#   이유: 최종 모델이 XGBoost(v9)인데 v4~v8의 항목 선택은 랜덤포레스트로 확인했으므로, XGBoost에서도 같은지 확인
#   XGBoost 설정은 v3와 같음. 같은 동네 409곳, 같은 분할(5분할 × 20번 반복)
#   판단: 좋아짐 = R² 차이가 반복마다 흔들리는 폭(v12 R² 표준편차)보다 크고, 100개 분할 중 80개 이상에서 오차가 작음
# 실행: python AptSaleXGB_v13.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python AptSaleXGB_v13.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록)
from apt_sale_data import X_COLS, load
from apt_sale_compare import AGE_COLS, AREA, add_age, add_area, compare

d = add_age(add_area(load()))
X12 = X_COLS + [AREA]                                                                               # v12의 8개
compare(d, old=('v12', X12), new=('v13', X12 + AGE_COLS), name='XGBoost v13 (전용면적 + 상주인구 연령대)', run='apt_sale_xgboost_v13_area_age',
        why='v12 XGBoost(7개 + 전용면적)에 상주인구 연령대 6개(10대~60대 이상 비율)를 추가해 좋아지는지 확인. 랜덤포레스트 v7과 같은 실험을 최종 모델 종류인 XGBoost로 다시 확인.',
        old_alg='xgboost', new_alg='xgboost')
