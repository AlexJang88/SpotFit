# 모델 버전 12: v6과 같은 실험을 XGBoost로 → 평균 전용면적을 더하면 좋아지는지 확인 (v3 = 7개 항목 vs v12 = 7개 + 평균 전용면적, 둘 다 XGBoost)
#   이유: 최종 모델이 XGBoost(v9)인데 v4~v8의 항목 선택은 랜덤포레스트로 확인했으므로, XGBoost에서도 같은지 확인
#   XGBoost 설정은 v3와 같음. 같은 동네 409곳, 같은 분할(5분할 × 20번 반복)
#   판단: 좋아짐 = R² 차이가 반복마다 흔들리는 폭(v3 R² 표준편차)보다 크고, 100개 분할 중 80개 이상에서 오차가 작음
# 실행: python AptSaleXGB_v12.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python AptSaleXGB_v12.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록)
from apt_sale_data import X_COLS, load
from apt_sale_compare import AREA, add_area, compare

d = add_area(load())
compare(d, old=('v3', X_COLS), new=('v12', X_COLS + [AREA]), name='XGBoost v12 (전용면적 추가)', run='apt_sale_xgboost_v12_area',
        why='v3 XGBoost(기본 7개)에 동네 평균 전용면적을 추가해 좋아지는지 확인. 랜덤포레스트 v6과 같은 실험을 최종 모델 종류인 XGBoost로 다시 확인.',
        old_alg='xgboost', new_alg='xgboost')
