# 모델 버전 21: v17과 같은 12개 항목을 랜덤포레스트로 학습 (v17 XGBoost vs v21 랜덤포레스트, 항목은 똑같음)
#   이유: 항목은 그대로 두고 모델만 바꿨을 때 XGBoost와 랜덤포레스트 중 어느 쪽이 12개 항목에서 더 나은지 확인
#   랜덤포레스트 설정은 v1과 같음(나무 500개, min_samples_leaf=5). 같은 동네 409곳, 같은 분할(5분할 × 20번 반복)
#   판단: 차이가 반복마다 흔들리는 폭(v17 R² 표준편차) 안이면 "두 모델 차이 없음"
# 실행: python AptSaleRF_v21.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python AptSaleRF_v21.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록)
from apt_sale_data import X_COLS, load
from apt_sale_compare import AGE_COLS, AREA, add_age, add_area, compare

d = add_age(add_area(load()))
X14 = X_COLS + [AREA] + AGE_COLS                                                                    # v13의 14개
X17 = [c for c in X14 if c not in ('CCTV 수', '50대 주민 비율(%)')]                                # v17의 12개
compare(d, old=('v17', X17), new=('v21', X17), name='랜덤포레스트 v21 (v17과 같은 12개 항목)', run='apt_sale_random_forest_v21_same_as_v17',
        why='v17 XGBoost(12개)와 같은 항목을 랜덤포레스트로 학습. 항목은 그대로 두고 모델만 바꿨을 때의 차이를 확인.',
        old_alg='xgboost', new_alg='random_forest')
