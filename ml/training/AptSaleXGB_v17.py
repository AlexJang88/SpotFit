# 모델 버전 17: v15(13개 항목)에서 50대 주민 비율만 빼도 성능이 그대로인지 확인 (v15 = 13개 vs v17 = 12개, 둘 다 XGBoost)
#   이유: 연령대 6개 비율은 합이 100%라서 50대는 나머지 5개로 계산됨 → 정보가 사라지지 않음. 하나씩 빼서 영향을 확인 (v15: CCTV 제외 OK, v16: 1인가구 제외는 나빠짐)
#   XGBoost 설정은 v3와 같음. 같은 동네 409곳, 같은 분할(5분할 × 20번 반복)
#   판단: 차이가 반복마다 흔들리는 폭(v15 R² 표준편차) 안이면 "빼도 성능 차이 없음" → 12개로 줄임
# 실행: python AptSaleXGB_v17.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python AptSaleXGB_v17.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록)
from apt_sale_data import X_COLS, load
from apt_sale_compare import AGE_COLS, AREA, add_age, add_area, compare

d = add_age(add_area(load()))
X14 = X_COLS + [AREA] + AGE_COLS                                                                    # v13의 14개
X15 = [c for c in X14 if c != 'CCTV 수']                                                            # v15의 13개
X17 = [c for c in X15 if c != '50대 주민 비율(%)']                                                  # v17의 12개
compare(d, old=('v15', X15), new=('v17', X17), name='XGBoost v17 (v15에서 50대 비율 제외)', run='apt_sale_xgboost_v17_drop_age50',
        why='v15 XGBoost(13개)에서 50대 주민 비율만 제외. 연령대 6개는 합이 100%라 50대는 나머지 5개로 계산되므로 정보 손실 없이 빼도 되는지 확인.',
        old_alg='xgboost', new_alg='xgboost')
