# 모델 버전 18: v17(12개 항목)에서 1인가구 비율만 빼도 성능이 그대로인지 확인 (v17 = 12개 vs v18 = 11개, 둘 다 XGBoost)
#   이유: v16(v15에서 1인가구 제외)에서는 R²가 내려갔음 → 50대를 뺀 최종 후보(v17) 기준으로 다시 확인
#   XGBoost 설정은 v3와 같음. 같은 동네 409곳, 같은 분할(5분할 × 20번 반복)
#   판단: 차이가 반복마다 흔들리는 폭(v17 R² 표준편차) 안이면 "빼도 성능 차이 없음" → 11개로 줄임
# 실행: python AptSaleXGB_v18.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python AptSaleXGB_v18.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록)
from apt_sale_data import X_COLS, load
from apt_sale_compare import AGE_COLS, AREA, add_age, add_area, compare

d = add_age(add_area(load()))
X14 = X_COLS + [AREA] + AGE_COLS                                                                    # v13의 14개
X17 = [c for c in X14 if c not in ('CCTV 수', '50대 주민 비율(%)')]                                # v17의 12개
X18 = [c for c in X17 if c != '1인가구 비율(%)']                                                    # v18의 11개
compare(d, old=('v17', X17), new=('v18', X18), name='XGBoost v18 (v17에서 1인가구 비율 제외)', run='apt_sale_xgboost_v18_drop_single_household',
        why='v17 XGBoost(12개)에서 1인가구 비율만 제외. v16(v15 기준)에서는 나빠졌으므로 50대를 뺀 v17 기준으로 다시 확인.',
        old_alg='xgboost', new_alg='xgboost')
