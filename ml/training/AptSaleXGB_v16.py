# 모델 버전 16: v15(13개 항목)에서 1인가구 비율만 빼도 성능이 그대로인지 확인 (v15 = 13개 vs v16 = 12개, 둘 다 XGBoost)
#   이유: v14에서 4개(50대·CCTV·1인가구·유동인구)를 한꺼번에 빼면 R²가 내려갔고, v15에서 CCTV는 빼도 괜찮았음 → 남은 3개 중 하나씩 확인
#   XGBoost 설정은 v3와 같음. 같은 동네 409곳, 같은 분할(5분할 × 20번 반복)
#   판단: 차이가 반복마다 흔들리는 폭(v15 R² 표준편차) 안이면 "빼도 성능 차이 없음" → 12개로 줄임
# 실행: python AptSaleXGB_v16.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python AptSaleXGB_v16.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록)
from apt_sale_data import X_COLS, load
from apt_sale_compare import AGE_COLS, AREA, add_age, add_area, compare

d = add_age(add_area(load()))
X14 = X_COLS + [AREA] + AGE_COLS                                                                    # v13의 14개
X15 = [c for c in X14 if c != 'CCTV 수']                                                            # v15의 13개
X16 = [c for c in X15 if c != '1인가구 비율(%)']                                                    # v16의 12개
compare(d, old=('v15', X15), new=('v16', X16), name='XGBoost v16 (v15에서 1인가구 비율 제외)', run='apt_sale_xgboost_v16_drop_single_household',
        why='v15 XGBoost(13개)에서 1인가구 비율만 제외. v14에서 4개를 한꺼번에 빼면 R²가 내려가서, 남은 후보를 하나씩 빼서 영향을 확인.',
        old_alg='xgboost', new_alg='xgboost')
