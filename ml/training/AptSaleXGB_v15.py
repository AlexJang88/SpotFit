# 모델 버전 15: v13(14개 항목)에서 CCTV 수만 빼도 성능이 그대로인지 확인 (v13 = 14개 vs v15 = 13개, 둘 다 XGBoost)
#   이유: v8에서는 4개(50대·CCTV·1인가구·유동인구)를 한꺼번에 뺐고 XGBoost에서는 R²가 0.017 내려갔음 → 하나씩 빼서 어느 항목 때문인지 확인
#   CCTV 수는 v13 중요도가 0.004로 가장 낮은 편이고, 랜덤포레스트 v4(CCTV 제외)에서는 7개 항목일 때 오차가 늘었음
#   XGBoost 설정은 v3와 같음. 같은 동네 409곳, 같은 분할(5분할 × 20번 반복)
#   판단: 차이가 반복마다 흔들리는 폭(v13 R² 표준편차) 안이면 "빼도 성능 차이 없음" → 13개로 줄임
# 실행: python AptSaleXGB_v15.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python AptSaleXGB_v15.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록)
from apt_sale_data import X_COLS, load
from apt_sale_compare import AGE_COLS, AREA, add_age, add_area, compare

d = add_age(add_area(load()))
X13 = X_COLS + [AREA] + AGE_COLS                                                                    # v13의 14개
X15 = [c for c in X13 if c != 'CCTV 수']                                                            # v15의 13개
compare(d, old=('v13', X13), new=('v15', X15), name='XGBoost v15 (v13에서 CCTV 제외)', run='apt_sale_xgboost_v15_drop_cctv',
        why='v13 XGBoost(14개)에서 CCTV 수만 제외. v14에서 4개를 한꺼번에 빼면 R²가 내려가서, 하나씩 빼서 영향을 확인.',
        old_alg='xgboost', new_alg='xgboost')
