# 모델 버전 14: v8과 같은 실험을 XGBoost로 → v7(=v13)의 14개 항목에서 4개를 빼도 성능이 그대로인지 확인 (v13 = 14개 vs v14 = 10개, 둘 다 XGBoost)
#   뺀 항목: 50대 주민 비율, CCTV 수, 1인가구 비율, 유동인구 수  ← 랜덤포레스트 v7 중요도가 낮았던 4개
#   이유: 4개를 뺀 판단(v8)은 랜덤포레스트로 했으므로, 최종 모델 종류인 XGBoost에서도 빼도 되는지 확인
#   XGBoost 설정은 v3와 같음. 같은 동네 409곳, 같은 분할(5분할 × 20번 반복). v14의 항목은 v9와 같음
#   판단: 차이가 반복마다 흔들리는 폭(v13 R² 표준편차) 안이면 "빼도 성능 차이 없음" → 10개 유지
# 실행: python AptSaleXGB_v14.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python AptSaleXGB_v14.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록)
from apt_sale_data import X_COLS, load
from apt_sale_compare import AGE_COLS, AREA, add_age, add_area, compare

d = add_age(add_area(load()))
X7 = X_COLS + [AREA] + AGE_COLS                                                                     # v7·v13의 14개
DROP = ['50대 주민 비율(%)', 'CCTV 수', '1인가구 비율(%)', '유동인구 수']
X14 = [c for c in X7 if c not in DROP]                                                              # v8·v9·v14의 10개
compare(d, old=('v13', X7), new=('v14', X14), name='XGBoost v14 (v13에서 중요도 낮은 4개 제외)', run='apt_sale_xgboost_v14_drop_low',
        why='v13 XGBoost(14개)에서 4개(50대 비율, CCTV, 1인가구 비율, 유동인구) 제외 실험. 랜덤포레스트 v8과 같은 실험을 최종 모델 종류인 XGBoost로 다시 확인.',
        old_alg='xgboost', new_alg='xgboost')
