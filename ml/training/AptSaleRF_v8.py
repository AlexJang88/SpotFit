# 모델 버전 8: v7(14개 항목)에서 중요도가 0.01 미만인 4개를 빼도 성능이 그대로인지 확인 (v7 = 14개 vs v8 = 10개)
#   뺀 항목: 50대 주민 비율(0.001), CCTV 수(0.002), 1인가구 비율(0.004), 유동인구 수(0.005)  ← v7 중요도
#   빼는 이유: 50대 비율은 6개 연령대 합이 100%라 나머지 5개로 계산되는 값이고, 1인가구 비율은 연령 구성과 정보가 겹침.
#             CCTV·유동인구는 v1부터 계속 하위권. 팀 규칙(docs/model-team-workflow.md)은 성능이 비슷하면 적은 특징을 우선함
#   판단: 차이가 반복마다 흔들리는 폭(v7 R² 표준편차) 안이면 "빼도 성능 차이 없음" → 10개로 줄임
# 실행: python AptSaleRF_v8.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python AptSaleRF_v8.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록)
from apt_sale_data import X_COLS, load
from apt_sale_compare import AGE_COLS, AREA, add_age, add_area, compare

d = add_age(add_area(load()))
X7 = X_COLS + [AREA] + AGE_COLS                                                                     # v7의 14개
DROP = ['50대 주민 비율(%)', 'CCTV 수', '1인가구 비율(%)', '유동인구 수']
X8 = [c for c in X7 if c not in DROP]                                                               # v8의 10개
compare(d, old=('v7', X7), new=('v8', X8), name='랜덤포레스트 v8 (v7에서 중요도 낮은 4개 제외)', run='apt_sale_random_forest_v8_drop_low',
        why='v7(14개)에서 중요도 0.01 미만 4개(50대 비율, CCTV, 1인가구 비율, 유동인구) 제외 실험. 50대는 연령대 합 100%라 중복, 1인가구는 연령 구성과 겹침, CCTV·유동인구는 계속 하위권.')
