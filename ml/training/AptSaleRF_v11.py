# 모델 버전 11: v8(10개 항목)에서 완공연도를 빼도 성능이 그대로인지 확인 (v8 = 10개 vs v11 = 9개), 모델은 둘 다 랜덤포레스트
#   뺀 항목: 완공연도 (v8 중요도 0.008로 10개 중 가장 낮음)
#   주의: 중요도가 낮다고 빼도 되는 건 아님. 완공연도가 평균 전용면적·연령대와 정보가 겹치면 중요도가 나뉘어 낮게 보일 수 있어서, 빼 본 모델로 직접 확인
#   판단: 차이가 반복마다 흔들리는 폭(v8 R² 표준편차) 안이면 "빼도 성능 차이 없음" → 9개로 줄임
# 실행: python AptSaleRF_v11.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python AptSaleRF_v11.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록)
from apt_sale_data import X_COLS, load
from apt_sale_compare import AGE_COLS, AREA, add_age, add_area, compare

d = add_age(add_area(load()))
DROP = ['50대 주민 비율(%)', 'CCTV 수', '1인가구 비율(%)', '유동인구 수']
X8 = [c for c in X_COLS + [AREA] + AGE_COLS if c not in DROP]                                       # v8의 10개
X11 = [c for c in X8 if c != '완공연도']                                                            # v11의 9개
compare(d, old=('v8', X8), new=('v11', X11), name='랜덤포레스트 v11 (v8에서 완공연도 제외)', run='apt_sale_random_forest_v11_drop_build_year',
        why='v8(10개)에서 중요도가 가장 낮은 완공연도(0.008)를 제외해도 성능이 유지되는지 확인. 평균 전용면적·연령대와 정보가 겹치는지 보기 위해 직접 빼서 비교.')
