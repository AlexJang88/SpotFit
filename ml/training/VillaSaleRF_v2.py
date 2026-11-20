# 빌라 매매 모델 버전 2: 아파트 최종(v8) 10개 항목 + 랜덤포레스트 → 빌라 v1(기본 7개)과 비교
#   10개: 층수, 거래일, 완공연도, 직장인구 수, 평균 전용면적, 10대·20대·30대·40대·60대 이상 주민 비율 (CCTV·1인가구·유동인구·50대 비율 제외)
#   층수·거래일·완공연도·평균 전용면적은 빌라 거래로 계산, 직장인구·연령대 비율은 동네 특징이라 아파트와 같은 값
#   이유: 아파트에서 효과가 컸던 전용면적(v6)·연령대(v7)가 빌라에도 통하는지, 아파트에서 뺀 항목이 빌라에서도 필요 없는지 확인
# 실행: python VillaSaleRF_v2.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python VillaSaleRF_v2.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록)
from apt_sale_data import X_COLS, load
from apt_sale_compare import AGE_COLS, AREA, add_age, add_area, compare

d = add_age(add_area(load(kind='villa'), kind='villa'))
DROP = ['50대 주민 비율(%)', 'CCTV 수', '1인가구 비율(%)', '유동인구 수']
X10 = [c for c in X_COLS + [AREA] + AGE_COLS if c not in DROP]                                      # 아파트 v8과 같은 10개 (순서도 같음)
compare(d, old=('v1', X_COLS), new=('v2', X10), name='랜덤포레스트 v2 (아파트 최종 10개 항목)', run='villa_sale_random_forest_v2_apt_final_features',
        why='빌라 v1(기본 7개)에 아파트 최종(v8) 10개 항목을 적용. 전용면적은 빌라 거래로 계산, 연령대는 2022Q4~2023Q3 제외·50대 제외(합 100%라 중복).',
        kind='villa')
