# 모델 버전 22: v17(12개 항목)에 '단지 평균 세대수'를 더하면 좋아지는지 확인 (v17 = 12개 vs v22 = 13개, 둘 다 XGBoost)
#   이유: v17에 K-apt 단지 정보 항목을 하나씩 더해 본 결과 단지 평균 세대수가 가장 컸음 (내 PC 시험: R² +0.027, 반복 20번 중 20번 앞섬)
#   단지 평균 세대수 = K-apt 서울 단지 중 아파트·주상복합의 행정동별 평균 세대수 (ml/etl/MapComplex.py, AptComplexDong.py)
#   K-apt 단지가 없는 동네는 값을 채우지 않고 빼서, v17과 v22 모두 같은 동네로 비교함 (그래서 v17 점수가 0.548과 조금 다를 수 있음)
#   XGBoost 설정은 v3와 같음. 같은 분할 방식(5분할 × 20번 반복)
#   판단: 차이가 반복마다 흔들리는 폭(v17 R² 표준편차) 안이면 "더해도 성능 차이 없음"
# 실행: python AptSaleXGB_v22.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python AptSaleXGB_v22.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록)
import os
import pandas as pd
from apt_sale_data import X_COLS, load
from apt_sale_compare import AGE_COLS, AREA, add_age, add_area, compare

NEW = '단지 평균 세대수'
d = add_age(add_area(load()))
cx = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'raw', '부동산', '행정동별_단지정보.csv'), encoding='utf-8-sig').set_index('행정동_코드')
d[NEW] = cx[NEW].reindex(d.index)
n0 = len(d)
d = d.dropna(subset=[NEW])
print(f'K-apt 단지 정보가 없는 동네 {n0 - len(d)}곳 제외 → {len(d)}곳')
X14 = X_COLS + [AREA] + AGE_COLS                                                                    # v13의 14개
X17 = [c for c in X14 if c not in ('CCTV 수', '50대 주민 비율(%)')]                                # v17의 12개
compare(d, old=('v17', X17), new=('v22', X17 + [NEW]), name='XGBoost v22 (v17에 단지 평균 세대수 추가)', run='apt_sale_xgboost_v22_add_complex_households',
        why='v17 XGBoost(12개)에 K-apt 단지 정보의 행정동별 평균 세대수(아파트·주상복합)를 추가. K-apt 단지가 없는 동네는 제외하고 두 버전을 같은 동네로 비교.',
        old_alg='xgboost', new_alg='xgboost')
