# 모델 버전 19: v17(12개 항목)에 '재건축 구역 수'를 더하면 좋아지는지 확인 (v17 = 12개 vs v19 = 13개, 둘 다 XGBoost)
#   이유: 재건축 구역이 많은 동네는 평당 금액이 높음 (구역 0개 3,884만원 → 3개 이상 6,942만원/평). v17 12개를 고려한 뒤에도 순위 상관 +0.16이 남음
#   재건축 구역 수 = 서울시 도시계획사업(서울플랜+) 정비사업 현황 중 재건축(공동·단독) 구역을 행정동별로 센 값, 구역이 없는 동네는 0 (ml/etl/AptUrbanPlanDong.py)
#   XGBoost 설정은 v3와 같음. 같은 동네 409곳, 같은 분할(5분할 × 20번 반복)
#   판단: 차이가 반복마다 흔들리는 폭(v17 R² 표준편차) 안이면 "더해도 성능 차이 없음"
# 실행: python AptSaleXGB_v19.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python AptSaleXGB_v19.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록)
import os
import pandas as pd
from apt_sale_data import X_COLS, load
from apt_sale_compare import AGE_COLS, AREA, add_age, add_area, compare

NEW = '재건축 구역 수'
d = add_age(add_area(load()))
urban = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'raw', '부동산', '도시계획사업_행정동별_아파트.csv'), encoding='utf-8-sig').set_index('행정동_코드')
d[NEW] = urban[NEW].reindex(d.index).fillna(0)
X14 = X_COLS + [AREA] + AGE_COLS                                                                    # v13의 14개
X17 = [c for c in X14 if c not in ('CCTV 수', '50대 주민 비율(%)')]                                # v17의 12개
compare(d, old=('v17', X17), new=('v19', X17 + [NEW]), name='XGBoost v19 (v17에 재건축 구역 수 추가)', run='apt_sale_xgboost_v19_add_rebuild_zones',
        why='v17 XGBoost(12개)에 재건축 구역 수(도시계획사업 정비사업 현황, 행정동별)를 추가. 재건축 구역이 많은 동네는 평당 금액이 높고, 12개를 고려한 뒤에도 연관이 남아서 확인.',
        old_alg='xgboost', new_alg='xgboost')
