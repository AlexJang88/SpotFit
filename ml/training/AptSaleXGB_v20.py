# 모델 버전 20: v3(항목 7개)에서 CCTV 수만 '주소로 매핑한 값'으로 바꿔도 좋아지는지 확인 (v3 vs v20, 둘 다 XGBoost, 항목 수는 7개로 같음)
#   이유: 기존 CCTV 수는 위경도로 가장 가까운 행정동 중심점에 배정해서 도로명주소로 확인된 CCTV의 26%가 다른 행정동에 들어가 있었음
#   새 CCTV 수 = 도로명주소·지번주소를 도로명주소 DB로 행정동에 붙인 카메라 대수 합 (ml/etl/MapCctv.py, 주소로 붙은 것 91.8%, 나머지는 좌표)
#   나머지 6개 항목(층수·거래일·완공연도·직장인구·1인가구·유동인구)과 XGBoost 설정은 v3와 같음. 같은 동네 409곳, 같은 분할(5분할 × 20번 반복)
#   판단: 차이가 반복마다 흔들리는 폭(v3 R² 표준편차) 안이면 "매핑을 바꿔도 성능 차이 없음"
# 실행: python AptSaleXGB_v20.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python AptSaleXGB_v20.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록)
import os
import pandas as pd
from apt_sale_data import X_COLS, load
from apt_sale_compare import compare

NEW = 'CCTV 수(주소 매핑)'
d = load()
cctv = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed', '행정동별_CCTV.csv'), encoding='utf-8-sig').set_index('행정동_코드')
d[NEW] = cctv['CCTV 수'].reindex(d.index)
assert d[NEW].notna().all(), '새 CCTV 수가 없는 동네가 있음'
X20 = [NEW if c == 'CCTV 수' else c for c in X_COLS]
compare(d, old=('v3', X_COLS), new=('v20', X20), name='XGBoost v20 (v3에서 CCTV 수를 주소 매핑 값으로 교체)', run='apt_sale_xgboost_v20_cctv_address_mapping',
        why='v3 XGBoost(7개)에서 CCTV 수만 위경도 가까운 중심점 방식 대신 주소(도로명·지번)로 행정동을 붙인 값으로 교체. 나머지 항목·설정·분할은 동일.',
        old_alg='xgboost', new_alg='xgboost')
