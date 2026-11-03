# 모델 버전 24: v23(13개 항목)에 '지역난방 단지 비율(%)'을 더하면 좋아지는지 확인 (v23 = 13개 vs v24 = 14개, 둘 다 XGBoost)
#   이유: v23에 K-apt 단지 정보 항목을 하나씩 더해 본 결과 지역난방 단지 비율이 가장 컸음 (내 PC 시험: R² +0.043, 반복 20번 중 20번 앞섬)
#   지역난방 단지 비율 = K-apt 서울 단지 중 아파트·주상복합에서 난방방식이 지역난방인 단지의 행정동별 비율 (ml/etl/AptComplexDong.py)
#   K-apt 단지가 없는 동네(16곳)는 값을 채우지 않고 빈 값으로 두며, XGBoost가 빈 값을 스스로 처리함 (compare의 allow_missing=True)
#   XGBoost 설정은 v3와 같음. 같은 동네 409곳, 같은 분할(5분할 × 20번 반복)
#   판단: 차이가 반복마다 흔들리는 폭(v23 R² 표준편차) 안이면 "더해도 성능 차이 없음"
# 실행: python AptSaleXGB_v24.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python AptSaleXGB_v24.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록)
import os
import pandas as pd
from apt_sale_data import X_COLS, load
from apt_sale_compare import AGE_COLS, AREA, add_age, add_area, compare

V23 = '단지 평균 세대수'
NEW = '지역난방 단지 비율(%)'
d = add_age(add_area(load()))
cx = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed', '행정동별_단지정보.csv'), encoding='utf-8-sig').set_index('행정동_코드')
d[V23] = cx[V23].reindex(d.index)
d[NEW] = cx[NEW].reindex(d.index)
print(f'K-apt 단지 정보가 없어 빈 값으로 두는 동네 {d[NEW].isna().sum()}곳 (전체 {len(d)}곳)')
X14 = X_COLS + [AREA] + AGE_COLS                                                                    # v13의 14개
X17 = [c for c in X14 if c not in ('CCTV 수', '50대 주민 비율(%)')]                                # v17의 12개
compare(d, old=('v23', X17 + [V23]), new=('v24', X17 + [V23, NEW]), name='XGBoost v24 (v23에 지역난방 단지 비율 추가)', run='apt_sale_xgboost_v24_add_district_heating',
        why='v23 XGBoost(13개)에 K-apt 단지 정보의 행정동별 지역난방 단지 비율(아파트·주상복합)을 추가. 빈 값은 채우지 않고 XGBoost가 처리.',
        old_alg='xgboost', new_alg='xgboost', allow_missing=True)
