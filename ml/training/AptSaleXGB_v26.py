# 모델 버전 26: v24(14개 항목)에서 '유동인구 수'를 빼도 되는지 확인 (v24 = 14개 vs v26 = 13개, 둘 다 XGBoost)
#   이유: v25의 항목 중요도에서 유동인구 수가 가장 낮게 나옴 → 빼도 점수가 유지되면 항목을 줄이는 쪽이 낫다는 팀 규칙에 맞음
#   K-apt 단지가 없는 동네(16곳)는 값을 채우지 않고 빈 값으로 두며, XGBoost가 빈 값을 스스로 처리함 (compare의 allow_missing=True)
#   XGBoost 설정은 v3와 같음. 같은 동네 409곳, 같은 분할(5분할 × 20번 반복)
#   판단: 차이가 반복마다 흔들리는 폭(v24 R² 표준편차) 안이면 "빼도 성능 차이 없음"
# 실행: python AptSaleXGB_v26.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python AptSaleXGB_v26.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록)
import os
import pandas as pd
from apt_sale_data import X_COLS, load
from apt_sale_compare import AGE_COLS, AREA, add_age, add_area, compare

V23 = '단지 평균 세대수'
V24 = '지역난방 단지 비율(%)'
DROP = '유동인구 수'
d = add_age(add_area(load()))
cx = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'raw', '부동산', '행정동별_단지정보.csv'), encoding='utf-8-sig').set_index('행정동_코드')
d[V23] = cx[V23].reindex(d.index)
d[V24] = cx[V24].reindex(d.index)
print(f'K-apt 단지 정보가 없어 빈 값으로 두는 동네 {d[V24].isna().sum()}곳 (전체 {len(d)}곳)')
X14 = X_COLS + [AREA] + AGE_COLS                                                                    # v13의 14개
X17 = [c for c in X14 if c not in ('CCTV 수', '50대 주민 비율(%)')]                                # v17의 12개
X24 = X17 + [V23, V24]
compare(d, old=('v24', X24), new=('v26', [c for c in X24 if c != DROP]), name='XGBoost v26 (v24에서 유동인구 수 제외)', run='apt_sale_xgboost_v26_drop_floating_pop',
        why='v24 XGBoost(14개)에서 항목 중요도가 가장 낮은 유동인구 수를 빼도 점수가 유지되는지 확인. 빈 값은 채우지 않고 XGBoost가 처리.',
        old_alg='xgboost', new_alg='xgboost', allow_missing=True)
