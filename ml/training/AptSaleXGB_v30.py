# 모델 버전 30: v25(15개 항목)에 '지역아동센터 수'를 더하면 좋아지는지 확인 (v25 = 15개 vs v30 = 16개, 둘 다 XGBoost)
#   이유: 지역아동센터는 저소득 가정 돌봄시설이라 많은 동네일수록 평당 금액이 낮을 것으로 봄 (v17 항목을 감안한 순위 상관 -0.19, 내 PC 간이 검사)
#   지역아동센터 수 = 행정동별_지역아동센터_최종.csv 최신 연도(2025)의 센터를 자치구+행정동 이름으로 동네별로 센 값. 센터가 없으면 0,
#   행정동별_아동인구 데이터_최종.csv(2025)에 이름이 붙지 않는 동네는 0이 아니라 빈 값으로 두고 XGBoost가 처리함
#   K-apt 단지가 없는 동네(22곳)는 값을 채우지 않고 빈 값으로 두며, XGBoost가 빈 값을 스스로 처리함 (compare의 allow_missing=True)
#   XGBoost 설정은 v3와 같음. 같은 동네 409곳, 같은 분할(5분할 × 20번 반복)
#   판단: 차이가 반복마다 흔들리는 폭(v25 R² 표준편차) 안이면 "더해도 성능 차이 없음"
# 실행: python AptSaleXGB_v30.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python AptSaleXGB_v30.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록)
import os
import re
import pandas as pd
from apt_sale_data import X_COLS, load
from apt_sale_compare import AGE_COLS, AREA, add_age, add_area, compare

V23 = '단지 평균 세대수'
V24 = '지역난방 단지 비율(%)'
V25 = '100세대당 경비 인원'
NEW = '지역아동센터 수'
d = add_age(add_area(load()))

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed') + os.sep
norm = lambda s: re.sub(r'제(?=\d)', '', str(s)).replace('·', '.').replace('ㆍ', '.').replace(' ', '')
key = lambda df: df['자치구'].astype(str) + '|' + df['행정동'].map(norm)
cn = pd.read_csv(P + '행정동별_지역아동센터_최종.csv', encoding='utf-8-sig')
cn = cn[cn['기준연도'] == cn['기준연도'].max()]
pop = pd.read_csv(P + '행정동별_아동인구 데이터_최종.csv', encoding='utf-8-sig')
pop = pop[pop['행정동'].notna() & pop['자치구'].notna() & (pop['연도'] == pop['연도'].max())]
k = key(d)
d[NEW] = k.map(key(cn).value_counts()).fillna(0).where(k.isin(key(pop)))
print(f'지역아동센터 수: 센터 있는 동네 {int((d[NEW] > 0).sum())}곳, 이름이 안 붙어 빈 값인 동네 {int(d[NEW].isna().sum())}곳 (전체 {len(d)}곳)')
cx = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed', '행정동별_단지정보.csv'), encoding='utf-8-sig').set_index('행정동_코드')
d[V23] = cx[V23].reindex(d.index)
d[V24] = cx[V24].reindex(d.index)
d[V25] = cx[V25].reindex(d.index)
print(f'K-apt 단지 정보가 없어 빈 값으로 두는 동네 {d[V25].isna().sum()}곳 (전체 {len(d)}곳)')
X14 = X_COLS + [AREA] + AGE_COLS                                                                    # v13의 14개
X17 = [c for c in X14 if c not in ('CCTV 수', '50대 주민 비율(%)')]                                # v17의 12개
compare(d, old=('v25', X17 + [V23, V24, V25]), new=('v30', X17 + [V23, V24, V25, NEW]), name='XGBoost v30 (v25에 지역아동센터 수 추가)', run='apt_sale_xgboost_v30_add_child_center',
        why='v25 XGBoost(15개)에 동네별 지역아동센터 수(2025)를 추가. 이름이 안 붙는 동네는 빈 값으로 두고 XGBoost가 처리. 빈 값은 채우지 않고 XGBoost가 처리.',
        old_alg='xgboost', new_alg='xgboost', allow_missing=True)
