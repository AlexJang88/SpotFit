# 모델 버전 28: v24(14개 항목)에 '일반교습학원 수'를 더하면 좋아지는지 확인 (v24 = 14개 vs v28 = 15개, 둘 다 XGBoost)
#   이유: 입시·보습 학원이 많은 동네(학군)가 평당 금액과 관련 있는지 확인
#   일반교습학원 수 = 서울시 상권분석서비스(점포-행정동)에서 업종이 '일반교습학원'(CS200001)인 전체_점포_수의 동네별 중앙값 (2025년 1분기~2026년 2분기 6개 분기)
#   K-apt 단지가 없는 동네(16곳)는 값을 채우지 않고 빈 값으로 두며, XGBoost가 빈 값을 스스로 처리함 (compare의 allow_missing=True)
#   XGBoost 설정은 v3와 같음. 같은 동네 409곳, 같은 분할(5분할 × 20번 반복)
#   판단: 차이가 반복마다 흔들리는 폭(v24 R² 표준편차) 안이면 "더해도 성능 차이 없음"
# 실행: python AptSaleXGB_v28.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python AptSaleXGB_v28.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록)
import os
import pandas as pd
from apt_sale_data import X_COLS, load
from apt_sale_compare import AGE_COLS, AREA, add_age, add_area, compare

V23 = '단지 평균 세대수'
V24 = '지역난방 단지 비율(%)'
NEW = '일반교습학원 수'
d = add_age(add_area(load()))
st = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'raw', '서울시 상권분석서비스(점포-행정동).csv'), encoding='cp949', usecols=['행정동_코드', '서비스_업종_코드', '전체_점포_수'])
d[NEW] = st[st['서비스_업종_코드'] == 'CS200001'].groupby('행정동_코드')['전체_점포_수'].median().reindex(d.index)
print(f'일반교습학원 수가 없어 빈 값으로 두는 동네 {d[NEW].isna().sum()}곳')
cx = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed', '행정동별_단지정보.csv'), encoding='utf-8-sig').set_index('행정동_코드')
d[V23] = cx[V23].reindex(d.index)
d[V24] = cx[V24].reindex(d.index)
print(f'K-apt 단지 정보가 없어 빈 값으로 두는 동네 {d[V24].isna().sum()}곳 (전체 {len(d)}곳)')
X14 = X_COLS + [AREA] + AGE_COLS                                                                    # v13의 14개
X17 = [c for c in X14 if c not in ('CCTV 수', '50대 주민 비율(%)')]                                # v17의 12개
X24 = X17 + [V23, V24]
compare(d, old=('v24', X24), new=('v28', X24 + [NEW]), name='XGBoost v28 (v24에 일반교습학원 수 추가)', run='apt_sale_xgboost_v28_add_academy_count',
        why='v24 XGBoost(14개)에 동네별 일반교습학원 점포 수(상권분석서비스 점포-행정동)를 추가. 빈 값은 채우지 않고 XGBoost가 처리. 빈 값은 채우지 않고 XGBoost가 처리.',
        old_alg='xgboost', new_alg='xgboost', allow_missing=True)
