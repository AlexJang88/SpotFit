# 모델 버전 10: v8과 같은 10개 항목을 릿지로 학습 → v8(랜덤포레스트)과 비교
#   이유: 팀 문서(docs/model-team-workflow.md)의 비교 후보 "Ridge: 선형 관계와 단순성 확인"을 바뀐 항목(전용면적·연령대 포함)에서도 확인
#   릿지 설정은 v2와 같음. 같은 동네 408곳, 같은 분할(5분할 × 20번 반복)
# 실행: python AptSaleRidge_v10.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python AptSaleRidge_v10.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록)
from apt_sale_data import X_COLS, load
from apt_sale_compare import AGE_COLS, AREA, add_age, add_area, compare

d = add_age(add_area(load()))
DROP = ['50대 주민 비율(%)', 'CCTV 수', '1인가구 비율(%)', '유동인구 수']
X8 = [c for c in X_COLS + [AREA] + AGE_COLS if c not in DROP]                                       # v8의 10개 (순서도 v8과 같음)
compare(d, old=('v8', X8), new=('v10', X8), name='릿지 v10 (v8과 같은 10개 항목)', run='apt_sale_ridge_v10_v8_features',
        why='v8과 같은 10개 항목을 릿지(v2와 같은 설정)로 학습해 랜덤포레스트 v8과 비교. 단순한 직선 공식으로 충분한지 바뀐 항목에서 다시 확인.',
        old_alg='random_forest', new_alg='ridge')
