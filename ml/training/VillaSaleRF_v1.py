# 빌라 매매 모델 버전 1: 아파트 v1과 같은 기본 7개 항목 + 랜덤포레스트 → 기준 모델(평균으로 찍기)과 비교
#   X: 층수, 거래일, 완공연도, 직장인구 수, CCTV 수, 1인가구 비율, 유동인구 수   y: 빌라 매매 평당 금액 (동네 중앙값, 만원)
#   거래 기준·동네 기준·분할·모델 설정은 아파트와 같음 (취소 거래·20㎡ 미만·지번 불일치 제외, 거래 10건 이상 동네, 5분할 × 20번 반복)
# 실행: python VillaSaleRF_v1.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python VillaSaleRF_v1.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록)
from apt_sale_data import X_COLS, load
from apt_sale_compare import compare

d = load(kind='villa')
compare(d, old=('baseline', X_COLS), new=('v1', X_COLS), name='랜덤포레스트 v1 (기본 7개)', run='villa_sale_random_forest_v1',
        why='빌라 매매 첫 모델. 아파트 v1과 같은 기본 7개 항목·같은 설정의 랜덤포레스트를 기준 모델(평균으로 찍기)과 비교.',
        old_alg='mean', new_alg='random_forest', kind='villa')
