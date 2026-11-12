# 아파트 매매 학습 데이터 (행정동 1곳 = 1행): 모델 v1, v2 ... 가 같은 데이터로 공정하게 비교되도록 여기서 한 번만 만든다
#   X: 층수, 거래일, 완공연도, 직장인구 수, CCTV 수, 1인가구 비율, 유동인구 수   y: 평당 금액 (동네 중앙값, 만원)
# 실행 전에 ml/etl 의 MergeSingleHousehold.py → MergeRent.py → ConvertSale.py → AptHypothesis.py 가 끝나 있어야 함 (평당금액_행정동표.csv 에서 동네 특징을 가져옴)
#   매매 거래는 data/raw/부동산/아파트_매매.csv (빌라는 빌라_매매.csv)를 읽음
import os
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed') + os.sep
OUT = DATA + 'output' + os.sep
JUSO = os.path.join(DATA, '..', 'interim', '부동산(도로명주소)') + os.sep                          # 행정동 변환한 실거래 파일 (중간 결과라 data/interim)
OLD = {11230515: 11230536, 11230533: 11230536, 11740525: 11740520, 11740526: 11740520, 11680675: 11680740}    # 새 행정동 코드 → 상권분석의 옛 코드 (AptHypothesis.py 와 같음)
MIN_N = 10          # 동네당 거래가 이만큼은 있어야 그 동네의 중앙값·평균을 믿고 사용
X_COLS = ['층수', '거래일(년)', '완공연도', '직장인구 수', 'CCTV 수', '1인가구 비율(%)', '유동인구 수']
Y = '평당 금액(만원)'

PROC = DATA + '../raw/부동산' + os.sep                                                              # 전처리가 끝난 실거래 파일 (취소 거래·지번 불일치 제외, 행정동_코드(상권분석) 포함)
PROC_VILLA = DATA + '../interim/부동산' + os.sep                                                  # 빌라 매매 파일은 raw가 아니라 interim/부동산
KINDS = {'apt': ('아파트', '아파트_매매.csv'), 'villa': ('빌라', '빌라_매매.csv')}                  # 종류: (한글 이름, 매매 파일: data/raw/부동산/)

def load(drop_missing=True, kind='apt'):     # drop_missing=False 이면 직장인구·1인가구가 빈 동네도 남김 (빈 값을 처리하는 모델용), kind='villa' 이면 빌라 매매
    # 매매 거래 → 행정동별 한 표 (전처리된 아파트_매매.csv / 빌라_매매.csv 사용: 취소 거래·지번 불일치는 이미 제외됨, 평수와 상관없이 작은 집도 모두 포함). 아파트·빌라 모두 같은 기준
    a = pd.read_csv((PROC if kind == 'apt' else PROC_VILLA) + KINDS[kind][1], low_memory=False, usecols=['행정동_코드(상권분석)', '계약일', Y, '층', '건축년도'])
    day = pd.to_datetime(a['계약일'], errors='coerce')
    a = pd.DataFrame({'행정동_코드': a['행정동_코드(상권분석)'].astype(int), Y: a[Y],                                                                   # 평당 금액 = 거래금액(만원) / 전용면적(평)
                      '층수': a['층'], '거래일(년)': day.dt.year + (day.dt.dayofyear - 1) / 365, '완공연도': a['건축년도']})
    g = a.groupby('행정동_코드').agg(**{'거래 수': (Y, 'size'), Y: (Y, 'median'), '층수': ('층수', 'mean'), '거래일(년)': ('거래일(년)', 'mean'), '완공연도': ('완공연도', 'mean')})    # y는 동네 중앙값, 항목은 동네 평균
    info = pd.read_csv(OUT + '../../raw/부동산/평당금액_행정동표.csv').set_index('행정동_코드')                                          # 동네 특징(직장인구·CCTV·1인가구·유동인구)과 동네 이름
    d = g[g['거래 수'] >= MIN_N].join(info[['자치구', '행정동', '직장인구 수', 'CCTV 수', '1인가구 비율(%)', '유동인구 수']])
    n0 = len(d); d = d.dropna(subset=X_COLS) if drop_missing else d                                                                            # 직장인구·1인가구가 비어 있는 동네(11곳)는 제외: 릿지가 빈 값을 못 다루고, 모든 모델을 같은 동네로 비교하기 위해서 (랜덤포레스트·XGBoost는 빈 값을 처리할 수 있음). 빈 값은 채우지 않음
    print(f'{KINDS[kind][0]} 매매 {len(a):,}건 → 행정동 {len(g)}곳 → 거래 {MIN_N}건 이상 {n0}곳 → 빈 값 제외 {len(d)}곳 | 평당 금액 중앙값의 중앙값 {d[Y].median():,.0f}만원')
    return d
