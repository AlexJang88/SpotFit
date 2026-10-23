# 아파트 매매 학습 데이터 (행정동 1곳 = 1행): 모델 v1, v2 ... 가 같은 데이터로 공정하게 비교되도록 여기서 한 번만 만든다
#   X: 층수, 거래일, 완공연도, 직장인구 수, CCTV 수, 1인가구 비율, 유동인구 수   y: 평당 금액 (동네 중앙값, 만원)
# 실행 전에 ml/etl 의 MergeSingleHousehold.py → MergeRent.py → ConvertSale.py → AptHypothesis.py 가 끝나 있어야 함 (평당금액_행정동표.csv 에서 동네 특징을 가져옴)
import os
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed') + os.sep
OUT = DATA + 'output' + os.sep
OLD = {11230515: 11230536, 11230533: 11230536, 11740525: 11740520, 11740526: 11740520, 11680675: 11680740}    # 새 행정동 코드 → 상권분석의 옛 코드 (AptHypothesis.py 와 같음)
MIN_N = 10          # 동네당 거래가 이만큼은 있어야 그 동네의 중앙값·평균을 믿고 사용
X_COLS = ['층수', '거래일(년)', '완공연도', '직장인구 수', 'CCTV 수', '1인가구 비율(%)', '유동인구 수']
Y = '평당 금액(만원)'

def load():
    # 아파트 매매 거래 → 행정동별 한 표 (AptHypothesis.py 의 deals() 와 같은 기준: 취소 거래·20㎡ 미만·지번 불일치 제외)
    cols = ['행정동_코드', '연결방식', 'cdealType', 'excluUseAr', 'dealAmount', 'dealYear', 'dealMonth', 'dealDay', 'buildYear', 'floor']
    a = pd.read_csv(DATA + '부동산(도로명주소)/apt_sale_juso.csv', low_memory=False, usecols=cols)
    a = a[a['cdealType'].isna() & (a['excluUseAr'] >= 20) & (a['연결방식'] == '지번 일치')]
    day = pd.to_datetime(a[['dealYear', 'dealMonth', 'dealDay']].set_axis(['year', 'month', 'day'], axis=1), errors='coerce')
    a = pd.DataFrame({'행정동_코드': a['행정동_코드'].astype(int).replace(OLD), Y: pd.to_numeric(a['dealAmount'].str.replace(',', ''), errors='coerce') / (a['excluUseAr'] / 3.3058),   # 거래금액(만원) / 전용면적(평)
                      '층수': a['floor'], '거래일(년)': day.dt.year + (day.dt.dayofyear - 1) / 365, '완공연도': a['buildYear']})
    g = a.groupby('행정동_코드').agg(**{'거래 수': (Y, 'size'), Y: (Y, 'median'), '층수': ('층수', 'mean'), '거래일(년)': ('거래일(년)', 'mean'), '완공연도': ('완공연도', 'mean')})    # y는 동네 중앙값, 항목은 동네 평균
    info = pd.read_csv(OUT + '평당금액_행정동표.csv').set_index('행정동_코드')                                          # 동네 특징(직장인구·CCTV·1인가구·유동인구)과 동네 이름
    d = g[g['거래 수'] >= MIN_N].join(info[['자치구', '행정동', '직장인구 수', 'CCTV 수', '1인가구 비율(%)', '유동인구 수']])
    n0 = len(d); d = d.dropna(subset=X_COLS)                                                                            # 직장인구·1인가구가 비어 있는 동네는 제외 (릿지·랜덤포레스트가 빈 값을 못 다룸)
    print(f'아파트 매매 {len(a):,}건 → 행정동 {len(g)}곳 → 거래 {MIN_N}건 이상 {n0}곳 → 빈 값 제외 {len(d)}곳 | 평당 금액 중앙값의 중앙값 {d[Y].median():,.0f}만원')
    return d
