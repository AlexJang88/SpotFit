# 아파트 전세 학습 데이터 (행정동 1곳 = 1행): AptJeonseRF_v1.py 와 같은 방법으로 v2부터는 여기서 한 번만 만든다
#   y: 평당 보증금 (전세 거래의 동네 중앙값, 만원)   전세만 사용: 월세는 보증금과 월세가 함께 있어 y를 한 숫자로 정하려면 전월세 전환율 같은 가정이 필요해 제외함
#   항목 출처 (output 표를 쓰지 않고 원자료에서 직접 만듦):
#     층수·거래일·완공연도 = data/processed/부동산(전처리)/아파트_전월세.csv 전세 거래의 동네 평균
#     1인가구 비율 = data/interim/1인가구/1인가구_전체.csv (1인가구수 ÷ 총인구, 2022~2025 합계. 동네는 이름으로 연결, 신사동은 강남·관악 따로)
#     직장인구·유동인구 = 서울시 상권분석서비스(직장인구-행정동, 길단위인구-행정동) 전체 분기 평균 (아파트 매매 v1과 같음)
#     생활방범 CCTV 수 = data/interim/기타/CCTV_생활방범_연도별/CCTV_생활방범_연도.csv 의 카메라 대수 합 (설치연도를 아는 CCTV만, 연도모름 파일은 뺌. 연도가 없는 11개 구는 0이 됨)
#     동네 이름(자치구·행정동) = 모델 입력이 아니라 예측표 표시용, 전월세 거래 파일의 이름을 씀
import os
from pathlib import Path
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed') + os.sep
OUT = DATA + 'output' + os.sep
RAW, INTERIM = DATA + '../raw/', DATA + '../interim/'
MIN_N = 10          # 동네당 거래가 이만큼은 있어야 그 동네의 중앙값·평균을 믿고 사용
X_COLS = ['층수', '거래일(년)', '완공연도', '직장인구 수', '생활방범 CCTV 수', '1인가구 비율(%)', '유동인구 수']
AREA = '평균 임대면적(㎡)'          # v3부터: 전세 거래의 임대면적(㎡) 동네 평균 (전월세 파일에는 전용면적 컬럼이 없고 임대면적만 있음)
BAGE = '평균 건물 나이(년)'         # v8부터: 전세 거래의 건물 나이(년, 계약연도 − 건축년도) 동네 평균
Y = '평당 보증금(만원)'
K = '행정동_코드'

def load():
    a = pd.read_csv(DATA + '부동산(전처리)/아파트_전월세.csv', low_memory=False, usecols=['행정동_코드(상권분석)', '자치구', '행정동', '계약일', '전월세구분', Y, '층', '건축년도', '임대면적(㎡)', '건물 나이(년)'])
    a = a[a['전월세구분'] == '전세']
    names = a.groupby('행정동_코드(상권분석)')[['자치구', '행정동']].agg(lambda s: s.mode().iloc[0])              # 동네 이름: 그 동네 거래에서 가장 많이 나온 이름 (표시용)
    names.index = names.index.astype(int)
    day = pd.to_datetime(a['계약일'], errors='coerce')
    a = pd.DataFrame({K: a['행정동_코드(상권분석)'].astype(int), Y: a[Y],                                                          # 평당 보증금 = 보증금(만원) / 임대면적(평)
                      '층수': a['층'], '거래일(년)': day.dt.year + (day.dt.dayofyear - 1) / 365, '완공연도': a['건축년도'], AREA: a['임대면적(㎡)'], BAGE: a['건물 나이(년)']})
    g = a.groupby(K).agg(**{'거래 수': (Y, 'size'), Y: (Y, 'median'), '층수': ('층수', 'mean'), '거래일(년)': ('거래일(년)', 'mean'), '완공연도': ('완공연도', 'mean'), AREA: (AREA, 'mean'), BAGE: (BAGE, 'mean')})    # y는 동네 중앙값, 항목은 동네 평균
    area = lambda name: pd.read_csv(RAW + f'서울시 상권분석서비스({name}).csv', encoding='cp949')
    feat = pd.DataFrame({'유동인구 수': area('길단위인구-행정동').groupby(K)['총_유동인구_수'].mean(),                              # 분기별 값의 전체 기간 평균 (매매 v1과 같음)
                         '직장인구 수': area('직장인구-행정동').groupby(K)['총_직장_인구_수'].mean()})
    one = pd.read_csv(INTERIM + '1인가구/1인가구_전체.csv', encoding='cp949').groupby(['행정동코드', '행정동명'])[['1인가구수', '총인구']].sum().reset_index()
    one['비율'] = one['1인가구수'] / one['총인구'] * 100                                                                       # 1인가구 비율 = 1인가구수 ÷ 총인구 (2022~2025 월별 합계)
    one['이름'] = one['행정동명'].str.replace('·', '').mask(one['행정동코드'] == 1121068, '관악신사동')                          # 코드 체계가 달라 이름으로 연결 (신사동은 강남·관악 2곳이라 따로 처리)
    nm = area('상주인구-행정동').drop_duplicates(K)[[K, '행정동_코드_명']]
    nm['이름'] = nm['행정동_코드_명'].str.replace('?', '').mask(nm[K] == 11620685, '관악신사동')
    feat['1인가구 비율(%)'] = one.merge(nm, on='이름').set_index(K)['비율']
    cc = pd.concat([pd.read_csv(f, encoding='utf-8-sig', usecols=[K, '카메라대수']) for f in sorted(Path(INTERIM + '기타/CCTV_생활방범_연도별').glob('CCTV_생활방범_[0-9]*.csv'))])   # 연도모름 파일은 제외
    cc = cc.dropna(subset=[K]).astype({K: int})
    feat['생활방범 CCTV 수'] = cc.groupby(K)['카메라대수'].sum().reindex(feat.index).fillna(0)                                  # 연도를 아는 CCTV가 없으면 0
    d = g[g['거래 수'] >= MIN_N].join(names).join(feat)
    n0 = len(d); d = d.dropna(subset=X_COLS)
    print(f'아파트 전세 {len(a):,}건 → 행정동 {len(g)}곳 → 거래 {MIN_N}건 이상 {n0}곳 → 빈 값 제외 {len(d)}곳 | 평당 보증금 중앙값의 중앙값 {d[Y].median():,.0f}만원')
    return d
