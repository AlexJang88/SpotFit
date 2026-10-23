# 모델 버전 1: 아파트 매매 평당 금액을 랜덤포레스트로 맞히기 (행정동 1곳 = 1행)
#   X: 층수, 거래일, 완공연도, 직장인구 수, CCTV 수, 1인가구 비율, 유동인구 수   y: 평당 금액 (동네 중앙값, 만원)
# 실행 전에 ml/etl 의 MergeSingleHousehold.py → MergeRent.py → ConvertSale.py → AptHypothesis.py 가 끝나 있어야 함 (평당금액_행정동표.csv 에서 동네 특징을 가져옴)
import os, platform, warnings
import joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib as mpl
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedKFold
warnings.simplefilter('ignore')
mpl.rc('font', family={'Windows': 'Malgun Gothic', 'Darwin': 'AppleGothic'}.get(platform.system(), 'NanumGothic'), size=11)
mpl.rcParams['axes.unicode_minus'] = False

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed') + os.sep
OUT = DATA + 'output' + os.sep
OLD = {11230515: 11230536, 11230533: 11230536, 11740525: 11740520, 11740526: 11740520, 11680675: 11680740}    # 새 행정동 코드 → 상권분석의 옛 코드 (AptHypothesis.py 와 같음)
MIN_N = 10          # 동네당 거래가 이만큼은 있어야 그 동네의 중앙값·평균을 믿고 사용
REPEAT = 20         # 5분할 교차검증을 반복하는 횟수 (클수록 결과가 안정적, 모델 학습은 5 × REPEAT번)
X_COLS = ['층수', '거래일(년)', '완공연도', '직장인구 수', 'CCTV 수', '1인가구 비율(%)', '유동인구 수']
Y = '평당 금액(만원)'

# 1) 아파트 매매 거래 → 행정동별 한 표 (AptHypothesis.py 의 deals() 와 같은 기준: 취소 거래·20㎡ 미만·지번 불일치 제외)
cols = ['행정동_코드', '연결방식', 'cdealType', 'excluUseAr', 'dealAmount', 'dealYear', 'dealMonth', 'dealDay', 'buildYear', 'floor']
a = pd.read_csv(DATA + '부동산(도로명주소)/apt_sale_juso.csv', low_memory=False, usecols=cols)
a = a[a['cdealType'].isna() & (a['excluUseAr'] >= 20) & (a['연결방식'] == '지번 일치')]
day = pd.to_datetime(a[['dealYear', 'dealMonth', 'dealDay']].set_axis(['year', 'month', 'day'], axis=1), errors='coerce')
a = pd.DataFrame({'행정동_코드': a['행정동_코드'].astype(int).replace(OLD), Y: pd.to_numeric(a['dealAmount'].str.replace(',', ''), errors='coerce') / (a['excluUseAr'] / 3.3058),   # 거래금액(만원) / 전용면적(평)
                  '층수': a['floor'], '거래일(년)': day.dt.year + (day.dt.dayofyear - 1) / 365, '완공연도': a['buildYear']})
g = a.groupby('행정동_코드').agg(**{'거래 수': (Y, 'size'), Y: (Y, 'median'), '층수': ('층수', 'mean'), '거래일(년)': ('거래일(년)', 'mean'), '완공연도': ('완공연도', 'mean')})    # y는 동네 중앙값, 항목은 동네 평균
info = pd.read_csv(OUT + '평당금액_행정동표.csv').set_index('행정동_코드')                                              # 동네 특징(직장인구·CCTV·1인가구·유동인구)과 동네 이름
d = g[g['거래 수'] >= MIN_N].join(info[['자치구', '행정동', '직장인구 수', 'CCTV 수', '1인가구 비율(%)', '유동인구 수']])
n0 = len(d); d = d.dropna(subset=X_COLS)                                                                                # 직장인구·1인가구가 비어 있는 동네는 제외 (랜덤포레스트가 빈 값을 못 다룸)
X, y = d[X_COLS], d[Y]
print(f'아파트 매매 {len(a):,}건 → 행정동 {len(g)}곳 → 거래 {MIN_N}건 이상 {n0}곳 → 빈 값 제외 {len(d)}곳 | 평당 금액 중앙값의 중앙값 {y.median():,.0f}만원')

# 2) 학습·평가: 동네 한 곳이 한 행이라 5분할하면 "학습에 쓰지 않은 동네"로 평가됨. 분할을 REPEAT번 바꿔서 반복
def rf(): return RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=0, n_jobs=-1)    # 동네가 적어서 잎 하나에 최소 5곳
cv = RepeatedKFold(n_splits=5, n_repeats=REPEAT, random_state=0)
pred, base = np.zeros((REPEAT, len(d))), np.zeros((REPEAT, len(d)))      # 반복마다 동네별 예측값 (모델 / 기준: 학습 동네 평균으로 찍기)
imp, k = np.zeros(len(X_COLS)), 0
for i, (tr, te) in enumerate(cv.split(X)):
    m = rf().fit(X.iloc[tr], y.iloc[tr])
    pred[i // 5, te], base[i // 5, te] = m.predict(X.iloc[te]), y.iloc[tr].mean()
    if i < 25:      # 중요도는 앞 5번 반복(25개 분할)만 계산 (시간 절약): 평가 동네의 항목을 섞으면 설명력이 얼마나 떨어지나
        imp += permutation_importance(m, X.iloc[te], y.iloc[te], n_repeats=5, random_state=0).importances_mean; k += 1
r2 = lambda p: 1 - ((y.values - p) ** 2).sum(axis=1) / ((y.values - y.mean()) ** 2).sum()
mae = lambda p: np.abs(y.values - p).mean(axis=1)
res = pd.DataFrame({'모델': ['랜덤포레스트 v1', '기준(평균으로 찍기)'], 'R² 평균': [r2(pred).mean(), r2(base).mean()], 'R² 표준편차': [r2(pred).std(), r2(base).std()],
                    '평균 오차(만원/평)': [mae(pred).mean(), mae(base).mean()]})
print(f'\n[학습에 쓰지 않은 동네에서 평가, 5분할 × {REPEAT}번 반복]'); print(res.round(3).to_string(index=False))
imp = pd.Series(imp / k, index=X_COLS).sort_values(ascending=False)
print('\n[순열 중요도: 항목을 섞으면 R²가 떨어지는 정도]'); print(imp.round(3).to_string())

# 3) 저장: 결과 표, 동네별 예측, 그림, 전체 데이터로 학습한 모델
os.makedirs(OUT, exist_ok=True)
res.round(3).to_csv(OUT + '아파트매매_RF_v1_성능.csv', index=False, encoding='utf-8-sig')
d.assign(예측=pred.mean(axis=0), 오차=lambda t: t['예측'] - t[Y]).reset_index().round(1).to_csv(OUT + '아파트매매_RF_v1_동네별예측.csv', index=False, encoding='utf-8-sig')
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(y, pred.mean(axis=0), s=12, alpha=.6); lim = [y.min(), y.max()]; ax[0].plot(lim, lim, 'k--', lw=1)       # 대각선 = 정확히 맞힌 경우
ax[0].set(xlabel='실제 평당 금액 (만원)', ylabel='예측 평당 금액 (만원)', title=f'동네별 실제 vs 예측 (R² {r2(pred).mean():.2f})')
imp[::-1].plot.barh(ax=ax[1], color='steelblue'); ax[1].set(xlabel='중요도 (섞었을 때 R² 감소량)', title='항목별 중요도')
fig.savefig(OUT + '아파트매매_RF_v1.png', dpi=100, bbox_inches='tight')
joblib.dump(rf().fit(X, y), OUT + '아파트매매_RF_v1.joblib')
print('\n저장:', os.path.abspath(OUT))
