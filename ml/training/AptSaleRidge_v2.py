# 모델 버전 2: 아파트 매매 평당 금액을 릿지 회귀(직선 공식)로 맞히기. v1(랜덤포레스트)과 같은 데이터·같은 분할로 평가해서 공정하게 비교함
#   X: 층수, 거래일, 완공연도, 직장인구 수, CCTV 수, 1인가구 비율, 유동인구 수   y: 평당 금액 (행정동 1곳 = 1행)
import os, platform, warnings
import joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib as mpl
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from apt_sale_data import OUT, X_COLS, Y, load
warnings.simplefilter('ignore')
mpl.rc('font', family={'Windows': 'Malgun Gothic', 'Darwin': 'AppleGothic'}.get(platform.system(), 'NanumGothic'), size=11)
mpl.rcParams['axes.unicode_minus'] = False

REPEAT = 20         # 5분할 교차검증을 반복하는 횟수 (v1과 같아야 같은 분할로 비교됨)
d = load()
X, y = d[X_COLS], d[Y]

# 릿지 = 항목마다 가중치를 곱해서 더하는 직선 공식. 가중치가 너무 커지지 않게 제한(alpha)을 걸어서 항목끼리 겹쳐도 안정적
# 항목마다 단위가 달라서(직장인구 수만 명 vs 층수 몇 층) 먼저 같은 눈금(평균 0, 표준편차 1)으로 맞추고, 제한 세기(alpha)는 학습 동네 안에서 자동으로 고름
def ridge(): return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))
cv = RepeatedKFold(n_splits=5, n_repeats=REPEAT, random_state=0)         # v1과 같은 분할
pred, base = np.zeros((REPEAT, len(d))), np.zeros((REPEAT, len(d)))      # 반복마다 동네별 예측값 (모델 / 기준: 학습 동네 평균으로 찍기)
coef, alpha = [], []
for i, (tr, te) in enumerate(cv.split(X)):
    m = ridge().fit(X.iloc[tr], y.iloc[tr])
    pred[i // 5, te], base[i // 5, te] = m.predict(X.iloc[te]), y.iloc[tr].mean()
    coef.append(m[-1].coef_); alpha.append(m[-1].alpha_)
r2 = lambda p: 1 - ((y.values - p) ** 2).sum(axis=1) / ((y.values - y.mean()) ** 2).sum()
mae = lambda p: np.abs(y.values - p).mean(axis=1)
res = pd.DataFrame({'모델': ['릿지 v2', '기준(평균으로 찍기)'], 'R² 평균': [r2(pred).mean(), r2(base).mean()], 'R² 표준편차': [r2(pred).std(), r2(base).std()],
                    '평균 오차(만원/평)': [mae(pred).mean(), mae(base).mean()]})
print(f'\n[학습에 쓰지 않은 동네에서 평가, 5분할 × {REPEAT}번 반복]'); print(res.round(3).to_string(index=False))
coef = pd.DataFrame(coef, columns=X_COLS)
cf = pd.DataFrame({'계수(만원/평, 1표준편차당)': coef.mean(), '분할마다 흔들림(표준편차)': coef.std()}).sort_values('계수(만원/평, 1표준편차당)', key=abs, ascending=False)
print(f'\n[항목별 계수: 그 항목이 평균보다 표준편차 1만큼 크면 평당 금액이 몇 만원 변하나 | 제한 세기 alpha 중앙값 {np.median(alpha):.1f}]'); print(cf.round(1).to_string())

# 저장: 결과 표, 동네별 예측, v1과의 비교 표, 그림, 전체 데이터로 학습한 모델
os.makedirs(OUT, exist_ok=True)
res.round(3).to_csv(OUT + '아파트매매_Ridge_v2_성능.csv', index=False, encoding='utf-8-sig')
d.assign(예측=pred.mean(axis=0), 오차=lambda t: t['예측'] - t[Y]).reset_index().round(1).to_csv(OUT + '아파트매매_Ridge_v2_동네별예측.csv', index=False, encoding='utf-8-sig')
cf.round(2).to_csv(OUT + '아파트매매_Ridge_v2_계수.csv', encoding='utf-8-sig')
v1 = OUT + '아파트매매_RF_v1_성능.csv'
if os.path.exists(v1):          # 지금까지 만든 모델들을 한 표에 모음 (마지막에 하나를 고를 때 사용)
    cmp = pd.concat([pd.read_csv(v1).iloc[[0]], res.iloc[[0]], res.iloc[[1]]]).round(3)
    cmp.to_csv(OUT + '아파트매매_모델비교.csv', index=False, encoding='utf-8-sig')
    print('\n[모델 비교]'); print(cmp.to_string(index=False))
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(y, pred.mean(axis=0), s=12, alpha=.6); lim = [y.min(), y.max()]; ax[0].plot(lim, lim, 'k--', lw=1)       # 대각선 = 정확히 맞힌 경우
ax[0].set(xlabel='실제 평당 금액 (만원)', ylabel='예측 평당 금액 (만원)', title=f'동네별 실제 vs 예측 (R² {r2(pred).mean():.2f})')
cf['계수(만원/평, 1표준편차당)'][::-1].plot.barh(ax=ax[1], xerr=cf['분할마다 흔들림(표준편차)'][::-1], color=['tomato' if v > 0 else 'steelblue' for v in cf['계수(만원/평, 1표준편차당)'][::-1]])
ax[1].set(xlabel='계수 (만원/평) · 막대 끝 선은 분할마다 흔들림', title='항목별 계수 (빨강: 클수록 비쌈, 파랑: 클수록 저렴)')
fig.savefig(OUT + '아파트매매_Ridge_v2.png', dpi=100, bbox_inches='tight')
joblib.dump(ridge().fit(X, y), OUT + '아파트매매_Ridge_v2.joblib')
print('\n저장:', os.path.abspath(OUT))
