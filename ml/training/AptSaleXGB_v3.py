# 모델 버전 3: 아파트 매매 평당 금액을 XGBoost(부스팅)로 맞히기. v1(랜덤포레스트), v2(릿지)와 같은 데이터·같은 분할로 평가해서 공정하게 비교함
#   X: 층수, 거래일, 완공연도, 직장인구 수, CCTV 수, 1인가구 비율, 유동인구 수   y: 평당 금액 (행정동 1곳 = 1행)
import os, platform, warnings
import joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib as mpl
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor
from apt_sale_data import OUT, X_COLS, Y, load
warnings.simplefilter('ignore')
mpl.rc('font', family={'Windows': 'Malgun Gothic', 'Darwin': 'AppleGothic'}.get(platform.system(), 'NanumGothic'), size=11)
mpl.rcParams['axes.unicode_minus'] = False

REPEAT = 20         # 5분할 교차검증을 반복하는 횟수 (v1, v2와 같아야 같은 분할로 비교됨)
NAME = 'XGBoost v3'
d = load()
X, y = d[X_COLS], d[Y]

# XGBoost = 트리를 하나씩 만들면서, 앞 트리가 틀린 부분을 다음 트리가 고쳐 가는 부스팅 모델
# 동네가 408곳뿐이라 기본 설정(트리 깊이 6, 학습 속도 0.3)은 외워서 과적합하기 쉬움 → 작게 설정: 얕은 트리(깊이 3), 천천히 학습(0.03), 트리마다 샘플·항목 일부만 사용(80%)
# (기본 설정 R² 0.370 → 이 설정 0.457로 미리 확인함)
def xgb(): return XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0, n_jobs=1, random_state=0)
cv = RepeatedKFold(n_splits=5, n_repeats=REPEAT, random_state=0)         # v1, v2와 같은 분할
pred, base = np.zeros((REPEAT, len(d))), np.zeros((REPEAT, len(d)))      # 반복마다 동네별 예측값 (모델 / 기준: 학습 동네 평균으로 찍기)
imp, k = np.zeros(len(X_COLS)), 0
for i, (tr, te) in enumerate(cv.split(X)):
    m = xgb().fit(X.iloc[tr], y.iloc[tr])
    pred[i // 5, te], base[i // 5, te] = m.predict(X.iloc[te]), y.iloc[tr].mean()
    if i < 25:      # 중요도는 앞 5번 반복(25개 분할)만 계산 (시간 절약): 평가 동네의 항목을 섞으면 설명력이 얼마나 떨어지나
        imp += permutation_importance(m, X.iloc[te], y.iloc[te], n_repeats=5, random_state=0).importances_mean; k += 1
r2 = lambda p: 1 - ((y.values - p) ** 2).sum(axis=1) / ((y.values - y.mean()) ** 2).sum()
mae = lambda p: np.abs(y.values - p).mean(axis=1)
res = pd.DataFrame({'모델': [NAME, '기준(평균으로 찍기)'], 'R² 평균': [r2(pred).mean(), r2(base).mean()], 'R² 표준편차': [r2(pred).std(), r2(base).std()],
                    '평균 오차(만원/평)': [mae(pred).mean(), mae(base).mean()]})
print(f'\n[학습에 쓰지 않은 동네에서 평가, 5분할 × {REPEAT}번 반복]'); print(res.round(3).to_string(index=False))
imp = pd.Series(imp / k, index=X_COLS).sort_values(ascending=False)
print('\n[순열 중요도: 항목을 섞으면 R²가 떨어지는 정도]'); print(imp.round(3).to_string())

# 저장: 결과 표, 동네별 예측, 모델 비교 표(v1, v2, v3 ... 를 한 표에 모음), 그림, 전체 데이터로 학습한 모델
os.makedirs(OUT, exist_ok=True)
res.round(3).to_csv(OUT + '아파트매매_XGB_v3_성능.csv', index=False, encoding='utf-8-sig')
d.assign(예측=pred.mean(axis=0), 오차=lambda t: t['예측'] - t[Y]).reset_index().round(1).to_csv(OUT + '아파트매매_XGB_v3_동네별예측.csv', index=False, encoding='utf-8-sig')
cp = OUT + '아파트매매_모델비교.csv'
old = pd.read_csv(cp) if os.path.exists(cp) else pd.DataFrame(columns=res.columns)
old = old[~old['모델'].isin([NAME, '기준(평균으로 찍기)'])]                  # 같은 모델을 다시 돌리면 덮어씀, 기준 행은 맨 아래로
cmp = pd.concat([old, res.iloc[[0]], res.iloc[[1]]]).round(3)
cmp.to_csv(cp, index=False, encoding='utf-8-sig')
print('\n[모델 비교]'); print(cmp.to_string(index=False))
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(y, pred.mean(axis=0), s=12, alpha=.6); lim = [y.min(), y.max()]; ax[0].plot(lim, lim, 'k--', lw=1)       # 대각선 = 정확히 맞힌 경우
ax[0].set(xlabel='실제 평당 금액 (만원)', ylabel='예측 평당 금액 (만원)', title=f'동네별 실제 vs 예측 (R² {r2(pred).mean():.2f})')
imp[::-1].plot.barh(ax=ax[1], color='steelblue'); ax[1].set(xlabel='중요도 (섞었을 때 R² 감소량)', title='항목별 중요도')
fig.savefig(OUT + '아파트매매_XGB_v3.png', dpi=100, bbox_inches='tight')
joblib.dump(xgb().fit(X, y), OUT + '아파트매매_XGB_v3.joblib')
print('\n저장:', os.path.abspath(OUT))
