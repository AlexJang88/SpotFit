# 아파트 매매 모델(릿지, 랜덤포레스트, XGBoost)의 항목 중요도를 같은 방식으로 계산해서 한 표로 비교
# 순열 중요도 = 학습에 쓰지 않은 동네에서 항목 하나의 값을 뒤섞었을 때 R²가 얼마나 떨어지나 (클수록 그 항목이 중요)
# 모델·분할·반복은 compare_apt_sale_models.py 와 같음 (v1~v3 스크립트의 중요도와 같은 값이 나와야 함)
import os, platform, warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib as mpl
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedKFold
from apt_sale_data import OUT, X_COLS, Y, load
from compare_apt_sale_models import ALGORITHMS, LABEL, build
warnings.simplefilter('ignore')
mpl.rc('font', family={'Windows': 'Malgun Gothic', 'Darwin': 'AppleGothic'}.get(platform.system(), 'NanumGothic'), size=11)
mpl.rcParams['axes.unicode_minus'] = False

FOLDS = 25          # 앞 25개 분할(5번 반복)에서 계산 (시간 절약, v1·v3 스크립트와 같음)
d = load()
X, y = d[X_COLS], d[Y]
cv = list(RepeatedKFold(n_splits=5, n_repeats=20, random_state=0).split(X))[:FOLDS]
imp = {}
for alg in ALGORITHMS:
    sc = np.zeros((FOLDS, len(X_COLS)))
    for i, (tr, te) in enumerate(cv):
        m = build(alg).fit(X.iloc[tr], y.iloc[tr])
        sc[i] = permutation_importance(m, X.iloc[te], y.iloc[te], n_repeats=5, random_state=0).importances_mean
    imp[LABEL[alg]] = pd.Series(sc.mean(axis=0), index=X_COLS)
    print(alg, '완료', flush=True)
raw = pd.DataFrame(imp)                                        # 모델마다 R² 감소량 (절댓값)
share = raw.clip(lower=0).div(raw.clip(lower=0).sum()).mul(100)      # 모델 안에서 전체 중요도 중 차지하는 비율(%): 모델끼리 크기를 비교하기 쉽게
rank = raw.rank(ascending=False).astype(int)
out = pd.concat({'중요도(R² 감소량)': raw, '비율(%)': share, '순위': rank}, axis=1).loc[raw.mean(axis=1).sort_values(ascending=False).index]
os.makedirs(OUT, exist_ok=True)
out.round(3).to_csv(OUT + '아파트매매_항목중요도_비교.csv', encoding='utf-8-sig')
print('\n[항목 중요도: 섞었을 때 R²가 떨어지는 정도]'); print(raw.loc[out.index].round(3).to_string())
print('\n[모델 안에서 차지하는 비율(%)]'); print(share.loc[out.index].round(0).to_string())
print('\n[순위]'); print(rank.loc[out.index].to_string())
fig, ax = plt.subplots(figsize=(9, 5))
share.loc[out.index][::-1].plot.barh(ax=ax, width=.8); ax.set(xlabel='모델 안에서 차지하는 중요도 비율 (%)', title='항목별 중요도 비교 (섞었을 때 R² 감소량 기준)')
fig.savefig(OUT + '아파트매매_항목중요도_비교.png', dpi=100, bbox_inches='tight')
print('\n저장:', os.path.abspath(OUT))
