# 월세 전환율 모델 알고리즘 비교: 데이터(data/processed/부동산/월세_전환율.csv)·항목 28개·정답(전환율 − 동 앵커)은 RentConversionFinalInput.py 앞부분을 그대로 실행해 사용
#   학습 ≤2024 → 시험 2025-01~2026-09. 결과: artifacts/algo_compare/wolse.csv
# 실행: python WolseAlgoCompare.py
# 월세 전환율 모델: 최종 스크립트(RentConversionFinalInput.py)의 데이터·항목 28개·정답(전환율 − 동 앵커) 준비 부분을 그대로 쓰고 알고리즘만 교체.
# 학습 ≤2024 → 시험 2025-01~2026-09
import os, time
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

TRAIN = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(TRAIN, '..', '..', 'artifacts', 'algo_compare') + os.sep
os.makedirs(OUT, exist_ok=True)
os.chdir(TRAIN)
path = os.path.join(TRAIN, 'RentConversionFinalInput.py')
src = open(path, encoding='utf-8').read(); src = src[:src.index('trT = d[')]
g = {'__name__': 'wolse_data', '__file__': path}
exec(compile(src, 'RentConversionFinalInput(앞부분)', 'exec'), g)
d, XA, LO, HI, TREES, CFG = g['d'], g['XA'], g['LO'], g['HI'], g['TREES'], g['CFG']
tr = d[d['ok'] & d['an_s'].notna() & (d['m'] < 2025 * 12)]; te = d[d['ok'] & d['an_s'].notna() & (d['m'] >= 2025 * 12)].copy(); ytr = tr['rc'] - tr['an_d']
print(f'학습 {len(tr):,}건, 시험 {len(te):,}건, 항목 {len(XA)}개, 트리 {TREES}', flush=True)
NJ = 8; imp = lambda *s: make_pipeline(SimpleImputer(strategy='median'), *s)
ALGOS = {
    'HistGradientBoosting (최종)': lambda: HistGradientBoostingRegressor(**CFG),
    'XGBoost': lambda: XGBRegressor(n_estimators=TREES, learning_rate=0.03, tree_method='hist', grow_policy='lossguide', max_leaves=255, max_depth=0, min_child_weight=50, n_jobs=NJ, random_state=0),
    'RandomForest': lambda: imp(RandomForestRegressor(n_estimators=100, min_samples_leaf=50, max_features=0.5, n_jobs=NJ, random_state=0)),
    'ExtraTrees': lambda: imp(ExtraTreesRegressor(n_estimators=100, min_samples_leaf=50, max_features=0.5, n_jobs=NJ, random_state=0)),
    '신경망 MLP (64-32)': lambda: imp(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=60, random_state=0)),
    'Ridge (선형)': lambda: imp(StandardScaler(), Ridge(alpha=1.0)),
    '동 앵커만 (학습 없음)': None,
}
r2 = lambda y, p: float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
rows = []
for name, make in ALGOS.items():
    t0 = time.time()
    if make is None: resid, secs = np.zeros(len(te)), 0.0
    else:
        model = make().fit(tr[XA], ytr); secs = time.time() - t0; resid = np.asarray(model.predict(te[XA]))
    rhat = np.clip(te['an_d'].values + resid, LO, HI); pR = rhat * te['gapJ'].values / 12; pD = te['J'].values - 12 * te['R'].values / rhat
    R, D = te['R'].values, te['D'].values; eR, eD = np.abs(pR - R), np.abs(pD - D)
    row = {'알고리즘': name, '월세 MAE(만원)': eR.mean(), '월세 오차율 중앙(%)': np.median(eR / R * 100), '월세 R²': r2(R, pR),
           '보증금 MAE(만원)': eD.mean(), '보증금 오차율 중앙(%)': np.median(eD / D * 100), '보증금 R²': r2(D, pD),
           '전환율 오차 중앙(%p)': np.median(np.abs(rhat - te['rc'].values) * 100), '학습 시간(초)': secs}
    rows.append(row); print(' | '.join(f'{k} {v:.3f}' if isinstance(v, float) else f'{k} {v}' for k, v in row.items()), flush=True)
    pd.DataFrame(rows).to_csv(OUT + 'wolse.csv', index=False, encoding='utf-8-sig')
