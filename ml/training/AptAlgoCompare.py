# 매매 V64 · 전세 V33 알고리즘 비교 (항목·정답·시간 순 시험은 최종 모델 스크립트의 Lab을 그대로 쓰고 알고리즘만 바꿈)
#   데이터: data/processed/부동산/매매_V64.csv, 전세_V33.csv (최종 모델이 읽는 입력 파일). 모델별 하이퍼파라미터 튜닝 없이 고정값
#   결과: artifacts/algo_compare/<sale|jeonse>.csv (기간별 오차율·10% 이내·평균 오차·R²·확인용 구간·시간)
# 실행: python AptAlgoCompare.py sale   /   python AptAlgoCompare.py jeonse
# 매매 V64 / 전세 V33: 항목·정답(현재 대비 로그 비율)·시간 순 시험은 최종 모델 스크립트(Lab)를 그대로 쓰고 알고리즘만 교체. 모델별로 기법 목록이 다름.
import importlib.util, os, sys, time
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

TRAIN = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(TRAIN, '..', '..', 'artifacts', 'algo_compare') + os.sep
os.makedirs(OUT, exist_ok=True)
NJ = 8
imp = lambda *steps: make_pipeline(SimpleImputer(strategy='median'), *steps)
XGB = lambda depth, nj: XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=depth, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0, n_jobs=nj, random_state=0)
JOBS = {
    'sale': ('AptSaleForecastV64Input.py', '매매 V64', {
        'XGBoost (최종, 깊이3)': lambda: XGB(3, 1), 'XGBoost 깊이 6': lambda: XGB(6, NJ),
        'HistGradientBoosting': lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_depth=3, random_state=0),
        'RandomForest': lambda: imp(RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5, n_jobs=NJ, random_state=0)),
        'ExtraTrees': lambda: imp(ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5, n_jobs=NJ, random_state=0)),
        'Ridge (선형)': lambda: imp(StandardScaler(), Ridge(alpha=1.0))}),
    'jeonse': ('AptJeonseForecastV33Input.py', '전세 V33', {
        'XGBoost (최종, 깊이3)': lambda: XGB(3, 1),
        'HistGradientBoosting': lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_depth=3, random_state=0),
        'GradientBoosting': lambda: imp(GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8, random_state=0)),
        'RandomForest': lambda: imp(RandomForestRegressor(n_estimators=300, min_samples_leaf=10, max_features=0.6, n_jobs=NJ, random_state=0)),
        'k-최근접 이웃 (k=50)': lambda: imp(StandardScaler(), KNeighborsRegressor(n_neighbors=50, n_jobs=NJ)),
        'Huber 선형 (이상치에 강함)': lambda: imp(StandardScaler(), HuberRegressor(max_iter=500)),
        'Ridge (선형)': lambda: imp(StandardScaler(), Ridge(alpha=1.0))}),
}

def load(file):
    spec = importlib.util.spec_from_file_location(file[:-3], os.path.join(TRAIN, file)); m = importlib.util.module_from_spec(spec)
    sys.modules[file[:-3]] = m; spec.loader.exec_module(m); return m

job = sys.argv[1]; file, label, algos = JOBS[job]; m = load(file); lab = m.Lab(); rows = []
col = ['오차율 중앙(%)', '10% 이내(%)', '평균 오차(만원/평)', '결정계수 R²']
for name, make in algos.items():
    m.xgb = make
    for h in m.STARTS:
        t0 = time.time(); res, _ = lab.tests(m.FEAT, h); res = res[res['방법'] == '예측 모델']; hold = res[res['시험연도'] > m.DEV_LAST[h]]
        row = {'모델': label, '알고리즘': name, '기간': m.NAMES[h], '시험 횟수': len(res), **{c: res[c].mean() for c in col},
               '확인용 오차율(%)': hold['오차율 중앙(%)'].mean(), '확인용 R²': hold['결정계수 R²'].mean(), '시간(초)': time.time() - t0}
        rows.append(row); print(f'{label} | {name} | {m.NAMES[h]} | 오차율 {row["오차율 중앙(%)"]:.2f}% 10%이내 {row["10% 이내(%)"]:.1f}% MAE {row["평균 오차(만원/평)"]:.1f} R² {row["결정계수 R²"]:.3f} | 확인용 {row["확인용 오차율(%)"]:.2f}%/{row["확인용 R²"]:.3f} | {row["시간(초)"]:.0f}초', flush=True)
        pd.DataFrame(rows).to_csv(OUT + f'{job}.csv', index=False, encoding='utf-8-sig')
