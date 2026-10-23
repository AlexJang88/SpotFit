# CCTV 수 하나로 행정동별 평당 금액을 맞히는 모델을 여러 개 비교 (시행 횟수를 크게: 5분할 교차검증을 100번 반복 = 500번 학습)
# 먼저 AptHypothesis.py 를 실행해서 평당금액_행정동표.csv 를 만들어야 함
import os, platform
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib as mpl
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures
mpl.rc('font', family={'Windows': 'Malgun Gothic', 'Darwin': 'AppleGothic'}.get(platform.system(), 'NanumGothic'), size=11)
mpl.rcParams['axes.unicode_minus'] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed', 'output') + os.sep
REPEAT = 100                                      # 교차검증 반복 횟수 (클수록 결과가 안정적)
log = FunctionTransformer(np.log1p)               # CCTV 수는 몇 개인 동네부터 수백 개인 동네까지 있어서 로그로 눌러서도 시도

def main():
    d = pd.read_csv(OUT + '평당금액_행정동표.csv').dropna(subset=['평당 금액 중앙값(만원)', 'CCTV 수'])
    X, y = d[['CCTV 수']], d['평당 금액 중앙값(만원)']          # X: 행정동별 CCTV 수, y: 행정동별 평당 금액(중앙값, 만원)
    rho, p = stats.spearmanr(X['CCTV 수'], y)
    print(f'행정동 {len(d)}곳 | 스피어만 연관도 {rho:+.2f} (p={p:.3g})')

    models = {'평균으로 찍기(기준)': DummyRegressor(),
              '선형회귀': LinearRegression(),
              '로그 변환 + 선형회귀': make_pipeline(log, LinearRegression()),
              '2차 곡선': make_pipeline(PolynomialFeatures(2), LinearRegression()),
              '3차 곡선': make_pipeline(PolynomialFeatures(3), LinearRegression()),
              '이웃 15개 평균(KNN)': KNeighborsRegressor(15),
              '랜덤포레스트(트리 500개)': RandomForestRegressor(500, min_samples_leaf=10, random_state=0),
              '그래디언트 부스팅(트리 300개)': GradientBoostingRegressor(n_estimators=300, learning_rate=0.02, max_depth=2, subsample=0.7, random_state=0)}
    cv = RepeatedKFold(n_splits=5, n_repeats=REPEAT, random_state=0)        # 같은 분할로 모든 모델을 비교
    rows, base = [], None
    for name, m in models.items():
        r = cross_validate(m, X, y, cv=cv, scoring={'r2': 'r2', 'mae': 'neg_mean_absolute_error'}, n_jobs=-1)
        if base is None: base = r['test_r2']
        rows.append([name, r['test_r2'].mean(), r['test_r2'].std(), (r['test_r2'] > base).mean() * 100, -r['test_mae'].mean()])
        print(f'{name:22s} R² {rows[-1][1]:+.3f} (±{rows[-1][2]:.3f})  기준보다 나은 비율 {rows[-1][3]:.0f}%  평균 오차 {rows[-1][4]:,.0f}만원/평', flush=True)
    res = pd.DataFrame(rows, columns=['모델', 'R² 평균', 'R² 표준편차', '기준보다 나은 비율(%)', '평균 오차(만원/평)']).sort_values('R² 평균', ascending=False)
    res.round(3).to_csv(OUT + 'CCTV_모델비교.csv', index=False, encoding='utf-8-sig')

    fig, ax = plt.subplots(figsize=(7.5, 5.5))                                # 점 = 행정동, 선 = 전체 데이터로 학습한 모델
    ax.scatter(X['CCTV 수'], y, s=12, alpha=.5, color='gray')
    g = pd.DataFrame({'CCTV 수': np.linspace(X['CCTV 수'].min(), X['CCTV 수'].max(), 300)})
    for name in ['선형회귀', '로그 변환 + 선형회귀', '랜덤포레스트(트리 500개)']:
        ax.plot(g['CCTV 수'], models[name].fit(X, y).predict(g), label=name)
    ax.set(xlabel='행정동별 CCTV 수', ylabel='행정동별 평당 금액 중앙값 (만원)', title=f'CCTV 수와 평당 금액 (스피어만 {rho:+.2f})'); ax.legend()
    fig.savefig(OUT + 'CCTV_모델.png', dpi=100, bbox_inches='tight')
    print('저장:', OUT + 'CCTV_모델비교.csv', '/', OUT + 'CCTV_모델.png')

if __name__ == '__main__':          # 병렬 실행(n_jobs=-1)에 필요
    main()
