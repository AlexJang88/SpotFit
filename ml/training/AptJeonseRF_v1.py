# 모델 버전 1: 아파트 전세 평당 보증금을 랜덤포레스트로 맞히기 (행정동 1곳 = 1행) — 아파트 매매 v1(AptSaleRF_v1.py)과 같은 구조
#   X: 층수, 거래일, 완공연도, 직장인구 수, 생활방범 CCTV 수, 1인가구 비율, 유동인구 수   y: 평당 보증금 (전세 거래의 동네 중앙값, 만원)
#   전세만 사용: 월세는 보증금과 월세가 함께 있어 y를 한 숫자로 정하려면 전월세 전환율 같은 가정이 필요해 제외함
#   항목 출처 (output 표를 쓰지 않고 원자료에서 직접 만듦):
#     층수·거래일·완공연도 = data/processed/부동산(전처리)/아파트_전월세.csv 전세 거래의 동네 평균
#     1인가구 비율 = data/interim/1인가구/1인가구_전체.csv (1인가구수 ÷ 총인구, 2022~2025 합계. 동네는 이름으로 연결, 신사동은 강남·관악 따로)
#     직장인구·유동인구 = 서울시 상권분석서비스(직장인구-행정동, 길단위인구-행정동) 전체 분기 평균 (아파트 매매 v1과 같음)
#     생활방범 CCTV 수 = data/interim/기타/CCTV_생활방범_연도별/CCTV_생활방범_연도.csv 의 카메라 대수 합 (설치연도를 아는 CCTV만, 연도모름 파일은 뺌. 연도가 없는 11개 구는 0이 됨)
#     동네 이름(자치구·행정동) = 모델 입력이 아니라 예측표에 어느 동네인지 보이려는 표시용, 전월세 거래 파일의 이름을 씀
# 실행 전에 ml/etl 의 MergeSingleHousehold.py, MapCctv.py, MergeRent.py → ConvertSale.py → SplitRent.py → CleanHousing.py 가 끝나 있어야 함
# 실행: python AptJeonseRF_v1.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python AptJeonseRF_v1.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록, Experiment: house-price-jeonse-comparison)
import argparse, hashlib, json, os, platform, subprocess, warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
import joblib, mlflow, mlflow.sklearn, numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib as mpl
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedKFold
warnings.simplefilter('ignore')
mpl.rc('font', family={'Windows': 'Malgun Gothic', 'Darwin': 'AppleGothic'}.get(platform.system(), 'NanumGothic'), size=11)
mpl.rcParams['axes.unicode_minus'] = False

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'processed') + os.sep
OUT = DATA + 'output' + os.sep
MIN_N = 10          # 동네당 거래가 이만큼은 있어야 그 동네의 중앙값·평균을 믿고 사용
REPEAT = 20         # 5분할 교차검증을 반복하는 횟수 (클수록 결과가 안정적, 모델 학습은 5 × REPEAT번)
X_COLS = ['층수', '거래일(년)', '완공연도', '직장인구 수', '생활방범 CCTV 수', '1인가구 비율(%)', '유동인구 수']
Y = '평당 보증금(만원)'
RAW, INTERIM = DATA + '../raw/', DATA + '../interim/'

# 1) 아파트 전세 거래 → 행정동별 한 표 (전처리된 아파트_전월세.csv 중 전월세구분이 '전세'인 거래)
a = pd.read_csv(DATA + '부동산(전처리)/아파트_전월세.csv', low_memory=False, usecols=['행정동_코드(상권분석)', '자치구', '행정동', '계약일', '전월세구분', Y, '층', '건축년도'])
a = a[a['전월세구분'] == '전세']
names = a.groupby('행정동_코드(상권분석)')[['자치구', '행정동']].agg(lambda s: s.mode().iloc[0])                  # 동네 이름: 그 동네 거래에서 가장 많이 나온 이름 (표시용)
names.index = names.index.astype(int)
day = pd.to_datetime(a['계약일'], errors='coerce')
a = pd.DataFrame({'행정동_코드': a['행정동_코드(상권분석)'].astype(int), Y: a[Y],                                                                   # 평당 보증금 = 보증금(만원) / 임대면적(평)
                  '층수': a['층'], '거래일(년)': day.dt.year + (day.dt.dayofyear - 1) / 365, '완공연도': a['건축년도']})
g = a.groupby('행정동_코드').agg(**{'거래 수': (Y, 'size'), Y: (Y, 'median'), '층수': ('층수', 'mean'), '거래일(년)': ('거래일(년)', 'mean'), '완공연도': ('완공연도', 'mean')})    # y는 동네 중앙값, 항목은 동네 평균

# 동네 특징 4개를 원자료에서 직접 만듦 (output/평당금액_행정동표.csv 를 쓰지 않음)
K = '행정동_코드'
daily = lambda name: pd.read_csv(RAW + f'서울시 상권분석서비스({name}).csv', encoding='cp949')
feat = pd.DataFrame({'유동인구 수': daily('길단위인구-행정동').groupby(K)['총_유동인구_수'].mean(),                  # 분기별 값의 전체 기간 평균 (매매 v1과 같음)
                     '직장인구 수': daily('직장인구-행정동').groupby(K)['총_직장_인구_수'].mean()})
one = pd.read_csv(INTERIM + '1인가구/1인가구_전체.csv', encoding='cp949').groupby(['행정동코드', '행정동명'])[['1인가구수', '총인구']].sum().reset_index()
one['비율'] = one['1인가구수'] / one['총인구'] * 100                                                              # 1인가구 비율 = 1인가구수 ÷ 총인구 (2022~2025 월별 합계)
one['이름'] = one['행정동명'].str.replace('·', '').mask(one['행정동코드'] == 1121068, '관악신사동')                   # 코드 체계가 달라 이름으로 연결 (신사동은 강남·관악 2곳이라 따로 처리)
nm = daily('상주인구-행정동').drop_duplicates(K)[[K, '행정동_코드_명']]
nm['이름'] = nm['행정동_코드_명'].str.replace('?', '').mask(nm[K] == 11620685, '관악신사동')
feat['1인가구 비율(%)'] = one.merge(nm, on='이름').set_index(K)['비율']
cc = pd.concat([pd.read_csv(f, encoding='utf-8-sig', usecols=['행정동_코드', '카메라대수']) for f in sorted(Path(INTERIM + '기타/CCTV_생활방범_연도별').glob('CCTV_생활방범_[0-9]*.csv'))])   # 연도모름 파일은 제외
cc = cc.dropna(subset=[K]).astype({K: int})
feat['생활방범 CCTV 수'] = cc.groupby(K)['카메라대수'].sum().reindex(feat.index).fillna(0)                          # 연도를 아는 CCTV가 없으면 0
d = g[g['거래 수'] >= MIN_N].join(names).join(feat)
n0 = len(d); d = d.dropna(subset=X_COLS)
X, y = d[X_COLS], d[Y]
print(f'아파트 전세 {len(a):,}건 → 행정동 {len(g)}곳 → 거래 {MIN_N}건 이상 {n0}곳 → 빈 값 제외 {len(d)}곳 | 평당 보증금 중앙값의 중앙값 {y.median():,.0f}만원')

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
res.round(3).to_csv(OUT + '아파트전세_RF_v1_성능.csv', index=False, encoding='utf-8-sig')
d.assign(예측=pred.mean(axis=0), 오차=lambda t: t['예측'] - t[Y]).reset_index().round(1).to_csv(OUT + '아파트전세_RF_v1_동네별예측.csv', index=False, encoding='utf-8-sig')
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(y, pred.mean(axis=0), s=12, alpha=.6); lim = [y.min(), y.max()]; ax[0].plot(lim, lim, 'k--', lw=1)       # 대각선 = 정확히 맞힌 경우
ax[0].set(xlabel='실제 평당 보증금 (만원)', ylabel='예측 평당 보증금 (만원)', title=f'동네별 실제 vs 예측 (R² {r2(pred).mean():.2f})')
imp[::-1].plot.barh(ax=ax[1], color='steelblue'); ax[1].set(xlabel='중요도 (섞었을 때 R² 감소량)', title='항목별 중요도')
fig.savefig(OUT + '아파트전세_RF_v1.png', dpi=100, bbox_inches='tight')
model = rf().fit(X, y)
joblib.dump(model, OUT + '아파트전세_RF_v1.joblib')
print('\n저장:', os.path.abspath(OUT))

# 4) MLflow 기록 (docs/model-team-workflow.md): Run 1개, 필수 태그, Registry 등록·champion·MinIO 업로드는 하지 않음
ap = argparse.ArgumentParser()
ap.add_argument('--tracking-uri'); ap.add_argument('--experiment', default='house-price-jeonse-comparison'); ap.add_argument('--task-id', default='unassigned')
a = ap.parse_args()
root = Path(__file__).resolve().parents[2]
ev = root / 'artifacts/apt_jeonse_model_comparison' / ('v1_' + datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')); ev.mkdir(parents=True)
d.reset_index().to_csv(ev / 'analysis_panel.csv', index=False, encoding='utf-8-sig')
split = pd.concat([d.reset_index()[['행정동_코드', '자치구', '행정동']].assign(repeat=r, validation_fold=-1) for r in range(REPEAT)], ignore_index=True)
for i, (_, te) in enumerate(cv.split(X)): split.loc[te + (i // 5) * len(d), 'validation_fold'] = i % 5
split.to_csv(ev / 'cv_split.csv', index=False, encoding='utf-8-sig')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
if a.tracking_uri: mlflow.set_tracking_uri(a.tracking_uri)
else: (root / 'artifacts/mlflow_apt_sale').mkdir(parents=True, exist_ok=True); mlflow.set_tracking_uri('sqlite:///' + (root / 'artifacts/mlflow_apt_sale/tracking.db').as_posix())
mlflow.set_experiment(a.experiment)
owner = subprocess.run(['git', 'config', 'user.name'], capture_output=True, text=True).stdout.strip() or 'unknown'
note = (f'아파트 전세(월세 제외) 동네 평당 보증금(중앙값) 기본 모델: X 7개(층수·거래일·완공연도·직장인구·생활방범 CCTV(연도를 아는 것만)·1인가구·유동인구, 동네 특징은 원자료에서 직접 계산), 랜덤포레스트. 월세는 보증금과 월세가 함께 있어 y를 한 숫자로 정하려면 전환율 가정이 필요해 제외. '
        f'결과: R² {r2(pred).mean():.3f} (기준 {r2(base).mean():.3f}), MAE {mae(pred).mean():.1f} (기준 {mae(base).mean():.1f}). 동네 단위 정적 집계이며 미래 예측·인과 모델이 아님.')
with mlflow.start_run(run_name='apt_jeonse_random_forest_v1_baseline'):
    mlflow.log_params({'target': '아파트 전세 동네 평당 보증금(만원, 중앙값)', 'algorithm': 'random_forest', 'version': 'v1', 'features': json.dumps(X_COLS, ensure_ascii=False), 'compared_with': 'baseline mean (평균으로 찍기)',
                       'removed_features': '[]', 'added_features': '[]', 'n_dongs': len(d), 'cv': f'RepeatedKFold(n_splits=5, n_repeats={REPEAT}, random_state=0)',
                       'estimator_params': json.dumps(rf().get_params(), default=str)})
    mlflow.log_metrics({'cv_r2_pooled': float(r2(pred).mean()), 'cv_r2_pooled_std': float(r2(pred).std()), 'cv_mae': float(mae(pred).mean()),
                        'baseline_cv_r2_pooled': float(r2(base).mean()), 'baseline_cv_mae': float(mae(base).mean())})
    mlflow.set_tags({'owner': owner, 'task_id': a.task_id, 'dataset_sha256': sha(ev / 'analysis_panel.csv'), 'split_sha256': sha(ev / 'cv_split.csv'), 'review_status': 'not_reviewed',
                     'selection_status': 'baseline_comparison', 'caution': 'static_dong_aggregate_not_future_forecast_not_causal', 'mlflow.note.content': note})
    for f in ['아파트전세_RF_v1_성능.csv', '아파트전세_RF_v1_동네별예측.csv', '아파트전세_RF_v1.png']: mlflow.log_artifact(OUT + f, artifact_path='evidence')
    for p in ev.iterdir(): mlflow.log_artifact(str(p), artifact_path='evidence')
    mlflow.log_artifact(__file__, artifact_path='source')
    mlflow.sklearn.log_model(model, name='model', serialization_format='cloudpickle', input_example=X.head(3))
print('MLflow:', mlflow.get_tracking_uri(), '| Experiment:', a.experiment)
