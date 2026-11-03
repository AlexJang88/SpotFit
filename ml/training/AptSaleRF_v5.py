# 모델 버전 5: 랜덤포레스트에 은행 수·지하철역 수를 더하면 좋아지는지 확인 (v1 = 7개 항목 vs v5 = 7개 + 은행 수 + 지하철역 수)
#   같은 동네 408곳, 같은 분할(5분할 × 20번 반복), 같은 랜덤포레스트 설정으로 두 모델을 나란히 학습해서 비교함
#   더하는 이유: AptHypothesis.py 집객시설 종류별 연관도에서 은행(+0.30)·지하철역(+0.25)이 가장 컸고, 지금 X에 없는 위치·교통 정보라서
#   은행 수·지하철역 수 = 상권분석 집객시설-행정동 자료의 분기별 시설 수를 전체 기간(2021년 1분기~2026년 2분기, 22개 분기) 평균. 빈칸은 시설 없음 → 0 (AptHypothesis.py 와 같은 방식)
#   판단: 좋아짐 = R² 차이가 반복마다 흔들리는 폭(v1 R² 표준편차)보다 크고, 100개 분할 중 80개 이상에서 오차가 작음
# 실행: python AptSaleRF_v5.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python AptSaleRF_v5.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록)
import argparse, hashlib, json, os, subprocess, warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
import joblib, mlflow, mlflow.sklearn, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from apt_sale_data import DATA, OUT, X_COLS, Y, load
warnings.simplefilter('ignore')

REPEAT = 20                                                 # v1~v4와 같아야 같은 분할로 비교됨
NAME = '랜덤포레스트 v5 (은행·지하철역 추가)'
ADD = ['은행 수', '지하철역 수']
X9 = X_COLS + ADD
def rf(): return RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=0, n_jobs=-1)      # v1과 같은 설정

ap = argparse.ArgumentParser()
ap.add_argument('--tracking-uri'); ap.add_argument('--experiment', default='house-price-forecast-comparison'); ap.add_argument('--task-id', default='unassigned')
a = ap.parse_args()

d = load()
fac = pd.read_csv(DATA + '../raw/서울시 상권분석서비스(집객시설-행정동).csv', encoding='cp949')
assert not fac.duplicated(['기준_년분기_코드', '행정동_코드']).any()
fac = fac[['행정동_코드', '은행_수', '지하철_역_수']].fillna(0).groupby('행정동_코드').mean().set_axis(ADD, axis=1)     # 행정동 코드는 상권분석 코드 (apt_sale_data 도 같은 코드로 맞춰 둠)
n0 = len(d); d = d.join(fac)
assert len(d) == n0 and d[ADD].notna().all().all(), '은행·지하철역 값이 없는 동네가 있음'
print(f'동네 {len(d)}곳 (v1과 같음) | 은행 수 평균 {d["은행 수"].mean():.2f}, 지하철역 수 평균 {d["지하철역 수"].mean():.2f}, 지하철역이 없는 동네 {(d["지하철역 수"] == 0).sum()}곳')
y = d[Y]
cv = list(RepeatedKFold(n_splits=5, n_repeats=REPEAT, random_state=0).split(d))
P = {'v1': np.zeros((REPEAT, len(d))), 'v5': np.zeros((REPEAT, len(d)))}
fold_mae = {k: [] for k in P}
for i, (tr, te) in enumerate(cv):
    for k, cols in zip(P, [X_COLS, X9]):
        p = rf().fit(d[cols].iloc[tr], y.iloc[tr]).predict(d[cols].iloc[te])
        P[k][i // 5, te] = p; fold_mae[k].append(np.abs(y.iloc[te] - p).mean())
    if i % 10 == 9: print(f'{i + 1}/{len(cv)} 분할 완료', flush=True)

r2 = lambda p: 1 - ((y.values - p) ** 2).sum(axis=1) / ((y.values - y.mean()) ** 2).sum()            # 반복마다 R² (v1~v4와 같은 정의)
mae = lambda p: np.abs(y.values - p).mean(axis=1)
a1, a5 = P['v1'], P['v5']
res = pd.DataFrame({'모델': ['랜덤포레스트 v1 (7개)', NAME], 'R² 평균': [r2(a1).mean(), r2(a5).mean()], 'R² 표준편차': [r2(a1).std(), r2(a5).std()],
                    '평균 오차(만원/평)': [mae(a1).mean(), mae(a5).mean()]})
diff = r2(a5) - r2(a1)                                       # 반복마다 (v5 R²) - (v1 R²): 같은 분할이라 바로 비교 가능
wins = int((np.array(fold_mae['v5']) < np.array(fold_mae['v1'])).sum())
band = r2(a1).std()
if diff.mean() > band and wins >= 80: verdict = '좋아짐 → 은행 수·지하철역 수 추가 효과 있음'
elif abs(diff.mean()) <= band: verdict = f'차이가 흔들리는 폭 안 → 뚜렷한 효과 없음 (오차가 더 작은 분할 {wins}/100)'
else: verdict = '나빠짐 → 추가하지 않는 게 나음' if diff.mean() < 0 else f'R²는 올랐지만 분할 {wins}/100에서만 오차가 작음 → 효과가 일관되지 않음'
print('\n[학습에 쓰지 않은 동네에서 평가, 5분할 × 20번 반복, 같은 분할]'); print(res.round(3).to_string(index=False))
print(f'\nR² 차이 (v5 - v1): 평균 {diff.mean():+.4f}, 반복 20번 중 v5가 더 좋음 {(diff > 0).sum()}번')
print(f'평균 오차 차이: {mae(a5).mean() - mae(a1).mean():+.1f}만원/평, 100개 분할 중 v5 오차가 더 작음 {wins}번')
print(f'기준(흔들리는 폭) = v1 R² 표준편차 {band:.4f} → 판정: {verdict}')

# 원래 약점 확인: 비싼 동네(평당 금액 상위 25%), 상업 비중이 큰 동네(점포 수 ÷ 거주인구 상위 25%)에서 오차가 줄었나
info = pd.read_csv(OUT + '평당금액_행정동표.csv').set_index('행정동_코드')
groups = {'전체': np.ones(len(d), bool), '비싼 동네(상위 25%)': (y >= y.quantile(.75)).values,
          '상업 비중 큰 동네(상위 25%)': (lambda s: (s >= s.quantile(.75)).values)(info.loc[d.index, '점포 수'] / info.loc[d.index, '거주인구 수'])}
weak = pd.DataFrame({g: {'동네 수': m.sum(), 'v1 평균 오차': np.abs(y.values - a1.mean(0))[m].mean(), 'v5 평균 오차': np.abs(y.values - a5.mean(0))[m].mean(),
                         'v1 치우침(예측-실제)': (a1.mean(0) - y.values)[m].mean(), 'v5 치우침(예측-실제)': (a5.mean(0) - y.values)[m].mean()} for g, m in groups.items()}).T
print('\n[약점 확인: 동네별 평균 예측 기준, 치우침이 음수면 낮게 예측]'); print(weak.round(1).to_string())

# 항목 중요도 (v5): 앞 25개 분할에서 순열 중요도 (v1~v3 스크립트와 같은 방식)
from sklearn.inspection import permutation_importance
imp = np.zeros(len(X9))
for tr, te in cv[:25]:
    m = rf().fit(d[X9].iloc[tr], y.iloc[tr])
    imp += permutation_importance(m, d[X9].iloc[te], y.iloc[te], n_repeats=5, random_state=0).importances_mean / 25
imp = pd.Series(imp, index=X9).sort_values(ascending=False)
print('\n[v5 항목 중요도: 섞으면 R²가 떨어지는 정도]'); print(imp.round(3).to_string())

# 저장: 성능 표, 반복별 차이, 약점 확인, 중요도, 동네별 예측, 모델 비교 표, 전체 데이터로 학습한 모델
os.makedirs(OUT, exist_ok=True)
res.round(3).to_csv(OUT + '아파트매매_RF_v5_성능.csv', index=False, encoding='utf-8-sig')
pd.DataFrame({'반복': range(1, REPEAT + 1), 'R² v1': r2(a1), 'R² v5': r2(a5), '차이': diff}).round(4).to_csv(OUT + '아파트매매_RF_v5_반복별차이.csv', index=False, encoding='utf-8-sig')
weak.round(1).to_csv(OUT + '아파트매매_RF_v5_약점확인.csv', encoding='utf-8-sig')
imp.round(4).rename('중요도(R² 감소량)').to_csv(OUT + '아파트매매_RF_v5_중요도.csv', encoding='utf-8-sig')
d.assign(예측=a5.mean(axis=0), 오차=lambda t: t['예측'] - t[Y]).reset_index().round(1).to_csv(OUT + '아파트매매_RF_v5_동네별예측.csv', index=False, encoding='utf-8-sig')
cp = OUT + '아파트매매_모델비교.csv'
old = pd.read_csv(cp)
cmp = pd.concat([old[~old['모델'].isin([NAME, '기준(평균으로 찍기)'])], res.iloc[[1]], old[old['모델'] == '기준(평균으로 찍기)']]).round(3)
cmp.to_csv(cp, index=False, encoding='utf-8-sig')
print('\n[모델 비교]'); print(cmp.to_string(index=False))
model = rf().fit(d[X9], y)
joblib.dump(model, OUT + '아파트매매_RF_v5.joblib')

# MLflow 기록 (docs/model-team-workflow.md): Run 1개, 필수 태그, Registry 등록·champion·MinIO 업로드는 하지 않음
root = Path(__file__).resolve().parents[2]
ev = root / 'artifacts/apt_sale_model_comparison' / ('v5_' + datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')); ev.mkdir(parents=True)
d.reset_index().to_csv(ev / 'analysis_panel.csv', index=False, encoding='utf-8-sig')                  # 학습에 실제로 쓴 표 (은행·지하철역 포함)
split = pd.concat([d.reset_index()[['행정동_코드', '자치구', '행정동']].assign(repeat=r, validation_fold=-1) for r in range(REPEAT)], ignore_index=True)
for i, (_, te) in enumerate(cv): split.loc[te + (i // 5) * len(d), 'validation_fold'] = i % 5
split.to_csv(ev / 'cv_split.csv', index=False, encoding='utf-8-sig')                                      # compare_apt_sale_models.py 와 같은 방식 → 분할이 같으면 같은 SHA-256
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
if a.tracking_uri: mlflow.set_tracking_uri(a.tracking_uri)
else: (root / 'artifacts/mlflow_apt_sale').mkdir(parents=True, exist_ok=True); mlflow.set_tracking_uri('sqlite:///' + (root / 'artifacts/mlflow_apt_sale/tracking.db').as_posix())
mlflow.set_experiment(a.experiment)
owner = subprocess.run(['git', 'config', 'user.name'], capture_output=True, text=True).stdout.strip() or 'unknown'
why = (f'v1 랜덤포레스트에 은행 수·지하철역 수 추가 실험. 이유: 집객시설 종류별 연관도 최상위(은행 +0.30, 지하철역 +0.25), 위치·교통 정보 보완. '
       f'결과: R² {r2(a1).mean():.3f} → {r2(a5).mean():.3f} (차이 {diff.mean():+.4f}, 기준 폭 {band:.4f}), MAE {mae(a1).mean():.1f} → {mae(a5).mean():.1f}, 오차가 더 작은 분할 {wins}/100. 판정: {verdict}. 동네 단위 정적 집계이며 미래 예측·인과 모델이 아님.')
with mlflow.start_run(run_name='apt_sale_random_forest_v5_bank_subway'):
    mlflow.log_params({'algorithm': 'random_forest', 'version': 'v5', 'features': json.dumps(X9, ensure_ascii=False), 'added_features': json.dumps(ADD, ensure_ascii=False),
                       'added_feature_source': '서울시 상권분석서비스(집객시설-행정동).csv 은행_수·지하철_역_수, 2021Q1~2026Q2 평균, 빈칸=0', 'compared_with': 'v1 random_forest 7 features',
                       'n_dongs': len(d), 'cv': 'RepeatedKFold(n_splits=5, n_repeats=20, random_state=0)', 'estimator_params': json.dumps(rf().get_params(), default=str)})
    mlflow.log_metrics({'cv_r2_pooled': float(r2(a5).mean()), 'cv_r2_pooled_std': float(r2(a5).std()), 'cv_mae': float(mae(a5).mean()), 'v1_cv_r2_pooled': float(r2(a1).mean()),
                        'v1_cv_mae': float(mae(a1).mean()), 'r2_diff_vs_v1': float(diff.mean()), 'fold_wins_vs_v1': wins})
    mlflow.set_tags({'owner': owner, 'task_id': a.task_id, 'dataset_sha256': sha(ev / 'analysis_panel.csv'), 'split_sha256': sha(ev / 'cv_split.csv'), 'review_status': 'not_reviewed',
                     'selection_status': 'feature_addition', 'caution': 'static_dong_aggregate_not_future_forecast_not_causal', 'mlflow.note.content': why})
    for f in ['성능', '반복별차이', '약점확인', '중요도', '동네별예측']: mlflow.log_artifact(OUT + f'아파트매매_RF_v5_{f}.csv', artifact_path='evidence')
    for f in ev.iterdir(): mlflow.log_artifact(str(f), artifact_path='evidence')
    mlflow.log_artifact(__file__, artifact_path='source')
    mlflow.sklearn.log_model(model, name='model', serialization_format='cloudpickle', input_example=d[X9].head(3))
print('\n저장:', os.path.abspath(OUT), '| MLflow:', mlflow.get_tracking_uri())
