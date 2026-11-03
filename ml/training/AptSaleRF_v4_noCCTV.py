# 모델 버전 4-D: 랜덤포레스트에서 CCTV 수를 빼도 성능이 그대로인지 확인 (v1 = 7개 항목 vs v4-D = CCTV 뺀 6개 항목)
#   같은 동네 408곳, 같은 분할(5분할 × 20번 반복)로 두 모델을 나란히 학습해서 반복마다 차이를 비교함
#   CCTV를 빼 보는 이유: 공공 CCTV(생활방범·교통단속·어린이보호 92.6%)라 아파트·주상복합 자체 CCTV가 없고, 세 모델 모두 중요도가 가장 낮았음
#   판단: R² 차이가 반복마다 흔들리는 폭(v1의 R² 표준편차) 안이면 "빼도 성능 차이 없음"
# 실행: python AptSaleRF_v4_noCCTV.py                              (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python AptSaleRF_v4_noCCTV.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록)
import argparse, hashlib, json, os, subprocess, warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
import joblib, mlflow, mlflow.sklearn, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from apt_sale_data import OUT, X_COLS, Y, load
warnings.simplefilter('ignore')

REPEAT = 20                                                 # v1~v3와 같아야 같은 분할로 비교됨
NAME = '랜덤포레스트 v4-D (CCTV 제외)'
X6 = [c for c in X_COLS if c != 'CCTV 수']
def rf(): return RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=0, n_jobs=-1)      # v1과 같은 설정

ap = argparse.ArgumentParser()
ap.add_argument('--tracking-uri'); ap.add_argument('--experiment', default='house-price-forecast-comparison'); ap.add_argument('--task-id', default='unassigned')
a = ap.parse_args()

d = load()
y = d[Y]
cv = list(RepeatedKFold(n_splits=5, n_repeats=REPEAT, random_state=0).split(d))
P = {'7개 (v1)': np.zeros((REPEAT, len(d))), '6개 (CCTV 제외)': np.zeros((REPEAT, len(d)))}
fold_mae = {k: [] for k in P}
for i, (tr, te) in enumerate(cv):
    for k, cols in zip(P, [X_COLS, X6]):
        p = rf().fit(d[cols].iloc[tr], y.iloc[tr]).predict(d[cols].iloc[te])
        P[k][i // 5, te] = p; fold_mae[k].append(np.abs(y.iloc[te] - p).mean())
    if i % 10 == 9: print(f'{i + 1}/{len(cv)} 분할 완료', flush=True)

r2 = lambda p: 1 - ((y.values - p) ** 2).sum(axis=1) / ((y.values - y.mean()) ** 2).sum()            # 반복마다 R² (v1~v3와 같은 정의)
mae = lambda p: np.abs(y.values - p).mean(axis=1)
res = pd.DataFrame({'모델': ['랜덤포레스트 v1 (7개)', NAME], 'R² 평균': [r2(v).mean() for v in P.values()], 'R² 표준편차': [r2(v).std() for v in P.values()],
                    '평균 오차(만원/평)': [mae(v).mean() for v in P.values()]})
a7, a6 = P.values()
diff = r2(a6) - r2(a7)                                       # 반복마다 (CCTV 뺀 모델 R²) - (7개 모델 R²): 같은 분할이라 바로 비교 가능
wins = int((np.array(fold_mae['6개 (CCTV 제외)']) <= np.array(fold_mae['7개 (v1)'])).sum())
same = abs(diff.mean()) < r2(a7).std()
verdict = ('빼도 성능 차이 없음 → CCTV 제외 가능' if same or diff.mean() > 0 else '빼면 성능이 떨어짐 → CCTV 유지')
print('\n[학습에 쓰지 않은 동네에서 평가, 5분할 × 20번 반복, 같은 분할]'); print(res.round(3).to_string(index=False))
print(f'\nR² 차이 (6개 - 7개): 평균 {diff.mean():+.4f}, 반복 20번 중 6개가 더 좋거나 같음 {(diff >= 0).sum()}번')
print(f'평균 오차 차이: {mae(a6).mean() - mae(a7).mean():+.1f}만원/평, 100개 분할 중 6개 모델 오차가 더 작거나 같음 {wins}번')
print(f'기준(흔들리는 폭) = v1 R² 표준편차 {r2(a7).std():.4f} → 판정: {verdict}')

# 저장: 성능 표, 반복별 차이, 동네별 예측, 모델 비교 표, 전체 데이터로 학습한 6개 항목 모델
os.makedirs(OUT, exist_ok=True)
res.round(3).to_csv(OUT + '아파트매매_RF_v4D_성능.csv', index=False, encoding='utf-8-sig')
pd.DataFrame({'반복': range(1, REPEAT + 1), 'R² 7개': r2(a7), 'R² 6개(CCTV 제외)': r2(a6), '차이': diff}).round(4).to_csv(OUT + '아파트매매_RF_v4D_반복별차이.csv', index=False, encoding='utf-8-sig')
d.assign(예측=a6.mean(axis=0), 오차=lambda t: t['예측'] - t[Y]).reset_index().round(1).to_csv(OUT + '아파트매매_RF_v4D_동네별예측.csv', index=False, encoding='utf-8-sig')
cp = OUT + '아파트매매_모델비교.csv'
old = pd.read_csv(cp)
base = old[old['모델'] == '기준(평균으로 찍기)']
cmp = pd.concat([old[~old['모델'].isin([NAME, '기준(평균으로 찍기)'])], res.iloc[[1]], base]).round(3)
cmp.to_csv(cp, index=False, encoding='utf-8-sig')
print('\n[모델 비교]'); print(cmp.to_string(index=False))
model = rf().fit(d[X6], y)
joblib.dump(model, OUT + '아파트매매_RF_v4D.joblib')

# MLflow 기록 (docs/model-team-workflow.md): Run 1개, 필수 태그, Registry 등록·champion·MinIO 업로드는 하지 않음
root = Path(__file__).resolve().parents[2]
if a.tracking_uri: mlflow.set_tracking_uri(a.tracking_uri)
else: (root / 'artifacts/mlflow_apt_sale').mkdir(parents=True, exist_ok=True); mlflow.set_tracking_uri('sqlite:///' + (root / 'artifacts/mlflow_apt_sale/tracking.db').as_posix())
mlflow.set_experiment(a.experiment)
ev = root / 'artifacts/apt_sale_model_comparison' / ('v4D_' + datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')); ev.mkdir(parents=True)
d.reset_index().to_csv(ev / 'analysis_panel.csv', index=False, encoding='utf-8-sig')                 # compare_apt_sale_models.py 와 같은 방식 → 같은 데이터면 같은 SHA-256
split = pd.concat([d.reset_index()[['행정동_코드', '자치구', '행정동']].assign(repeat=r, validation_fold=-1) for r in range(REPEAT)], ignore_index=True)
for i, (_, te) in enumerate(cv): split.loc[te + (i // 5) * len(d), 'validation_fold'] = i % 5
split.to_csv(ev / 'cv_split.csv', index=False, encoding='utf-8-sig')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
owner = subprocess.run(['git', 'config', 'user.name'], capture_output=True, text=True).stdout.strip() or 'unknown'
why = (f'v1 랜덤포레스트에서 CCTV 수 제거 실험. 이유: 공공 CCTV(생활방범·교통단속·어린이보호 92.6%)라 아파트·주상복합 자체 CCTV가 없고, 세 모델 순열 중요도 최하위. '
       f'결과: R² {r2(a7).mean():.3f} → {r2(a6).mean():.3f} (차이 {diff.mean():+.4f}, 기준 폭 {r2(a7).std():.4f}), MAE {mae(a7).mean():.1f} → {mae(a6).mean():.1f}. 판정: {verdict}. 동네 단위 정적 집계이며 미래 예측·인과 모델이 아님.')
with mlflow.start_run(run_name='apt_sale_random_forest_v4D_no_cctv'):
    mlflow.log_params({'algorithm': 'random_forest', 'version': 'v4-D', 'features': json.dumps(X6, ensure_ascii=False), 'removed_feature': 'CCTV 수', 'compared_with': 'v1 random_forest 7 features',
                       'n_dongs': len(d), 'cv': 'RepeatedKFold(n_splits=5, n_repeats=20, random_state=0)', 'estimator_params': json.dumps(rf().get_params(), default=str)})
    mlflow.log_metrics({'cv_r2_pooled': float(r2(a6).mean()), 'cv_r2_pooled_std': float(r2(a6).std()), 'cv_mae': float(mae(a6).mean()), 'v1_cv_r2_pooled': float(r2(a7).mean()),
                        'v1_cv_mae': float(mae(a7).mean()), 'r2_diff_vs_v1': float(diff.mean()), 'fold_wins_vs_v1': wins})
    mlflow.set_tags({'owner': owner, 'task_id': a.task_id, 'dataset_sha256': sha(ev / 'analysis_panel.csv'), 'split_sha256': sha(ev / 'cv_split.csv'), 'review_status': 'not_reviewed', 'selection_status': 'feature_ablation', 'caution': 'static_dong_aggregate_not_future_forecast_not_causal', 'mlflow.note.content': why})
    for f in ['아파트매매_RF_v4D_성능.csv', '아파트매매_RF_v4D_반복별차이.csv', '아파트매매_RF_v4D_동네별예측.csv']: mlflow.log_artifact(OUT + f, artifact_path='evidence')
    for f in ev.iterdir(): mlflow.log_artifact(str(f), artifact_path='evidence')
    mlflow.log_artifact(__file__, artifact_path='source')
    mlflow.sklearn.log_model(model, name='model', serialization_format='cloudpickle', input_example=d[X6].head(3))
print('\n저장:', os.path.abspath(OUT), '| MLflow:', mlflow.get_tracking_uri())
