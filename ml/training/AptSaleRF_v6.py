# 모델 버전 6: 랜덤포레스트에 평균 전용면적을 더하면 좋아지는지 확인 (v1 = 7개 항목 vs v6 = 7개 + 평균 전용면적)
#   같은 동네 408곳, 같은 분할(5분할 × 20번 반복), 같은 랜덤포레스트 설정으로 두 모델을 나란히 학습해서 비교함
#   평균 전용면적 = 동네 아파트 매매 거래의 전용면적(excluUseAr, ㎡) 평균. 거래 기준은 apt_sale_data.py 와 같음 (취소 거래·20㎡ 미만·지번 불일치 제외)
#   더하는 이유: 지금 X에 집 크기 정보가 없고, 기존 7개를 고려한 뒤에도 평당 금액과의 연관이 남음(순위 편상관 +0.30)
#   판단: 좋아짐 = R² 차이가 반복마다 흔들리는 폭(v1 R² 표준편차)보다 크고, 100개 분할 중 80개 이상에서 오차가 작음
# 실행: python AptSaleRF_v6.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python AptSaleRF_v6.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록)
import argparse, hashlib, json, os, subprocess, warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
import joblib, mlflow, mlflow.sklearn, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedKFold
from apt_sale_data import DATA, OLD, OUT, X_COLS, Y, load
warnings.simplefilter('ignore')

REPEAT = 20                                                 # v1~v5와 같아야 같은 분할로 비교됨
NAME = '랜덤포레스트 v6 (전용면적 추가)'
ADD = '평균 전용면적(㎡)'
X8 = X_COLS + [ADD]
def rf(): return RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=0, n_jobs=-1)      # v1과 같은 설정

ap = argparse.ArgumentParser()
ap.add_argument('--tracking-uri'); ap.add_argument('--experiment', default='house-price-forecast-comparison'); ap.add_argument('--task-id', default='unassigned')
a = ap.parse_args()

# 데이터: v1과 같은 408곳 + 동네 평균 전용면적
d = load()
t = pd.read_csv(DATA + '부동산(도로명주소)/apt_sale_juso.csv', low_memory=False, usecols=['행정동_코드', '연결방식', 'cdealType', 'excluUseAr'])
t = t[t['cdealType'].isna() & (t['excluUseAr'] >= 20) & (t['연결방식'] == '지번 일치')]          # apt_sale_data.py 와 같은 거래 기준
area = t.groupby(t['행정동_코드'].astype(int).replace(OLD))['excluUseAr'].mean().rename(ADD)
n0 = len(d); d = d.join(area)
assert len(d) == n0 and d[ADD].notna().all(), '전용면적이 없는 동네가 있음'
print(f'동네 {len(d)}곳 (v1과 같음) | 평균 전용면적 {d[ADD].min():.0f}~{d[ADD].max():.0f}㎡ (중앙값 {d[ADD].median():.0f}㎡) | 평당 금액과 연관도 {d[ADD].corr(d[Y], method="spearman"):+.2f}')
y = d[Y]

# 학습·평가: 같은 분할에서 v1(7개)과 v6(8개)을 나란히
cv = list(RepeatedKFold(n_splits=5, n_repeats=REPEAT, random_state=0).split(d))
P = {'v1': np.zeros((REPEAT, len(d))), 'v6': np.zeros((REPEAT, len(d)))}
fold_mae = {k: [] for k in P}
for i, (tr, te) in enumerate(cv):
    for k, cols in zip(P, [X_COLS, X8]):
        p = rf().fit(d[cols].iloc[tr], y.iloc[tr]).predict(d[cols].iloc[te])
        P[k][i // 5, te] = p; fold_mae[k].append(np.abs(y.iloc[te] - p).mean())
    if i % 10 == 9: print(f'{i + 1}/{len(cv)} 분할 완료', flush=True)

r2 = lambda p: 1 - ((y.values - p) ** 2).sum(axis=1) / ((y.values - y.mean()) ** 2).sum()            # 반복마다 R² (v1~v5와 같은 정의)
mae = lambda p: np.abs(y.values - p).mean(axis=1)
a1, a6 = P['v1'], P['v6']
res = pd.DataFrame({'모델': ['랜덤포레스트 v1 (7개)', NAME], 'R² 평균': [r2(a1).mean(), r2(a6).mean()], 'R² 표준편차': [r2(a1).std(), r2(a6).std()],
                    '평균 오차(만원/평)': [mae(a1).mean(), mae(a6).mean()]})
diff = r2(a6) - r2(a1)                                       # 반복마다 (v6 R²) - (v1 R²): 같은 분할이라 바로 비교 가능
wins = int((np.array(fold_mae['v6']) < np.array(fold_mae['v1'])).sum())
band = r2(a1).std()
if diff.mean() > band and wins >= 80: verdict = '좋아짐 → 전용면적 추가 효과 있음'
elif abs(diff.mean()) <= band: verdict = f'차이가 흔들리는 폭 안 → 뚜렷한 효과 없음 (오차가 더 작은 분할 {wins}/100)'
else: verdict = '나빠짐 → 추가하지 않는 게 나음' if diff.mean() < 0 else f'R²는 올랐지만 분할 {wins}/100에서만 오차가 작음 → 효과가 일관되지 않음'
print('\n[학습에 쓰지 않은 동네에서 평가, 5분할 × 20번 반복, 같은 분할]'); print(res.round(3).to_string(index=False))
print(f'\nR² 차이 (v6 - v1): 평균 {diff.mean():+.4f}, 반복 20번 중 v6가 더 좋음 {(diff > 0).sum()}번')
print(f'평균 오차 차이: {mae(a6).mean() - mae(a1).mean():+.1f}만원/평, 100개 분할 중 v6 오차가 더 작음 {wins}번')
print(f'기준(흔들리는 폭) = v1 R² 표준편차 {band:.4f} → 판정: {verdict}')

# 약점 확인: 비싼 동네(평당 금액 상위 25%)에서 오차가 줄었나
top = (y >= y.quantile(.75)).values
weak = pd.DataFrame({g: {'동네 수': m.sum(), 'v1 평균 오차': np.abs(y.values - a1.mean(0))[m].mean(), 'v6 평균 오차': np.abs(y.values - a6.mean(0))[m].mean(),
                         'v1 치우침(예측-실제)': (a1.mean(0) - y.values)[m].mean(), 'v6 치우침(예측-실제)': (a6.mean(0) - y.values)[m].mean()}
                     for g, m in {'전체': np.ones(len(d), bool), '비싼 동네(상위 25%)': top}.items()}).T
print('\n[약점 확인: 동네별 평균 예측 기준, 치우침이 음수면 낮게 예측]'); print(weak.round(1).to_string())

# 항목 중요도(앞 25개 분할) + 전용면적 방향: 다른 항목은 그대로 두고 전용면적만 바꿨을 때 예측 평당 금액
imp = np.zeros(len(X8))
for tr, te in cv[:25]:
    m = rf().fit(d[X8].iloc[tr], y.iloc[tr])
    imp += permutation_importance(m, d[X8].iloc[te], y.iloc[te], n_repeats=5, random_state=0).importances_mean / 25
imp = pd.Series(imp, index=X8).sort_values(ascending=False)
print('\n[v6 항목 중요도: 섞으면 R²가 떨어지는 정도]'); print(imp.round(3).to_string())
model = rf().fit(d[X8], y)
way = pd.Series({f'{v:.0f}㎡': model.predict(d[X8].assign(**{ADD: v})).mean() for v in d[ADD].quantile([.1, .25, .5, .75, .9])}, name='예측 평당 금액 평균(만원)')
print('\n[전용면적 방향: 모든 동네의 전용면적을 이 값으로 바꿨을 때 예측 평당 금액 평균]'); print(way.round(0).to_string())

# 저장: 성능 표, 반복별 차이, 약점 확인, 중요도, 방향, 동네별 예측, 모델 비교 표, 전체 데이터로 학습한 모델
os.makedirs(OUT, exist_ok=True)
res.round(3).to_csv(OUT + '아파트매매_RF_v6_성능.csv', index=False, encoding='utf-8-sig')
pd.DataFrame({'반복': range(1, REPEAT + 1), 'R² v1': r2(a1), 'R² v6': r2(a6), '차이': diff}).round(4).to_csv(OUT + '아파트매매_RF_v6_반복별차이.csv', index=False, encoding='utf-8-sig')
weak.round(1).to_csv(OUT + '아파트매매_RF_v6_약점확인.csv', encoding='utf-8-sig')
imp.round(4).rename('중요도(R² 감소량)').to_csv(OUT + '아파트매매_RF_v6_중요도.csv', encoding='utf-8-sig')
way.round(1).rename_axis('전용면적').to_csv(OUT + '아파트매매_RF_v6_면적방향.csv', encoding='utf-8-sig')
d.assign(예측=a6.mean(axis=0), 오차=lambda t: t['예측'] - t[Y]).reset_index().round(1).to_csv(OUT + '아파트매매_RF_v6_동네별예측.csv', index=False, encoding='utf-8-sig')
cp = OUT + '아파트매매_모델비교.csv'
old = pd.read_csv(cp)
cmp = pd.concat([old[~old['모델'].isin([NAME, '기준(평균으로 찍기)'])], res.iloc[[1]], old[old['모델'] == '기준(평균으로 찍기)']]).round(3)
cmp.to_csv(cp, index=False, encoding='utf-8-sig')
print('\n[모델 비교]'); print(cmp.to_string(index=False))
joblib.dump(model, OUT + '아파트매매_RF_v6.joblib')

# MLflow 기록 (docs/model-team-workflow.md): Run 1개, 필수 태그, Registry 등록·champion·MinIO 업로드는 하지 않음
root = Path(__file__).resolve().parents[2]
ev = root / 'artifacts/apt_sale_model_comparison' / ('v6_' + datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')); ev.mkdir(parents=True)
d.reset_index().to_csv(ev / 'analysis_panel.csv', index=False, encoding='utf-8-sig')                  # 학습에 실제로 쓴 표 (전용면적 포함)
split = pd.concat([d.reset_index()[['행정동_코드', '자치구', '행정동']].assign(repeat=r, validation_fold=-1) for r in range(REPEAT)], ignore_index=True)
for i, (_, te) in enumerate(cv): split.loc[te + (i // 5) * len(d), 'validation_fold'] = i % 5
split.to_csv(ev / 'cv_split.csv', index=False, encoding='utf-8-sig')                                      # compare_apt_sale_models.py 와 같은 방식 → 분할이 같으면 같은 SHA-256
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
if a.tracking_uri: mlflow.set_tracking_uri(a.tracking_uri)
else: (root / 'artifacts/mlflow_apt_sale').mkdir(parents=True, exist_ok=True); mlflow.set_tracking_uri('sqlite:///' + (root / 'artifacts/mlflow_apt_sale/tracking.db').as_posix())
mlflow.set_experiment(a.experiment)
owner = subprocess.run(['git', 'config', 'user.name'], capture_output=True, text=True).stdout.strip() or 'unknown'
why = (f'v1 랜덤포레스트에 동네 평균 전용면적 추가 실험. 이유: 집 크기 정보가 X에 없고, 기존 7개 고려 후에도 평당 금액과 연관이 남음(순위 편상관 +0.30). '
       f'결과: R² {r2(a1).mean():.3f} → {r2(a6).mean():.3f} (차이 {diff.mean():+.4f}, 기준 폭 {band:.4f}), MAE {mae(a1).mean():.1f} → {mae(a6).mean():.1f}, 오차가 더 작은 분할 {wins}/100. 판정: {verdict}. 동네 단위 정적 집계이며 미래 예측·인과 모델이 아님.')
with mlflow.start_run(run_name='apt_sale_random_forest_v6_area'):
    mlflow.log_params({'algorithm': 'random_forest', 'version': 'v6', 'features': json.dumps(X8, ensure_ascii=False), 'added_features': json.dumps([ADD], ensure_ascii=False),
                       'added_feature_source': 'apt_sale_juso.csv excluUseAr 동네 평균 (취소·20㎡ 미만·지번 불일치 제외)', 'compared_with': 'v1 random_forest 7 features',
                       'n_dongs': len(d), 'cv': 'RepeatedKFold(n_splits=5, n_repeats=20, random_state=0)', 'estimator_params': json.dumps(rf().get_params(), default=str)})
    mlflow.log_metrics({'cv_r2_pooled': float(r2(a6).mean()), 'cv_r2_pooled_std': float(r2(a6).std()), 'cv_mae': float(mae(a6).mean()), 'v1_cv_r2_pooled': float(r2(a1).mean()),
                        'v1_cv_mae': float(mae(a1).mean()), 'r2_diff_vs_v1': float(diff.mean()), 'fold_wins_vs_v1': wins})
    mlflow.set_tags({'owner': owner, 'task_id': a.task_id, 'dataset_sha256': sha(ev / 'analysis_panel.csv'), 'split_sha256': sha(ev / 'cv_split.csv'), 'review_status': 'not_reviewed',
                     'selection_status': 'feature_addition', 'caution': 'static_dong_aggregate_not_future_forecast_not_causal', 'mlflow.note.content': why})
    for f in ['성능', '반복별차이', '약점확인', '중요도', '면적방향', '동네별예측']: mlflow.log_artifact(OUT + f'아파트매매_RF_v6_{f}.csv', artifact_path='evidence')
    for f in ev.iterdir(): mlflow.log_artifact(str(f), artifact_path='evidence')
    mlflow.log_artifact(__file__, artifact_path='source')
    mlflow.sklearn.log_model(model, name='model', serialization_format='cloudpickle', input_example=d[X8].head(3))
print('\n저장:', os.path.abspath(OUT), '| MLflow:', mlflow.get_tracking_uri())
