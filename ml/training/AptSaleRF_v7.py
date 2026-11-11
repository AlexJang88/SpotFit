# 모델 버전 7: v6(7개 + 평균 전용면적)에 상주인구 연령대 비율 6개를 더하면 좋아지는지 확인 (v6 = 8개 항목 vs v7 = 8개 + 연령대 6개)
#   같은 동네 408곳, 같은 분할(5분할 × 20번 반복), 같은 랜덤포레스트 설정으로 두 모델을 나란히 학습해서 비교함
#   연령대 비율 = 상주인구(그 동네에 사는 주민)의 연령대별 인구 ÷ 6개 연령대 합 × 100. 특정 연령대를 고르지 않고 6개를 모두 넣음
#     상주인구 연령대 값은 몇 번만 바뀜(같은 값이 여러 분기 반복). 값이 이상한 구간(2022년 4분기~2023년 3분기: 서울 20대가 급감 후 회복)은
#     AgingHypothesis.py 와 같이 빼고, 나머지 분기의 비율을 동네별로 평균함
#   판단: 좋아짐 = R² 차이가 반복마다 흔들리는 폭(v6 R² 표준편차)보다 크고, 100개 분할 중 80개 이상에서 오차가 작음
# 실행: python AptSaleRF_v7.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python AptSaleRF_v7.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록)
import argparse, hashlib, json, os, subprocess, warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
import joblib, mlflow, mlflow.sklearn, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedKFold
from apt_sale_data import DATA, OUT, PROC, X_COLS, Y, load
warnings.simplefilter('ignore')

REPEAT = 20                                                 # v1~v6와 같아야 같은 분할로 비교됨
NAME = '랜덤포레스트 v7 (전용면적 + 상주인구 연령대)'
AREA = '평균 전용면적(㎡)'
AGES = {'10': '10대', '20': '20대', '30': '30대', '40': '40대', '50': '50대', '60_이상': '60대 이상'}
AGE_COLS = [f'{v} 주민 비율(%)' for v in AGES.values()]
X8 = X_COLS + [AREA]                                        # v6의 항목
X14 = X8 + AGE_COLS                                         # v7의 항목
def rf(): return RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=0, n_jobs=-1)      # v1~v6과 같은 설정

ap = argparse.ArgumentParser()
ap.add_argument('--tracking-uri'); ap.add_argument('--experiment', default='house-price-forecast-comparison'); ap.add_argument('--task-id', default='unassigned')
a = ap.parse_args()

# ---------- 데이터: v1과 같은 408곳 + 평균 전용면적(v6과 같은 방법) + 연령대 비율 ----------
d = load()
t = pd.read_csv(PROC + '아파트_매매.csv', low_memory=False, usecols=['행정동_코드(상권분석)', '전용면적(㎡)'])          # apt_sale_data.py 와 같은 거래 파일 (취소 거래·지번 불일치는 이미 제외됨)
d = d.join(t.groupby(t['행정동_코드(상권분석)'].astype(int))['전용면적(㎡)'].mean().rename(AREA))

p = pd.read_csv(DATA + '../raw/부동산/서울시 상권분석서비스(상주인구-행정동).csv', encoding='cp949')
assert not p.duplicated(['기준_년분기_코드', '행정동_코드']).any()
raw = [f'연령대_{k}_상주인구_수' for k in AGES]
tot = p.groupby('기준_년분기_코드')[raw].sum()                                                   # 분기별 서울 전체 연령대 인구
start = pd.Series(tot.index, index=tot.index).groupby((tot.diff().abs().sum(axis=1) > 0).cumsum()).transform('min')   # 값이 바뀐 분기 = 그 자료의 시작 분기
end = pd.Series(tot.index, index=tot.index).groupby(start).max()
print('상주인구 연령대 자료 구간:', ', '.join(f'{s}~{e}' for s, e in end.items()))
BAD = 20224                                                                                         # 이상한 구간의 시작 분기 (2022년 4분기)
assert BAD in end.index and end[BAD] == 20233, '이상한 구간이 2022년 4분기~2023년 3분기가 아님: 자료가 바뀌었는지 확인 필요'
print('서울 전체 20대 인구(천 명) 구간별:', (tot.groupby(start).first()['연령대_20_상주인구_수'] / 1000).round(0).to_dict(), '→', f'{BAD}~{end[BAD]} 구간 제외')
p = p[p['기준_년분기_코드'].map(start) != BAD]
share = p[raw].div(p[raw].sum(axis=1), axis=0).mul(100).set_axis(AGE_COLS, axis=1)                # 분기마다 연령대 비율(%)
age = share.groupby(p['행정동_코드']).mean()                                                       # 동네별 평균 (남은 분기)
n0 = len(d); d = d.join(age)
assert len(d) == n0 and d[X14].notna().all().all(), '전용면적 또는 연령대 값이 없는 동네가 있음'
print(f'동네 {len(d)}곳 (v1~v6과 같음) | 사용한 분기 {p["기준_년분기_코드"].nunique()}개 | 연령대 비율 합 {d[AGE_COLS].sum(axis=1).min():.1f}~{d[AGE_COLS].sum(axis=1).max():.1f}%')
print('[연령대 비율과 평당 금액의 연관도]', d[AGE_COLS].corrwith(d[Y], method='spearman').round(2).to_dict())
y = d[Y]

# ---------- 학습·평가: 같은 분할에서 v6(8개)과 v7(14개)을 나란히 ----------
cv = list(RepeatedKFold(n_splits=5, n_repeats=REPEAT, random_state=0).split(d))
P = {'v6': np.zeros((REPEAT, len(d))), 'v7': np.zeros((REPEAT, len(d)))}
fold_mae = {k: [] for k in P}
for i, (tr, te) in enumerate(cv):
    for k, cols in zip(P, [X8, X14]):
        p_ = rf().fit(d[cols].iloc[tr], y.iloc[tr]).predict(d[cols].iloc[te])
        P[k][i // 5, te] = p_; fold_mae[k].append(np.abs(y.iloc[te] - p_).mean())
    if i % 10 == 9: print(f'{i + 1}/{len(cv)} 분할 완료', flush=True)

r2 = lambda p: 1 - ((y.values - p) ** 2).sum(axis=1) / ((y.values - y.mean()) ** 2).sum()            # 반복마다 R² (v1~v6과 같은 정의)
mae = lambda p: np.abs(y.values - p).mean(axis=1)
a6, a7 = P['v6'], P['v7']
res = pd.DataFrame({'모델': ['랜덤포레스트 v6 (8개)', NAME], 'R² 평균': [r2(a6).mean(), r2(a7).mean()], 'R² 표준편차': [r2(a6).std(), r2(a7).std()],
                    '평균 오차(만원/평)': [mae(a6).mean(), mae(a7).mean()]})
diff = r2(a7) - r2(a6)                                       # 반복마다 (v7 R²) - (v6 R²): 같은 분할이라 바로 비교 가능
wins = int((np.array(fold_mae['v7']) < np.array(fold_mae['v6'])).sum())
band = r2(a6).std()
if diff.mean() > band and wins >= 80: verdict = '좋아짐 → 연령대 추가 효과 있음'
elif abs(diff.mean()) <= band: verdict = f'차이가 흔들리는 폭 안 → 뚜렷한 효과 없음 (오차가 더 작은 분할 {wins}/100)'
else: verdict = '나빠짐 → 추가하지 않는 게 나음' if diff.mean() < 0 else f'R²는 올랐지만 분할 {wins}/100에서만 오차가 작음 (80 미만)'
print('\n[학습에 쓰지 않은 동네에서 평가, 5분할 × 20번 반복, 같은 분할]'); print(res.round(3).to_string(index=False))
print(f'\nR² 차이 (v7 - v6): 평균 {diff.mean():+.4f}, 반복 20번 중 v7이 더 좋음 {(diff > 0).sum()}번')
print(f'평균 오차 차이: {mae(a7).mean() - mae(a6).mean():+.1f}만원/평, 100개 분할 중 v7 오차가 더 작음 {wins}번')
print(f'기준(흔들리는 폭) = v6 R² 표준편차 {band:.4f} → 판정: {verdict}')

# 약점 확인: 비싼 동네(평당 금액 상위 25%)에서 오차가 줄었나
top = (y >= y.quantile(.75)).values
weak = pd.DataFrame({g: {'동네 수': m.sum(), 'v6 평균 오차': np.abs(y.values - a6.mean(0))[m].mean(), 'v7 평균 오차': np.abs(y.values - a7.mean(0))[m].mean(),
                         'v6 치우침(예측-실제)': (a6.mean(0) - y.values)[m].mean(), 'v7 치우침(예측-실제)': (a7.mean(0) - y.values)[m].mean()}
                     for g, m in {'전체': np.ones(len(d), bool), '비싼 동네(상위 25%)': top}.items()}).T
print('\n[약점 확인: 동네별 평균 예측 기준, 치우침이 음수면 낮게 예측]'); print(weak.round(1).to_string())

# 항목 중요도(앞 25개 분할). 연령대끼리 정보가 겹쳐 중요도가 나뉘므로 6개 합도 같이 봄
imp = np.zeros(len(X14))
for tr, te in cv[:25]:
    m = rf().fit(d[X14].iloc[tr], y.iloc[tr])
    imp += permutation_importance(m, d[X14].iloc[te], y.iloc[te], n_repeats=5, random_state=0).importances_mean / 25
imp = pd.Series(imp, index=X14).sort_values(ascending=False)
print('\n[v7 항목 중요도: 섞으면 R²가 떨어지는 정도]'); print(imp.round(3).to_string())
print(f'연령대 6개 중요도 합: {imp[AGE_COLS].sum():.3f}')

# ---------- 저장: 성능 표, 반복별 차이, 약점 확인, 중요도, 동네별 예측, 모델 비교 표, 전체 데이터로 학습한 모델 ----------
os.makedirs(OUT, exist_ok=True)
res.round(3).to_csv(OUT + '아파트매매_RF_v7_성능.csv', index=False, encoding='utf-8-sig')
pd.DataFrame({'반복': range(1, REPEAT + 1), 'R² v6': r2(a6), 'R² v7': r2(a7), '차이': diff}).round(4).to_csv(OUT + '아파트매매_RF_v7_반복별차이.csv', index=False, encoding='utf-8-sig')
weak.round(1).to_csv(OUT + '아파트매매_RF_v7_약점확인.csv', encoding='utf-8-sig')
imp.round(4).rename('중요도(R² 감소량)').to_csv(OUT + '아파트매매_RF_v7_중요도.csv', encoding='utf-8-sig')
d.assign(예측=a7.mean(axis=0), 오차=lambda t: t['예측'] - t[Y]).reset_index().round(2).to_csv(OUT + '아파트매매_RF_v7_동네별예측.csv', index=False, encoding='utf-8-sig')
cp = OUT + '아파트매매_모델비교.csv'
old = pd.read_csv(cp)
cmp = pd.concat([old[~old['모델'].isin([NAME, '기준(평균으로 찍기)'])], res.iloc[[1]], old[old['모델'] == '기준(평균으로 찍기)']]).round(3)
cmp.to_csv(cp, index=False, encoding='utf-8-sig')
print('\n[모델 비교]'); print(cmp.to_string(index=False))
model = rf().fit(d[X14], y)
joblib.dump(model, OUT + '아파트매매_RF_v7.joblib')

# ---------- MLflow 기록 (docs/model-team-workflow.md): Run 1개, 필수 태그, Registry 등록·champion·MinIO 업로드는 하지 않음 ----------
root = Path(__file__).resolve().parents[2]
ev = root / 'artifacts/apt_sale_model_comparison' / ('v7_' + datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')); ev.mkdir(parents=True)
d.reset_index().to_csv(ev / 'analysis_panel.csv', index=False, encoding='utf-8-sig')                  # 학습에 실제로 쓴 표 (전용면적·연령대 포함)
split = pd.concat([d.reset_index()[['행정동_코드', '자치구', '행정동']].assign(repeat=r, validation_fold=-1) for r in range(REPEAT)], ignore_index=True)
for i, (_, te) in enumerate(cv): split.loc[te + (i // 5) * len(d), 'validation_fold'] = i % 5
split.to_csv(ev / 'cv_split.csv', index=False, encoding='utf-8-sig')                                      # compare_apt_sale_models.py 와 같은 방식 → 분할이 같으면 같은 SHA-256
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
if a.tracking_uri: mlflow.set_tracking_uri(a.tracking_uri)
else: (root / 'artifacts/mlflow_apt_sale').mkdir(parents=True, exist_ok=True); mlflow.set_tracking_uri('sqlite:///' + (root / 'artifacts/mlflow_apt_sale/tracking.db').as_posix())
mlflow.set_experiment(a.experiment)
owner = subprocess.run(['git', 'config', 'user.name'], capture_output=True, text=True).stdout.strip() or 'unknown'
why = (f'v6 랜덤포레스트(7개 + 평균 전용면적)에 상주인구 연령대 비율 6개 추가 실험. 이유: 기존 항목을 고려한 뒤에도 연령 구성과 평당 금액의 연관이 남음(10대 +0.39, 40대 +0.37, 60대 이상 -0.31), '
       f'특정 연령대를 고르지 않고 6개 모두 넣음. 상주인구 연령대 자료 중 이상한 구간(2022Q4~2023Q3) 제외. '
       f'결과: R² {r2(a6).mean():.3f} → {r2(a7).mean():.3f} (차이 {diff.mean():+.4f}, 기준 폭 {band:.4f}), MAE {mae(a6).mean():.1f} → {mae(a7).mean():.1f}, 오차가 더 작은 분할 {wins}/100. 판정: {verdict}. 동네 단위 정적 집계이며 미래 예측·인과 모델이 아님.')
with mlflow.start_run(run_name='apt_sale_random_forest_v7_area_age'):
    mlflow.log_params({'algorithm': 'random_forest', 'version': 'v7', 'features': json.dumps(X14, ensure_ascii=False), 'added_features': json.dumps(AGE_COLS, ensure_ascii=False),
                       'added_feature_source': '서울시 상권분석서비스(상주인구-행정동).csv 연령대별 인구 ÷ 6개 연령대 합, 2022Q4~2023Q3 구간 제외 후 분기 평균', 'compared_with': 'v6 random_forest 8 features',
                       'n_dongs': len(d), 'cv': 'RepeatedKFold(n_splits=5, n_repeats=20, random_state=0)', 'estimator_params': json.dumps(rf().get_params(), default=str)})
    mlflow.log_metrics({'cv_r2_pooled': float(r2(a7).mean()), 'cv_r2_pooled_std': float(r2(a7).std()), 'cv_mae': float(mae(a7).mean()), 'v6_cv_r2_pooled': float(r2(a6).mean()),
                        'v6_cv_mae': float(mae(a6).mean()), 'r2_diff_vs_v6': float(diff.mean()), 'fold_wins_vs_v6': wins, 'age_importance_sum': float(imp[AGE_COLS].sum())})
    mlflow.set_tags({'owner': owner, 'task_id': a.task_id, 'dataset_sha256': sha(ev / 'analysis_panel.csv'), 'split_sha256': sha(ev / 'cv_split.csv'), 'review_status': 'not_reviewed',
                     'selection_status': 'feature_addition', 'caution': 'static_dong_aggregate_not_future_forecast_not_causal', 'mlflow.note.content': why})
    for f in ['성능', '반복별차이', '약점확인', '중요도', '동네별예측']: mlflow.log_artifact(OUT + f'아파트매매_RF_v7_{f}.csv', artifact_path='evidence')
    for f in ev.iterdir(): mlflow.log_artifact(str(f), artifact_path='evidence')
    mlflow.log_artifact(__file__, artifact_path='source')
    mlflow.sklearn.log_model(model, name='model', serialization_format='cloudpickle', input_example=d[X14].head(3))
print('\n저장:', os.path.abspath(OUT), '| MLflow:', mlflow.get_tracking_uri())
