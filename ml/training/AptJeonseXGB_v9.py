# 모델 버전 9: v7(XGBoost, 12개 항목)에 '재건축 구역 수'를 더하면 좋아지는지 확인 (v7 = 12개 vs v9 = 13개, 둘 다 XGBoost, 아파트 전세 평당 보증금)
#   이유: 재건축 구역이 많은 동네는 개발 기대 등으로 가격이 다를 수 있음. 아파트 매매 v19에서 같은 항목을 시험했고(+0.003, 효과 없음) 전세에서도 같은지 확인
#   재건축 구역 수 = 서울시 도시계획사업(서울플랜+) 정비사업 현황 중 재건축(공동·단독) 구역을 행정동별로 센 값, 구역이 없는 동네는 0
#     (data/processed/부동산(전처리)/도시계획사업_행정동별_아파트.csv, ml/etl/AptUrbanPlanDong.py. 아파트 매매 v19와 같은 정의)
#   X (v7 12개): 층수, 거래일, 완공연도, 직장인구 수, 생활방범 CCTV 수, 1인가구 비율, 평균 임대면적, 연령대 비율 5개(10·20·30·40·60대 이상) + 재건축 구역 수   y: 평당 보증금 (전세 거래의 동네 중앙값, 만원)
#   연령대 비율 = 서울시 상권분석서비스(상주인구-행정동) 연령대별 상주인구 ÷ 6개 연령대 합계 × 100, 2022년 4분기~2023년 3분기를 뺀 분기 평균 (apt_sale_compare.add_age)
#   평균 임대면적 = 전세 거래의 임대면적(㎡) 동네 평균 (전월세 파일에는 전용면적 컬럼이 없고 임대면적만 있음)
#   XGBoost 설정은 v2·v4~v8과 같음. 같은 동네 409곳, 같은 분할(5분할 × 20번 반복). 데이터 준비는 apt_jeonse_data.py
#   판단: 차이가 반복마다 흔들리는 폭(v7 R² 표준편차) 안이면 "더해도 성능 차이 없음"
# 실행: python AptJeonseXGB_v9.py                                          (내 PC의 로컬 MLflow에만 기록, 시험용)
#       python AptJeonseXGB_v9.py --tracking-uri http://192.168.0.3:5000   (팀 공유 MLflow 서버에 기록, Experiment: house-price-jeonse-comparison)
import argparse, hashlib, json, os, subprocess, warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
import joblib, mlflow, mlflow.sklearn, numpy as np, pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedKFold
from apt_jeonse_data import AREA, DATA, OUT, X_COLS, Y, load
from apt_sale_compare import AGE_COLS, REPEAT, add_age, xgb
warnings.simplefilter('ignore')

NEW = '재건축 구역 수'
ap = argparse.ArgumentParser()
ap.add_argument('--tracking-uri'); ap.add_argument('--experiment', default='house-price-jeonse-comparison'); ap.add_argument('--task-id', default='unassigned')
a = ap.parse_args()
d = add_age(load())
urban = pd.read_csv(DATA + '부동산(전처리)/도시계획사업_행정동별_아파트.csv', encoding='utf-8-sig').set_index('행정동_코드')
d[NEW] = urban[NEW].reindex(d.index).fillna(0)                                              # 구역이 없는 동네는 0 (아파트 매매 v19와 같음)
print(f'{NEW}: 파일 {len(urban)}곳 중 모델 동네에 있는 곳 {int(d.index.isin(urban.index).sum())}곳, 구역이 1개 이상인 동네 {int((d[NEW] > 0).sum())}곳 / 전체 {len(d)}곳, 최대 {int(d[NEW].max())}개, 평당 보증금과 순위상관 {d[[NEW, Y]].corr(method="spearman").iloc[0, 1]:+.2f}')
X7 = [c for c in X_COLS + [AREA] if c != '유동인구 수'] + [c for c in AGE_COLS if c != '50대 주민 비율(%)']; X9 = X7 + [NEW]
assert d[X9].notna().all().all(), '빈 값이 있는 동네가 있음'
y = d[Y]

# 학습·평가: 같은 분할에서 v7(12개)과 v9(13개)를 나란히 학습
cv = list(RepeatedKFold(n_splits=5, n_repeats=REPEAT, random_state=0).split(d))
P = {'v7': np.zeros((REPEAT, len(d))), 'v9': np.zeros((REPEAT, len(d)))}; fold_mae = {'v7': [], 'v9': []}; base = np.zeros((REPEAT, len(d)))
for i, (tr, te) in enumerate(cv):
    base[i // 5, te] = y.iloc[tr].mean()
    for k, cols in [('v7', X7), ('v9', X9)]:
        p = xgb().fit(d[cols].iloc[tr], y.iloc[tr]).predict(d[cols].iloc[te]); P[k][i // 5, te] = p; fold_mae[k].append(np.abs(y.iloc[te] - p).mean())
    if i % 10 == 9: print(f'{i + 1}/{len(cv)} 분할 완료', flush=True)
r2 = lambda p: 1 - ((y.values - p) ** 2).sum(axis=1) / ((y.values - y.mean()) ** 2).sum()
mae = lambda p: np.abs(y.values - p).mean(axis=1)
p7, p9 = P['v7'], P['v9']
res = pd.DataFrame({'모델': ['XGBoost v7 (12개)', 'XGBoost v9 (v7에 재건축 구역 수 추가, 13개)', '기준(평균으로 찍기)'], 'R² 평균': [r2(p7).mean(), r2(p9).mean(), r2(base).mean()],
                    'R² 표준편차': [r2(p7).std(), r2(p9).std(), r2(base).std()], '평균 오차(만원/평)': [mae(p7).mean(), mae(p9).mean(), mae(base).mean()]})
diff = r2(p9) - r2(p7); wins = int((np.array(fold_mae['v9']) < np.array(fold_mae['v7'])).sum()); band = r2(p7).std()
if diff.mean() > band and wins >= 80: verdict = '좋아짐 (v9가 확실히 나음)'
elif abs(diff.mean()) <= band: verdict = f'차이가 흔들리는 폭 안 → 뚜렷한 차이 없음 (v9 오차가 더 작은 분할 {wins}/100)'
else: verdict = 'v9가 확실히 못함' if diff.mean() < 0 else f'R²는 올랐지만 분할 {wins}/100에서만 오차가 작음 (80 미만)'
print('\n[학습에 쓰지 않은 동네에서 평가, 5분할 × 20번 반복, 같은 분할]'); print(res.round(3).to_string(index=False))
print(f'\nR² 차이 (v9 - v7): 평균 {diff.mean():+.4f}, 반복 20번 중 v9가 더 좋음 {(diff > 0).sum()}번')
print(f'평균 오차 차이: {mae(p9).mean() - mae(p7).mean():+.1f}만원/평, 100개 분할 중 v9 오차가 더 작음 {wins}번')
print(f'기준(흔들리는 폭) = v7 R² 표준편차 {band:.4f} → 판정: {verdict}')
imp = np.zeros(len(X9))
for tr, te in cv[:25]:              # 중요도는 앞 25개 분할 (매매와 같음)
    m = xgb().fit(d[X9].iloc[tr], y.iloc[tr])
    imp += permutation_importance(m, d[X9].iloc[te], y.iloc[te], n_repeats=5, random_state=0).importances_mean / 25
imp = pd.Series(imp, index=X9).sort_values(ascending=False)
print('\n[v9 항목 중요도: 섞으면 R²가 떨어지는 정도]'); print(imp.round(3).to_string())

# 저장
os.makedirs(OUT, exist_ok=True)
f = OUT + '아파트전세_XGB_v9_'
res.round(3).to_csv(f + '성능.csv', index=False, encoding='utf-8-sig')
imp.round(4).rename('중요도(R² 감소량)').to_csv(f + '중요도.csv', encoding='utf-8-sig')
d.assign(예측=p9.mean(axis=0), 오차=lambda t: t['예측'] - t[Y]).reset_index().round(1).to_csv(f + '동네별예측.csv', index=False, encoding='utf-8-sig')
model = xgb().fit(d[X9], y)
joblib.dump(model, OUT + '아파트전세_XGB_v9.joblib')
print('\n저장:', os.path.abspath(OUT))

# MLflow 기록 (docs/model-team-workflow.md): Run 1개, 필수 태그, Registry 등록·champion·MinIO 업로드는 하지 않음
root = Path(__file__).resolve().parents[2]
ev = root / 'artifacts/apt_jeonse_model_comparison' / ('v9_' + datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')); ev.mkdir(parents=True)
d.reset_index().to_csv(ev / 'analysis_panel.csv', index=False, encoding='utf-8-sig')
split = pd.concat([d.reset_index()[['행정동_코드', '자치구', '행정동']].assign(repeat=r, validation_fold=-1) for r in range(REPEAT)], ignore_index=True)
for i, (_, te) in enumerate(cv): split.loc[te + (i // 5) * len(d), 'validation_fold'] = i % 5
split.to_csv(ev / 'cv_split.csv', index=False, encoding='utf-8-sig')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
if a.tracking_uri: mlflow.set_tracking_uri(a.tracking_uri)
else: (root / 'artifacts/mlflow_apt_sale').mkdir(parents=True, exist_ok=True); mlflow.set_tracking_uri('sqlite:///' + (root / 'artifacts/mlflow_apt_sale/tracking.db').as_posix())
mlflow.set_experiment(a.experiment)
owner = subprocess.run(['git', 'config', 'user.name'], capture_output=True, text=True).stdout.strip() or 'unknown'
note = (f'아파트 전세(월세 제외) 동네 평당 보증금(중앙값): v7(XGBoost, X 12개)에 재건축 구역 수(서울시 도시계획사업(서울플랜+) 정비사업 현황의 재건축(공동·단독) 구역을 행정동별로 센 값, 구역 없는 동네는 0, 아파트 매매 v19와 같은 정의)를 더해 비교. '
        f'결과: R² {r2(p7).mean():.3f} → {r2(p9).mean():.3f} (차이 {diff.mean():+.4f}, 기준 폭 {band:.4f}), MAE {mae(p7).mean():.1f} → {mae(p9).mean():.1f}, 오차가 더 작은 분할 {wins}/100. 판정: {verdict}. '
        f'평균 임대면적은 전세 거래 임대면적의 동네 평균(전월세 파일에 전용면적이 없어 임대면적 사용). 생활방범 CCTV는 설치연도를 아는 것만(연도 없는 11개 구는 0). 동네 단위 정적 집계이며 미래 예측·인과 모델이 아님.')
with mlflow.start_run(run_name='apt_jeonse_xgboost_v9_add_rebuild_zones'):
    mlflow.log_params({'target': '아파트 전세 동네 평당 보증금(만원, 중앙값)', 'algorithm': 'xgboost', 'version': 'v9', 'features': json.dumps(X9, ensure_ascii=False), 'compared_with': 'v7 xgboost 12 features',
                       'removed_features': '[]', 'added_features': json.dumps([NEW], ensure_ascii=False), 'n_dongs': len(d), 'cv': f'RepeatedKFold(n_splits=5, n_repeats={REPEAT}, random_state=0)',
                       'estimator_params': json.dumps(xgb().get_params(), default=str)})
    mlflow.log_metrics({'cv_r2_pooled': float(r2(p9).mean()), 'cv_r2_pooled_std': float(r2(p9).std()), 'cv_mae': float(mae(p9).mean()), 'v7_cv_r2_pooled': float(r2(p7).mean()),
                        'v7_cv_mae': float(mae(p7).mean()), 'r2_diff_vs_v7': float(diff.mean()), 'fold_wins_vs_v7': wins})
    mlflow.set_tags({'owner': owner, 'task_id': a.task_id, 'dataset_sha256': sha(ev / 'analysis_panel.csv'), 'split_sha256': sha(ev / 'cv_split.csv'), 'review_status': 'not_reviewed',
                     'selection_status': 'feature_ablation', 'caution': 'static_dong_aggregate_not_future_forecast_not_causal', 'mlflow.note.content': note})
    for s in ['성능', '중요도', '동네별예측']: mlflow.log_artifact(f + s + '.csv', artifact_path='evidence')
    for p in ev.iterdir(): mlflow.log_artifact(str(p), artifact_path='evidence')
    mlflow.log_artifact(__file__, artifact_path='source')
    mlflow.sklearn.log_model(model, name='model', serialization_format='cloudpickle', input_example=d[X9].head(3))
print('MLflow:', mlflow.get_tracking_uri(), '| Experiment:', a.experiment)
