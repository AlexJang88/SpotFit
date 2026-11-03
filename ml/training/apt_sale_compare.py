# 매매 모델(아파트 v8부터, 빌라 v1부터) 버전 비교에 공통으로 쓰는 코드
#   add_area, add_age : 동네 항목 추가 (AptSaleRF_v6.py, AptSaleRF_v7.py 와 같은 방법)
#   compare           : 같은 분할(5분할 × 20번 반복)에서 이전 버전과 새 버전 모델(랜덤포레스트·XGBoost·릿지, 이전 버전 자리에 기준 모델 가능)을
#                       나란히 학습·비교하고, 결과 저장 + MLflow 기록. kind='villa' 이면 빌라 매매
import argparse, hashlib, json, os, subprocess, warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
import joblib, mlflow, mlflow.sklearn, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedKFold
from apt_sale_data import DATA, JUSO, KINDS, OLD, OUT, PROC, Y
warnings.simplefilter('ignore')

REPEAT = 20
AREA = '평균 전용면적(㎡)'
AGES = {'10': '10대', '20': '20대', '30': '30대', '40': '40대', '50': '50대', '60_이상': '60대 이상'}
AGE_COLS = [f'{v} 주민 비율(%)' for v in AGES.values()]
def rf(): return RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=0, n_jobs=-1)      # v1~v8과 같은 설정
def xgb():                                                                                          # v3와 같은 설정 (동네가 적어 얕은 트리·느린 학습)
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0, n_jobs=1, random_state=0)
def ridge():                                                                                        # v2와 같은 설정 (항목 단위를 맞춘 뒤 직선 공식, 규제 세기는 학습 동네 안에서 자동 선택)
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))
def mean():                                                                                         # 기준 모델: 학습 동네의 평균으로 찍기 (아파트 v1~v3의 기준과 같음)
    from sklearn.dummy import DummyRegressor
    return DummyRegressor(strategy='mean')
MODELS = {'random_forest': ('랜덤포레스트', 'RF', rf), 'xgboost': ('XGBoost', 'XGB', xgb), 'ridge': ('릿지', 'Ridge', ridge), 'mean': ('기준(평균으로 찍기)', 'Mean', mean)}   # 이름: (한글 이름, 파일 이름용 약자, 모델 만드는 함수)

def add_area(d, kind='apt'):        # 동네 매매 거래의 평균 전용면적 (apt_sale_data.py 와 같은 거래 기준, kind='villa' 이면 빌라 거래)
    t = pd.read_csv(PROC + KINDS[kind][1], low_memory=False, usecols=['행정동_코드(상권분석)', '전용면적(㎡)'])
    return d.join(t.groupby(t['행정동_코드(상권분석)'].astype(int))['전용면적(㎡)'].mean().rename(AREA))

def add_age(d):         # 상주인구 연령대 비율 6개 (값이 이상한 2022년 4분기~2023년 3분기는 빼고 분기 평균)
    p = pd.read_csv(DATA + '../raw/서울시 상권분석서비스(상주인구-행정동).csv', encoding='cp949')
    raw = [f'연령대_{k}_상주인구_수' for k in AGES]
    tot = p.groupby('기준_년분기_코드')[raw].sum()
    start = pd.Series(tot.index, index=tot.index).groupby((tot.diff().abs().sum(axis=1) > 0).cumsum()).transform('min')   # 값이 바뀐 분기 = 그 자료의 시작 분기
    end = pd.Series(tot.index, index=tot.index).groupby(start).max()
    assert end.get(20224) == 20233, '이상한 구간이 2022년 4분기~2023년 3분기가 아님: 자료가 바뀌었는지 확인 필요'
    p = p[p['기준_년분기_코드'].map(start) != 20224]
    share = p[raw].div(p[raw].sum(axis=1), axis=0).mul(100).set_axis(AGE_COLS, axis=1)
    return d.join(share.groupby(p['행정동_코드']).mean())

def compare(d, old, new, name, run, why, old_alg='random_forest', new_alg='random_forest', kind='apt', allow_missing=False):
    """old, new = ('버전', [X 목록]). 같은 분할에서 두 버전을 나란히 학습해 비교하고 new 결과를 저장·기록. old_alg, new_alg = MODELS의 이름"""
    (ko_o, _, mk_o), (ko_n, ab_n, mk_n) = MODELS[old_alg], MODELS[new_alg]
    pre = KINDS[kind][0] + '매매'                                                                     # 저장 파일 이름 앞부분: 아파트매매 / 빌라매매
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracking-uri'); ap.add_argument('--experiment', default='house-price-forecast-comparison'); ap.add_argument('--task-id', default='unassigned')
    a = ap.parse_args()
    (vo, Xo), (vn, Xn) = old, new
    assert allow_missing or d[Xo + Xn].notna().all().all(), '빈 값이 있는 동네가 있음'                       # allow_missing=True는 빈 값을 처리하는 XGBoost·랜덤포레스트일 때만 (릿지는 불가)
    y = d[Y]
    print(f'동네 {len(d)}곳 | {vo} {len(Xo)}개 항목 vs {vn} {len(Xn)}개 항목 | 뺀 항목 {[c for c in Xo if c not in Xn]} | 더한 항목 {[c for c in Xn if c not in Xo]}')

    # 학습·평가
    cv = list(RepeatedKFold(n_splits=5, n_repeats=REPEAT, random_state=0).split(d))
    P = {vo: np.zeros((REPEAT, len(d))), vn: np.zeros((REPEAT, len(d)))}
    fold_mae = {vo: [], vn: []}
    for i, (tr, te) in enumerate(cv):
        for k, cols, mk in [(vo, Xo, mk_o), (vn, Xn, mk_n)]:
            p = mk().fit(d[cols].iloc[tr], y.iloc[tr]).predict(d[cols].iloc[te])
            P[k][i // 5, te] = p; fold_mae[k].append(np.abs(y.iloc[te] - p).mean())
        if i % 10 == 9: print(f'{i + 1}/{len(cv)} 분할 완료', flush=True)
    r2 = lambda p: 1 - ((y.values - p) ** 2).sum(axis=1) / ((y.values - y.mean()) ** 2).sum()            # 반복마다 R² (v1~v7과 같은 정의)
    mae = lambda p: np.abs(y.values - p).mean(axis=1)
    po, pn = P[vo], P[vn]
    res = pd.DataFrame({'모델': [ko_o if old_alg == 'mean' else f'{ko_o} {vo} ({len(Xo)}개)', name], 'R² 평균': [r2(po).mean(), r2(pn).mean()], 'R² 표준편차': [r2(po).std(), r2(pn).std()],
                        '평균 오차(만원/평)': [mae(po).mean(), mae(pn).mean()]})
    diff = r2(pn) - r2(po)
    wins = int((np.array(fold_mae[vn]) < np.array(fold_mae[vo])).sum())
    band = r2(po).std()
    if diff.mean() > band and wins >= 80: verdict = f'좋아짐 ({vn}이 확실히 나음)'
    elif abs(diff.mean()) <= band: verdict = f'차이가 흔들리는 폭 안 → 뚜렷한 차이 없음 ({vn} 오차가 더 작은 분할 {wins}/100)'
    else: verdict = f'나빠짐 ({vn}이 확실히 못함)' if diff.mean() < 0 else f'R²는 올랐지만 분할 {wins}/100에서만 오차가 작음 (80 미만)'
    print('\n[학습에 쓰지 않은 동네에서 평가, 5분할 × 20번 반복, 같은 분할]'); print(res.round(3).to_string(index=False))
    print(f'\nR² 차이 ({vn} - {vo}): 평균 {diff.mean():+.4f}, 반복 20번 중 {vn}이 더 좋음 {(diff > 0).sum()}번')
    print(f'평균 오차 차이: {mae(pn).mean() - mae(po).mean():+.1f}만원/평, 100개 분할 중 {vn} 오차가 더 작음 {wins}번')
    print(f'기준(흔들리는 폭) = {vo} R² 표준편차 {band:.4f} → 판정: {verdict}')

    top = (y >= y.quantile(.75)).values
    weak = pd.DataFrame({g: {'동네 수': m.sum(), f'{vo} 평균 오차': np.abs(y.values - po.mean(0))[m].mean(), f'{vn} 평균 오차': np.abs(y.values - pn.mean(0))[m].mean(),
                             f'{vo} 치우침(예측-실제)': (po.mean(0) - y.values)[m].mean(), f'{vn} 치우침(예측-실제)': (pn.mean(0) - y.values)[m].mean()}
                         for g, m in {'전체': np.ones(len(d), bool), '비싼 동네(상위 25%)': top}.items()}).T
    print('\n[약점 확인: 동네별 평균 예측 기준, 치우침이 음수면 낮게 예측]'); print(weak.round(1).to_string())

    imp = np.zeros(len(Xn))
    for tr, te in cv[:25]:              # 중요도는 앞 25개 분할 (v1~v7과 같음)
        m = mk_n().fit(d[Xn].iloc[tr], y.iloc[tr])
        imp += permutation_importance(m, d[Xn].iloc[te], y.iloc[te], n_repeats=5, random_state=0).importances_mean / 25
    imp = pd.Series(imp, index=Xn).sort_values(ascending=False)
    print(f'\n[{vn} 항목 중요도: 섞으면 R²가 떨어지는 정도]'); print(imp.round(3).to_string())

    # 저장
    f = OUT + f'{pre}_{ab_n}_{vn}_'
    res.round(3).to_csv(f + '성능.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame({'반복': range(1, REPEAT + 1), f'R² {vo}': r2(po), f'R² {vn}': r2(pn), '차이': diff}).round(4).to_csv(f + '반복별차이.csv', index=False, encoding='utf-8-sig')
    weak.round(1).to_csv(f + '약점확인.csv', encoding='utf-8-sig')
    imp.round(4).rename('중요도(R² 감소량)').to_csv(f + '중요도.csv', encoding='utf-8-sig')
    d.assign(예측=pn.mean(axis=0), 오차=lambda t: t['예측'] - t[Y]).reset_index().round(2).to_csv(f + '동네별예측.csv', index=False, encoding='utf-8-sig')
    cp = OUT + f'{pre}_모델비교.csv'                                                                   # 버전들을 한 표에 모음 (기준 모델 행은 맨 아래)
    c = pd.read_csv(cp) if os.path.exists(cp) else pd.DataFrame(columns=res.columns)
    base = res.iloc[[0]] if old_alg == 'mean' else c[c['모델'] == '기준(평균으로 찍기)']
    cmp = pd.concat([c[~c['모델'].isin([name, '기준(평균으로 찍기)'])], res.iloc[[1]], base]).infer_objects().round(3)     # 새 표일 때 숫자 열이 글자형이 되는 것 방지
    cmp.to_csv(cp, index=False, encoding='utf-8-sig')
    print('\n[모델 비교]'); print(cmp.to_string(index=False))
    model = mk_n().fit(d[Xn], y)
    joblib.dump(model, f[:-1] + '.joblib')

    # MLflow 기록 (docs/model-team-workflow.md): Run 1개, 필수 태그, Registry 등록·champion·MinIO 업로드는 하지 않음
    root = Path(__file__).resolve().parents[2]
    ev = root / f'artifacts/{kind}_sale_model_comparison' / (vn + '_' + datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')); ev.mkdir(parents=True)
    d.reset_index().to_csv(ev / 'analysis_panel.csv', index=False, encoding='utf-8-sig')
    split = pd.concat([d.reset_index()[['행정동_코드', '자치구', '행정동']].assign(repeat=r, validation_fold=-1) for r in range(REPEAT)], ignore_index=True)
    for i, (_, te) in enumerate(cv): split.loc[te + (i // 5) * len(d), 'validation_fold'] = i % 5
    split.to_csv(ev / 'cv_split.csv', index=False, encoding='utf-8-sig')
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if a.tracking_uri: mlflow.set_tracking_uri(a.tracking_uri)
    else: (root / 'artifacts/mlflow_apt_sale').mkdir(parents=True, exist_ok=True); mlflow.set_tracking_uri('sqlite:///' + (root / 'artifacts/mlflow_apt_sale/tracking.db').as_posix())
    mlflow.set_experiment(a.experiment)
    owner = subprocess.run(['git', 'config', 'user.name'], capture_output=True, text=True).stdout.strip() or 'unknown'
    note = (f'{why} 결과: R² {r2(po).mean():.3f} → {r2(pn).mean():.3f} (차이 {diff.mean():+.4f}, 기준 폭 {band:.4f}), MAE {mae(po).mean():.1f} → {mae(pn).mean():.1f}, '
            f'오차가 더 작은 분할 {wins}/100. 판정: {verdict}. 동네 단위 정적 집계이며 미래 예측·인과 모델이 아님.')
    with mlflow.start_run(run_name=run):
        mlflow.log_params({'target': f'{KINDS[kind][0]} 매매 동네 평당 금액(만원, 중앙값)', 'algorithm': new_alg, 'version': vn, 'features': json.dumps(Xn, ensure_ascii=False), 'compared_with': f'{vo} {old_alg} {len(Xo)} features',
                           'removed_features': json.dumps([c for c in Xo if c not in Xn], ensure_ascii=False), 'added_features': json.dumps([c for c in Xn if c not in Xo], ensure_ascii=False),
                           'n_dongs': len(d), 'cv': f'RepeatedKFold(n_splits=5, n_repeats={REPEAT}, random_state=0)', 'estimator_params': json.dumps(mk_n().get_params(), default=str)})
        mlflow.log_metrics({'cv_r2_pooled': float(r2(pn).mean()), 'cv_r2_pooled_std': float(r2(pn).std()), 'cv_mae': float(mae(pn).mean()), f'{vo}_cv_r2_pooled': float(r2(po).mean()),
                            f'{vo}_cv_mae': float(mae(po).mean()), f'r2_diff_vs_{vo}': float(diff.mean()), f'fold_wins_vs_{vo}': wins})
        mlflow.set_tags({'owner': owner, 'task_id': a.task_id, 'dataset_sha256': sha(ev / 'analysis_panel.csv'), 'split_sha256': sha(ev / 'cv_split.csv'), 'review_status': 'not_reviewed',
                         'selection_status': 'baseline_comparison' if old_alg == 'mean' else 'feature_ablation', 'caution': 'static_dong_aggregate_not_future_forecast_not_causal', 'mlflow.note.content': note})
        for s in ['성능', '반복별차이', '약점확인', '중요도', '동네별예측']: mlflow.log_artifact(f + s + '.csv', artifact_path='evidence')
        for p in ev.iterdir(): mlflow.log_artifact(str(p), artifact_path='evidence')
        mlflow.log_artifact(__file__, artifact_path='source')
        mlflow.sklearn.log_model(model, name='model', serialization_format='cloudpickle', input_example=d[Xn].head(3))
    print('\n저장:', os.path.abspath(OUT), '| MLflow:', mlflow.get_tracking_uri())
