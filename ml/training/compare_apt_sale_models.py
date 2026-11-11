"""아파트 매매 평당 금액 알고리즘 비교 → MLflow 기록 (docs/model-team-workflow.md 규칙)

같은 데이터(행정동 408곳)·같은 분할(5분할 × 20번 반복)로 후보(ridge, random_forest, xgboost)와 기준 모델(평균)을 비교하고,
선정 규칙(최저 MAE의 1 표준오차 안에서 단순한 모델 우선, 기준 대비 5% 이상 개선, 80% 이상 폴드에서 기준보다 좋음)을 적용한다.
부모 Run 1개 + 후보 Run으로 기록한다. Registry 등록·champion 별칭·서비스 MinIO 업로드는 하지 않는다(검증·릴리스 담당 절차).
"""
import argparse, hashlib, html, json, subprocess, warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
import mlflow, mlflow.sklearn, numpy as np, pandas as pd, sklearn, xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from apt_sale_data import OUT, PROC, X_COLS, Y, MIN_N, load
warnings.simplefilter('ignore')

ALGORITHMS = ['ridge', 'random_forest', 'xgboost']        # 단순한 순서 (선정 규칙에서 단순한 모델을 우선)
SPLITS, REPEAT, SEED = 5, 20, 0                           # 5분할 × 20번 반복 = 100폴드, 난수 seed
LABEL = {'ridge': '릿지(직선 공식)', 'random_forest': '랜덤포레스트', 'xgboost': 'XGBoost(부스팅)'}

def build(name):        # 알고리즘별 모델 (AptSaleRF_v1.py, AptSaleRidge_v2.py, AptSaleXGB_v3.py 와 같은 설정)
    if name == 'ridge': return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))
    if name == 'random_forest': return RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=0, n_jobs=-1)
    return XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0, n_jobs=1, random_state=0)

def scores(y, p): return dict(mae=float(mean_absolute_error(y, p)), rmse=float(np.sqrt(mean_squared_error(y, p))), r2=float(r2_score(y, p)))
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tracking-uri', help='공유 MLflow 서버 주소. 없으면 내 PC의 로컬 SQLite에만 기록(시험용)')
    ap.add_argument('--experiment', default='house-price-forecast-comparison')
    ap.add_argument('--owner', default=subprocess.run(['git', 'config', 'user.name'], capture_output=True, text=True).stdout.strip() or 'unknown')
    ap.add_argument('--task-id', default='unassigned', help='작업 번호 (문서 8장 필수 태그)')
    a = ap.parse_args()

    d = load()                                                              # 행정동 1곳 = 1행 (408곳)
    X, y = d[X_COLS], d[Y]
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')
    out = root / 'artifacts/apt_sale_model_comparison' / stamp; out.mkdir(parents=True)
    d.reset_index().to_csv(out / 'analysis_panel.csv', index=False, encoding='utf-8-sig')            # 학습에 실제로 쓴 표 (이 파일의 SHA-256이 dataset_sha256)

    cv = RepeatedKFold(n_splits=SPLITS, n_repeats=REPEAT, random_state=SEED)
    folds = list(cv.split(X))
    split = pd.concat([d.reset_index()[['행정동_코드', '자치구', '행정동']].assign(repeat=r, validation_fold=-1) for r in range(REPEAT)], ignore_index=True)
    for i, (_, te) in enumerate(folds): split.loc[te + (i // SPLITS) * len(d), 'validation_fold'] = i % SPLITS            # 반복마다 동네가 어느 폴드에서 평가됐는지
    split.to_csv(out / 'cv_split.csv', index=False, encoding='utf-8-sig')
    dataset_sha, split_sha = sha(out / 'analysis_panel.csv'), sha(out / 'cv_split.csv')

    base_fold = [scores(y.iloc[te], np.repeat(y.iloc[tr].mean(), len(te))) for tr, te in folds]            # 기준 모델: 학습 동네 평균으로 찍기
    base_mae = np.array([b['mae'] for b in base_fold])
    rows, all_folds, models = [], [], {}
    for alg in ALGORITHMS:
        P = np.zeros((REPEAT, len(d))); rec = []
        for i, (tr, te) in enumerate(folds):
            m = build(alg); t0 = perf_counter(); m.fit(X.iloc[tr], y.iloc[tr]); fit = perf_counter() - t0
            t0 = perf_counter(); p = m.predict(X.iloc[te]); pred = perf_counter() - t0
            P[i // SPLITS, te] = p
            rec.append(dict(algorithm=alg, repeat=i // SPLITS, fold=i % SPLITS, fit_seconds=fit, predict_seconds=pred, **scores(y.iloc[te], p)))
        r = pd.DataFrame(rec); all_folds += rec
        pooled = float(np.mean(1 - ((y.values - P) ** 2).sum(axis=1) / ((y.values - y.mean()) ** 2).sum()))     # 반복마다 전체 동네의 R²를 구해 평균 (AptSaleRF_v1.py 등과 같은 정의)
        rows.append(dict(algorithm=alg, cv_mae=float(r.mae.mean()), cv_mae_std=float(r.mae.std(ddof=1)), cv_rmse=float(r.rmse.mean()), cv_r2=float(r.r2.mean()), cv_r2_pooled=pooled,
                         fit_seconds=float(r.fit_seconds.mean()), predict_seconds=float(r.predict_seconds.mean()), baseline_cv_mae=float(base_mae.mean()),
                         fold_wins=int((r.mae.to_numpy() < base_mae).sum()), n_folds=len(folds)))
        rows[-1]['relative_improvement'] = 1 - rows[-1]['cv_mae'] / rows[-1]['baseline_cv_mae']
        models[alg] = build(alg).fit(X, y)                                                          # 전체 데이터로 학습한 후보 모델 (Run 아티팩트로 저장)
        print(f'{alg:14s} MAE {rows[-1]["cv_mae"]:7.1f} ±{rows[-1]["cv_mae_std"]:.1f}  R²(폴드평균) {rows[-1]["cv_r2"]:.3f}  R²(반복평균) {pooled:.3f}  기준 대비 {rows[-1]["relative_improvement"]:.1%}  승리 {rows[-1]["fold_wins"]}/{len(folds)}', flush=True)
    table = pd.DataFrame(rows)

    # 선정 규칙(문서 7장, compare_child_algorithms.py 와 같은 공식): 최저 MAE + 1 표준오차(MAE 표준편차 / √(분할 수)) 안에서 단순한 알고리즘 우선
    best = table.loc[table.cv_mae.idxmin()]
    threshold = best.cv_mae + best.cv_mae_std / np.sqrt(SPLITS)
    near = table[table.cv_mae <= threshold].copy(); near['complexity'] = near.algorithm.map({n: i for i, n in enumerate(ALGORITHMS)})
    chosen = near.sort_values(['complexity', 'cv_mae']).iloc[0]
    eligible = chosen.relative_improvement >= .05 and chosen.fold_wins >= int(np.ceil(.8 * len(folds)))
    table['decision'] = np.where(table.algorithm == chosen.algorithm, 'exploratory_candidate' if eligible else 'selection_deferred', np.where(table.cv_mae > threshold, 'outside_one_se_band', 'less_simple_within_band'))
    reason = (f'최저 CV MAE {best.cv_mae:.1f}({best.algorithm}), 허용 상한 {threshold:.1f}(1-SE). 범위 내 단순성 우선으로 {chosen.algorithm} 선정. '
              f'MAE {chosen.cv_mae:.1f}±{chosen.cv_mae_std:.1f}, 기준 대비 {chosen.relative_improvement:.1%} 개선, {chosen.fold_wins}/{len(folds)} 폴드에서 기준보다 좋음. '
              + ('탐색 후보 유지.' if eligible else '5% 개선·80% 폴드 승리 조건 미충족: 공식 선정 보류.') + ' 독립 최종 검증 전이므로 운영 배포는 보류.')
    print('\n선정:', reason)
    table.to_csv(out / 'comparison.csv', index=False, encoding='utf-8-sig'); pd.DataFrame(all_folds).to_csv(out / 'fold_results.csv', index=False, encoding='utf-8-sig')
    (out / 'selection_reasons.json').write_text(json.dumps(dict(target='아파트 매매 동네 평당 금액(만원), 행정동 1곳=1행', chosen=chosen.algorithm, eligible=bool(eligible), reason=reason,
                                                                  candidates=table.to_dict('records')), ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    (out / 'report.html').write_text('<meta charset="utf-8"><h1>아파트 매매 평당 금액 알고리즘 비교</h1><p>' + html.escape(reason) + '</p><p>동네 단위 정적 집계 기반 교차검증이며 미래 예측·인과 모델이 아닙니다. 1-SE 규칙은 유의성 검정이 아니고 5%/80% 문턱은 조정 가능한 운영 기준입니다.</p>'
                                     + table.round(3).to_html(index=False), encoding='utf-8')

    # ---------- MLflow 기록: 부모 Run(실험 조건) + 후보 Run(알고리즘별 결과·모델) ----------
    tracking = root / 'artifacts/mlflow_apt_sale'
    if a.tracking_uri: mlflow.set_tracking_uri(a.tracking_uri)
    else: tracking.mkdir(parents=True, exist_ok=True); mlflow.set_tracking_uri('sqlite:///' + (tracking / 'tracking.db').as_posix())
    if mlflow.get_experiment_by_name(a.experiment) is None: mlflow.create_experiment(a.experiment, artifact_location=None if a.tracking_uri else (tracking / 'runs').as_uri())
    mlflow.set_experiment(a.experiment)
    common = dict(owner=a.owner, task_id=a.task_id, dataset_sha256=dataset_sha, split_sha256=split_sha, review_status='not_reviewed',
                  caution='static_dong_aggregate_not_future_forecast_not_causal')
    note = ('아파트 매매 동네 평당 금액(행정동 1곳=1행, 동네 중앙값) 알고리즘 비교. 동네 단위 정적 집계이며 미래 예측·인과 모델이 아님. 동일 검증에서 여러 후보를 비교한 성능은 낙관적일 수 있고, '
            '독립 테스트 데이터가 없어 선정 후보도 운영 승인이 아님. 직장인구·1인가구가 빈 11개 동네는 모든 후보 공통으로 제외(' + str(len(d)) + '곳 사용). 빈 값은 채우지 않음.')
    with mlflow.start_run(run_name='apt_sale_algorithm_comparison_' + stamp) as parent:
        mlflow.log_params({'data_apt_sale': 'raw/부동산/아파트_매매.csv', 'data_apt_sale_sha256': sha(PROC + '아파트_매매.csv'),
                           'data_dong_features': '평당금액_행정동표.csv', 'data_dong_features_sha256': sha(OUT + '../../raw/부동산/평당금액_행정동표.csv'), 'dataset_sha256': dataset_sha, 'split_sha256': split_sha,
                           'reference_period': '거래 2021-01~2026-09 (상권분석 자료는 2021년 1분기~2026년 2분기 평균)', 'n_dongs': len(d), 'min_deals_per_dong': MIN_N, 'cv': f'RepeatedKFold(n_splits={SPLITS}, n_repeats={REPEAT})',
                           'seed': SEED, 'n_folds': len(folds), 'target': '평당 금액(만원) = 거래금액 ÷ (전용면적 ÷ 3.3058), 동네 중앙값', 'features': json.dumps(X_COLS, ensure_ascii=False),
                           'missing_policy': '빈 값은 채우지 않음. 직장인구·1인가구가 빈 행정동 11곳은 모든 후보에서 제외(릿지가 빈 값을 처리하지 못하고 후보를 같은 408곳으로 비교하기 위해). 랜덤포레스트·XGBoost는 빈 값을 처리할 수 있음', 'candidates': json.dumps(ALGORITHMS), 'baseline': 'dummy_mean(학습 동네 평균)',
                           'selection': 'one_standard_error_then_simplicity', 'min_relative_improvement': .05, 'min_fold_win_rate': .8,
                           'package_versions': json.dumps(dict(mlflow=mlflow.__version__, sklearn=sklearn.__version__, xgboost=xgboost.__version__, pandas=pd.__version__, numpy=np.__version__))})
        mlflow.log_metrics({'baseline_cv_mae': float(base_mae.mean()), 'baseline_cv_mae_std': float(base_mae.std(ddof=1)), 'baseline_cv_r2': float(np.mean([b['r2'] for b in base_fold]))})
        mlflow.set_tags({**common, 'selection_status': 'exploratory_candidate' if eligible else 'selection_deferred', 'mlflow.note.content': note + ' 선정: ' + reason})
        mlflow.log_artifact(__file__, artifact_path='source')
        for _, row in table.iterrows():
            alg = row.algorithm
            with mlflow.start_run(run_name=f'apt_sale_{alg}', nested=True) as child:
                mlflow.log_params({'algorithm': alg, 'algorithm_ko': LABEL[alg], 'estimator_params': json.dumps(build(alg).steps[-1][1].get_params() if alg == 'ridge' else build(alg).get_params(), default=str),
                                   'feature_count': len(X_COLS), 'features': json.dumps(X_COLS, ensure_ascii=False)})
                mlflow.log_metrics({k: float(row[k]) for k in ['cv_mae', 'cv_mae_std', 'cv_rmse', 'cv_r2', 'cv_r2_pooled', 'fit_seconds', 'predict_seconds', 'baseline_cv_mae', 'relative_improvement']} | {'fold_wins': int(row.fold_wins), 'n_folds': int(row.n_folds)})
                mlflow.log_dict([r for r in all_folds if r['algorithm'] == alg], 'fold_results.json')
                why = (f'{LABEL[alg]}: MAE {row.cv_mae:.1f}±{row.cv_mae_std:.1f}, 기준 대비 {row.relative_improvement:.1%} 개선, {int(row.fold_wins)}/{len(folds)} 폴드에서 기준보다 좋음. 판정 {row.decision}. ' + ('' if alg == chosen.algorithm else f'선정 알고리즘은 {chosen.algorithm}. ') + '한계: ' + note)
                mlflow.set_tags({**common, 'selection_status': row.decision if alg != chosen.algorithm else ('exploratory_candidate' if eligible else 'selection_deferred'), 'decision': row.decision, 'mlflow.note.content': why})
                mlflow.sklearn.log_model(models[alg], name='model', serialization_format='cloudpickle', input_example=X.head(3))     # Registry 등록·배포는 하지 않음
        for path in out.iterdir(): mlflow.log_artifact(str(path), artifact_path='evidence')
        print('\nMLflow 부모 Run:', parent.info.run_id, '| 실험:', a.experiment, '| 서버:', mlflow.get_tracking_uri())
    print('로컬 증빙 파일:', out)

if __name__ == '__main__':
    main()
