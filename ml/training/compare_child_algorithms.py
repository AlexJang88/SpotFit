"""알고리즘·시설 순위 조합을 교차검증하여 선정 근거를 남긴다.

중요: 폴드마다 상위 시설을 다시 선정하므로 조합은 고정 시설명이 아닌
상관순위 위치(예: 1위+3위)라는 학습 절차이다. 최종 시설명도 별도로 저장한다.
"""
from pathlib import Path
from datetime import datetime, timezone, timedelta
from itertools import combinations
from time import perf_counter
import argparse
import hashlib
import json
import html
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

ALGORITHMS=['ridge','tree','random_forest','extra_trees','hist_gradient_boosting']


def build(name):
    """탐색 전에 고정한 설정. 결과를 보고 임의로 파라미터를 변경하지 않는다."""
    estimators={'ridge':lambda:Ridge(alpha=10),
        'tree':lambda:DecisionTreeRegressor(max_depth=4,min_samples_leaf=5,random_state=42),
        'random_forest':lambda:RandomForestRegressor(n_estimators=80,min_samples_leaf=5,random_state=42,n_jobs=-1),
        'extra_trees':lambda:ExtraTreesRegressor(n_estimators=80,min_samples_leaf=5,random_state=42,n_jobs=-1),
        'hist_gradient_boosting':lambda:HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,min_samples_leaf=15,random_state=42)}
    # 결측 대체와 스케일은 각 폴드의 학습 부분에서만 fit한다.
    steps=[SimpleImputer(strategy='median')]
    if name=='ridge': steps.append(StandardScaler())
    return make_pipeline(*steps,estimators[name]())


def top_features(x,y):
    """관측률 70% 이상, 비상수인 시설을 학습 내 절대 Spearman 순위로 정렬."""
    rows=[]
    for c in x:
        if x[c].notna().mean()>=.7 and x[c].nunique()>1:
            r=x[c].corr(y,method='spearman')
            if pd.notna(r): rows.append((c,r))
    rows.sort(key=lambda t:(-abs(t[1]),t[0]))
    if len(rows)<5: raise ValueError('학습 폴드의 적격 시설이 5개 미만입니다.')
    return [c for c,r in rows[:5]]


def scores(y,p):
    return dict(mae=float(mean_absolute_error(y,p)),rmse=float(np.sqrt(mean_squared_error(y,p))),r2=float(r2_score(y,p)))


def main():
    root=Path(__file__).resolve().parents[2]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--split-file',type=Path,default=root/'artifacts/child_feature_subsets/20261023_095054/split.csv')
    p.add_argument('--ages',nargs='+',default=['infant','elementary','middle','high'],choices=['infant','elementary','middle','high'])
    p.add_argument('--tracking-uri')
    p.add_argument('--folds',type=int,default=5)
    p.add_argument('--evaluate-reference-test',action='store_true',help='이미 본 기존 테스트를 참고 평가. 기본은 미평가')
    a=p.parse_args()
    if a.folds<3:p.error('교차검증은 최소 3폴드가 필요합니다.')
    d=pd.read_csv(a.data,dtype={'dong_code':str})
    d=d[d.year==d.year.max()].copy()
    s=pd.read_csv(a.split_file,dtype={'dong_code':str})
    if set(d.dong_code)!=set(s.dong_code) or not d.dong_code.is_unique or not s.dong_code.is_unique:
        raise ValueError('기존 분할과 데이터 동 코드가 일치해야 합니다.')
    if set(s.split)!= {'train','validation','test'}:raise ValueError('분할 라벨 오류')
    d=d.merge(s[['dong_code','split']],on='dong_code',validate='one_to_one').sort_values('dong_code').reset_index(drop=True)
    dev=d[d.split!='test'].reset_index(drop=True);test=d[d.split=='test'].copy()
    facilities=[c for c in d if c.endswith('_수')]
    stamp=datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S_%f')
    out=root/'artifacts/child_algorithm_comparison'/stamp;out.mkdir(parents=True)
    tracking=root/'artifacts/mlflow_child_subsets';tracking.mkdir(parents=True,exist_ok=True)
    mlflow.set_tracking_uri(a.tracking_uri or 'sqlite:///'+(tracking/'tracking.db').as_posix())
    experiment='child-population-algorithm-comparison'
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment,artifact_location=None if a.tracking_uri else (tracking/'comparison_runs').as_uri())
    mlflow.set_experiment(experiment)
    folds=list(KFold(n_splits=a.folds,shuffle=True,random_state=42).split(dev))
    fold_assignment=dev[['dong_code','gu','dong','year']].copy()
    for i,(_,v) in enumerate(folds):fold_assignment.loc[v,'validation_fold']=i
    fold_assignment.to_csv(out/'cv_split.csv',index=False,encoding='utf-8-sig')
    combos=[c for n in range(1,6) for c in combinations(range(5),n)]
    all_summary=[];all_folds=[];decisions=[]
    with mlflow.start_run(run_name='algorithm_comparison_'+stamp) as parent:
        mlflow.log_params({'data_sha256':hashlib.sha256(a.data.read_bytes()).hexdigest(),
            'split_sha256':hashlib.sha256(a.split_file.read_bytes()).hexdigest(),'folds':a.folds,
            'development_n':len(dev),'excluded_test_n':len(test),'selection':'one_standard_error_then_simplicity',
            'min_relative_improvement':.05,'reference_test':a.evaluate_reference_test})
        mlflow.set_tag('mlflow.note.content','시설 순위 조합×5종 알고리즘 교차검증. 테스트는 기존에 열람한 참고자료이며 새 독립 검증이 아님. 선정 후보도 운영 승인 아님.')
        mlflow.log_artifact(__file__,artifact_path='source')
        for age in a.ages:
            y=dev['children_'+age];x=dev[facilities]
            if y.isna().any():raise ValueError('타깃 결측')
            top_by_fold=[top_features(x.iloc[t],y.iloc[t]) for t,v in folds]
            baseline=[]
            for fold,(t,v) in enumerate(folds):
                score=scores(y.iloc[v],np.repeat(y.iloc[t].mean(),len(v)))
                baseline.append(score['mae'])
                all_folds.append(dict(age=age,algorithm='dummy_mean',combo='none',fold=fold,features='none',fit_seconds=0,predict_seconds=0,**score))
            summaries=[]
            for alg in ALGORITHMS:
                for combo in combos:
                    cid='-'.join(str(i+1) for i in combo);records=[]
                    for fold,(t,v) in enumerate(folds):
                        cols=[top_by_fold[fold][i] for i in combo]
                        m=build(alg);start=perf_counter();m.fit(x.iloc[t][cols],y.iloc[t]);fit=perf_counter()-start
                        start=perf_counter();pred=m.predict(x.iloc[v][cols]);predict=perf_counter()-start
                        row=dict(age=age,algorithm=alg,combo=cid,fold=fold,features=' | '.join(cols),fit_seconds=fit,predict_seconds=predict,**scores(y.iloc[v],pred))
                        records.append(row);all_folds.append(row)
                    r=pd.DataFrame(records)
                    summary=dict(age=age,algorithm=alg,combo=cid,feature_count=len(combo),cv_mae=float(r.mae.mean()),
                        cv_mae_std=float(r.mae.std(ddof=1)),cv_rmse=float(r.rmse.mean()),cv_r2=float(r.r2.mean()),
                        fit_seconds=float(r.fit_seconds.mean()),predict_seconds=float(r.predict_seconds.mean()),
                        baseline_cv_mae=float(np.mean(baseline)),fold_wins=int((r.mae.to_numpy()<baseline).sum()))
                    summary['relative_improvement']=1-summary['cv_mae']/summary['baseline_cv_mae']
                    with mlflow.start_run(run_name=f'{age}_{alg}_rank{cid}',nested=True) as child:
                        summary['run_id']=child.info.run_id
                        mlflow.log_params({'age':age,'algorithm':alg,'rank_subset':cid,'feature_count':len(combo),
                            'estimator_params':json.dumps(build(alg).steps[-1][1].get_params(),default=str)})
                        mlflow.log_metrics({k:v for k,v in summary.items() if isinstance(v,(float,int))})
                        mlflow.log_dict(records,'fold_results.json')
                        mlflow.set_tag('mlflow.note.content','순위 위치 조합이므로 실제 시설은 폴드별로 다를 수 있음. fold_results.json 참조. 동일 검증에서 다수 후보를 탐색한 성능은 낙관적일 수 있음.')
                    summaries.append(summary)
                print(age,alg,'완료',flush=True)
            table=pd.DataFrame(summaries)
            best=table.loc[table.cv_mae.idxmin()]
            # 최소 MAE의 1 표준오차 이내 후보 중 단순한 모델·적은 특징을 우선한다.
            # 통계적 동등성 검정이 아닌 사전 고정한 실무 선택 규칙이다.
            threshold=best.cv_mae+best.cv_mae_std/np.sqrt(a.folds)
            near=table[table.cv_mae<=threshold].copy()
            near['complexity']=near.algorithm.map({n:i for i,n in enumerate(ALGORITHMS)})
            chosen=near.sort_values(['complexity','feature_count','cv_mae']).iloc[0]
            eligible=chosen.relative_improvement>=.05 and chosen.fold_wins>=int(np.ceil(.8*a.folds))
            reason=(f'최저 CV MAE {best.cv_mae:.2f}, 허용 상한 {threshold:.2f}(1-SE). '
                f'범위 내 단순성 우선으로 {chosen.algorithm}, 순위 {chosen.combo} 선정. '
                f'MAE {chosen.cv_mae:.2f}±{chosen.cv_mae_std:.2f}, 기준 대비 {chosen.relative_improvement:.1%} 개선, '
                f'{chosen.fold_wins}/{a.folds} 폴드에서 기준 초과 성능. '
                +('탐색 후보 유지.' if eligible else '5% 개선·80% 폴드 승리 조건 미충족: 공식 선정 보류.')
                +' 독립 최종 검증 전이므로 운영 배포는 항상 보류.')
            table['decision']=np.where(table.run_id==chosen.run_id,'exploratory_candidate' if eligible else 'selection_deferred',
                np.where(table.cv_mae>threshold,'outside_one_se_band','less_simple_within_band'))
            table.to_csv(out/(age+'_comparison.csv'),index=False,encoding='utf-8-sig');all_summary.extend(table.to_dict('records'))
            cols=[top_features(x,y)[int(i)-1] for i in chosen.combo.split('-')]
            model=build(chosen.algorithm);model.fit(x[cols],y)
            joblib.dump({'model':model,'features':cols,'target':'children_'+age},out/(age+'_candidate.joblib'))
            decision=dict(age=age,algorithm=chosen.algorithm,features=cols,run_id=chosen.run_id,eligible=bool(eligible),reason=reason)
            with mlflow.start_run(run_id=chosen.run_id,nested=True):
                mlflow.set_tags({'selection_status':'exploratory_candidate' if eligible else 'selection_deferred','mlflow.note.content':reason})
                mlflow.log_artifact(str(out/(age+'_candidate.joblib')),artifact_path='candidate')
                # 모델 표준 저장까지 하되 Registry 버전 생성과 배포는 자동으로 하지 않는다.
                mlflow.sklearn.log_model(model,name='model',serialization_format='cloudpickle',input_example=x[cols].head(3))
                if a.evaluate_reference_test:
                    pred=model.predict(test[cols]);metrics=scores(test['children_'+age],pred)
                    mlflow.log_metrics({'reference_test_'+k:v for k,v in metrics.items()})
                    decision['reference_test']=metrics
                    prediction=test[['dong_code','dong','gu','year']].copy()
                    prediction['actual']=test['children_'+age];prediction['predicted']=pred
                    prediction.to_csv(out/(age+'_reference_predictions.csv'),index=False,encoding='utf-8-sig')
            decisions.append(decision)
        pd.DataFrame(all_summary).to_csv(out/'comparison.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(all_folds).to_csv(out/'fold_results.csv',index=False,encoding='utf-8-sig')
        (out/'selection_reasons.json').write_text(json.dumps(decisions,ensure_ascii=False,indent=2),encoding='utf-8')
        report='<meta charset="utf-8"><h1>알고리즘·시설 조합 선정 근거</h1><p>개발 자료 내 교차검증 탐색 결과. 미래 예측·독립 최종 평가 아님. 1-SE 규칙은 유의성 검정이 아니며 5%/80% 문턱은 조정 가능한 운영 기준입니다.</p>'
        report+=''.join('<h2>'+v['age']+'</h2><p>'+html.escape(v['reason'])+'</p><p>'+html.escape(str(v['features']))+'</p>' for v in decisions)
        report+=pd.DataFrame(all_summary).sort_values(['age','cv_mae']).to_html(index=False)
        (out/'report.html').write_text(report,encoding='utf-8')
        for path in out.iterdir():
            if path.suffix!='.joblib':mlflow.log_artifact(str(path),artifact_path='evidence')
    print('OUTPUT',out,flush=True)


if __name__=='__main__':main()
