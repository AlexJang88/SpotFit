"""Leakage-controlled exhaustive GROUP subset forecasting benchmark.

Run with --smoke to test one subset; full runs resume completed subsets.
Model artifacts are plain dictionaries (no __main__ class pickle dependency).
"""
from __future__ import annotations
import os
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable]='1'
import argparse
import json
import time
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.stats import spearmanr, rankdata
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/child_crime_borough_v2'
MODELS=ROOT/'models/experiments/child_crime_borough_v2'
CORE=['y','log_crime','log_child']
CV_YEARS=[2022,2023,2024]
HOLDOUT=2025

def metrics(y,p,years):
    p=np.asarray(p);y=np.asarray(y)
    raw=np.expm1(y);pr=np.expm1(np.clip(p,0,20))
    scores={'RMSE':float(np.sqrt(mean_squared_error(y,p))),
            'MAE':float(mean_absolute_error(y,p)), 'R2':float(r2_score(y,p)),
            'raw_RMSE':float(np.sqrt(mean_squared_error(raw,pr))),
            'raw_MAE':float(mean_absolute_error(raw,pr)),
            'raw_MAPE_pct':float(np.mean(np.abs(pr-raw)/raw)*100)}
    rank_corr=[];grade=[];rank_err=[];top=[]
    for year in np.unique(years):
        actual=y[years==year];pred=p[years==year]
        ar=rankdata(actual,method='ordinal');br=rankdata(pred,method='ordinal')
        rank_corr.append(spearmanr(actual,pred).statistic)
        grade.extend((np.floor((ar-1)/5)==np.floor((br-1)/5)).tolist())
        rank_err.extend(np.abs(ar-br).tolist())
        top.append(len(set(np.argsort(actual)[:5])&set(np.argsort(pred)[:5]))/5)
    scores.update(Spearman=float(np.mean(rank_corr)),rank_MAE=float(np.mean(rank_err)),
                  exact_grade_accuracy_pct=float(np.mean(grade)*100),
                  lowest_risk_top5_overlap_pct=float(np.mean(top)*100))
    return scores

def make_pairs(panel,h):
    target=panel[['gu','year','y','log_crime','log_child']].rename(columns={c:c+'_future' for c in CORE})
    target['year']=target.year-h
    d=panel.merge(target,on=['gu','year'],validate='one_to_one')
    d['target_year']=d.year+h
    return d[(d.year>=2015)&d.y.notna()&d.y_future.notna()].sort_values(['year','gu']).reset_index(drop=True)

def context(train,test,columns,gus):
    assert train.target_year.max()<=test.year.min(), 'Future labels leaked'
    raw=train[columns].to_numpy(float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        median=np.nanmedian(raw,axis=0)
    median=np.nan_to_num(median)
    filled=np.where(np.isfinite(raw),raw,median)
    mean=filled.mean(0);std=filled.std(0);std[std<1e-8]=1
    x=(filled-mean)/std
    z=(train.gu.to_numpy()[:,None]==np.array(gus)[None,:]).astype(float)
    test_z=(test.gu.to_numpy()[:,None]==np.array(gus)[None,:]).astype(float)
    yr=(train.target_year.to_numpy()[:,None]==np.unique(train.target_year)[None,:]).astype(float)
    return {'train':train,'test':test,'columns':columns,'gus':gus,'median':median,'mean':mean,'std':std,
            'x':x,'z':z,'test_z':test_z,'yr':yr,
            'future':train[[c+'_future' for c in CORE]].to_numpy(float),
            'current':train[CORE].to_numpy(float)}

def transform(c,frame,idx):
    raw=frame[c['columns']].to_numpy(float)[:,idx]
    raw=np.where(np.isfinite(raw),raw,c['median'][idx])
    return (raw-c['mean'][idx])/c['std'][idx]

def fit(c,idx,method,alpha=10):
    x=c['x'][:,idx];z=c['z'];future=c['future'];current=c['current']
    model={'method':method,'alpha':alpha,'features':[c['columns'][i] for i in idx],
           'median':c['median'][idx],'mean':c['mean'][idx],'std':c['std'][idx],'gus':c['gus']}
    if method.startswith('panel'):
        # Balanced training panel: double-demeaning gives exact borough/year FE.
        yr=c['yr'];y=future[:,0:1]-current[:,0:1]
        gm=z.T@x/z.sum(0)[:,None];tm=yr.T@x/yr.sum(0)[:,None];overall=x.mean(0)
        yd=y-z@(z.T@y/z.sum(0)[:,None])-yr@(yr.T@y/yr.sum(0)[:,None])+y.mean(0)
        xd=x-z@gm-yr@tm+overall
        coef=solve(xd.T@xd+alpha*np.eye(x.shape[1]),xd.T@yd,assume_a='pos')
        resid=y-x@coef;mu=resid.mean(0)
        gu_effect=z.T@resid/z.sum(0)[:,None]-mu
        shocks=yr.T@resid/yr.sum(0)[:,None]-mu
        shock=0.5*shocks[-1] if method=='panel_last_shock' else np.zeros(1)
        model.update(coef=coef,intercept=mu,gu_effect=gu_effect,shock=shock,training_year_shocks=shocks)
        return model
    if method=='components': y=future[:,1:]-current[:,1:]
    elif method=='recursive': y=future[:,[0,2]]-current[:,[0,2]]
    elif method.endswith('residual'): y=future[:,0:1]-current[:,0:1]
    else: y=future[:,0:1]
    design=np.column_stack([x,z])
    if method.startswith('extra') or method.startswith('hist'):
        if method.startswith('extra'):
            estimator=ExtraTreesRegressor(n_estimators=40,max_depth=5,min_samples_leaf=4,max_features=0.8,n_jobs=1,random_state=42)
        else:
            estimator=HistGradientBoostingRegressor(max_iter=60,max_leaf_nodes=5,min_samples_leaf=10,l2_regularization=10,learning_rate=.05,early_stopping=False,random_state=42)
        estimator.fit(design,y.ravel());model['estimator']=estimator
    else:
        xm=design.mean(0);ym=y.mean(0);xc=design-xm
        coef=solve(xc.T@xc+alpha*np.eye(design.shape[1]),xc.T@(y-ym),assume_a='pos')
        model.update(coef=coef,intercept=ym-xm@coef)
    return model

def raw_predict(model,frame):
    x=frame[model['features']].to_numpy(float)
    x=np.where(np.isfinite(x),x,model['median'])
    x=(x-model['mean'])/model['std']
    z=(frame.gu.to_numpy()[:,None]==np.array(model['gus'])[None,:]).astype(float)
    if model['method'].startswith('panel'):
        return x@model['coef']+model['intercept']+z@model['gu_effect']+model['shock']
    design=np.column_stack([x,z])
    if 'estimator' in model:return model['estimator'].predict(design)[:,None]
    return design@model['coef']+model['intercept']

def predict(model,frame,h):
    method=model['method'];detail={}
    if method=='recursive':
        state=frame.copy()
        for _ in range(h):
            step=raw_predict(model,state)
            prev=state[CORE].copy()
            state['y']=np.maximum(0,state.y.to_numpy()+step[:,0])
            state['log_child']=np.maximum(np.log(2),state.log_child.to_numpy()+step[:,1])
            child=np.expm1(state.log_child)
            state['log_crime']=np.log1p(np.expm1(np.minimum(state.y,20))*child/10000)
            for col in CORE:
                state[col+'_lag2']=state[col+'_lag1'].to_numpy()
                state[col+'_lag1']=prev[col].to_numpy()
                state[col+'_change']=state[col]-state[col+'_lag1']
            # Other numeric features freeze at origin; no future observations.
            state['year']+=1
        detail={'crime':np.expm1(state.log_crime.to_numpy()),'child':np.expm1(state.log_child.to_numpy())}
        return state.y.to_numpy(),detail
    result=raw_predict(model,frame)
    if method=='components':
        crime=np.expm1(np.clip(frame.log_crime.to_numpy()+result[:,0],0,20))
        child=np.expm1(np.clip(frame.log_child.to_numpy()+result[:,1],np.log(2),20))
        return np.log1p(crime/child*10000),{'crime':crime,'child':child}
    y=result[:,0]
    if method.endswith('residual') or method.startswith('panel'): y=y+frame.y.to_numpy()
    return np.clip(y,0,20),detail

def configurations(h):
    methods=['ridge_direct','ridge_residual','components','panel_mean_shock','panel_last_shock']
    if h==3:methods+=['recursive']
    return [(m,a) for m in methods for a in (1,10,100)]+[(m,0) for m in ('extra_direct','extra_residual','hist_direct','hist_residual')]

def evaluate_subset(mask,h,groups,columns,contexts,one_contexts):
    names=list(groups);active=[names[i] for i in range(len(names)) if mask&(1<<i)]
    features=list(dict.fromkeys(CORE+[c for name in active for c in groups[name]]))
    idx=[columns.index(c) for c in features];results=[]
    for method,alpha in configurations(h):
        truth=[];pred=[];years=[];folds=[]
        for k,c in enumerate(contexts):
            use=one_contexts[k] if method=='recursive' else c
            model=fit(use,idx,method,alpha)
            p,_=predict(model,c['test'],h)
            actual=c['test'].y_future.to_numpy();year=c['test'].target_year.to_numpy()
            truth.extend(actual);pred.extend(p);years.extend(year)
            folds.append({'year':int(year[0]),'RMSE':float(np.sqrt(np.mean((actual-p)**2))), 'n_train':len(use['train'])})
        scores=metrics(np.array(truth),np.array(pred),np.array(years))
        results.append({'horizon':h,'mask':mask,'groups':active,'n_features':len(features),'features':features,
                        'method':method,'alpha':alpha,'folds':folds,**scores})
    return results

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--jobs',type=int,default=2)
    args=ap.parse_args();start=time.time()
    panel=pd.read_csv(OUT/'panel.csv');groups=json.loads((OUT/'feature_groups.json').read_text(encoding='utf-8'))
    columns=list(dict.fromkeys(CORE+[c for cols in groups.values() for c in cols]));gus=sorted(panel.gu.unique())
    summary=[];one_pairs=make_pairs(panel,1)
    for h in [1,3]:
        dest=OUT/('smoke' if args.smoke else f'{h}y');dest.mkdir(parents=True,exist_ok=True)
        pairs=make_pairs(panel,h);contexts=[];one_contexts=[];base_y=[];base_p=[];base_year=[]
        for target_year in CV_YEARS:
            origin=target_year-h;test=pairs[pairs.target_year==target_year]
            train=pairs[pairs.target_year<=origin]
            assert len(test)==25 and train.groupby('year').size().eq(25).all()
            contexts.append(context(train,test,columns,gus))
            one_contexts.append(context(one_pairs[one_pairs.target_year<=origin],test,columns,gus))
            base_y.extend(test.y_future);base_p.extend(test.y);base_year.extend(test.target_year)
        baseline=metrics(np.array(base_y),np.array(base_p),np.array(base_year))
        checkpoint=dest/f'completed_{h}y.jsonl';results=[];complete=set()
        if checkpoint.exists():
            for line in checkpoint.read_text(encoding='utf-8').splitlines():
                block=json.loads(line);results.extend(block['results']);complete.add(block['mask'])
        masks=[0,511] if args.smoke else list(range(2**len(groups)))
        tasks=[m for m in masks if m not in complete]
        print(f'START horizon={h}, subsets={len(masks)}, remaining={len(tasks)}, models_per_subset={len(configurations(h))}, baseline={baseline["RMSE"]:.6f}',flush=True)
        jobs=joblib.Parallel(n_jobs=args.jobs,return_as='generator_unordered')(
            joblib.delayed(evaluate_subset)(m,h,groups,columns,contexts,one_contexts) for m in tasks)
        for block in jobs:
            results.extend(block);complete.add(block[0]['mask'])
            with checkpoint.open('a',encoding='utf-8') as stream:stream.write(json.dumps({'mask':block[0]['mask'],'results':block},ensure_ascii=False)+'\n')
            if len(complete)%16==0 or args.smoke:
                best=min(results,key=lambda x:x['RMSE'])
                print(f'{h}y subsets={len(complete)}/{len(masks)}, CV_best={best["RMSE"]:.6f}, elapsed={time.time()-start:.0f}s',flush=True)
        results.sort(key=lambda x:(x['RMSE'],x['MAE'],x['n_features'],x['method'],x['alpha']))
        winner=results[0]
        records=[{'stage':0,'method':'Persistence','RMSE':baseline['RMSE'],'improvement_pct':0,'groups':[]}]
        best=baseline['RMSE']
        # Deterministic complexity order, independent of parallel completion order.
        for row in sorted(results,key=lambda x:(len(x['groups']),x['mask'],x['method'],x['alpha'])):
            if row['RMSE']<best-1e-12:
                records.append({'stage':len(records),'previous_RMSE':best,'step_improvement_pct':(best-row['RMSE'])/best*100,
                                'improvement_pct':(baseline['RMSE']-row['RMSE'])/baseline['RMSE']*100,**row})
                best=row['RMSE']
        pd.DataFrame(results).to_csv(dest/'all_comparisons.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(records).to_csv(dest/'improvement_history.csv',index=False,encoding='utf-8-sig')
        by_method=pd.DataFrame(results).sort_values('RMSE').drop_duplicates('method')
        by_method.to_csv(dest/'best_by_method_cv.csv',index=False,encoding='utf-8-sig')
        # Winner is locked from CV before touching holdout outcomes.
        target_test=pairs[pairs.target_year==HOLDOUT];origin=HOLDOUT-h
        hold_base=metrics(target_test.y_future.to_numpy(),target_test.y.to_numpy(),target_test.target_year.to_numpy())
        cols=winner['features'];idx=[columns.index(c) for c in cols]
        train=(one_pairs if winner['method']=='recursive' else pairs)
        train=train[train.target_year<=origin]
        model=fit(context(train,target_test,columns,gus),idx,winner['method'],winner['alpha'])
        p,detail=predict(model,target_test,h)
        hold=metrics(target_test.y_future.to_numpy(),p,target_test.target_year.to_numpy())
        hold['direction_accuracy_pct']=float(np.mean(np.sign(p-target_test.y.to_numpy())==np.sign(target_test.y_future-target_test.y))*100)
        predictions=target_test[['gu','year','target_year','y','y_future']].copy()
        predictions['predicted_y']=p;predictions['actual_exposure']=np.expm1(target_test.y_future)
        predictions['predicted_exposure']=np.expm1(p)
        for key,val in detail.items():
            actual=np.expm1(target_test['log_'+key+'_future'].to_numpy())
            predictions['predicted_'+key]=val;predictions['actual_'+key]=actual
            hold[key+'_MAPE_pct']=float(np.mean(np.abs(val-actual)/actual)*100)
        predictions.to_csv(dest/'holdout_predictions.csv',index=False,encoding='utf-8-sig')
        accepted=winner['RMSE']<baseline['RMSE'] and hold['RMSE']<hold_base['RMSE']
        result={'horizon':h,'cv_years':CV_YEARS,'holdout_year':HOLDOUT,'baseline_cv':baseline,'winner_cv':winner,
                'baseline_holdout':hold_base,'winner_holdout':hold,'configuration_count':len(results),
                'cv_improvement_pct':(baseline['RMSE']-winner['RMSE'])/baseline['RMSE']*100,
                'holdout_improvement_pct':(hold_base['RMSE']-hold['RMSE'])/hold_base['RMSE']*100,
                'acceptance_check_passed':accepted,'deployment':'not deployed; separate experiment artifacts',
                'features_required':cols,'as_of_assumption':'crime and population through origin year observed; all other groups lag1; no historical publication-vintage verification',
                'recursive_assumption':'1-year y residual and log-child residual, fixed models, predicted lags updated each step, all other features frozen at origin',
                'search_scope':'exhaustive 2^9 semantic GROUP subsets over fixed candidate model grid, not individual-column powerset',
                'grade_definition':'annual relative quintiles, A=lowest exposure, E=highest; no adjacent grades',
                'panel_definition':'penalized slope with exact unpenalized borough and target-year fixed effects; future common shock assumed zero or half last observed training shock'}
        (dest/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        if not args.smoke:
            # Only the CV-selected candidate is fitted on all available targets.
            deployment_train=(one_pairs if winner['method']=='recursive' else pairs)
            latest=panel[(panel.year==2025)&panel.y.notna()].copy();latest['target_year']=2025+h
            final=fit(context(deployment_train,latest,columns,gus),idx,winner['method'],winner['alpha'])
            final.update(horizon=h,training_max_target_year=int(deployment_train.target_year.max()),metadata=result)
            MODELS.mkdir(parents=True,exist_ok=True)
            joblib.dump(final,MODELS/f'cv_best_{h}y.joblib')
            future,_=predict(final,latest,h)
            fc=latest[['gu','year','target_year']].copy();fc['predicted_y']=future;fc['predicted_exposure']=np.expm1(future)
            fc['baseline_y']=latest.y;fc['baseline_exposure']=latest.exposure
            fc['recommended_y']=future if accepted else latest.y.to_numpy()
            fc.to_csv(dest/'future_predictions.csv',index=False,encoding='utf-8-sig')
        summary.append(result)
        print('FINISH',h,'CV',winner['RMSE'],'holdout',hold['RMSE'],'baseline',hold_base['RMSE'],'accepted',accepted,flush=True)
    if not args.smoke:
        (OUT/'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('ELAPSED',round(time.time()-start),flush=True)

if __name__=='__main__': main()
