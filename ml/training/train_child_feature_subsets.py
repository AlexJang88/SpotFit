"""연령별 상위 5개 시설의 모든 부분집합(31개)을 동일 분할로 비교한다.

현재 시설의 반복 연도 누수를 피하기 위해 최신 한 연도만 사용한다.
학습자료에서만 특징을 선정하고 검증 MAE로 모델을 선택한다.
테스트 점수는 모델을 선택한 후에만 계산하며 미래 예측 성능이 아니다.
"""
from pathlib import Path
from itertools import combinations
from datetime import datetime, timezone, timedelta
import argparse
import hashlib
import json
import os
import sys
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn


def metrics(y, pred):
    """인구수는 회귀이므로 분류 정확도가 아닌 명 단위 오차와 R²를 기록한다."""
    return {'mae':float(mean_absolute_error(y,pred)),
            'rmse':float(np.sqrt(mean_squared_error(y,pred))), 'r2':float(r2_score(y,pred))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--data',type=Path)
    p.add_argument('--tracking-uri',help='생략하면 프로젝트 내 로컬 SQLite. 서버 URI는 명시적으로 지정.')
    p.add_argument('--ages',nargs='+',default=['infant','elementary','middle','high'],choices=['infant','elementary','middle','high'])
    p.add_argument('--register-best',action='store_true',help='연령별 검증 우승 모델만 레지스트리에 등록')
    p.add_argument('--export-version',help='README 방식 models/<이름>/<버전>/model.joblib로 우승 모델 내보내기. 업로드는 별도.')
    a=p.parse_args();root=a.root
    source=a.data or sorted((root/'data/processed/child_population').glob('facility_child_counts_*/analysis_panel.csv'))[-1]
    frame=pd.read_csv(source,dtype={'dong_code':str})
    year=int(frame.year.max());frame=frame[frame.year==year].reset_index(drop=True)
    assert frame.dong_code.is_unique
    # 동일한 동 분할을 모든 연령·조합에 적용한다. 행정동 ID는 입력 특징이 아니다.
    tr,hold=train_test_split(np.arange(len(frame)),test_size=.4,random_state=42)
    va,te=train_test_split(hold,test_size=.5,random_state=42)
    assert not(set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te))
    stamp=datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')
    out=root/'artifacts/child_feature_subsets'/stamp;out.mkdir(parents=True,exist_ok=False)
    tracking=root/'artifacts/mlflow_child_subsets';tracking.mkdir(parents=True,exist_ok=True)
    uri=a.tracking_uri or 'sqlite:///'+(tracking/'tracking.db').as_posix()
    mlflow.set_tracking_uri(uri)
    experiment='child-population-feature-subsets'
    if mlflow.get_experiment_by_name(experiment) is None:
        # 로컬 실험만 저장 경로를 고정하고 원격 실험은 서버 설정을 사용한다.
        mlflow.create_experiment(experiment,artifact_location=(tracking/'runs').as_uri() if not a.tracking_uri else None)
    mlflow.set_experiment(experiment)
    split=frame[['dong_code','dong','gu','year']].copy();split['split']='train'
    split.loc[va,'split']='validation';split.loc[te,'split']='test'
    split.to_csv(out/'split.csv',index=False,encoding='utf-8-sig')
    features=[c for c in frame if c.endswith('_수')]
    winners=[];all_rows=[]
    with mlflow.start_run(run_name='subsets_'+stamp) as parent:
        parent_id=parent.info.run_id
        mlflow.log_params({'year':year,'seed':42,'train_n':len(tr),'validation_n':len(va),'test_n':len(te),
            'data_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'target':'same_year_child_count','selection':'validation_mae'})
        mlflow.set_tags({'caution':'static_facilities_not_future_forecast','split':'random_disjoint_dongs_not_gu_holdout'})
        mlflow.log_artifact(str(Path(__file__)),artifact_path='source')
        for age in a.ages:
            target='children_'+age
            y=frame[target]
            assert y.notna().all()
            # 유효값 70% 미만 지표는 결측 선택편향 위험 때문에 후보에서 제외한다.
            # abs 상관으로 고르되 부호를 보존하며, 음의 상관도 예측 특징이 될 수 있다.
            ranking=[]
            for f in features:
                x=frame.loc[tr,f]
                rho=x.corr(y.loc[tr],method='spearman') if x.nunique()>1 else np.nan
                ranking.append({'feature':f,'rho':rho,'coverage':x.notna().mean(),
                    'eligible':bool(x.notna().mean()>=.7 and x.nunique()>1 and pd.notna(rho))})
            rank=pd.DataFrame(ranking);rank['abs_rho']=rank.rho.abs()
            rank=rank.sort_values('abs_rho',ascending=False)
            top=rank[rank.eligible].head(5).feature.tolist()
            assert len(top)==5
            rank.to_csv(out/(age+'_feature_ranking.csv'),index=False,encoding='utf-8-sig')
            models=[]
            for size in range(1,6):
                for combo in combinations(top,size):
                    cols=list(combo)
                    # 결측 중앙값은 학습자료에서만 학습. 모든 조합에 동일 RF 설정 적용.
                    model=make_pipeline(SimpleImputer(strategy='median'),RandomForestRegressor(
                        n_estimators=80,min_samples_leaf=5,max_features=1.0,random_state=42,n_jobs=-1))
                    model.fit(frame.loc[tr,cols],y.loc[tr])
                    score=metrics(y.loc[va],model.predict(frame.loc[va,cols]))
                    with mlflow.start_run(run_name=f'{age}_{len(models)+1:02d}_{size}features',nested=True) as run:
                        mlflow.log_params({'age':age,'features':json.dumps(cols,ensure_ascii=False),'feature_count':size,
                            'algorithm':'RandomForestRegressor','n_estimators':80,'min_samples_leaf':5,'seed':42})
                        mlflow.log_metrics({'validation_'+k:v for k,v in score.items()})
                        # 31개 모두 모델과 입력 열을 보존하고 우승 모델은 표준 MLflow 형식도 저장한다.
                        path=out/f'{age}_{len(models)+1:02d}.joblib'
                        joblib.dump({'model':model,'features':cols,'target':target},path)
                        mlflow.log_artifact(str(path),artifact_path='pipeline')
                        row={'age':age,'run_id':run.info.run_id,'features':' | '.join(cols),'feature_count':size,
                             **{'validation_'+k:v for k,v in score.items()}}
                        all_rows.append(row);models.append((score['mae'],cols,model,run.info.run_id))
            # 검증으로 우승 모델을 고정하며 테스트 결과로 다시 선택하지 않는다.
            _,cols,best,run_id=min(models,key=lambda item:item[0])
            test_pred=best.predict(frame.loc[te,cols]);score=metrics(y.loc[te],test_pred)
            baseline=DummyRegressor(strategy='mean').fit(np.zeros((len(tr),1)),y.loc[tr])
            base_score=metrics(y.loc[te],baseline.predict(np.zeros((len(te),1))))
            with mlflow.start_run(run_id=run_id,nested=True):
                mlflow.log_metrics({'test_'+k:v for k,v in score.items()})
                mlflow.log_metrics({'baseline_test_'+k:v for k,v in base_score.items()})
                mlflow.set_tag('validation_winner','true')
                # 직접 생성한 모델에만 cloudpickle을 사용한다. 외부 비신뢰 모델 로드는 금지한다.
                info=mlflow.sklearn.log_model(best,name='model',serialization_format='cloudpickle',input_example=frame.loc[tr,cols].head(3),
                    pip_requirements=[f'scikit-learn=={sklearn.__version__}',f'pandas=={pd.__version__}',f'numpy=={np.__version__}'])
                version=None
                if a.register_best:
                    name='spotfit-child-count-'+age
                    version=mlflow.register_model(info.model_uri,name).version
                    mlflow.MlflowClient().set_model_version_tag(name,version,'validation_status','exploratory_not_production')
                winners.append({'age':age,'run_id':run_id,'model_uri':info.model_uri,'version':version,
                    'features':' | '.join(cols),**{'test_'+k:v for k,v in score.items()},'baseline_test_mae':base_score['mae']})
            pred=frame.loc[te,['dong_code','dong','gu','year']].copy()
            pred['actual']=y.loc[te];pred['predicted']=test_pred
            pred.to_csv(out/(age+'_test_predictions.csv'),index=False,encoding='utf-8-sig')
            print(age,cols,score,flush=True)
        comparison=pd.DataFrame(all_rows).sort_values(['age','validation_mae'])
        comparison.to_csv(out/'comparison.csv',index=False,encoding='utf-8-sig')
        winner=pd.DataFrame(winners);winner.to_csv(out/'winners.csv',index=False,encoding='utf-8-sig')
        (out/'report.html').write_text('<meta charset="utf-8"><h1>시설 조합별 아동 인구수 회귀 비교</h1><p>2025년 동별 분할. 학습 내 상관 상위 5개(관측률 70% 이상). 검증 MAE로 선택. 미래 예측·인과효과가 아니며 공간 인접에 따른 낙관 가능성이 있습니다. 테스트로 조합을 재선택하지 마세요.</p><h2>검증 우승 모델의 테스트 성능</h2>'+winner.to_html(index=False)+'<h2>모든 조합의 검증 성능</h2>'+comparison.to_html(index=False),encoding='utf-8')
        for path in out.glob('*.csv'): mlflow.log_artifact(str(path),artifact_path='results')
        mlflow.log_artifact(str(out/'report.html'))
    (out/'tracking.json').write_text(json.dumps({'tracking_uri':uri,'parent_run_id':parent_id,'source':str(source)},indent=2),encoding='utf-8')
    print('OUTPUT',out,flush=True)
    if a.export_version:
        from export_child_models import export_models
        export_models(root,out,a.export_version)


if __name__=='__main__':
    # 새 평가가 기본이다. 과거 RF-only 재현은 명시적인 legacy 옵션으로만 실행한다.
    if '--legacy-rf-only' in sys.argv:
        sys.argv.remove('--legacy-rf-only')
        main()
    else:
        from compare_child_algorithms import main as compare_main
        compare_main()
