"""행정동 외식업: t분기 특징으로 t+1분기의 주말 일평균 매출 우세를 예측한다.

실행: python ml/training/train_weekday_preference.py
모델 선택은 2024년 검증으로만 수행하고 2025년은 사후 평가한다.
과거 EDA에서 2025년을 확인했으므로 완전히 독립적인 최종 평가라고 주장하지 않는다.
"""
from pathlib import Path
import sys
import argparse
import json
import html
import platform
import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, brier_score_loss, confusion_matrix
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'data_analyze/attraction'))
sys.path.insert(0, str(ROOT / 'data_analyze/weekday_sales'))
from attraction_analyze import prepare
from weekday_sales_analyze import read, check_unique, day_counts, KEY, SALES


def next_quarter(value):
    """연도 경계를 포함한 바로 다음 분기. 누락 분기를 건너뛰어 연결하지 않는다."""
    year, q = divmod(int(value), 10)
    return (year+1)*10+1 if q==4 else year*10+q+1


def labels(raw):
    """외식업×행정동×분기로 매출을 집계하고 요일별 일수 차이를 보정해 정답을 만든다."""
    files=sorted(raw.glob('서울시 상권분석서비스(추정매출-행정동)*.csv'))
    sales=pd.concat([read(p) for p in files],ignore_index=True)
    sales=sales[sales['서비스_업종_코드'].str.startswith('CS1',na=False)].copy()
    check_unique(sales,KEY+['서비스_업종_코드'])
    for c in SALES:
        sales[c]=pd.to_numeric(sales[c],errors='coerce').where(lambda s:s>=0)
    g=sales.groupby(KEY); d=g[SALES].sum(min_count=1)
    for c in SALES:
        d.loc[g[c].count()!=g.size(),c]=np.nan
    d=d.reset_index()
    days=d[KEY[0]].map(day_counts)
    d['weekday_daily']=d[SALES[0]]/days.map(lambda x:x[0])
    d['weekend_daily']=d[SALES[1]]/days.map(lambda x:x[1])
    # 양쪽 매출이 0이면 우세를 정의할 실질적 정보가 없어 제외한다.
    d=d.dropna(subset=['weekday_daily','weekend_daily'])
    d=d[(d.weekday_daily+d.weekend_daily)>0].copy()
    d['target']=(d.weekend_daily>d.weekday_daily).astype(int)
    d['near_tie']=((d.weekend_daily-d.weekday_daily).abs()/(d.weekend_daily+d.weekday_daily)<.05)
    return d


def metrics(y,p):
    pred=(p>=.5).astype(int)
    return {'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),
            'macro_f1':f1_score(y,pred,average='macro',zero_division=0),
            'roc_auc':roc_auc_score(y,p) if len(np.unique(y))==2 else None,
            'brier':brier_score_loss(y,p),'n':len(y),'weekend_rate':float(np.mean(y))}


def build(kind):
    # 중앙값/스케일/결측 표시 모두 fit한 과거 자료만으로 정한다.
    imputer=SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)
    if kind=='majority':
        return make_pipeline(imputer,DummyClassifier(strategy='most_frequent'))
    if kind=='logistic':
        return make_pipeline(imputer,StandardScaler(),LogisticRegression(max_iter=3000,random_state=42))
    return make_pipeline(imputer,HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,
        min_samples_leaf=40,l2_regularization=5,early_stopping=False,random_state=42))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,default=ROOT/'data/raw')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/weekday_preference')
    args=parser.parse_args()
    # 이전 분석의 시설 비율 정의를 재사용한다. 매출은 준비 함수 결과에 있지만 X에 넣지 않는다.
    features,_,_,quality,_=prepare(args.raw,'food')
    features=features.replace([np.inf,-np.inf],np.nan)
    populations=['총_직장_인구_수','총_유동인구_수','총_상주인구_수']
    facilities=[c for c in features if c.endswith('_총계대비비율')]
    feature_sets={'population':populations,'population_bank':populations+['은행_총계대비비율'],
                  'population_all_facilities':populations+facilities}
    # 점포 수는 2025년부터만 있어 과거 학습에 부적합하므로 제외한다.
    # 관광명소는 과거 이력이 없어 제외. 행정동 코드는 식별용이지 입력 특징이 아니다.
    features['target_quarter']=features[KEY[0]].map(next_quarter)
    y=labels(args.raw)
    target=y.rename(columns={KEY[0]:'target_quarter'})
    panel=features.merge(target[['target_quarter',KEY[1],'target','weekday_daily','weekend_daily','near_tie']],
                         on=['target_quarter',KEY[1]],how='inner',validate='one_to_one')
    # 바로 이전 분기 우세를 유지하는 단순 기준모델도 비교한다. 매출을 ML 입력에는 사용하지 않는다.
    panel=panel.merge(y[KEY+['target']].rename(columns={'target':'previous_target'}),on=KEY,
                      how='left',validate='one_to_one')
    train=panel[panel.target_quarter<=20234].copy()
    validation=panel[panel.target_quarter.between(20241,20244)].copy()
    test=panel[panel.target_quarter.between(20251,20254)].copy()
    for name,part in [('train',train),('validation',validation),('test',test)]:
        if part.empty or part.target.nunique()<2:
            raise ValueError(f'{name}: 표본 또는 두 클래스가 부족합니다.')
        assert (part[KEY[0]]<part.target_quarter).all()
    assert train.target_quarter.max()<validation.target_quarter.min()<test.target_quarter.min()
    out=args.output/pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    scores=[]; fitted={}
    # 후보 수를 작게 고정한다. 테스트 성능으로 조합을 고르거나 튜닝하지 않는다.
    for set_name,cols in feature_sets.items():
        for kind in ['logistic','tree']:
            name=set_name+'__'+kind
            model=build(kind); model.fit(train[cols],train.target)
            p=model.predict_proba(validation[cols])[:,1]
            scores.append({'model':name,'feature_set':set_name,'kind':kind,**metrics(validation.target,p)})
            fitted[name]=(model,cols)
    table=pd.DataFrame(scores).sort_values(['macro_f1','brier','model'],ascending=[False,True,True])
    chosen=table.iloc[0]
    cols=feature_sets[chosen.feature_set]
    # 검증 선택 이후 train+validation으로 학습하고 테스트는 한 번 평가한다.
    development=pd.concat([train,validation],ignore_index=True)
    selected=build(chosen.kind); selected.fit(development[cols],development.target)
    p=selected.predict_proba(test[cols])[:,1]
    evaluation=[{'model':'selected','split':'test',**metrics(test.target,p)}]
    dummy=build('majority'); dummy.fit(development[cols],development.target)
    evaluation.append({'model':'majority','split':'test',**metrics(test.target,dummy.predict_proba(test[cols])[:,1])})
    previous=test.dropna(subset=['previous_target'])
    if len(previous):
        evaluation.append({'model':'previous_quarter_persistence','split':'test_available_previous',
                           **metrics(previous.target,previous.previous_target.to_numpy())})
        available_p=selected.predict_proba(previous[cols])[:,1]
        evaluation.append({'model':'selected_same_persistence_sample','split':'test_available_previous',
                           **metrics(previous.target,available_p)})
    test_out=test[KEY+['행정동_코드_명','target_quarter','target','weekday_daily','weekend_daily','near_tie']].copy()
    test_out['weekend_probability']=p; test_out['prediction']=(p>=.5).astype(int)
    # 중요도는 검증기간에서 이미 학습된 후보로 계산한다. 테스트를 보고 특징을 다시 선택하지 않는다.
    validation_model=fitted[chosen.model][0]
    importance=permutation_importance(validation_model,validation[cols],validation.target,
                                      scoring='f1_macro',n_repeats=5,random_state=42)
    importance_table=pd.DataFrame({'feature':cols,'mean_macro_f1_drop':importance.importances_mean,
                                   'std':importance.importances_std}).sort_values('mean_macro_f1_drop',ascending=False)
    # 평가용 모델과 실제 추론용 재학습 모델을 구분하여 2025 라벨이 평가 모델에 들어가지 않게 한다.
    joblib.dump({'pipeline':selected,'features':cols,'training_target_max':20244},out/'evaluation_model.joblib')
    final=build(chosen.kind); final.fit(panel[cols],panel.target)
    joblib.dump({'pipeline':final,'features':cols,'classes':{0:'주중 우세(동률 포함)',1:'주말 우세'},
                 'training_target_max':int(panel.target_quarter.max())},out/'model.joblib')
    latest=features[features[KEY[0]]==features[KEY[0]].max()].copy()
    prediction=latest[KEY+['행정동_코드_명','target_quarter']].copy()
    prediction['weekend_probability']=final.predict_proba(latest[cols])[:,1]
    prediction['prediction']=np.where(prediction.weekend_probability>=.5,'주말 우세','주중 우세')
    prediction['missing_feature_count']=latest[cols].isna().sum(axis=1)
    split_info=pd.DataFrame([{'split':n,'n':len(d),'target_min':int(d.target_quarter.min()),
                             'target_max':int(d.target_quarter.max()),'weekend_rate':d.target.mean(),
                             'near_tie_rate':d.near_tie.mean()} for n,d in [('train',train),('validation',validation),('test',test)]])
    for name,frame in [('validation_scores',table),('test_scores',pd.DataFrame(evaluation)),
                        ('test_predictions',test_out),('latest_predictions',prediction),
                        ('feature_importance_validation',importance_table),('split_summary',split_info),
                        ('training_panel',panel[KEY+['target_quarter','target']+list(dict.fromkeys(populations+facilities))])]:
        frame.to_csv(out/f'{name}.csv',index=False,encoding='utf-8-sig')
    metadata={'chosen':chosen.model,'features':cols,'versions':{'python':platform.python_version(),
              'sklearn':sklearn.__version__,'pandas':pd.__version__},'seed':42,
              'source_files':[{'name':f.name,'size':f.stat().st_size,'mtime_ns':f.stat().st_mtime_ns} for f in args.raw.glob('*행정동*.csv')]}
    (out/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    notes=('정답: 다음 분기 주말 하루평균 매출>주중 하루평균 매출이면 1, 그 외(동률 포함) 0. '
           '분기 달력 월~금/토~일 일수를 사용하고 공휴일은 별도 조정하지 않습니다. '
           '같은 행정동이 여러 기간에 존재하는 미래 시점 예측이며, 새로운 행정동 일반화 평가는 아닙니다. '
           '관측 분기와 실제 공개 시점은 다를 수 있어 t자료가 t+1 시작 전에 공개된다는 보장이 없습니다. '
           '공개일 이력 확보 전에는 실시간 운영 성능으로 해석하지 마세요. '
           '시설 결측은 미제공으로 유지하고 학습 시점에서만 대체합니다. 시설 비율은 합계 100% 구성비가 아닙니다. '
           '2025년은 이전 EDA에서 이미 확인한 자료로 완전히 독립적인 최종 시험이 아닙니다. '
           '확률은 미보정 모델 출력으로 정확한 신뢰도나 보장 확률이 아닙니다. '
           '최신 예측은 보유 자료의 다음 분기 추정이며 실행일 이후의 미래일 필요는 없습니다. '
           '최종 model.joblib은 평가 완료 후 모든 관측 라벨로 재학습했으므로 표의 평가용 모델과 구분합니다. '
           'joblib은 신뢰할 수 있는 자체 생성 파일만 로드하세요.')
    report='<!doctype html><meta charset="utf-8"><title>주중·주말 우세 예측</title><h1>다음 분기 주중·주말 우세 분류</h1>'
    report+='<p>'+html.escape(notes)+'</p><h2>검증기간 선택 모델</h2><p>'+html.escape(chosen.model)+'</p>'
    report+='<h2>시간 분리·클래스 분포</h2>'+split_info.to_html(index=False)
    report+='<h2>테스트 및 기준모델 비교</h2>'+pd.DataFrame(evaluation).to_html(index=False)
    report+='<h2>혼동행렬 (행=정답, 열=예측, 순서: 주중·주말)</h2>'+pd.DataFrame(confusion_matrix(test.target,p>=.5,labels=[0,1]),index=['주중','주말'],columns=['주중','주말']).to_html()
    report+='<h2>검증 조합 비교</h2>'+table.to_html(index=False)+'<h2>검증 순열 중요도</h2>'+importance_table.to_html(index=False)
    report+='<h2>최신 보유 분기의 다음 분기 추정</h2>'+prediction.to_html(index=False)
    (out/'report.html').write_text(report,encoding='utf-8')
    print(out); print(chosen.model); print(pd.DataFrame(evaluation).to_string(index=False))


if __name__=='__main__':
    main()
