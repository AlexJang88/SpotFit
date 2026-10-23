"""6개월/1년 후 우세 분류 비교. 관측 분기 종료 시 자료가 이용 가능하다는 가정의 백테스트.

실행: python ml/training/compare_weekday_horizons.py
2023년으로 후보 선택을 고정하고, 2025년 동일 행정동·정답 분기를 비교한다.
"""
import argparse
import json
import html
import numpy as np
import pandas as pd
import joblib
from train_weekday_preference import ROOT, prepare, labels, KEY, build, metrics
from improve_weekday_preference import add_history, quarter_offset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=__import__('pathlib').Path,default=ROOT/'data/raw')
    parser.add_argument('--output',type=__import__('pathlib').Path,default=ROOT/'artifacts/weekday_horizon_comparison')
    args=parser.parse_args()
    raw,_,_,_,_=prepare(args.raw,'food')
    raw=raw.replace([np.inf,-np.inf],np.nan)
    sales=labels(args.raw)
    target=sales.rename(columns={KEY[0]:'target_quarter'})[['target_quarter',KEY[1],'target']]
    panels={}; all_features={}; sets={}
    for horizon in [2,4]:
        f=raw.copy(); f['target_quarter']=f[KEY[0]].map(lambda q:quarter_offset(q,horizon))
        f,history=add_history(f,sales)
        # 예측 정답 분기의 1년 전 우세. 2분기 후는 t-2, 4분기 후는 t 자료이므로 미래 정보가 아니다.
        seasonal=sales[KEY+['target']].rename(columns={KEY[0]:'target_quarter','target':'seasonal_target'})
        seasonal['target_quarter']=seasonal.target_quarter.map(lambda q:quarter_offset(q,4))
        f=f.merge(seasonal,on=['target_quarter',KEY[1]],how='left',validate='one_to_one')
        base=['총_직장_인구_수','총_유동인구_수','총_상주인구_수']+[c for c in f if c.endswith('_총계대비비율')]
        sets[horizon]={'history_logistic':('logistic',history), 'history_tree':('tree',history),
                       'combined_tree':('tree',base+history)}
        all_features[horizon]=f
        p=f.merge(target,on=['target_quarter',KEY[1]],validate='one_to_one')
        panels[horizon]=p[p.preference_lag0.notna()].copy()
    # 양 기간에서 입력 이력이 존재하는 동일 정답 행정동·분기만 검증/평가한다.
    shared=panels[2][['target_quarter',KEY[1]]].merge(panels[4][['target_quarter',KEY[1]]],
                                                   on=['target_quarter',KEY[1]],validate='one_to_one')
    cv=[]; split_audit=[]; predictions=[]; selected={}
    out=args.output/pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    for horizon,p in panels.items():
        candidates=sets[horizon]
        # 1년 모델의 첫 평가(2025Q1) 기준시점은 2024Q1이다.
        # 따라서 2024년 전체 정답으로 모델을 선택하면 누수: 선택용 정답은 2023년으로 고정한다.
        for q in [20231,20232,20233,20234]:
            origin=quarter_offset(q,-horizon)
            train=p[p.target_quarter<=origin]
            val=p[p.target_quarter==q].merge(shared,on=['target_quarter',KEY[1]],validate='one_to_one')
            if train.target.nunique()!=2 or val.empty: raise ValueError('검증 표본 부족')
            assert train.target_quarter.max()<=origin<q
            split_audit.append(dict(horizon=horizon,stage='validation',origin=origin,target=q,
                                    training_target_max=int(train.target_quarter.max()),n_train=len(train),n_eval=len(val)))
            for name,(kind,cols) in candidates.items():
                model=build(kind).fit(train[cols],train.target)
                cv.append(dict(horizon=horizon,model=name,quarter=q,**metrics(val.target,model.predict_proba(val[cols])[:,1])))
        ranking=pd.DataFrame(cv).query('horizon == @horizon').groupby('model')[['macro_f1','brier']].mean().sort_values(
            ['macro_f1','brier'],ascending=[False,True])
        name=ranking.index[0]; selected[horizon]=name; kind,cols=candidates[name]
        for q in [20251,20252,20253,20254]:
            origin=quarter_offset(q,-horizon)
            train=p[p.target_quarter<=origin]
            test=p[p.target_quarter==q].merge(shared,on=['target_quarter',KEY[1]],validate='one_to_one')
            assert train.target_quarter.max()<=origin<q
            split_audit.append(dict(horizon=horizon,stage='test',origin=origin,target=q,
                                    training_target_max=int(train.target_quarter.max()),n_train=len(train),n_eval=len(test)))
            model=build(kind).fit(train[cols],train.target)
            prediction=test[KEY+['행정동_코드_명','target_quarter','target','preference_lag0','seasonal_target']].copy()
            prediction['horizon']=horizon; prediction['weekend_probability']=model.predict_proba(test[cols])[:,1]
            predictions.append(prediction)
        # 추론 산출물은 전체 관측 정답으로 학습한다. 백테스트 분기 모델과 구분하여 보관한다.
        final=build(kind).fit(p[cols],p.target)
        joblib.dump(dict(pipeline=final,features=cols,horizon_quarters=horizon,selected=name,
                         training_target_max=int(p.target_quarter.max()),threshold=.5),out/f'model_{horizon*3}months.joblib')
        available=all_features[horizon].dropna(subset=['preference_lag0'])
        latest=available[available[KEY[0]]==available[KEY[0]].max()]
        future=latest[KEY+['행정동_코드_명','target_quarter']].copy()
        future['weekend_probability']=final.predict_proba(latest[cols])[:,1]
        future['prediction']=(future.weekend_probability>=.5).astype(int)
        future.to_csv(out/f'latest_predictions_{horizon*3}months.csv',index=False,encoding='utf-8-sig')
    pred=pd.concat(predictions,ignore_index=True)
    scores=[]; by_quarter=[]; transition=[]
    for horizon,d in pred.groupby('horizon'):
        for name,column in [('model','weekend_probability'),('persistence','preference_lag0'),('year_ago_same_quarter','seasonal_target')]:
            part=d.dropna(subset=[column])
            scores.append(dict(months=horizon*3,method=name,**metrics(part.target,part[column].to_numpy())))
            for q,sub in part.groupby('target_quarter'):
                by_quarter.append(dict(months=horizon*3,method=name,quarter=q,**metrics(sub.target,sub[column].to_numpy())))
            # 변화는 각 예측 기준시점 대비 정의하므로 6개월/1년의 변화 표본은 다를 수 있다.
            change=part.target.ne(part.preference_lag0)
            flags=(part[column]>=.5).ne(part.preference_lag0)
            correct=int((change & flags).sum())
            transition.append(dict(months=horizon*3,method=name,actual_changes=int(change.sum()),
                                   predicted_changes=int(flags.sum()),correct_changes=correct,
                                   precision=correct/flags.sum() if flags.sum() else 0,
                                   recall=correct/change.sum() if change.sum() else 0))
    scores=pd.DataFrame(scores)
    # 公平性検証: MLの比較対象は期間が違っても同じ正解キーとする。
    a=pred[pred.horizon==2]; b=pred[pred.horizon==4]
    paired=a.merge(b,on=['target_quarter',KEY[1]],suffixes=('_6m','_12m'),validate='one_to_one')
    assert len(paired)==len(a)==len(b)
    assert (paired.target_6m==paired.target_12m).all()
    for name,frame in [('validation_scores',pd.DataFrame(cv)),('split_audit',pd.DataFrame(split_audit)),
                       ('test_scores',scores),('test_by_quarter',pd.DataFrame(by_quarter)),
                       ('transition_scores',pd.DataFrame(transition)),('test_predictions',pred)]:
        frame.to_csv(out/f'{name}.csv',index=False,encoding='utf-8-sig')
    (out/'metadata.json').write_text(json.dumps({'selected':selected,'same_test_keys':len(paired),
        'selection_targets':[20231,20232,20233,20234],'test_reused':True,
        'availability_assumption':'source quarter data available at quarter end; publication lag unverified'},ensure_ascii=False,indent=2),encoding='utf-8')
    notes=('6개월=2분기、1년=4분기 뒤 단일 분기의 주말/주중 하루평균 매출 우세를 예측합니다. '
           '앞으로 6개월/1년 전체 기간의 매출 합계 예측이 아닙니다. 1=주말 우세, 0=주중 우세(동률 포함). '
           '2023년 4개 분기 확장창으로 후보를 선택하며, 각 학습 정답은 예측 기준시점 이하로 제한했습니다. '
           '2025년 각 분기별로 당시 사용 가능한 정답으로 재학습하는 순차 백테스트입니다. '
           '동일 정답 분기·행정동에서 비교하지만 예측 기준시점과 학습량은 기간에 따라 달라집니다. '
           '원본의 실제 공개 지연·수정 이력은 반영하지 못했으므로 운영 성능 보장은 아닙니다. '
           '2025년은 이미 사용한 평가 자료이며 새 기간 검증이 필요합니다. '
           '전환은 각 기간의 시작점 대비 정의되어 전환 표본 수가 다를 수 있습니다. '
           '확률은 미보정 값이며 model 파일은 전체 관측 정답으로 재학습한 추론용입니다. '
           '최신 예측도 보유 매출 시점에서 출발하므로 실행일 이후 미래일 필요는 없습니다.')
    report='<!doctype html><meta charset="utf-8"><title>6개월·1년 비교</title><h1>예측 기간 비교</h1><p>'+html.escape(notes)+'</p>'
    report+='<h2>선택 모델</h2><p>'+html.escape(str(selected))+'</p><h2>동일 평가 표본</h2>'+scores.to_html(index=False)
    report+='<h2>분기별 평가</h2>'+pd.DataFrame(by_quarter).to_html(index=False)
    report+='<h2>유형 전환 탐지</h2>'+pd.DataFrame(transition).to_html(index=False)
    report+='<h2>시점 누수 점검</h2>'+pd.DataFrame(split_audit).to_html(index=False)
    (out/'report.html').write_text(report,encoding='utf-8')
    print(out); print(selected); print(scores.to_string(index=False)); print(pd.DataFrame(transition).to_string(index=False))


if __name__=='__main__': main()
