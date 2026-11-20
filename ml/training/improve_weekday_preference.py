"""과거 매출 특징과 확장창 시간 검증을 추가한 v2 실험. 기존 모델은 덮어쓰지 않는다.

실행: python ml/training/improve_weekday_preference.py
2025년은 이전 실험에서 확인했으므로 재사용 평가이며 독립적인 최종 검증이 아니다.
"""
from pathlib import Path
import argparse
import json
import html
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score
from train_weekday_preference import ROOT, prepare, labels, KEY, next_quarter, build, metrics


def quarter_offset(q, offset):
    """행 순서가 아닌 실제 분기 차이로 연결해 누락된 분기를 건너뛰는 누수를 방지한다."""
    year, quarter = divmod(int(q), 10)
    ordinal = year * 4 + quarter - 1 + offset
    return (ordinal // 4) * 10 + ordinal % 4 + 1


def add_history(features, sales):
    """기준분기 t 및 t-1~t-3의 정답 구성 정보를 사용한다. t+1은 입력하지 않는다."""
    sales = sales.copy()
    total = sales.weekday_daily + sales.weekend_daily
    # 0으로 나누거나 극단적인 비율이 되는 것을 피하려고 0~1의 일평균 점유율을 사용한다.
    sales['daily_share'] = sales.weekend_daily / total
    history = []
    result = features.copy()
    for lag in range(4):
        previous = sales[KEY + ['daily_share', 'target']].copy()
        previous[KEY[0]] = previous[KEY[0]].map(lambda q: quarter_offset(q, lag))
        rename = {'daily_share': f'sales_share_lag{lag}', 'target': f'preference_lag{lag}'}
        previous = previous.rename(columns=rename)
        result = result.merge(previous, on=KEY, how='left', validate='one_to_one')
        history += list(rename.values())
    result['share_change_1q'] = result.sales_share_lag0 - result.sales_share_lag1
    cols = [f'sales_share_lag{i}' for i in range(4)]
    result['share_mean_4q'] = result[cols].mean(axis=1)
    result['history_quarters_available'] = result[cols].count(axis=1)
    # 계절을 주기형 숫자로 표현한다. 미래 매출 대신 예측 분기의 달력만 사용한다.
    q = result.target_quarter % 10
    result['season_sin'] = np.sin(2*np.pi*q/4)
    result['season_cos'] = np.cos(2*np.pi*q/4)
    history += ['share_change_1q', 'share_mean_4q', 'history_quarters_available', 'season_sin', 'season_cos']
    return result, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=ROOT/'data/raw')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/weekday_preference_v2')
    args = parser.parse_args()
    f, _, _, _, _ = prepare(args.raw, 'food')
    f = f.replace([np.inf, -np.inf], np.nan)
    f['target_quarter'] = f[KEY[0]].map(next_quarter)
    sales = labels(args.raw)
    f, history = add_history(f, sales)
    base = ['총_직장_인구_수', '총_유동인구_수', '총_상주인구_수'] + [c for c in f if c.endswith('_총계대비비율')]
    panel = f.merge(sales.rename(columns={KEY[0]:'target_quarter'})[['target_quarter', KEY[1], 'target']],
                    on=['target_quarter', KEY[1]], validate='one_to_one')
    # 개선 모델과 유지 기준을 동일 표본으로 비교한다. 직전 매출 미확인 지역은 명시적으로 제외한다.
    excluded = panel[panel.preference_lag0.isna()].copy()
    panel = panel[panel.preference_lag0.notna()].copy()
    candidates = {'base_tree': ('tree', base), 'history_logistic': ('logistic', history),
                  'history_tree': ('tree', history), 'combined_tree': ('tree', base+history)}
    folds = []
    for quarter in [20241, 20242, 20243, 20244]:
        train = panel[panel.target_quarter < quarter]
        val = panel[panel.target_quarter == quarter]
        assert train.target_quarter.max() < val.target_quarter.min()
        for name, (kind, cols) in candidates.items():
            model = build(kind).fit(train[cols], train.target)
            p = model.predict_proba(val[cols])[:, 1]
            folds.append({'model':name, 'quarter':quarter, **metrics(val.target, p)})
        folds.append({'model':'persistence', 'quarter':quarter, **metrics(val.target, val.preference_lag0.to_numpy())})
    cv = pd.DataFrame(folds)
    rank = cv.groupby('model')[['macro_f1', 'accuracy', 'brier']].mean().reset_index().sort_values(
        ['macro_f1','brier','model'], ascending=[False,True,True])
    # 유지 기준도 선택 후보에 포함한다. ML이 검증에서 이기지 못하면 억지로 채택하지 않는다.
    winner = rank.iloc[0]['model']
    ml_winner = rank[rank.model != 'persistence'].iloc[0]['model']
    development = panel[panel.target_quarter <= 20244]
    test = panel[panel.target_quarter.between(20251,20254)]
    kind, cols = candidates[ml_winner]
    model = build(kind).fit(development[cols], development.target)
    baseline = build('tree').fit(development[base], development.target)
    probabilities = {'selected_ml':model.predict_proba(test[cols])[:,1],
                     'base_tree_retrained':baseline.predict_proba(test[base])[:,1],
                     'persistence':test.preference_lag0.to_numpy()}
    changed = test.target.to_numpy() != test.preference_lag0.to_numpy()
    evaluations = []
    for name,p in probabilities.items():
        for group,mask in [('all',np.ones(len(test),dtype=bool)),('changed',changed),('unchanged',~changed)]:
            if mask.any():
                evaluations.append({'model':name,'group':group,**metrics(test.target.to_numpy()[mask],p[mask])})
    evaluation = pd.DataFrame(evaluations)
    # 예측된 전환 자체의 정밀도/재현율: 그대로 유지하는 모델은 전환을 한 번도 찾지 못한다.
    transitions = []
    for name,p in probabilities.items():
        flag = (p>=.5) != test.preference_lag0.to_numpy()
        tp = int((flag & changed).sum())
        transitions.append({'model':name,'actual_changes':int(changed.sum()),'predicted_changes':int(flag.sum()),
                            'correct_changes':tp,'change_precision':tp/flag.sum() if flag.sum() else 0,
                            'change_recall':tp/changed.sum() if changed.sum() else 0})
    # 지역을 묶어서 복원추출한다. 동일 행정동의 여러 분기가 독립인 것처럼 처리하지 않는다.
    # 공간적으로 인접한 서로 다른 동의 의존성은 여전히 미보정이다.
    test_predictions = test[KEY+['행정동_코드_명','target_quarter','target','preference_lag0']].copy()
    for name,p in probabilities.items(): test_predictions[name+'_probability'] = p
    score = pd.DataFrame({'dong':test[KEY[1]].to_numpy(),
        'gain':((probabilities['selected_ml']>=.5)==test.target.to_numpy()).astype(int)
               -(probabilities['persistence']==test.target.to_numpy()).astype(int)})
    agg = score.groupby('dong').gain.agg(['sum','count']).to_numpy()
    rng = np.random.default_rng(42)
    bootstrap=[]
    for _ in range(1000):
        sample=agg[rng.integers(0,len(agg),len(agg))]
        bootstrap.append(sample[:,0].sum()/sample[:,1].sum())
    interval=np.quantile(bootstrap,[.025,.975]).tolist()
    out=args.output/pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    joblib.dump({'pipeline':model,'features':cols,'training_target_max':20244},out/'evaluation_model.joblib')
    final=build(kind).fit(panel[cols],panel.target)
    joblib.dump({'pipeline':final,'features':cols,'training_target_max':int(panel.target_quarter.max()),
                 'validation_policy':winner,'classes':{0:'주중 우세',1:'주말 우세'}},out/'model.joblib')
    # 매출이 없는 최신 인구분기로 억지 추론하지 않는다. 지역별 최신 행이 아닌 공통 최신 분기로 제한한다.
    available=f[f.preference_lag0.notna()]
    latest=available[available[KEY[0]]==available[KEY[0]].max()]
    prediction=latest[KEY+['행정동_코드_명','target_quarter']].copy()
    prediction['weekend_probability']=final.predict_proba(latest[cols])[:,1]
    prediction['ml_prediction']=(prediction.weekend_probability>=.5).astype(int)
    prediction['recommended_prediction']=latest.preference_lag0.to_numpy().astype(int) if winner=='persistence' else prediction.ml_prediction
    for name,frame in [('rolling_validation',cv),('validation_ranking',rank),('test_scores',evaluation),
                       ('transition_scores',pd.DataFrame(transitions)),('test_predictions',test_predictions),
                       ('latest_predictions',prediction),('excluded_missing_previous_sales',excluded),
                       ('training_panel',panel[KEY+['target_quarter','target']+list(dict.fromkeys(base+history))])]:
        frame.to_csv(out/f'{name}.csv',index=False,encoding='utf-8-sig')
    metadata={'selected_policy':winner,'selected_ml':ml_winner,'features':cols,'accuracy_gain_ci95_vs_persistence':interval,
              'test_is_reused':True,'threshold':.5,'seed':42,'bootstrap_unit':'administrative_dong'}
    (out/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    notes=('기존 v1과 결과는 보존했습니다. 이번 실험은 2024년 4개 분기 확장창 검증으로 모델을 선택하고 '
           '2025년을 재사용 평가했습니다. 이전 평가를 보고 개선했으므로 독립 시험이 아니며 개선의 확증에는 새 기간이 필요합니다. '
           '직전 매출이 없는 표본은 제외하며 모든 기준모델도 같은 표본으로 비교합니다. '
           '자료 공개일을 모르는 상태에서 직전 분기 매출을 사용하므로 분기 시작 시 실제 이용 가능한 정보인지 확인해야 합니다. '
           '매출 점유율은 주말 일평균/(주말+평일 일평균)입니다. 미래 정답은 특징에 포함하지 않습니다. '
           '확률은 보정되지 않았고 계절 변수는 예측 분기의 달력 정보만 사용합니다. '
           '최신 추정도 실행일 이후 미래가 아니라 매출이 존재하는 최신 분기의 바로 다음 분기입니다. '
           'model.joblib은 전체 라벨로 재학습한 ML 후보이며 recommended_prediction은 검증에서 선택된 정책을 따릅니다. '
           '신뢰구간은 행정동 단위 부트스트랩으로 동간 공간 의존성까지 해결하지는 않습니다.')
    report='<!doctype html><meta charset="utf-8"><title>주중·주말 예측 v2</title><h1>성능 개선 실험 v2</h1><p>'+html.escape(notes)+'</p>'
    report+=f'<h2>검증 선택</h2><p>정책: {winner}, ML 후보: {ml_winner}</p>'+rank.to_html(index=False)
    report+='<h2>같은 표본의 평가</h2>'+evaluation.to_html(index=False)
    report+=f'<p>ML−유지 기준 정확도 차이의 95% 지역 부트스트랩 구간: {interval[0]*100:+.2f}~{interval[1]*100:+.2f}%p</p>'
    report+='<h2>우세 전환 탐지</h2>'+pd.DataFrame(transitions).to_html(index=False)
    report+='<h2>분기별 검증</h2>'+cv.to_html(index=False)+'<h2>최신 추정</h2>'+prediction.to_html(index=False)
    (out/'report.html').write_text(report,encoding='utf-8')
    print(out); print(rank.to_string(index=False)); print(evaluation.to_string(index=False)); print(pd.DataFrame(transitions).to_string(index=False)); print('CI',interval)


if __name__=='__main__': main()
