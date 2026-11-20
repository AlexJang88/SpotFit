"""Reproduce recorded 2025 evaluation and diagnose feature-score errors."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ml.rebuild_traffic_accident_v93_from_processed import load_processed, model_config
from ml.train_traffic_accident_v72_indicator_forecast import supervised, model_candidates, deterministic
from ml.train_traffic_accident_v74_all_available import supervised as expanded, candidates
from ml.train_traffic_accident_v71_fixed_rubric import evaluate

ROOT = Path(__file__).resolve().parents[1]


def main():
    panel, wt, mt = load_processed()
    info = json.loads((ROOT / 'models/교통사고/93차.어린이교통사고수_2차모델/model_info.json').read_text(encoding='utf-8'))
    summaries, districts, contributions, interventions = [], [], [], []
    for h in (2, 3):
        features, specs, co, we = model_config(wt, mt, h)
        coef, weights = [co[f] for f in features], [we[f] for f in features]
        old = supervised(panel, features, h)
        numeric = [c for c in old if c.startswith(('현재__', '전년__', '증감__', '3년평균__'))]
        raw, _, _ = expanded(panel, features, h)
        cols = [c for c in raw if c not in {'연도', '자치구', '목표연도'} and not c.startswith('목표__')]
        pool = candidates(len(cols))
        names = sorted(panel['자치구'].unique())
        pred = pd.DataFrame({'연도': 2025, '자치구': names})
        for f in features:
            spec = specs[f]
            methods = [spec] if isinstance(spec, str) else list(spec[:2])
            results = {}
            for method in set(methods):
                if method in ('Persistence', 'Mean2', 'Mean3', 'DampedTrend3'):
                    hist = panel[panel['연도'] <= 2025-h]
                    results[method] = np.array([deterministic(hist[hist['자치구']==d].sort_values('연도')[f], method) for d in names])
                    continue
                data = raw if method in pool else old
                target = '목표__'+f
                train = data[(data['목표연도'] < 2025) & data[target].notna()]
                valid = data[data['목표연도']==2025].sort_values('자치구')
                own = [prefix+f for prefix in ('현재__', '전년__', '증감__', '3년평균__')]
                estimator = pool[method] if method in pool else model_candidates(numeric, own)[method]
                xtrain, xvalid = (train[cols], valid[cols]) if method in pool else (train, valid)
                fitted = deepcopy(estimator).fit(xtrain, train[target])
                results[method] = np.clip(fitted.predict(xvalid), train[target].quantile(.01), train[target].quantile(.99))
            pred[f] = results[spec] if isinstance(spec, str) else results[spec[0]]*spec[2]+results[spec[1]]*(1-spec[2])
        actual = panel[panel['연도']==2025][['연도','자치구',*features]].sort_values('자치구').reset_index(drop=True)
        metrics, detail = evaluate(actual, pred, features, coef, weights)
        expected = info['결과'][str(h)]['93차_2025독립']
        for key in metrics:
            assert np.isclose(metrics[key], expected[key], atol=1e-9), (h,key,metrics[key],expected[key])
        wrong = detail['실제_안전등급'] != detail['예측_안전등급']
        short = detail[['자치구','실제_안전등급','예측_안전등급','실제_종합위험점수','예측_종합위험점수']].copy()
        short.insert(0,'예측기간',h)
        short['오답'] = wrong
        districts.append(short)
        for f in features:
            errors = detail['예측_가중점수_'+f] - detail['실제_가중점수_'+f]
            for i, row in detail.iterrows():
                contributions.append({'예측기간':h,'자치구':row['자치구'],'오답':bool(wrong.iloc[i]),'지표':f,'가중점수오차':errors.iloc[i], '실제값':actual.loc[i,f], '예측값':pred.loc[i,f]})
            oracle = pred.copy()
            oracle[f] = actual[f].to_numpy()
            om, od = evaluate(actual, oracle, features, coef, weights)
            correct = od['실제_안전등급']==od['예측_안전등급']
            interventions.append({'예측기간':h,'지표':f,'오답지역_평균절대가중점수오차':errors[wrong].abs().mean(),'실제값대체_정확도':om['정확등급정확도'],'오답중개선수':int((wrong & correct).sum()),'정답중악화수':int((~wrong & ~correct).sum()),'순개선수':int(correct.sum()-(~wrong).sum())})
        summaries.append({'예측기간':h, **metrics})
        print(json.dumps({'horizon':h,'metrics':metrics,'wrong':short[short['오답']].to_dict('records')},ensure_ascii=False),flush=True)
    out = ROOT/'artifacts/v93_error_audit'
    out.mkdir(parents=True,exist_ok=True)
    rank = pd.DataFrame(interventions).sort_values(['예측기간','순개선수','오답지역_평균절대가중점수오차'],ascending=[True,False,False])
    path = out/'93차_등급오답분석.xlsx'
    with pd.ExcelWriter(path,engine='openpyxl') as writer:
        pd.DataFrame(summaries).to_excel(writer,sheet_name='성능재현',index=False)
        pd.concat(districts).to_excel(writer,sheet_name='자치구별정오답',index=False)
        rank.to_excel(writer,sheet_name='지표별진단',index=False)
        pd.DataFrame(contributions).to_excel(writer,sheet_name='지표별오차상세',index=False)
        pd.DataFrame({'주의사항':['2025년 기존 평가를 재현한 사후 진단이며 새로운 독립검증이 아님','실제값 대체는 원인 탐색용 가상 실험이며 실현 가능한 성능 향상 보장이 아님','등급은 25개 자치구 상대순위 기반이므로 한 지표 수정이 여러 지역에 영향을 줄 수 있음']}).to_excel(writer,sheet_name='분석범위',index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes='A2'
            sheet.auto_filter.ref=sheet.dimensions
            for cells in sheet.columns:
                sheet.column_dimensions[cells[0].column_letter].width=min(65,max(14,max(len(str(c.value or '')) for c in cells[:60])+4))
    print(rank.groupby('예측기간').head(6).to_json(orient='records',force_ascii=False),flush=True)
    print(path,flush=True)


if __name__ == '__main__':
    main()
