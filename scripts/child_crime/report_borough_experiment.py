"""Verify the locked CV winners and write a readable comparison report."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from ml.training.compare_child_crime_borough_v2 import *

LABELS={
 'ridge_direct':'Ridge 직접 Y 예측','ridge_residual':'Ridge Persistence 잔차',
 'components':'범죄·아동인구 분리 Ridge','recursive':'1년 잔차모델 3회 순차 적용',
 'panel_mean_shock':'자치구·연도 고정효과(미래 공통충격 0)',
 'panel_last_shock':'자치구·연도 고정효과(최근 충격 50%)',
 'extra_direct':'ExtraTrees 직접','extra_residual':'ExtraTrees 잔차',
 'hist_direct':'HistGradientBoosting 직접','hist_residual':'HistGradientBoosting 잔차'}

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def main():
    panel=pd.read_csv(OUT/'panel.csv');groups=json.loads((OUT/'feature_groups.json').read_text(encoding='utf-8'))
    columns=list(dict.fromkeys(CORE+[c for v in groups.values() for c in v]));gus=sorted(panel.gu.unique())
    results=json.loads((OUT/'results.json').read_text(encoding='utf-8'));lines=[];summary=[];tests=[]
    assert not panel.duplicated(['year','gu']).any()
    assert panel.groupby('year').size().eq(25).all()
    comparable=panel.dropna(subset=['businesses','mdis_total'])
    assert np.array_equal(comparable.businesses,comparable.mdis_total)
    tests.extend(['400-row panel: 16 years x 25 boroughs; no duplicate keys',
                  f'MDIS record counts equal official borough business totals: {len(comparable)} borough-years'])
    lines+=['# 아동 지역범죄 노출 예측 실험 v2','',
      'Y = log(1 + 자치구 5대범죄 발생건수 / 0~17세 주민등록 아동인구 × 10,000). 아동 피해 발생률이 아닌 지역 노출 대리지표다.',
      '', '## 데이터와 검증 조건','',
      '- 범죄 정답은 2014~2025년 서울 25개 자치구. 인구는 행정동을 합산한 뒤 자치구 공표 합계와 일치 검증.',
      '- 기존 파일의 자치구 합계가 있는 통계는 해당 합계행만 사용하여 동·구 합계 중복을 방지. 비율은 분자/분모로 계산.',
      '- 모델 입력 기준연도는 2015년부터. 원자료 패널은 2010~2025년 400행이며 범죄 정답 없는 연도는 학습에서 제외.',
      '- 2022·2023·2024년을 순차 검증, 2025년은 최종 평가. h년 예측 시 학습 정답 연도 ≤ 예측기준연도=평가연도−h.',
      '- 범죄·인구는 기준연도까지 관측됐다고 가정. 그 외 자료는 1년 지연 입력. 과거 실제 공표일·수정 전 빈티지는 확보하지 않아 완전한 당시 실서비스 재현은 아니다.',
      '- 가구·인구이동·조직형태 지연값과 일부 초기 범죄 lag는 초기에 결측. 학습 구간의 중앙값만 사용하며 전 기간 보간/미래값 역채움은 하지 않음.',
      '- 9개 의미 기반 Feature 그룹의 512개 부분집합 전수 비교. 개별 60개 열의 2^60 조합 전수탐색은 아님. y, log(1+범죄건수), log(1+아동인구)는 공통 입력.',
      '- Ridge alpha=1/10/100; ExtraTrees 40트리·최대깊이5; HistGradientBoosting 60회·최대잎5. 하이퍼파라미터는 사전 고정.',
      '- 1년 19×512=9,728개, 3년 22×512=11,264개 설정, 설정당 3개 시간순 fold. 합계 20,992개 설정.',
      '- 기존 서비스 모델과 다른 평가 연도/정보 지연 조건이므로 기존 발표 지표와 직접 개선율 비교 금지.',
      '', '## 원자료 주의사항','',
      '- `1인 가구` 파일은 가구형태별 통계로, 실제 1인가구 수는 없음. 가구구조 묶음으로 사용.',
      '- 조직형태 XLS는 상위 분류명이 빠져 같은 항목명이 반복되어 세부 분류를 추측하지 않음. MDIS의 명시적 조직형태 코드로 대체.',
      '- 집객시설 폴더는 전국사업체조사. 업종별 사업체 레코드를 집계하며 비공개 종사자수(*)를 0으로 치환하지 않음. 집계 사업체 총수는 공표 자치구 합계와 일치.',
      '- 사업체수는 2019년 823,624→2020년 1,211,053으로 급변. 전후 통계 비교에 구조적 단절 가능성을 고려해야 함.',
      '- 인허가일·폐업일로 연말 업소수를 복원. 날짜 미상 폐업/비정상 상태 5개 단란주점 제외. 과거 이전 주소·휴업 이력은 복원하지 못함.',
      '', '## 점수와 정확도의 해석','',
      '- 주 선택 기준: 검증 구간을 합친 log-Y RMSE. 작은 값이 좋음. 최종 평가의 원래 노출지수 MAPE도 별도 제시.',
      '- R²와 Spearman은 정답률(%)이 아님. 회귀 예측에는 단일 정답률이 없으므로 오차와 순위·등급 지표를 함께 사용.',
      '- 등급은 매년 25개 구 상대 5분위: 노출이 낮은 5개 구=A, 다음 5개=B, … 높은 5개=E. 인접등급 인정 없이 완전일치만 계산.',
      '- 안전 상위5 일치율은 실제 노출이 가장 낮은 5개 구와 예측 상위5의 교집합/5. 방향 정확도는 기준연도 대비 증가·감소 일치율.',
      '- 검증 조합 탐색 규모에 비해 시간 구간이 적어 선택 과적합 위험이 있음. 최종 평가는 25개 구×1개 연도이며 일반화 확정 근거가 아님.', '']
    for result in results:
        h=result['horizon'];dest=OUT/f'{h}y';pairs=make_pairs(panel,h);test=pairs[pairs.target_year==2025];origin=2025-h
        selected=result['winner_cv'];b=result['baseline_cv'];hb=result['baseline_holdout'];hs=result['winner_holdout']
        df=pd.read_csv(dest/'all_comparisons.csv')
        # Actual completion-order record reconstruction; model order within each task is fixed.
        records=[{'stage':0,'method':'Persistence','RMSE':b['RMSE'],'improvement_pct':0}];running=b['RMSE']
        all_results=[]
        for line in (dest/f'completed_{h}y.jsonl').read_text(encoding='utf-8').splitlines():
            block=json.loads(line)['results'];all_results.extend(block)
            for row in block:
                if row['RMSE']<running-1e-12:
                    records.append({'stage':len(records),'previous_RMSE':running,
                      'step_improvement_pct':(running-row['RMSE'])/running*100,
                      'improvement_pct':(b['RMSE']-row['RMSE'])/b['RMSE']*100,**row})
                    running=row['RMSE']
        pd.DataFrame(records).to_csv(dest/'execution_improvements.csv',index=False,encoding='utf-8-sig')
        method_rows=[]
        for method in sorted(df.method.unique()):
            candidate=min((r for r in all_results if r['method']==method),key=lambda r:(r['RMSE'],r['MAE'],r['n_features'],r['alpha']))
            train_pairs=make_pairs(panel,1) if method=='recursive' else pairs
            train=train_pairs[train_pairs.target_year<=origin]
            c=context(train,test,columns,gus);idx=[columns.index(col) for col in candidate['features']]
            model=fit(c,idx,method,candidate['alpha']);p,detail=predict(model,test,h)
            m=metrics(test.y_future.to_numpy(),p,test.target_year.to_numpy())
            row={'method':method,'label':LABELS[method],'alpha':candidate['alpha'],'groups':candidate['groups'],
                 'CV_RMSE':candidate['RMSE'],'CV_improvement_pct':(b['RMSE']-candidate['RMSE'])/b['RMSE']*100,
                 **{'holdout_'+k:v for k,v in m.items()},'holdout_improvement_pct':(hb['RMSE']-m['RMSE'])/hb['RMSE']*100}
            for key,val in detail.items():
                actual=np.expm1(test['log_'+key+'_future'].to_numpy())
                row[key+'_MAPE_pct']=float(np.mean(np.abs(val-actual)/actual)*100)
            method_rows.append(row)
            # Perturbing unavailable future labels must never change predictions.
            tampered=test.copy()
            for col in ['y_future','log_crime_future','log_child_future']:
                tampered[col]=9999
            p2,_=predict(model,tampered,h)
            assert np.allclose(p,p2),method
            tests.append(f'{h}y/{method}: future-label perturbation invariant')
            if method==selected['method']:
                assert np.isclose(m['RMSE'],hs['RMSE']), (method,m['RMSE'],hs['RMSE'])
        methods=pd.DataFrame(method_rows).sort_values('CV_RMSE');methods.to_csv(dest/'method_comparison_holdout.csv',index=False,encoding='utf-8-sig')
        # Holdout comparison is descriptive; it never changes the CV winner.
        lines += [f'## {h}년 예측','',f"CV 선택 모델: **{LABELS[selected['method']]}**, alpha={selected['alpha']}, 입력 {selected['n_features']}개.",
                  '', '선택 묶음: '+(', '.join(selected['groups']) or '공통 입력만'),'',
                  table(['지표','Persistence','CV 선택 모델'],[
                    ['CV Y-RMSE',f"{b['RMSE']:.6f}",f"{selected['RMSE']:.6f}"],
                    ['2025 Y-RMSE',f"{hb['RMSE']:.6f}",f"{hs['RMSE']:.6f}"],
                    ['2025 Y-MAE',f"{hb['MAE']:.6f}",f"{hs['MAE']:.6f}"],
                    ['2025 R²',f"{hb['R2']:.4f}",f"{hs['R2']:.4f}"],
                    ['2025 원 노출지수 MAPE(%)',f"{hb['raw_MAPE_pct']:.2f}",f"{hs['raw_MAPE_pct']:.2f}"],
                    ['2025 순위 Spearman',f"{hb['Spearman']:.4f}",f"{hs['Spearman']:.4f}"],
                    ['2025 순위 평균오차(위)',f"{hb['rank_MAE']:.2f}",f"{hs['rank_MAE']:.2f}"],
                    ['2025 등급 완전일치(%)',f"{hb['exact_grade_accuracy_pct']:.1f}",f"{hs['exact_grade_accuracy_pct']:.1f}"],
                    ['2025 안전 상위5 일치(%)',f"{hb['lowest_risk_top5_overlap_pct']:.1f}",f"{hs['lowest_risk_top5_overlap_pct']:.1f}"]]),'',
                  f"CV RMSE 개선 {result['cv_improvement_pct']:.2f}%, 2025 RMSE 개선 {result['holdout_improvement_pct']:.2f}%. 음수는 악화.",
                  '', '판정: '+('기준모델 대비 CV·최종 평가 모두 개선. 배포는 하지 않음.' if result['acceptance_check_passed'] else '최종 후보가 기준모델 우위를 입증하지 못함. 공식 예측 권고는 Persistence 유지; 저장 모델은 실험용.'),'',
                  '### 기법별 비교','',
                  '각 기법 안에서 CV로 최선 조합을 고정한 뒤 2025 성능을 표시. 2025 결과를 보고 우승 모델을 다시 고르지 않음.','',
                  table(['기법','alpha','CV RMSE','CV 개선(%)','2025 RMSE','2025 개선(%)'],[
                      [r['label'],r['alpha'],f"{r['CV_RMSE']:.5f}",f"{r['CV_improvement_pct']:.2f}",f"{r['holdout_RMSE']:.5f}",f"{r['holdout_improvement_pct']:.2f}"] for r in methods.to_dict('records')]),'',
                  '### 실제 실행 중 최고성능 갱신 이력','',
                  table(['단계','기법','추가 그룹 전체','CV RMSE','최초기준 대비 개선(%)'],[
                      [r['stage'],LABELS.get(r['method'],r['method']),', '.join(r.get('groups',[])) or '공통 입력',f"{r['RMSE']:.6f}",f"{r['improvement_pct']:.2f}"] for r in records]),'']
        ablations=[]
        names=list(groups)
        for group in selected['groups']:
            reduced=selected['mask'] & ~(1<<names.index(group))
            peer=next(r for r in all_results if r['mask']==reduced and r['method']==selected['method'] and r['alpha']==selected['alpha'])
            ablations.append({'removed_group':group,'without_RMSE':peer['RMSE'],'with_RMSE':selected['RMSE'],
                              'conditional_gain_pct':(peer['RMSE']-selected['RMSE'])/peer['RMSE']*100})
        pd.DataFrame(ablations).to_csv(dest/'winner_group_ablation.csv',index=False,encoding='utf-8-sig')
        lines+=['### 최종 조합에서 한 그룹씩 제외한 비교','',
                '동일 기법·alpha를 유지한 조건부 비교다. 그룹 간 상관관계가 있어 인과적 중요도로 해석할 수 없다.','',
                table(['제외 그룹','제외 시 RMSE','포함 시 RMSE','포함 개선(%)'],[
                    [r['removed_group'],f"{r['without_RMSE']:.6f}",f"{r['with_RMSE']:.6f}",f"{r['conditional_gain_pct']:.2f}"] for r in ablations]),'']
        summary.append({'horizon':h,'method':selected['method'],'groups':'+'.join(selected['groups']),
                        'CV_RMSE':selected['RMSE'],'CV_improvement_pct':result['cv_improvement_pct'],
                        'holdout_improvement_pct':result['holdout_improvement_pct'],
                        **{'holdout_'+k:v for k,v in hs.items()},'accepted':result['acceptance_check_passed']})
        # Stored model reload must reproduce the saved future forecasts.
        saved=joblib.load(MODELS/f'cv_best_{h}y.joblib')
        latest=panel[panel.year==2025];fp,_=predict(saved,latest,h)
        expected=pd.read_csv(dest/'future_predictions.csv')
        assert np.allclose(fp,expected.predicted_y)
        tests.append(f'{h}y: joblib reload reproduces future predictions')
        leaked_train=pairs[pairs.target_year<=origin+1]
        try:context(leaked_train,test,columns,gus)
        except AssertionError:tests.append(f'{h}y: future training target correctly rejected')
        else:raise AssertionError('Leakage guard missing')
    lines+=['## 시도한 전체 Feature','', '자치구 식별자는 모든 모델에 제공. 일반 Ridge/트리는 구 더미, 패널은 비정규화 고정효과로 처리.','',
            '공통 입력: `y`, `log_crime`, `log_child`. 인구구조의 log_child는 공통 입력과 중복 제거.','']
    for group,features in groups.items():
        lines += [f'### {group}','']+['- `'+f+'`' for f in features]+['']
    lines+=['업종 코드: G 도소매, I 숙박음식, H 운수창고, P 교육, Q 보건복지, R 예술스포츠여가, S 기타서비스, C 제조, L 부동산. 조직형태: 1 개인, 2 회사법인, 3 회사외법인, 4 비법인단체, 5 국가·지방자치단체.','',
             '## 재현 및 저장물','',
             '전처리: `python scripts/child_crime/build_borough_experiment_panel.py`',
             '', '학습: `python ml/training/compare_child_crime_borough_v2.py --jobs 2` (완료 그룹은 JSONL에서 재개. 입력/설정 변경 시 새 실험 디렉터리를 사용해야 함.)',
             '', '보고서: `python scripts/child_crime/report_borough_experiment.py`',
             '', '`all_comparisons.csv`: 전체 설정 / `execution_improvements.csv`: 실행 중 개선 기록 / `improvement_history.csv`: 그룹 복잡도 순서로 재구성한 개선 기록 / `method_comparison_holdout.csv`: 기법별 비교 / `winner_group_ablation.csv`: 그룹 제거 비교.',
             '', '최종 후보만 `models/experiments/child_crime_borough_v2/cv_best_1y.joblib`, `cv_best_3y.joblib` 저장. 서비스 v1 미변경. 딕셔너리 모델이며 학습 스크립트의 `predict(model, frame, horizon)`으로 추론.',
             '', '2025년까지 재학습한 2026·2028 전망은 각 `future_predictions.csv`에 저장. 검증 탈락 시 `recommended_y`는 Persistence를 사용하며 `predicted_y`는 실험 후보 예측이다.','',
             '## 검증 실행 결과','']+['- '+t for t in tests]
    (OUT/'experiment_report.md').write_text('\n'.join(lines),encoding='utf-8')
    pd.DataFrame(summary).to_csv(OUT/'final_summary.csv',index=False,encoding='utf-8-sig')
    (OUT/'verification.json').write_text(json.dumps({'checks_passed':tests},ensure_ascii=False,indent=2),encoding='utf-8')
    feature_names={
      'y':'현재 log(1+아동인구 1만명당 5대범죄)', 'log_crime':'log(1+5대범죄 건수)',
      'log_child':'log(1+0~17세 인구)', 'child_share':'전체인구 중 0~17세 비율',
      'age0_5_share':'0~5세 비율','age6_12_share':'6~12세 비율','age13_17_share':'13~17세 비율',
      'elderly_share':'65세 이상 비율','유흥주점':'연말 유흥주점 수','단란주점':'연말 단란주점 수',
      '유흥주점_per10k':'전체인구 1만명당 유흥주점','단란주점_per10k':'전체인구 1만명당 단란주점',
      'workers':'전체 사업체 종사자수','self_family_share':'자영업주·무급가족 종사자 비율',
      'regular_share':'상용근로자 비율','temporary_share':'임시·일용근로자 비율','other_workers_share':'기타종사자 비율',
      'businesses':'사업체 수','business_density':'면적 1㎢당 사업체수','worker_density':'면적 1㎢당 종사자수',
      'workers_per_business':'사업체당 평균 종사자수','households':'총가구 수','ordinary_share':'총가구 중 일반가구 비율',
      'household_size':'일반가구당 가구원수','foreign_household_share':'총가구 중 외국인가구 비율',
      'institution_members':'집단시설가구 가구원수','births_rate':'인구 1천명당 출생신고','deaths_rate':'인구 1천명당 사망신고',
      'natural_growth':'인구 대비 출생−사망 비율',
    }
    for code,name in {'G':'도소매','I':'숙박음식','H':'운수창고','P':'교육','Q':'보건복지','R':'예술스포츠여가','S':'기타서비스','C':'제조','L':'부동산'}.items():
        feature_names['industry_'+code+'_share']=name+'업 사업체 비율'
    for code,name in {'1':'개인사업체','2':'회사법인','3':'회사외법인','4':'비법인단체','5':'국가·지방자치단체'}.items():
        feature_names['org_'+code+'_share']=name+' 사업체 비율'
    dictionary=[]
    for col in columns:
        base=col.removesuffix('_available_lag1');desc=feature_names.get(base,base)
        if base.startswith('migration_'):desc='인구 1천명당 '+base.removeprefix('migration_').removesuffix('_rate')
        for suffix,tag in [('_lag1',' 1년 전'),('_lag2',' 2년 전'),('_change',' 전년 차이')]:
            if base.endswith(suffix):desc=feature_names.get(base.removesuffix(suffix),base.removesuffix(suffix))+tag
        dictionary.append({'column':col,'description':desc,'group':'공통' if col in CORE else next(g for g,v in groups.items() if col in v),
                           'availability_lag_years':1 if col.endswith('_available_lag1') else 0,
                           'selected_1y':col in results[0]['winner_cv']['features'],
                           'selected_3y':col in results[1]['winner_cv']['features']})
    pd.DataFrame(dictionary).to_csv(OUT/'feature_dictionary.csv',index=False,encoding='utf-8-sig')
    with pd.ExcelWriter(OUT/'모델비교_전체결과.xlsx',engine='openpyxl') as writer:
        pd.DataFrame(summary).to_excel(writer,sheet_name='최종성능',index=False)
        pd.DataFrame(dictionary).to_excel(writer,sheet_name='전체60개Feature',index=False)
        for h in [1,3]:
            for filename,label in [('method_comparison_holdout','기법비교'),('execution_improvements','개선이력'),
                                    ('winner_group_ablation','Feature제거비교'),('holdout_predictions','2025실제예측'),
                                    ('future_predictions','미래예측'),('all_comparisons','전체실험')]:
                data=pd.read_csv(OUT/f'{h}y'/f'{filename}.csv')
                data.to_excel(writer,sheet_name=f'{h}년_{label}',index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes='A2';sheet.auto_filter.ref=sheet.dimensions
    import hashlib
    manifest={}
    for path in [OUT/'panel.csv',OUT/'feature_groups.json',ROOT/'ml/training/compare_child_crime_borough_v2.py',
                 ROOT/'scripts/child_crime/build_borough_experiment_panel.py']:
        manifest[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT/'reproducibility_hashes.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False))
    print('CHECKS',len(tests))

if __name__=='__main__':main()
