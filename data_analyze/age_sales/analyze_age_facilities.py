"""연령대 가설 1만 분석: 행정동 시설 비율과 외식업 연령별 매출 비중.

실행: python data_analyze/age_sales/analyze_age_facilities.py
매출 건수는 고유 고객 수가 아니며 금액 비중은 소비자 수 비중이 아니다.
"""
from pathlib import Path
import argparse
import sys
import html
import warnings
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'data_analyze/attraction'))
from attraction_analyze import prepare, correlate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,default=ROOT/'data/raw')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'results')
    args=parser.parse_args()
    # 기존 분석과 동일한 외식업·시설 비율 정의 및 중복키 검증을 재사용한다.
    data,targets,hypotheses,quality,industries=prepare(args.raw,'food')
    targets=[c for c in targets if c.endswith('_총계대비비율')]
    hypotheses=[(1,feature,'unspecified') for number,feature,_ in hypotheses if number==3]
    # 2026년 시설 자료만으로 매출이 있다고 오인하지 않도록 연령 매출 관측 분기로 제한한다.
    age_columns=[f for _,f,_ in hypotheses]
    data=data[data[age_columns].notna().any(axis=1)].copy()
    result=correlate(data,targets,hypotheses)
    valid=result[result.status.eq('ok')]
    summary=valid.groupby(['feature','target']).agg(
        median_rho=('spearman_rho','median'),min_rho=('spearman_rho','min'),max_rho=('spearman_rho','max'),
        quarters=('quarter','nunique'),min_n=('n','min'),max_n=('n','max'),
        positive_fraction=('spearman_rho',lambda s:(s>0).mean()),
        negative_fraction=('spearman_rho',lambda s:(s<0).mean())).reset_index()
    summary['direction_consistency']=summary[['positive_fraction','negative_fraction']].max(axis=1)
    # 説明用の基準であり、統計的有意性や因果関係の採択ではない。
    summary['exploratory_candidate']=summary.median_rho.abs().ge(.3)&summary.direction_consistency.ge(.8)
    summary=summary.sort_values('median_rho',key=lambda s:s.abs(),ascending=False)
    out=args.output/('hypothesis1_'+pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True,exist_ok=False)
    # 品質指標は実際に分析した行に合わせて再計算する。
    quality=pd.DataFrame([{'facility_ratio':c,'rows':len(data),'missing':int(data[c].isna().sum()),
                           'ratio_over_100':int(data[c].gt(100).sum())} for c in targets])
    for name,frame in [('analysis_data',data[['기준_년분기_코드','행정동_코드','행정동_코드_명']+age_columns+targets]),
                        ('correlations_by_quarter',result),('summary',summary),('facility_quality',quality),
                        ('selected_industries',industries)]:
        frame.to_csv(out/f'{name}.csv',index=False,encoding='utf-8-sig')
    sns.set_theme(style='whitegrid')
    fonts={f.name for f in font_manager.fontManager.ttflist}
    font=next((f for f in ['Malgun Gothic','NanumGothic','Noto Sans CJK KR','AppleGothic'] if f in fonts),None)
    if font: plt.rcParams['font.family']=font
    else: warnings.warn('한글 폰트를 설치하세요.')
    plt.rcParams['axes.unicode_minus']=False
    report='<!doctype html><meta charset="utf-8"><title>소비 연령 가설 1</title><h1>집객시설 비율과 연령별 외식업 매출 비중</h1>'
    report+='<p>행정동·같은 분기 간 Spearman 탐색 분석. 시설이 소비 연령을 결정한다는 인과 검증이나 연령대 군집 분류는 아닙니다. '
    report+='시설별 수/원본 집객시설 수 비율을 사용하며 세부합과 총계가 달라 합계 100% 구성비가 아닙니다. '
    report+='시설 결측을 0으로 채우지 않으므로 관측 시설이 있는 일부 지역에 편향될 수 있습니다. '
    report+='연령 비중=해당 연령 매출/연령이 확인된 전체 매출(금액·건수 별도). '
    report+='연령 비중은 합계 100%이므로 한 연령의 증가가 다른 연령의 감소와 연결되며 독립 증거가 아닙니다. '
    report+='건수는 고유 고객 수가 아니고 금액은 객단가의 영향을 받습니다. 시설 비율 증가도 실제 시설 수 증가와 다릅니다. '
    report+='지역 규모·업종 구성·공간 의존성을 통제하지 않았으며 q값도 공간 의존성 미보정 참고값입니다. '
    report+='|ρ 중앙값|≥0.3 및 방향 일치≥80%는 요약용 탐색 기준일 뿐 가설 채택 기준이 아닙니다.</p>'
    for unit in ['건수','금액']:
        sub=summary[summary.feature.str.contains(f'_매출_{unit}_')]
        report+=f'<h2>연령별 매출 {unit} 비중</h2>'
        notable=sub[sub.exploratory_candidate]
        report+=f'<p>유효 조합 {len(sub)}개 중 탐색 기준 해당 {len(notable)}개. 아래는 절대 중앙값 순 상위 8개이며 전체 조합도 별도 표에 제공합니다.</p>'
        report+=notable.head(8).to_html(index=False)
        if not sub.empty:
            matrix=sub.pivot(index='target',columns='feature',values='median_rho')
            fig,ax=plt.subplots(figsize=(13,10))
            sns.heatmap(matrix,annot=True,fmt='.2f',vmin=-1,vmax=1,center=0,cmap='RdBu_r',ax=ax)
            ax.set_title(f'시설 비율 ↔ 연령별 매출 {unit} 비중: 분기별 ρ 중앙값')
            fig.tight_layout(); fig.savefig(out/f'heatmap_{unit}.png',dpi=150); plt.close(fig)
            report+=f'<img src="heatmap_{unit}.png" style="max-width:100%">'
        report+='<details><summary>전체 조합</summary>'+sub.to_html(index=False)+'</details>'
    report+='<h2>시설 자료 품질</h2>'+quality.to_html(index=False)+'<p>2~10번 가설은 이번 실행에서 분석하지 않았습니다.</p>'
    (out/'report.html').write_text(report,encoding='utf-8')
    print(out)
    print(summary[summary.feature.str.contains('_매출_건수_')].head(8).to_string(index=False))


if __name__=='__main__': main()
