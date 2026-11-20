"""7번 대체 분석: 지역별 시간대/주중주말 매출 구성과 연령별 매출 구성의 관계.

시간×연령 교차 집계가 아니므로 특정 시간대 소비자의 연령을 추론할 수 없다.
"""
from pathlib import Path
import argparse
import html
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
from analyze_age_hypotheses import ROOT, read, check_unique, KEY, strict_sum, share


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,default=ROOT/'data/raw')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'results')
    args=parser.parse_args()
    sales=pd.concat([read(p) for p in sorted(args.raw.glob('서울시 상권분석서비스(추정매출-행정동)*.csv'))],ignore_index=True)
    sales=sales[sales['서비스_업종_코드'].str.startswith('CS1',na=False)].copy()
    check_unique(sales,KEY+['서비스_업종_코드'])
    age=[c for c in sales if c.startswith('연령대_') and c.endswith(('_금액','_건수'))]
    hours=[c for c in sales if c.startswith('시간대_')]
    days=[f'{d}_매출_{u}' for d in ['주중','주말'] for u in ['금액','건수']]
    cols=age+hours+days
    for c in cols: sales[c]=pd.to_numeric(sales[c],errors='coerce').where(lambda s:s>=0)
    data=strict_sum(sales,cols).merge(sales.groupby(KEY)['행정동_코드_명'].first().reset_index(),on=KEY,validate='one_to_one')
    # 건수 원본은 시작 시각이 '건수'로 잘못 표기되어 있다. 금액 구간과 종료 시각 대응을 명시한다.
    # 이 해석은 원본 데이터 사전 확인 전 잠정 해석이며 보고서와 매핑 CSV에 공개한다.
    intervals=['00~06','06~11','11~14','14~17','17~21','21~24']
    mapping=[]; pairs=[]
    for unit in ['금액','건수']:
        agecols=[c for c in age if c.endswith(unit)]
        den=data[agecols].sum(axis=1,min_count=len(agecols)); targets=[]
        for c in agecols:
            name=c+'_비중'; data[name]=share(data[c],den); targets.append(name)
        young=f'10_30대_매출_{unit}_비중'
        youngcols=[c for c in agecols if any(c.startswith(f'연령대_{a}_') for a in [10,20,30])]
        data[young]=share(data[youngcols].sum(axis=1,min_count=3),den); targets.append(young)
        source_hours=[]
        for interval in intervals:
            source=f'시간대_{interval}_매출_{unit}'
            if source not in data:
                source=f'시간대_건수~{interval.split("~")[1]}_매출_건수'
            if source not in data: raise ValueError(f'시간대 컬럼 없음: {source}')
            source_hours.append(source)
            mapping.append(dict(unit=unit,interval=interval,source=source,interpretation='잠정 종료시각 대응' if '건수~' in source else '원본 구간'))
        hden=data[source_hours].sum(axis=1,min_count=6)
        for interval,source in zip(intervals,source_hours):
            f=f'시간대_{interval}_매출_{unit}_비중'; data[f]=share(data[source],hden)
            pairs.extend(('time',unit,f,t) for t in targets)
        dcols=[f'{d}_매출_{unit}' for d in ['주중','주말']]
        dden=data[dcols].sum(axis=1,min_count=2)
        for source in dcols:
            f=source+'_비중'; data[f]=share(data[source],dden)
            pairs.extend(('weekday_weekend',unit,f,t) for t in targets)
    rows=[]
    for q,frame in data.groupby(KEY[0]):
        for group,unit,f,t in pairs:
            pair=frame[[f,t]].replace([np.inf,-np.inf],np.nan).dropna()
            status='ok' if len(pair)>=20 and (pair.nunique()>=2).all() else 'insufficient_or_constant'
            rho,p=spearmanr(pair[f],pair[t]) if status=='ok' else (np.nan,np.nan)
            rows.append(dict(group=group,unit=unit,quarter=q,feature=f,target=t,n=len(pair),excluded=len(frame)-len(pair),rho=rho,p_value=p,status=status))
    result=pd.DataFrame(rows); result['q_bh']=np.nan
    ordered=result.p_value.dropna().sort_values()
    if len(ordered):
        adj=ordered.to_numpy()*len(ordered)/np.arange(1,len(ordered)+1)
        result.loc[ordered.index,'q_bh']=np.minimum.accumulate(adj[::-1])[::-1].clip(0,1)
    valid=result[result.status=='ok']
    summary=valid.groupby(['group','unit','feature','target']).agg(median_rho=('rho','median'),min_rho=('rho','min'),max_rho=('rho','max'),
        quarters=('quarter','nunique'),min_n=('n','min'),max_n=('n','max'),positive=('rho',lambda x:(x>0).mean()),negative=('rho',lambda x:(x<0).mean())).reset_index()
    out=args.output/('hypothesis7_revised_'+pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True,exist_ok=False)
    for name,frame in [('analysis_data',data),('correlations',result),('summary',summary),('time_column_mapping',pd.DataFrame(mapping))]:
        frame.to_csv(out/f'{name}.csv',index=False,encoding='utf-8-sig')
    sns.set_theme(style='whitegrid'); fonts={f.name for f in font_manager.fontManager.ttflist}
    for font in ['Malgun Gothic','NanumGothic','Noto Sans CJK KR','AppleGothic']:
        if font in fonts: plt.rcParams['font.family']=font; break
    plt.rcParams['axes.unicode_minus']=False
    notes=('행정동 외식업의 분기별 지역 간 상관입니다. 연령×시간×요일 교차 자료가 아니므로 금·토 야간 소비자의 나이를 검증하지 않습니다. '
           '젊은 층은 10·20·30대 매출 합산 비중이며 고유 소비자 수나 평균 나이가 아닙니다. '
           '시간대 비중은 6개 시간대 합계, 주중주말 비중은 두 구분 합계, 연령 비중은 연령 확인 매출 합계가 분모입니다. '
           '각 비중은 합계 제약을 가지며 주중·주말 관계는 서로 보완적이라 독립 증거가 아닙니다. '
           '구간 길이와 주중/주말 일수 차이를 보정한 시간당·일평균 매출 비교는 아닙니다. '
           '분기별 Spearman 중앙값은 통합 상관계수가 아니며 인과성·공간 의존성·상권 규모를 통제하지 않았습니다. '
           '시간대 건수 원본은 시작 시각이 훼손된 컬럼명이라 종료 시각을 금액 구간과 대응했습니다. '
           '건수 시간대 결과는 이 대응을 전제로 한 잠정 결과이며 time_column_mapping.csv를 확인하세요. '
           '원본 결측은 0 대체하지 않고 관측 업종 중 결측이 있으면 합계도 제외했습니다. q값은 공간 의존성 미보정 참고값입니다.')
    index='<!doctype html><meta charset="utf-8"><h1>가설 7 변경 분석</h1><p>'+html.escape(notes)+'</p>'
    for group,title in [('time','시간대별 매출 구성과 연령 구성'),('weekday_weekend','주중·주말 매출 구성과 연령 구성')]:
        folder=out/group; folder.mkdir(); sub=summary[summary.group==group]
        sub.to_csv(folder/'summary.csv',index=False,encoding='utf-8-sig')
        focus=sub[sub.target.str.startswith('10_30대')]
        report=f'<!doctype html><meta charset="utf-8"><h1>{title}</h1><p>{html.escape(notes)}</p><h2>젊은 층 비중과의 관계</h2>'+focus.to_html(index=False)
        index+=f'<h2><a href="{group}/report.html">{title}</a></h2>'+focus.to_html(index=False)
        for unit in ['금액','건수']:
            s=sub[sub.unit==unit]; matrix=s.pivot(index='feature',columns='target',values='median_rho')
            if matrix.empty: continue
            fig,ax=plt.subplots(figsize=(14,7))
            sns.heatmap(matrix,annot=True,fmt='.2f',vmin=-1,vmax=1,cmap='RdBu_r',ax=ax)
            ax.set_title(title+f' / {unit} / 분기별 ρ 중앙값')
            fig.tight_layout(); fig.savefig(folder/f'heatmap_{unit}.png',dpi=150); plt.close(fig)
            report+=f'<img src="heatmap_{unit}.png" style="max-width:100%">'
        report+='<h2>전체 관계</h2>'+sub.to_html(index=False)
        (folder/'report.html').write_text(report,encoding='utf-8')
    index+='<p>5번 1인 가구 가설은 자료 확보 전까지 보류합니다.</p>'
    (out/'report.html').write_text(index,encoding='utf-8')
    print(out); print(summary[summary.target.str.startswith('10_30대')][['unit','feature','median_rho','min_rho','max_rho','min_n','max_n']].to_string(index=False))


if __name__=='__main__': main()
