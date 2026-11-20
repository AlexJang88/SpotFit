"""연령 매출 가설 2~10: 행정동 외식업 기준. 자료가 없는 가설은 대리지표로 바꾸지 않는다.

실행: python data_analyze/age_sales/analyze_age_hypotheses.py
"""
from pathlib import Path
import sys
import argparse
import html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'data_analyze/weekday_sales'))
from weekday_sales_analyze import read, check_unique, KEY

TITLES={2:'업종 비율과 소비 연령',3:'상주인구 연령 구성과 소비 연령',4:'유동인구와 젊은 소비층 비중',
        5:'1인 가구와 소비 연령',6:'아파트 가격과 젊은 소비층 비중',7:'금·토 야간 소비 연령',
        8:'영업·폐업 점포 평균 영업기간과 소비 연령',9:'소비지출과 소비 연령',10:'직장인구와 소비 연령'}


def share(numerator,denominator):
    """0분모와 결측은 0%가 아니라 계산 불가능한 값으로 남긴다."""
    return numerator/denominator.where(denominator>0)*100


def strict_sum(frame,cols):
    """관측 업종 중 한 업종이라도 결측이면 부분합을 전체값으로 취급하지 않는다."""
    g=frame.groupby(KEY); result=g[cols].sum(min_count=1)
    for col in cols: result.loc[g[col].count()!=g.size(),col]=np.nan
    return result.reset_index()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,default=ROOT/'data/raw')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'results')
    args=parser.parse_args()
    def load(kind):
        d=read(args.raw/f'서울시 상권분석서비스({kind}-행정동).csv')
        check_unique(d,KEY+(['서비스_업종_코드'] if '서비스_업종_코드' in d else []))
        return d
    files=sorted(args.raw.glob('서울시 상권분석서비스(추정매출-행정동)*.csv'))
    sales=pd.concat([read(p) for p in files],ignore_index=True)
    sales=sales[sales['서비스_업종_코드'].str.startswith('CS1',na=False)].copy()
    check_unique(sales,KEY+['서비스_업종_코드'])
    agecols=[c for c in sales if c.startswith('연령대_') and c.endswith(('_매출_금액','_매출_건수'))]
    for c in agecols: sales[c]=pd.to_numeric(sales[c],errors='coerce').where(lambda s:s>=0)
    data=strict_sum(sales,agecols)
    names=sales.groupby(KEY)['행정동_코드_명'].first().reset_index()
    data=data.merge(names,on=KEY,validate='one_to_one')
    targets=[]
    for unit in ['금액','건수']:
        cols=[c for c in agecols if c.endswith(unit)]
        den=data[cols].sum(axis=1,min_count=len(cols))
        for c in cols:
            name=c+'_비중'; data[name]=share(data[c],den); targets.append(name)
        # 개방형 60대 이상 구간 때문에 임의 평균 나이는 만들지 않는다.
        # '젊다'는 10·20·30대 합산 비중으로 사전 정의하고 각 연령 비중도 함께 검토한다.
        young=[c for c in cols if any(c.startswith(f'연령대_{a}_') for a in (10,20,30))]
        name=f'10_30대_매출_{unit}_비중'
        data[name]=share(data[young].sum(axis=1,min_count=len(young)),den); targets.append(name)
    tests={}; notes={}
    def merge(d,cols):
        nonlocal data
        data=data.merge(d[KEY+cols],on=KEY,how='left',validate='one_to_one')
    shops=load('점포'); shops=shops[shops['서비스_업종_코드'].str.startswith('CS1',na=False)].copy()
    shops['전체_점포_수']=pd.to_numeric(shops['전체_점포_수'],errors='coerce').where(lambda s:s>=0)
    den=strict_sum(shops,['전체_점포_수']).rename(columns={'전체_점포_수':'관측외식업점포합'})
    shops=shops.merge(den,on=KEY,validate='many_to_one')
    shops['ratio']=share(shops['전체_점포_수'],shops['관측외식업점포합'])
    shops['feature']='업종점포비율_'+shops['서비스_업종_코드']+'_'+shops['서비스_업종_코드_명']
    wide=shops.pivot(index=KEY,columns='feature',values='ratio').reset_index()
    tests[2]=[c for c in wide if c not in KEY]; merge(wide,tests[2])
    notes[2]='외식업 내부의 관측 점포 수 대비 각 외식업종 점포 비율. 누락 업종 행을 0으로 채우지 않습니다. 2025년 4분기만이 아니라 2025년 네 분기를 사용합니다. 비중의 공통 분모와 업종별 고객 구성 차이에 주의하세요.'
    residents=load('상주인구')
    resident_cols=[c for c in residents if c.startswith('연령대_') and c.endswith('_상주인구_수')]
    tests[3]=[]
    for c in resident_cols:
        name=c+'_비중'; residents[name]=share(residents[c],residents['총_상주인구_수']); tests[3].append(name)
    merge(residents,tests[3])
    notes[3]='연령별 상주인구/총 상주인구와 연령별 매출비중을 비교합니다. 특히 같은 연령끼리의 관계를 별도 표로 제공합니다. 거주자와 실제 소비자가 동일 인물이라는 의미는 아닙니다.'
    flow=load('길단위인구'); tests[4]=['총_유동인구_수']; merge(flow,tests[4])
    notes[4]='젊은 층=10·20·30대 합산 매출비중으로 정의합니다. 유동인구와 이 비중의 양의 관계가 가설 예상 방향입니다. 고유 방문자 수나 평균 나이의 직접 측정은 아닙니다.'
    notes[5]='분석 불가: 원본에 총 가구·아파트 가구·비아파트 가구 수만 있고 1인 가구 수가 없습니다. 비아파트 가구 또는 인구/가구 평균으로 대체하지 않았습니다. 행정동·시점별 1인 가구 수 자료가 필요합니다.'
    apartment=load('아파트'); tests[6]=['아파트_평균_시가','아파트_평균시가_면적대비']
    apartment['아파트_평균_시가']=apartment['아파트_평균_시가'].where(lambda s:s>0)
    apartment[tests[6][1]]=apartment[tests[6][0]]/apartment['아파트_평균_면적'].where(lambda s:s>0)
    merge(apartment,tests[6]); notes[6]='주 지표는 원본 평균 시가, 보조는 평균 시가/평균 면적입니다. 낮은 가격에서 젊은 층 비중이 높다는 가설은 가격과 10~30대 비중의 음의 관계를 예상합니다. 면적대비 값은 개별 실거래 단가 평균이 아닙니다.'
    notes[7]='분석 불가: 요일별·시간대별·연령별 매출은 각각의 주변 합계이고 연령×요일×시간대 교차 집계가 없습니다. 금·토 매출과 야간 매출을 더하거나 곱해 금·토 야간 연령을 만들 수 없습니다. 교차 집계와 야간의 명시적 시간 정의가 필요합니다.'
    change=load('상권변화지표'); tests[8]=['운영_영업_개월_평균','폐업_영업_개월_평균']; merge(change,tests[8])
    notes[8]='지역 내 영업 중인 점포와 폐업 점포의 평균 영업기간 지표입니다. 폐업률·개별 매장 수명 예측이 아닙니다. 서울 전체 평균은 분기 내 상수라 제외했습니다. 설명변수의 업종 범위와 외식업 매출 대상이 일치한다고 가정하지 않습니다.'
    spend=load('소비'); tests[9]=['지출_총금액']
    for c in list(spend.columns):
        if c.endswith('_지출_총금액'):
            tests[9].append(c); name=c+'_총지출대비비중'; spend[name]=share(spend[c],spend['지출_총금액']); tests[9].append(name)
    merge(spend,tests[9]); notes[9]='전체 및 항목별 지출 규모와 항목별 총지출 대비 비중을 분리합니다. 지출의 모집단·산출 방식이 추정매출과 같다고 단정하지 않습니다. 지출과 매출은 중복·공통 규모 요인이 있어 인과 예측변수로 자동 채택하지 않습니다.'
    workers=load('직장인구'); tests[10]=['총_직장_인구_수']; merge(workers,tests[10])
    notes[10]='주 분석은 직장인구와 연령별 소비비중입니다. 특정 연령 매출 절대 규모도 보조로 제공합니다. 매출 건수는 고유 소비자 수가 아니며 규모 변수 간 상관이 커질 수 있습니다.'
    out=args.output/('hypotheses2_10_'+pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y%m%d_%H%M%S_%f')); out.mkdir(parents=True,exist_ok=False)
    data.to_csv(out/'analysis_data.csv',index=False,encoding='utf-8-sig')
    # 분기 반복 관측을 독립 표본으로 합치지 않고 동일 분기 행정동 간 관계를 구한다.
    rows=[]
    for h,features in tests.items():
        use_targets=targets+(agecols if h==10 else [])
        for quarter,frame in data.groupby(KEY[0]):
            for f in features:
                for t in use_targets:
                    pair=frame[[f,t]].replace([np.inf,-np.inf],np.nan).dropna()
                    status='ok' if len(pair)>=20 and (pair.nunique()>=2).all() else 'insufficient_or_constant'
                    rho,p=spearmanr(pair[f],pair[t]) if status=='ok' else (np.nan,np.nan)
                    rows.append(dict(hypothesis=h,quarter=quarter,feature=f,target=t,n=len(pair),excluded=len(frame)-len(pair),rho=rho,p_value=p,status=status))
    results=pd.DataFrame(rows)
    # 이번 2~10번의 모든 유효 검정을 하나의 family로 BH 보정한다. 공간 의존성 미보정 참고값이다.
    results['q_bh']=np.nan; ordered=results.p_value.dropna().sort_values()
    if len(ordered):
        q=ordered.to_numpy()*len(ordered)/np.arange(1,len(ordered)+1)
        results.loc[ordered.index,'q_bh']=np.minimum.accumulate(q[::-1])[::-1].clip(0,1)
    results.to_csv(out/'all_correlations.csv',index=False,encoding='utf-8-sig')
    sns.set_theme(style='whitegrid')
    fonts={f.name for f in font_manager.fontManager.ttflist}
    for f in ['Malgun Gothic','NanumGothic','Noto Sans CJK KR','AppleGothic']:
        if f in fonts: plt.rcParams['font.family']=f; break
    plt.rcParams['axes.unicode_minus']=False
    intro=('행정동·외식업 매출, 같은 분기의 지역 간 Spearman 탐색 분석입니다. '
           '연령 확인 매출 합계를 분모로 금액·건수 비중을 각각 계산합니다. 건수는 고유 고객 수가 아닙니다. '
           '비중은 합계 제약으로 서로 독립이 아니며 면적·상권규모·공간 의존성은 미통제입니다. '
           '결측은 0으로 채우지 않고 쌍별 제외합니다. rho 중앙값은 통합 상관계수가 아닙니다. '
           '|중앙값|≥0.3, 방향 일치≥80%는 요약용 기준이지 통계적 가설 채택 기준이 아닙니다.')
    index='<!doctype html><meta charset="utf-8"><h1>연령대 가설 2~10 통합 결과</h1><p>'+intro+'</p>'
    for h in range(2,11):
        folder=out/f'hypothesis{h}'; folder.mkdir()
        report=f'<!doctype html><meta charset="utf-8"><h1>{h}. {TITLES[h]}</h1><p>{intro}</p><p>{html.escape(notes[h])}</p>'
        index+=f'<h2><a href="hypothesis{h}/report.html">{h}. {TITLES[h]}</a></h2><p>{html.escape(notes[h])}</p>'
        if h not in tests:
            report+='<h2>판정: 자료 부족으로 분석 불가</h2>'
        else:
            subset=results[results.hypothesis==h]; valid=subset[subset.status=='ok']
            summary=valid.groupby(['feature','target']).agg(median_rho=('rho','median'),min_rho=('rho','min'),max_rho=('rho','max'),
                quarters=('quarter','nunique'),min_n=('n','min'),max_n=('n','max'),positive=('rho',lambda s:(s>0).mean()),negative=('rho',lambda s:(s<0).mean())).reset_index()
            summary['direction_consistency']=summary[['positive','negative']].max(axis=1)
            summary['exploratory_candidate']=summary.median_rho.abs().ge(.3)&summary.direction_consistency.ge(.8)
            summary=summary.sort_values('median_rho',key=lambda s:s.abs(),ascending=False)
            subset.to_csv(folder/'correlations.csv',index=False,encoding='utf-8-sig'); summary.to_csv(folder/'summary.csv',index=False,encoding='utf-8-sig')
            data[KEY+['행정동_코드_명']+tests[h]].groupby(KEY[0])[tests[h]].count().to_csv(folder/'coverage.csv',encoding='utf-8-sig')
            focus=summary[summary.target.str.endswith('_건수_비중')]
            if h in [4,6]: focus=summary[summary.target=='10_30대_매출_건수_비중']
            if h==3:
                focus=summary[[f.split('_')[1]==t.split('_')[1] and t.endswith('_건수_비중') for f,t in zip(summary.feature,summary.target)]]
            display=focus.head(8)
            report+='<h2>주요 결과 (건수 비중 중심)</h2>'+display.to_html(index=False)
            index+=display[['feature','target','median_rho','quarters','min_n','max_n']].to_html(index=False)
            print('HYPOTHESIS',h); print(display[['feature','target','median_rho','quarters']].head(5).to_string(index=False))
            if h in [4,6]:
                expected=1 if h==4 else -1
                for r in focus.itertuples():
                    message='예상 방향 경향' if r.median_rho*expected>=.3 and r.direction_consistency>=.8 else ('반대 방향 경향' if r.median_rho*expected<=-.3 and r.direction_consistency>=.8 else '뚜렷한 근거 부족')
                    report+=f'<p>{html.escape(r.feature)}: {message}. 이는 설명용 탐색 판단이며 인과 검증이 아닙니다.</p>'
            for unit in ['금액','건수']:
                s=summary[summary.target.str.endswith(f'_{unit}_비중')]
                if s.empty: continue
                matrix=s.pivot(index='feature',columns='target',values='median_rho')
                fig,ax=plt.subplots(figsize=(14,max(5,len(matrix)*.42)))
                sns.heatmap(matrix,annot=True,fmt='.2f',vmin=-1,vmax=1,cmap='RdBu_r',ax=ax)
                ax.set_title(f'가설 {h} / 매출 {unit} 비중 / 분기별 상관 중앙값')
                fig.tight_layout(); fig.savefig(folder/f'heatmap_{unit}.png',dpi=150); plt.close(fig)
                report+=f'<img src="heatmap_{unit}.png" style="max-width:100%">'
            report+='<h2>전체 관계</h2>'+summary.to_html(index=False)
        (folder/'report.html').write_text(report,encoding='utf-8')
    (out/'report.html').write_text(index,encoding='utf-8'); print('OUTPUT',out)


if __name__=='__main__': main()
