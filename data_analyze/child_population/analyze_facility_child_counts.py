"""시설 수와 같은 연도 아동 거주 인구수의 관계. 증가율 분석과 별도 저장한다."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GROUPS = {'infant':'영유아(0~5세)', 'elementary':'초등 학령기(6~11세)',
          'middle':'중등 학령기(12~14세)', 'high':'고등 학령기(15~17세)'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    root = parser.parse_args().root
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S_%f')
    out = root/'data/processed/child_population'/('facility_child_counts_'+stamp)
    out.mkdir(parents=True, exist_ok=False)

    def save(frame, name):
        # Excel에서도 한글이 깨지지 않도록 UTF-8 BOM을 붙여 저장한다.
        frame.to_csv(out/name, index=False, encoding='utf-8-sig')

    # 前回の検証済み4年齢群パネルを再利用。施設の現況集計は使用しない。
    candidates = sorted((root/'data/processed/child_population').glob('*/population_panel_2021_2025.csv'), reverse=True)
    source = next(p for p in candidates if all('children_'+g in pd.read_csv(p,nrows=0).columns for g in GROUPS))
    pop = pd.read_csv(source,dtype={'dong_code':str})
    assert not pop.duplicated(['dong_code','year']).any()
    facility_path = root/'data/raw/서울시 상권분석서비스(집객시설-행정동).csv'
    raw = pd.read_csv(facility_path,encoding='cp949',dtype=str)
    fs = list(raw.columns[3:])
    for f in fs:
        raw[f] = pd.to_numeric(raw[f].str.replace(',','',regex=False),errors='raise')
    raw['dong_code'] = raw['행정동_코드']+'00'
    raw['year'] = raw['기준_년분기_코드'].str[:4].astype(int)
    assert not raw.duplicated(['dong_code','기준_년분기_코드']).any()
    # 시설 수는 분기마다 누적해 더할 값이 아니다. 매년 4분기 한 관측만 선택한다.
    annual = raw[raw['기준_년분기_코드'].str.endswith('4')]
    joined = pop.merge(annual,on=['dong_code','year'],how='outer',validate='one_to_one',indicator=True)
    save(joined[joined['_merge']!='both'],'unmatched_rows.csv')
    data = joined[joined['_merge']=='both'].drop(columns='_merge').copy()
    save(data,'analysis_panel.csv')
    quality = raw.groupby('dong_code')[fs].nunique(dropna=False)
    save(pd.DataFrame({'feature':fs,'missing_rows':raw[fs].isna().sum().values,
        'dongs_with_temporal_change':quality.gt(1).sum().values}),'facility_quality.csv')
    rows=[]
    for year,frame in data.groupby('year'):
        for g,label in GROUPS.items():
            for f in fs:
                # 원본 빈칸의 의미가 확정되지 않았으므로 기본 분석에서는 결측 유지.
                # 0 대입 결과는 별도 민감도 분석이며 기본값과 혼합하지 않는다.
                for policy in ['observed_only','blank_as_zero_sensitivity']:
                    x=frame[f] if policy=='observed_only' else frame[f].fillna(0)
                    pair=pd.DataFrame({'x':x,'y':frame['children_'+g]}).dropna()
                    if len(pair)<30 or pair.x.nunique()<2 or pair.y.nunique()<2:
                        continue
                    common=dict(year=int(year),age=label,feature=f,missing_policy=policy,n=len(pair))
                    for method,fn in [('spearman',stats.spearmanr),('pearson',stats.pearsonr),('kendall',stats.kendalltau)]:
                        r,p=fn(pair.x,pair.y)
                        rows.append(dict(**common,method=method,coefficient=r,p_nominal=p))
                    # 아동수가 타깃이므로 아동비율을 보정변수로 넣지 않는다.
                    # 전체 인구수에는 아동도 포함되므로 아래 결과는 규모를 조건부로
                    # 고정한 관계이며 독립적 인과효과가 아니다. 구 차이도 함께 보정한다.
                    base=frame.loc[pair.index]
                    z=np.column_stack([np.ones(len(pair)),base.total_population.rank(),
                        pd.get_dummies(base.gu,drop_first=True,dtype=float).to_numpy()])
                    ranked=pair.rank().to_numpy()
                    resid=ranked-z@np.linalg.lstsq(z,ranked,rcond=None)[0]
                    r=np.corrcoef(resid.T)[0,1] if resid.std(axis=0).min()>1e-8 else np.nan
                    rows.append(dict(**common,method='partial_rank_population_gu',coefficient=r,p_nominal=np.nan))
    result=pd.DataFrame(rows)
    result['q_bh']=np.nan
    # 연도·연령군·방법·결측정책별 다중검정 보정. 공간 의존 보정은 아니므로 참고용.
    for _,b in result[result.p_nominal.notna()].groupby(['year','age','method','missing_policy']):
        order=np.argsort(b.p_nominal.to_numpy());p=b.p_nominal.to_numpy()[order]
        q=np.minimum(1,np.minimum.accumulate((p*len(p)/np.arange(1,len(p)+1))[::-1])[::-1])
        result.loc[b.index.to_numpy()[order],'q_bh']=q
    save(result,'method_comparison_by_year.csv')
    primary=result.query("method=='spearman' and missing_policy=='observed_only'")
    summary=primary.groupby(['age','feature']).agg(years=('year','nunique'),median_rho=('coefficient','median'),
        min_rho=('coefficient','min'),max_rho=('coefficient','max'),min_n=('n','min')).reset_index()
    adjusted=result.query("method=='partial_rank_population_gu' and missing_policy=='observed_only'").groupby(['age','feature']).coefficient.median().rename('median_partial_rank')
    summary=summary.merge(adjusted,on=['age','feature'])
    summary['abs_median']=summary.median_rho.abs()
    summary=summary.sort_values(['age','abs_median'],ascending=[True,False])
    save(summary,'summary.csv')
    plt.rcParams['font.family']='Malgun Gothic'
    plt.rcParams['axes.unicode_minus']=False
    images=[]
    for g,label in GROUPS.items():
        fig,axes=plt.subplots(1,2,figsize=(15,11))
        for ax,method,title in zip(axes,['spearman','partial_rank_population_gu'],['단순 Spearman','전체 인구·구 보정 부분 순위 상관']):
            t=result[(result.age==label)&(result.method==method)&(result.missing_policy=='observed_only')].pivot(index='feature',columns='year',values='coefficient').reindex(fs)
            im=ax.imshow(np.ma.masked_invalid(t.to_numpy()),vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto')
            ax.set_xticks(range(len(t.columns)),t.columns)
            ax.set_yticks(range(len(t)),t.index)
            for i in range(len(t)):
                for j in range(len(t.columns)):
                    v=t.iloc[i,j]
                    ax.text(j,i,'—' if pd.isna(v) else f'{v:.2f}',ha='center',va='center',fontsize=8,color='white' if abs(v)>.6 else 'black')
            ax.set_title(title)
            fig.colorbar(im,ax=ax,shrink=.6)
        fig.suptitle(label+' 거주 인구수와 시설 수 / 연도별 횡단면 비교')
        fig.tight_layout();name=g+'_heatmap.png';fig.savefig(out/name,dpi=150);plt.close(fig);images.append(name)
    report='''<meta charset="utf-8"><title>아동 거주 인구수·시설 수</title>
    <style>body{font-family:Malgun Gothic;margin:30px;line-height:1.7}img{width:100%;max-width:1500px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:5px}</style>
    <h1>집객시설 수와 아동 거주 인구수의 관계</h1>
    <p>타깃은 증가율이 아닌 같은 연도의 0~5 / 6~11 / 12~14 / 15~17세 인구수입니다. 실제 재학 상태는 아닙니다.</p>
    <h2>분석 방법·해석</h2><ul>
    <li>매년 4분기 시설 수와 같은 연도 인구를 코드로 1:1 연결. 정확한 인구 기준월과 과거 경계 변경은 추가 확인 필요.</li>
    <li>시설 지표는 원본 전 기간 동일합니다. 연도 라벨이 과거 시설 스냅샷임을 보장하지 않으며 시설 증설 효과나 예측력을 뜻하지 않습니다.</li>
    <li>Spearman은 순위 단조관계, Pearson은 선형관계, Kendall은 순서 일치도를 비교합니다.</li>
    <li>부분 순위 상관은 전체 인구 규모와 자치구 차이를 보정합니다. 아동 비율은 타깃과 직접 연결되므로 보정하지 않았습니다. 전체 인구는 아동을 포함하므로 부분-전체 관계와 과도한 보정 가능성에 유의하세요.</li>
    <li>빈칸 제외가 기본이고 0 대입은 민감도 분석만 제공합니다. 표본 30 미만·상수는 미산출. 지표별 유효 지역이 다릅니다.</li>
    <li>연도별 독립 횡단면으로 산출하고 반복 관측을 합쳐 표본 수를 부풀리지 않았습니다. 연도 중앙값은 기술 요약입니다. p/q는 공간 의존 미보정 참고값입니다.</li>
    <li>높은 상관은 인구가 많은 곳에 시설이 많은 현상 또는 역인과일 수 있습니다. 육아환경의 질·향후 인구 증가·인과효과·가중치로 바로 해석하지 않습니다.</li></ul>'''
    report+=f'<p>결합 {len(data):,}행, {data.dong_code.nunique()}개 행정동</p>'
    report+='<h2>연령군별 상위 5개: 연도별 계수 중앙값</h2>'+summary.groupby('age',sort=False).head(5).round(3).to_html(index=False)
    report+=''.join(f'<img src="{im}">' for im in images)
    report+='<h2>전체 결과 (유효 표본 n 포함)</h2>'+result.round(4).to_html(index=False)
    (out/'report.html').write_text(report,encoding='utf-8')
    (out/'provenance.json').write_text(json.dumps({'population':str(source),'facilities':str(facility_path),'target':'same_year_child_population_count'},ensure_ascii=False,indent=2),encoding='utf-8')
    print('OUTPUT',out)
    print(data.groupby('year').size().to_string())
    print(summary.groupby('age',sort=False).head(3).round(3).to_string(index=False))


if __name__=='__main__':
    main()
