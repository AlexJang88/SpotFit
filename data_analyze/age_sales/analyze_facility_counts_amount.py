"""시설 절대 수와 연령별 외식업 매출 절대 금액: 분기별 상관 및 규모 보정 탐색.

실행: python data_analyze/age_sales/analyze_facility_counts_amount.py
기존 비율 분석을 수정하지 않고 새 디렉터리에 결과를 저장한다.
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
from analyze_age_hypotheses import ROOT, read, check_unique, KEY, strict_sum


def adjusted(frame, feature, target, controls):
    """동일한 완전 관측 표본으로 보정 전/후를 비교한다.

    부분 순위상관: X·Y와 통제변수 각각을 순위로 바꾼 뒤 통제변수 회귀 잔차끼리 상관.
    보조 OLS: log1p 변환 후 표준화한 매출을 시설 수 및 통제변수에 회귀한다.
    시설 종류를 모두 한꺼번에 넣지 않고 시설별로 같은 통제변수 집합을 사용한다.
    """
    pair=frame[[feature,target]+controls].replace([np.inf,-np.inf],np.nan).dropna()
    result={'n':len(pair),'raw_same_sample':np.nan,'partial_rank_rho':np.nan,
            'standardized_log_beta':np.nan,'condition_number':np.nan,'status':'insufficient'}
    if len(pair)<max(30,10*(len(controls)+1)): return result
    if (pair[[feature,target]].nunique()<2).any(): result['status']='constant'; return result
    # 상수 통제변수는 절편과 중복되므로 제거한다. 남은 설계행렬도 랭크를 검사한다.
    used=[c for c in controls if pair[c].nunique()>1]
    ranks=pair[[feature,target]+used].rank(method='average')
    z=np.column_stack([np.ones(len(pair)),ranks[used].to_numpy()])
    if np.linalg.matrix_rank(z)<z.shape[1]: result['status']='singular_controls'; return result
    x=ranks[feature].to_numpy(); y=ranks[target].to_numpy()
    rx=x-z@np.linalg.lstsq(z,x,rcond=None)[0]; ry=y-z@np.linalg.lstsq(z,y,rcond=None)[0]
    if np.linalg.norm(rx)<1e-8 or np.linalg.norm(ry)<1e-8: result['status']='zero_residual'; return result
    result['raw_same_sample']=spearmanr(pair[feature],pair[target]).statistic
    result['partial_rank_rho']=np.corrcoef(rx,ry)[0,1]
    logs=np.log1p(pair[[feature]+used+[target]])
    standardized=(logs-logs.mean())/logs.std(ddof=0)
    design=np.column_stack([np.ones(len(pair)),standardized[[feature]+used].to_numpy()])
    result['condition_number']=np.linalg.cond(design)
    if np.linalg.matrix_rank(design)<design.shape[1]: result['status']='singular_regression'; return result
    if result['condition_number']>1e8: result['status']='ill_conditioned'; return result
    result['standardized_log_beta']=np.linalg.lstsq(design,standardized[target].to_numpy(),rcond=None)[0][1]
    result['status']='ok'; return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,default=ROOT/'data/raw')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'results')
    args=parser.parse_args()
    def load(kind): return read(args.raw/f'서울시 상권분석서비스({kind}-행정동).csv')
    sales=pd.concat([read(p) for p in sorted(args.raw.glob('서울시 상권분석서비스(추정매출-행정동)*.csv'))],ignore_index=True)
    sales=sales[sales['서비스_업종_코드'].str.startswith('CS1',na=False)].copy()
    check_unique(sales,KEY+['서비스_업종_코드'])
    targets=[c for c in sales if c.startswith('연령대_') and c.endswith('_매출_금액')]
    for c in targets: sales[c]=pd.to_numeric(sales[c],errors='coerce').where(lambda s:s>=0)
    data=strict_sum(sales,targets).merge(sales.groupby(KEY)['행정동_코드_명'].first().reset_index(),on=KEY,validate='one_to_one')
    facility=load('집객시설'); check_unique(facility,KEY)
    facilities=[c for c in facility if c.endswith('_수')]
    # 총 집객시설 수도 포함하되 세부항목 합계와 동일하다고 가정하지 않는다.
    data=data.merge(facility[KEY+facilities],on=KEY,how='left',validate='one_to_one')
    controls=[]
    for kind,c in [('상주인구','총_상주인구_수'),('직장인구','총_직장_인구_수'),('길단위인구','총_유동인구_수')]:
        d=load(kind); check_unique(d,KEY)
        data=data.merge(d[KEY+[c]],on=KEY,how='left',validate='one_to_one'); controls.append(c)
    shops=load('점포'); shops=shops[shops['서비스_업종_코드'].str.startswith('CS1',na=False)].copy()
    check_unique(shops,KEY+['서비스_업종_코드'])
    shops['전체_점포_수']=pd.to_numeric(shops['전체_점포_수'],errors='coerce').where(lambda s:s>=0)
    data=data.merge(strict_sum(shops,['전체_점포_수']),on=KEY,how='left',validate='one_to_one')
    for c in facilities+controls+['전체_점포_수']:
        data[c]=pd.to_numeric(data[c],errors='coerce').where(lambda s:s>=0)
    out=args.output/('facility_counts_amount_'+pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True,exist_ok=False)
    data.to_csv(out/'analysis_data.csv',index=False,encoding='utf-8-sig')
    raw_rows=[]; adjusted_rows=[]
    for quarter,frame in data.groupby(KEY[0]):
        for feature in facilities:
            for target in targets:
                pair=frame[[feature,target]].dropna()
                status='ok' if len(pair)>=20 and (pair.nunique()>=2).all() else 'insufficient_or_constant'
                rho,p=spearmanr(pair[feature],pair[target]) if status=='ok' else (np.nan,np.nan)
                raw_rows.append(dict(quarter=quarter,feature=feature,target=target,n=len(pair),rho=rho,p_value=p,status=status))
                for mode,cs in [('population',controls),('population_shops',controls+['전체_점포_수'])]:
                    # 점포 통제는 원본이 존재하는 분기만. 2021~2024에 2025년 값을 붙이지 않는다.
                    if frame[cs].dropna().empty: continue
                    adjusted_rows.append(dict(quarter=quarter,mode=mode,feature=feature,target=target,
                                              **adjusted(frame,feature,target,cs)))
    raw=pd.DataFrame(raw_rows); adj=pd.DataFrame(adjusted_rows)
    raw['q_bh']=np.nan; ordered=raw.p_value.dropna().sort_values()
    if len(ordered):
        q=ordered.to_numpy()*len(ordered)/np.arange(1,len(ordered)+1)
        raw.loc[ordered.index,'q_bh']=np.minimum.accumulate(q[::-1])[::-1].clip(0,1)
    summary=raw[raw.status=='ok'].groupby(['feature','target']).agg(median_rho=('rho','median'),min_rho=('rho','min'),
        max_rho=('rho','max'),quarters=('quarter','nunique'),min_n=('n','min'),max_n=('n','max')).reset_index()
    corrected=adj[adj.status=='ok'].groupby(['mode','feature','target']).agg(
        raw_same_sample=('raw_same_sample','median'),partial_rank_rho=('partial_rank_rho','median'),
        standardized_log_beta=('standardized_log_beta','median'),quarters=('quarter','nunique'),
        min_n=('n','min'),max_n=('n','max'),min_partial=('partial_rank_rho','min'),max_partial=('partial_rank_rho','max')).reset_index()
    quality=pd.DataFrame([{'feature':f,'rows':len(data),'missing':int(data[f].isna().sum()),'observed_zero':int(data[f].eq(0).sum())} for f in facilities+controls+['전체_점포_수']])
    for name,frame in [('raw_correlations',raw),('raw_summary',summary),('adjusted_by_quarter',adj),
                       ('adjusted_summary',corrected),('data_quality',quality)]:
        frame.to_csv(out/f'{name}.csv',index=False,encoding='utf-8-sig')
    sns.set_theme(style='whitegrid'); fonts={f.name for f in font_manager.fontManager.ttflist}
    for font in ['Malgun Gothic','NanumGothic','Noto Sans CJK KR','AppleGothic']:
        if font in fonts: plt.rcParams['font.family']=font; break
    plt.rcParams['axes.unicode_minus']=False
    def heat(frame,value,title,filename):
        if frame.empty: return
        matrix=frame.pivot(index='feature',columns='target',values=value)
        fig,ax=plt.subplots(figsize=(14,11)); sns.heatmap(matrix,annot=True,fmt='.2f',vmin=-1,vmax=1,cmap='RdBu_r',ax=ax)
        ax.set_title(title); fig.tight_layout(); fig.savefig(out/filename,dpi=150); plt.close(fig)
    heat(summary,'median_rho','시설 수 ↔ 연령별 매출금액: 분기별 Spearman 중앙값','raw_heatmap.png')
    for mode in ['population','population_shops']:
        s=corrected[corrected['mode']==mode]
        heat(s,'partial_rank_rho',f'{mode}: 규모 통제 후 부분 순위상관 중앙값',mode+'_heatmap.png')
        heat(s,'raw_same_sample',f'{mode}: 통제 후와 동일 표본의 보정 전 상관',mode+'_same_sample_raw.png')
    # 최신 매출 분기에서 시설별·연령별 산점도를 모두 제공해 유리한 사례만 선택하지 않는다.
    latest=data[data[KEY[0]]==data[KEY[0]].max()]
    for feature in facilities:
        fig,axes=plt.subplots(2,3,figsize=(15,9))
        for ax,target in zip(axes.flat,targets):
            pair=latest[[feature,target]].dropna()
            ax.scatter(pair[feature],pair[target],alpha=.4,s=12)
            ax.set_xscale('symlog',linthresh=1); ax.set_yscale('symlog',linthresh=1)
            ax.set_xlabel(feature); ax.set_ylabel(target); ax.set_title(f'n={len(pair)}')
        fig.suptitle(f'{int(latest[KEY[0]].iloc[0])}: {feature} / 0을 유지하는 symlog 축')
        fig.tight_layout(); fig.savefig(out/f'scatter_{feature}.png',dpi=120); plt.close(fig)
    notes=('타겟은 연령별 외식업 매출 절대 금액만 사용하며 비중·건수는 제외했습니다. '
           '원본 업종 행이 없는 경우 그 업종의 실제 매출이 0인지 알 수 없으며 관측 업종을 합산합니다. '
           '시설 결측은 0으로 바꾸지 않으며 표본이 시설마다 다릅니다. '
           '기본 Spearman은 분기별 행정동 간 비교입니다. 인구 보정은 상주·직장·유동인구, '
           '인구+점포 보정은 외식업 전체 점포 수를 추가합니다. 점포 자료는 2025년만 매출과 겹칩니다. '
           '보정 전후는 각 모드의 동일 완전관측 표본으로 비교하세요. 전체기간 기본 상관과 2025년 보정값의 차이를 순수 보정 효과로 해석하지 마세요. '
           '부분 순위상관은 순위값을 통제변수에 회귀한 잔차 간 Pearson 상관입니다. '
           '표준화 log1p OLS 계수는 보조 회귀이며 인과효과·탄력성·매출 증가율이 아닙니다. '
           '유동인구 등이 시설 효과의 매개변수일 수 있어 보정값이 무조건 더 올바른 것도 아닙니다. '
           '면적·입지·공간 의존성·업종 세부 구성을 모두 통제하지 못했습니다. '
           '분기별 계수 중앙값은 통합 상관이 아니며 다른 연령보다 계수가 크다고 연령별 효과 차이가 입증되지는 않습니다. '
           '원시 상관의 p/q는 공간 의존성 미보정 참고값입니다. 보정 분석은 유의성 검정 없이 기술적 계수만 제공합니다.')
    report='<!doctype html><meta charset="utf-8"><title>시설 수와 연령별 매출금액</title><h1>시설 절대 수와 연령별 매출 절대 금액</h1><p>'+html.escape(notes)+'</p>'
    report+='<h2>동일 표본 보정 전·후 비교</h2>'+corrected.to_html(index=False)+'<h2>기본 상관</h2>'+summary.to_html(index=False)
    for p in sorted(out.glob('*.png')): report+=f'<h3>{html.escape(p.stem)}</h3><img src="{p.name}" style="max-width:100%">'
    report+='<h2>결측·0값 현황</h2>'+quality.to_html(index=False)
    (out/'report.html').write_text(report,encoding='utf-8')
    print(out)
    print(corrected[(corrected['mode']=='population_shops')&corrected.feature.isin(['대학교_수','은행_수','집객시설_수'])].to_string(index=False))


if __name__=='__main__': main()
