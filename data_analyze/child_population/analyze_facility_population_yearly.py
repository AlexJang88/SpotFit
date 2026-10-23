"""집객시설 수준과 다음 연도 아동 인구 증가율의 연도별 탐색 분석.

분기 시설은 재고(stock)이므로 합산하지 않고 시작 연도 4분기를 사용한다.
시점 표시는 과거여도 값이 전 기간 같다면 역사적 시설 변화로 해석하지 않는다.
"""
from pathlib import Path
from datetime import datetime
import argparse
import html
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GROUPS = {'infant': '영유아(0~5세)', 'elementary': '초등 학령기(6~11세)',
          'middle': '중등 학령기(12~14세)', 'high': '고등 학령기(15~17세)'}


def bh(p):
    """同一 연도·연령군·방법의 다중 비교를 BH로 보정한다(공간 의존은 미해결)."""
    a = np.asarray(p, dtype=float)
    order = np.argsort(a)
    result = np.empty(len(a))
    result[order] = np.minimum(1, np.minimum.accumulate(
        (a[order] * len(a) / np.arange(1, len(a)+1))[::-1])[::-1])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root
    out = root / 'data/processed/child_population' / ('facilities_yearly_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True, exist_ok=False)

    def save(d, name):
        d.to_csv(out/name, index=False, encoding='utf-8-sig')

    # 파생 인구 파일을 명시적으로 탐색하고 4개 연령군이 존재하는 최신 파일만 선택한다.
    candidates = sorted((root/'data/processed/child_population').glob('*/population_panel_2021_2025.csv'), reverse=True)
    source = next(p for p in candidates if all('children_'+g in pd.read_csv(p, nrows=0).columns for g in GROUPS))
    pop = pd.read_csv(source, dtype={'dong_code': str})
    assert not pop.duplicated(['dong_code','year']).any()
    rawpath = root/'data/raw/서울시 상권분석서비스(집객시설-행정동).csv'
    d = pd.read_csv(rawpath, encoding='cp949', dtype=str)
    features = list(d.columns[3:])
    for f in features:
        d[f] = pd.to_numeric(d[f].str.replace(',', '', regex=False), errors='raise')
    # 상권 서비스 8자리 코드는 주민등록 10자리 코드의 앞 8자리와 대응한다.
    # 이름만으로 연결하지 않으며 미대응과 코드 중복을 별도 검증한다.
    d['dong_code'] = d['행정동_코드'] + '00'
    d['year'] = d['기준_년분기_코드'].str[:4].astype(int)
    d['quarter'] = d['기준_년분기_코드'].str[-1].astype(int)
    assert not d.duplicated(['dong_code','year','quarter']).any()
    variation = d.groupby('dong_code')[features].nunique(dropna=False)
    save(variation.reset_index(), 'facility_temporal_unique_counts.csv')
    save(pd.DataFrame({'feature': features, 'missing_rows': d[features].isna().sum().values,
                       'dongs_with_temporal_change': (variation > 1).sum().values}), 'feature_quality.csv')
    annual = d[d.quarter == 4].copy()
    save(annual, 'annual_facilities_q4.csv')
    # 정확히 t+1년을 연결한다. 단순 shift는 관측 공백을 1년으로 오인할 수 있다.
    end = pop[['dong_code','year'] + ['children_'+g for g in GROUPS]].copy()
    end['year'] -= 1
    end = end.rename(columns={'children_'+g: 'end_'+g for g in GROUPS})
    panel = pop.merge(end, on=['dong_code','year'], validate='one_to_one')
    for g in GROUPS:
        panel['growth_'+g] = (panel['end_'+g] / panel['children_'+g].replace(0,np.nan)-1)*100
    joined = panel.merge(annual, on=['dong_code','year'], how='left', validate='one_to_one', indicator=True)
    save(joined[joined['_merge'] != 'both'], 'unmatched_population_rows.csv')
    data = joined[joined['_merge'] == 'both'].drop(columns='_merge').copy()
    save(data, 'analysis_panel.csv')
    save(annual[~annual.dong_code.isin(pop.dong_code)], 'unmatched_facility_rows.csv')
    # 연도별 횡단면을 따로 계산하여 동일 동을 독립 표본으로 반복 합산하지 않는다.
    rows = []
    for year, frame in data.groupby('year'):
        for g, label in GROUPS.items():
            for policy in ['observed_only', 'blank_as_zero_sensitivity']:
                for f in features:
                    for scale in ['count', 'per1000_total_population']:
                        x = frame[f].copy()
                        if policy != 'observed_only':
                            x = x.fillna(0)
                        if scale != 'count':
                            x = x / frame.total_population.replace(0,np.nan)*1000
                        pair = pd.DataFrame({'x':x, 'y':frame['growth_'+g]}).replace([np.inf,-np.inf],np.nan).dropna()
                        if len(pair)<30 or pair.x.nunique()<2 or pair.y.nunique()<2:
                            continue
                        for method, func in [('spearman',stats.spearmanr),('pearson',stats.pearsonr),('kendall',stats.kendalltau)]:
                            r,p = func(pair.x,pair.y)
                            rows.append(dict(year=int(year),end_year=int(year)+1,age=label,feature=f,scale=scale,
                                             missing_policy=policy,method=method,coefficient=r,p_nominal=p,n=len(pair)))
                        # 부분 순위 상관은 규모·기초 아동비율·구의 공통 차이를 보정한다.
                        # 인과관계가 되지는 않으며 p값은 별도로 산출하지 않는다.
                        base = frame.loc[pair.index]
                        controls = pd.DataFrame({'population':base.total_population.rank(),
                            'child_share':(base['children_'+g]/base.total_population).rank()})
                        controls = pd.concat([controls,pd.get_dummies(base.gu,drop_first=True,dtype=float)],axis=1)
                        z = np.column_stack([np.ones(len(pair)),controls.to_numpy(float)])
                        ranks = pair.rank().to_numpy()
                        resid = ranks-z@np.linalg.lstsq(z,ranks,rcond=None)[0]
                        r = np.corrcoef(resid.T)[0,1] if np.min(resid.std(axis=0))>1e-8 else np.nan
                        rows.append(dict(year=int(year),end_year=int(year)+1,age=label,feature=f,scale=scale,
                            missing_policy=policy,method='partial_rank',coefficient=r,p_nominal=np.nan,n=len(pair)))
    result = pd.DataFrame(rows)
    result['q_bh'] = np.nan
    valid = result[result.p_nominal.notna()]
    for _, block in valid.groupby(['year','age','missing_policy','scale','method']):
        result.loc[block.index,'q_bh'] = bh(block.p_nominal)
    save(result,'method_comparison_by_year.csv')
    main = result.query("method == 'spearman' and scale == 'count' and missing_policy == 'observed_only'")
    stability = main.groupby(['age','feature']).agg(years=('year','nunique'),median_rho=('coefficient','median'),
                 min_rho=('coefficient','min'),max_rho=('coefficient','max'),min_n=('n','min')).reset_index()
    stability['same_direction'] = (stability.min_rho>0)|(stability.max_rho<0)
    stability['abs_median'] = stability.median_rho.abs()
    stability = stability.sort_values(['age','abs_median'],ascending=[True,False])
    save(stability,'year_stability_summary.csv')
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    plots = []
    for i, (g,label) in enumerate(GROUPS.items()):
        table = main[main.age==label].pivot(index='feature',columns='year',values='coefficient').reindex(features)
        fig,ax = plt.subplots(figsize=(8,11))
        im=ax.imshow(np.ma.masked_invalid(table.to_numpy()),vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto')
        ax.set_xticks(range(len(table.columns)),[f'{y}→{y+1}' for y in table.columns])
        ax.set_yticks(range(len(table)),table.index)
        for a in range(len(table)):
            for b in range(len(table.columns)):
                v=table.iloc[a,b]
                ax.text(b,a,'미산출' if pd.isna(v) else f'{v:.3f}',ha='center',va='center',fontsize=9)
        ax.set_title(label+' : 시설 수 vs 다음 연도 인구 증가율\nSpearman / 빈칸 제외 / 인과·미래예측 아님')
        fig.colorbar(im,ax=ax);fig.tight_layout()
        name=f'{i+1}_{g}_heatmap.png';fig.savefig(out/name,dpi=150);plt.close(fig);plots.append(name)
    notes = '''<h2>해석 및 전처리</h2><ul>
    <li>연도 t의 4분기 시설 수와 t→t+1년 인구 증감률을 연결. 분기 시설 수는 합산하지 않음.</li>
    <li>중요: 원본에서 모든 동의 시설 지표가 2021~2026 전 기간 동일(결측 포함)합니다. 연도별 시설 변화 효과를 검증한 결과가 아닙니다. 실제 과거 스냅샷인지 제공기관 확인이 필요합니다.</li>
    <li>0~5, 6~11, 12~14, 15~17세 구분은 실제 재학 상태가 아닙니다. 증감률은 출생·연령 이동·전출입을 포함하며 순유입률이 아닙니다.</li>
    <li>인구는 기존 핵심 파일 파생 패널 사용. 8자리 행정동 코드에 00을 붙여 10자리 코드로 일치 결합. 미대응 행 별도 보존. 과거 경계 변경과 인구 파일의 정확한 기준월은 추가 확인 필요.</li>
    <li>빈칸은 미상으로 보존해 지표별 유효 표본만 사용. 0으로 가정한 결과는 민감도 분석으로만 별도 제공. 철도역 등 상수·전부 결측, n&lt;30 지표는 미산출. 시설 세부항목 합계가 전체 집객시설 수와 같지 않아 구성비로 해석하지 않음.</li>
    <li>Spearman은 치우친 수량의 단조 관계, Pearson은 극단값에 민감한 선형 관계, Kendall은 순서 일치, 부분 순위 상관은 인구 규모·기초 아동비율·구 차이를 보정합니다.</li>
    <li>시설 수와 전체 인구 1000명당 시설 수를 별도 비교. 지표별 표본 n이 다르며 결측 지역이 무작위라는 보장은 없습니다.</li>
    <li>각 연도별 횡단면 분석이며 전체 연도를 독립 표본처럼 합산하지 않음. p/BH q는 공간 의존을 보정하지 않은 탐색용입니다. 연도 중앙값은 요약일 뿐 메타분석이나 학습 가중치가 아닙니다.</li>
    </ul>'''
    report='<meta charset="utf-8"><title>집객시설·아동 인구 연도별 분석</title><style>body{font-family:Malgun Gothic;margin:36px;line-height:1.7}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ddd;padding:5px}img{max-width:900px;width:100%}</style><h1>집객시설과 연도별 아동 인구 증가율</h1>'+notes
    report+=f'<p>결합 표본 {len(data):,}행 / {data.dong_code.nunique()}개 동. 인구 출처: {html.escape(str(source))}</p>'
    report+='<h2>연령군별 연도 중앙 상관계수 상위 5개 (절댓값 순)</h2>'+stability.groupby('age',sort=False).head(5).round(3).to_html(index=False)
    report+=''.join(f'<img src="{name}">' for name in plots)
    report+='<h2>연도별 방법 비교 (원수량·결측 제외)</h2>'+result.query("scale == 'count' and missing_policy == 'observed_only'").round(4).to_html(index=False)
    (out/'report.html').write_text(report,encoding='utf-8')
    (out/'provenance.json').write_text(json.dumps({'population_source':str(source),'facility_source':str(rawpath),'temporally_varying_dongs':int((variation>1).any(axis=1).sum())},ensure_ascii=False,indent=2),encoding='utf-8')
    print('OUTPUT',out)
    print('MATCHED',len(data),data.dong_code.nunique(),'UNMATCHED',len(joined)-len(data))
    print(stability.groupby('age',sort=False).head(3).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
