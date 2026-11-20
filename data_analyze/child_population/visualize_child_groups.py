"""기존 4개 연령구간 분석 결과를 다시 학습하지 않고 시각화한다.

실행: python data_analyze/child_population/visualize_child_groups.py
선택 편향을 줄이기 위해 주 그림은 매핑 품질 경고가 없는 동일 6개 지표로 고정한다.
"""
from pathlib import Path
from datetime import datetime
import html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
base = ROOT / 'data/processed/child_population'
# 단일 연령 분석의 최신 결과를 잘못 읽지 않도록 4개 그룹 컬럼까지 검사한다.
sources = []
for folder in sorted(base.iterdir()):
    path = folder / 'analysis_current_facilities_vs_2024_2025_growth.csv'
    if path.exists() and 'growth_high' in pd.read_csv(path, nrows=0).columns:
        sources.append(folder)
if not sources:
    raise ValueError('먼저 analyze_child_age_groups.py를 실행하세요.')
source = sources[-1]
out = source / ('visualizations_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
out.mkdir()
data = pd.read_csv(source / 'analysis_current_facilities_vs_2024_2025_growth.csv')
results = pd.read_csv(source / 'method_comparison.csv')
groups = [('infant','영유아','0~5세'),('elementary','초등 학령기','6~11세'),('middle','중등 학령기','12~14세'),('high','고등 학령기','15~17세')]
labels = {'cctv_locations':'CCTV 설치지점 수','emergency_bells':'안전비상벨 수','parks':'공원 수','schools':'학교 수','school_zone_count':'어린이보호구역 수','child_centers':'지역아동센터 수'}
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,'font.size':11})
warning = '현재 시설(주로 2026년) vs 2024→2025 인구 증감 | 인과·미래 예측 아님 | 실제 재학생 아닌 만 나이 기준'
def finish(fig, name):
    fig.text(.5,.015,warning,ha='center',fontsize=9,color='#80552d')
    fig.tight_layout(rect=[0,.05,1,.94])
    fig.savefig(out/name,dpi=170,facecolor='white')
    plt.close(fig)

# 1. 동일 변수·동일 색 범위를 사용해야 연령별 색 차이가 과장되지 않는다.
spearman=results[results.method=='spearman']
matrix=spearman.pivot(index='feature',columns='age',values='coefficient').reindex(index=labels,columns=[g[1] for g in groups])
fig,ax=plt.subplots(figsize=(12,7))
im=ax.imshow(matrix,cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto')
ax.set_xticks(range(4),[g[1]+'\n'+g[2] for g in groups]);ax.set_yticks(range(len(labels)),labels.values())
for i,f in enumerate(labels):
    for j,(_,g,_) in enumerate(groups):
        row=spearman[(spearman.feature==f)&(spearman.age==g)].iloc[0]
        ax.text(j,i,f'{row.coefficient:+.3f}\nn={int(row.n)}',ha='center',va='center',color='white' if abs(row.coefficient)>.55 else '#152d35')
fig.colorbar(im,ax=ax,label='Spearman ρ: 음의 관계 ← 0 → 양의 관계')
fig.suptitle('연령구간별 시설 수와 인구 증가율의 관계',fontsize=19)
finish(fig,'01_age_heatmap.png')

# 2. 관계의 방향이 방법마다 유지되는지 확인. 부분상관도 인과효과는 아니다.
fig,axes=plt.subplots(2,2,figsize=(15,10),sharex=True)
methods=[('spearman','Spearman'),('pearson','Pearson'),('kendall','Kendall'),('partial','부분 순위 상관')]
for ax,(_,g,ages) in zip(axes.flat,groups):
    for j,(method,title) in enumerate(methods):
        table=results[(results.age==g)&(results.method==('spearman' if method=='partial' else method))].set_index('feature')
        vals=table.reindex(labels)['partial_rank' if method=='partial' else 'coefficient']
        ax.plot(vals,np.arange(len(labels))+(j-1.5)*.12,'o',label=title,markersize=5)
    ax.set_yticks(range(len(labels)),labels.values());ax.axvline(0,color='gray',lw=1)
    ax.set_xlim(-1,1);ax.grid(axis='x',alpha=.2);ax.set_title(f'{g} ({ages})');ax.set_xlabel('계수 (Kendall과 다른 방법의 크기는 직접 동등 비교하지 않음)',fontsize=9)
axes.flat[0].legend(fontsize=9)
fig.suptitle('분석 방법 비교 · 같은 지표, 같은 연령구간',fontsize=19)
finish(fig,'02_methods.png')

# 3. 점 하나는 행정동. 극단값을 삭제하거나 추세선을 인과관계처럼 그리지 않는다.
fig,axes=plt.subplots(2,2,figsize=(13,9))
for ax,(key,g,ages) in zip(axes.flat,groups):
    pair=data[['emergency_bells',f'growth_{key}']].dropna()
    ax.scatter(pair.iloc[:,0],pair.iloc[:,1],alpha=.5,s=17,color='#287b68')
    ax.axhline(0,color='gray',lw=1);ax.set_xlabel('현재 안전비상벨 수');ax.set_ylabel('2024→2025 인구 증감률 (%)')
    row=spearman[(spearman.age==g)&(spearman.feature=='emergency_bells')].iloc[0]
    ax.set_title(f'{g} ({ages}) · ρ={row.coefficient:+.3f}, n={int(row.n)}')
fig.suptitle('실제 행정동별 분포 · 안전비상벨 사례',fontsize=19)
finish(fig,'03_scatter.png')

# 4. 평균만 보여주면 소수 급증 지역에 좌우될 수 있어 중앙값·범위를 함께 제시한다.
fig,ax=plt.subplots(figsize=(11,6))
values=[data[f'growth_{key}'].dropna().to_numpy() for key,_,_ in groups]
ax.boxplot(values,tick_labels=[g[1]+'\n'+g[2] for g in groups],showmeans=True)
ax.axhline(0,color='gray',lw=1);ax.set_ylabel('2024→2025 인구 증감률 (%)')
fig.suptitle('연령구간별 인구 증가율 분포 · 이상값도 표시',fontsize=19)
finish(fig,'04_growth_distribution.png')
summary=pd.DataFrame([{'구분':g,'연령':ages,'행정동수':len(v),'증감률_중앙값':np.median(v),'증감률_평균':np.mean(v)} for (_,g,ages),v in zip(groups,values)])
summary.to_csv(out/'growth_summary.csv',index=False,encoding='utf-8-sig')
body='''<!doctype html><meta charset="utf-8"><title>아동 연령별 시각화</title><style>body{font-family:Malgun Gothic,sans-serif;max-width:1300px;margin:30px auto;padding:15px;line-height:1.8;background:#f6f8f5}img{width:100%;background:white;border-radius:12px}section{background:white;padding:24px;margin:25px 0;border-radius:12px}.warning{background:#fff0d3;padding:18px}td,th{border:1px solid #ccc;padding:8px}table{border-collapse:collapse}</style><h1>영유아·초등·중등·고등 학령기 비교</h1>'''
body+=f'<p>원 분석 폴더: {html.escape(source.name)}</p><p class="warning">{warning}<br>증감에는 출생·연령 이동·전출입이 모두 포함됩니다. 시설이 인구 변화를 유발했다는 뜻이 아닙니다.</p>'
body+='<p>주 그림은 품질 경고가 없는 동일한 6개 시설 수 지표로 비교합니다. 정원·면적·인구 대비 지표와 매핑 누락이 많은 병원·관광·음식점은 원 보고서에서 표본 수와 함께 확인하세요.</p>'
for title,file,note in [('① 연령별 관계','01_age_heatmap.png','색 범위는 -1~1로 고정했습니다. 0에 가까우면 단조 관계가 약합니다. n은 유효 행정동 수입니다.'),('② 방법별 비교','02_methods.png','Pearson은 선형 관계, Spearman은 순위 관계, Kendall은 순서 일치, 부분 순위 상관은 인구 규모·초기 아동 비율·자치구를 보정합니다. 어떤 방법도 인과를 증명하지 않습니다.'),('③ 산점도','03_scatter.png','점 하나가 행정동입니다. 이상값도 제외하지 않았습니다. 패널별 세로축 범위가 다를 수 있습니다.'),('④ 증가율 분포','04_growth_distribution.png','상자 중앙선은 중앙값, 삼각형은 평균입니다. 수염 밖 점도 실제 관측값으로 유지했습니다.')]:
    body+=f'<section><h2>{title}</h2><p>{note}</p><img src="{file}" alt="{title}"></section>'
body+=summary.round(3).to_html(index=False)+'<p><a href="../report.html">전체 전처리·통계 결과 보기</a></p>'
(out/'dashboard.html').write_text(body,encoding='utf-8')
print(out)
print(matrix.round(3).to_string())
