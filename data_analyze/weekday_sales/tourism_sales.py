"""가설 5: 주소로 확정 매핑한 관광명소 등록 수와 최신 분기 외식업 주말 매출.

명소 목록/행정구역 대응표의 시점이 매출 시점과 일치하지 않을 수 있어 예측 검증이 아니다.
"""
import re
import html
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns


def run(raw, output):
    from weekday_sales_analyze import read, check_unique, KEY, SALES, day_counts
    paths = list(raw.glob('*관광*.csv'))
    if len(paths) != 1:
        raise ValueError('관광 CSV가 정확히 하나 있어야 합니다.')
    attractions = read(paths[0])
    # 서로 다른 언어의 콘텐츠 ID가 다르므로 ID만으로는 장소 중복을 제거할 수 없다.
    # 한국어 등록 항목으로 범위를 제한한다. 다른 언어에만 있는 장소는 포함되지 않는다.
    ko = attractions[attractions['언어'].eq('ko')].copy()
    for col in ['상호명', '주소', '신주소']:
        ko[col] = ko[col].fillna('').astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
    ko['duplicate'] = ko.duplicated(['상호명', '주소', '신주소'])
    duplicates = ko[ko.duplicate].copy()
    ko = ko[~ko.duplicate].copy()
    mapping = pd.read_excel(raw / '행정동_법정동_매핑.xlsx', sheet_name='KIKmix', dtype=str)
    # 말소된 이력 및 시/구 행을 제외한다. 최신 대응표이며 과거 분기 경계를 복원하지 않는다.
    mapping = mapping[mapping['시도명'].eq('서울특별시') & mapping['말소일자'].isna()
                      & mapping['읍면동명'].notna() & mapping['동리명'].notna()].copy()
    mapping['code'] = mapping['행정동코드'].str.strip()
    if not mapping['code'].str.fullmatch(r'\d{8}00').all():
        raise ValueError('행정동 코드 10자리→8자리 변환 조건을 확인하세요.')
    mapping['code'] = mapping['code'].str[:8]
    lookup = {key: sorted(group.code.unique()) for key, group in mapping.groupby(['시군구명', '동리명'])}
    audit = []
    for _, row in ko.iterrows():
        # 신주소의 괄호 안 법정동도 사용한다. 도로명만 있으면 외부 지오코딩 없이 추정하지 않는다.
        address = row['주소'] + ' ' + row['신주소']
        districts = set(re.findall(r'(?<![가-힣])([가-힣]+구)(?![가-힣])', address))
        hits = []
        for (district, legal), codes in lookup.items():
            if district in districts and re.search(r'(?<![가-힣0-9])'+re.escape(legal)+r'(?![가-힣0-9])', address):
                hits.append((district, legal, codes))
        candidates = sorted({code for _, _, codes in hits for code in codes})
        # 다대일은 허용하지만 일대다는 임의 배정하지 않는다.
        status = 'mapped' if len(candidates) == 1 else ('ambiguous' if candidates else 'unmatched')
        if not row['상호명']:
            status = 'missing_name'
        audit.append({'고유번호': row['고유번호'], '상호명': row['상호명'], '주소': row['주소'],
                      '신주소': row['신주소'], 'status': status,
                      'matched_legal': ';'.join(d+' '+l for d,l,_ in hits),
                      'candidate_codes': ';'.join(candidates),
                      '행정동_코드': candidates[0] if status == 'mapped' else None})
    audit = pd.DataFrame(audit)
    files = sorted(raw.glob('서울시 상권분석서비스(추정매출-행정동)*.csv'))
    if not files:
        raise FileNotFoundError('행정동 매출 CSV 없음')
    sales = pd.concat([read(p) for p in files], ignore_index=True)
    sales = sales[sales['서비스_업종_코드'].str.startswith('CS1', na=False)].copy()
    latest = int(sales[KEY[0]].max())
    sales = sales[sales[KEY[0]].eq(latest)].copy()
    check_unique(sales, KEY+['서비스_업종_코드'])
    industries = sales[['서비스_업종_코드', '서비스_업종_코드_명']].drop_duplicates()
    for col in SALES:
        sales[col] = pd.to_numeric(sales[col], errors='coerce').where(lambda s:s>=0)
    grouped = sales.groupby(KEY)
    data = grouped[SALES].sum(min_count=1)
    for col in SALES:
        data.loc[grouped[col].count()!=grouped.size(), col] = np.nan
    data = data.join(grouped['행정동_코드_명'].first()).reset_index()
    mapped = audit[audit.status.eq('mapped')]
    count = mapped.groupby('행정동_코드').size()
    feature = '확정매핑_관광명소등록수'
    # 0은 현재 목록에서 확인된 등록이 없다는 뜻. 지역의 실제 관광명소가 없다는 뜻은 아니다.
    data[feature] = data['행정동_코드'].map(count).fillna(0).astype(int)
    ambiguous_codes = {c for s in audit.loc[audit.status.eq('ambiguous'), 'candidate_codes'] for c in s.split(';') if c}
    data['미확정명소_후보지역'] = data['행정동_코드'].isin(ambiguous_codes)
    denominator = data[SALES].sum(axis=1, min_count=2).where(lambda s:s>0)
    data['주말_매출비중'] = data[SALES[1]]/denominator*100
    weekdays, weekends = day_counts(latest)
    data['주말_하루평균_매출'] = data[SALES[1]]/weekends
    data['주말_평일_일평균비'] = (data[SALES[1]]/weekends)/(data[SALES[0]]/weekdays).where(lambda s:s>0)
    metrics = [SALES[1], '주말_하루평균_매출', '주말_매출비중', '주말_평일_일평균비']
    # 선택편향을 숨기지 않기 위해 전체·등록 양수 지역·모호한 매핑 후보 제외를 함께 제공한다.
    samples = {'전체_등록0포함': data, '등록1개이상': data[data[feature]>0],
               '모호매핑후보제외': data[~data['미확정명소_후보지역']]}
    rows = []
    for sample, frame in samples.items():
        for metric in metrics:
            pair = frame[[feature,metric]].replace([np.inf,-np.inf],np.nan).dropna()
            status = 'ok' if len(pair)>=20 and (pair.nunique()>=2).all() else 'insufficient_or_constant'
            rho,p = spearmanr(pair[feature],pair[metric]) if status=='ok' else (np.nan,np.nan)
            rows.append(dict(sample=sample,quarter=latest,metric=metric,n=len(pair),spearman_rho=rho,p_value=p,status=status))
    result = pd.DataFrame(rows)
    ordered = result.p_value.dropna().sort_values(); result['q_bh']=np.nan
    if len(ordered):
        q=ordered.to_numpy()*len(ordered)/np.arange(1,len(ordered)+1)
        result.loc[ordered.index,'q_bh']=np.minimum.accumulate(q[::-1])[::-1].clip(0,1)
    out=output/('hypothesis5_'+pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True,exist_ok=False)
    for name,frame in [('analysis_data',data),('correlations',result),('mapping_audit',audit),
                       ('duplicate_records',duplicates),('selected_industries',industries),
                       ('mapped_not_in_sales',mapped[~mapped['행정동_코드'].isin(data['행정동_코드'])])]:
        frame.to_csv(out/f'{name}.csv',index=False,encoding='utf-8-sig')
    sns.set_theme(style='whitegrid')
    fonts={f.name for f in font_manager.fontManager.ttflist}
    font=next((f for f in ['Malgun Gothic','NanumGothic','Noto Sans CJK KR','AppleGothic'] if f in fonts),None)
    if font: plt.rcParams['font.family']=font
    else: warnings.warn('한글 폰트가 없습니다.')
    plt.rcParams['axes.unicode_minus']=False
    valid=result[result.status.eq('ok')]
    if not valid.empty:
        fig,ax=plt.subplots(figsize=(12,5))
        sns.heatmap(valid.pivot(index='sample',columns='metric',values='spearman_rho'),annot=True,fmt='.2f',vmin=-1,vmax=1,cmap='RdBu_r',ax=ax)
        ax.set_title(f'{latest} 매출과 현재 목록의 확정매핑 명소 등록 수 (시점 불일치 주의)')
        fig.tight_layout(); fig.savefig(out/'heatmap.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    for ax,metric in zip(axes,[SALES[1],'주말_매출비중']):
        sns.scatterplot(data=data,x=feature,y=metric,ax=ax,alpha=.5)
    fig.tight_layout(); fig.savefig(out/'scatter.png',dpi=150); plt.close(fig)
    notes=('관광명소 CSV의 한국어 등록 항목만 사용하며 상호명·주소·신주소가 같은 행은 중복 제거했습니다. '
           '같은 장소의 명칭 변형은 남을 수 있으므로 실제 고유 명소 수가 아니라 등록 수의 대리지표입니다. '
           '주소의 구+법정동을 최신 유효 대응표에 연결하며 행정동 후보가 하나일 때만 배정했습니다. '
           '목록에 관광시설 외 보관소 등의 콘텐츠도 포함될 수 있으며 원본 분류를 그대로 사용했습니다. '
           '0개는 관광명소가 없다는 뜻이 아닌 확정매핑 등록이 없다는 뜻입니다. 미확정 항목 때문에 과소집계될 수 있습니다. '
           '명소 개폐업일·과거 이력이 없어 최신 매출 분기만 비교했습니다. 현재 목록/대응표와 과거 매출 시점 불일치가 있어 '
           '정식 가설 검증·예측 학습 근거로 사용하면 안 됩니다. 등록 양수 지역 분석도 선택편향이 있습니다. '
           '미확정 후보지역 제외 분석도 주소 미추출 항목의 영향을 제거하지 못합니다. '
           '외식업 매출 기준이며 점포수·입지·지역 규모와 공간 의존성을 통제하지 않았습니다. '
           '총매출과 하루평균, 매출비중과 일평균비는 순위가 같을 수 있어 독립 증거가 아닙니다. p/q는 참고값입니다.')
    report='<!doctype html><meta charset="utf-8"><title>가설5 관광명소와 주말매출</title><h1>가설 5: 관광명소가 많으면 주말 매출이 높은가?</h1>'
    report+='<p><b>제한된 등록 목록과 시점이 다른 매출을 비교한 예비 탐색 결과입니다.</b></p><p>'+html.escape(notes)+'</p>'
    report+=f'<h2>매핑 현황</h2><p>원본 {len(attractions)}행, 한국어 {len(ko)+len(duplicates)}행, 중복 제외 {len(duplicates)}행, 비교 매출 분기 {latest}, 확정등록 양수 지역 {(data[feature]>0).sum()}개.</p>'
    report+=audit.status.value_counts().rename_axis('상태').reset_index(name='건수').to_html(index=False)
    report+='<h2>결과 요약</h2><ul>'
    for row in result[(result['sample']=='전체_등록0포함') & result.metric.isin([SALES[1],'주말_매출비중'])].itertuples():
        verdict='양의 관계 경향' if row.spearman_rho>=.3 else ('음의 관계 경향' if row.spearman_rho<=-.3 else '뚜렷한 관계 근거 부족')
        if row.status!='ok': verdict='분석 불가'
        report+=f'<li>{html.escape(row.metric)}: ρ={row.spearman_rho:+.3f}, n={row.n}, {verdict}. 시점·매핑 제약으로 가설 판단은 보류합니다.</li>'
    report+='</ul><p>|ρ| 0.3은 설명용 탐색 기준이지 통계적 채택 기준이 아닙니다.</p>'+result.to_html(index=False)
    report+=''.join(f'<img src="{p.name}" style="max-width:100%">' for p in out.glob('*.png'))
    (out/'report.html').write_text(report,encoding='utf-8')
    print(out); print(audit.status.value_counts().to_string()); print(result.to_string(index=False))
