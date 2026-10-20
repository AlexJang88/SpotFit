"""가설 4: 시설별 원본 총계 대비 비율과 외식업 평일/주말 매출의 탐색 분석."""
from pathlib import Path
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
    # 기존 가설과 동일한 인코딩·키 검증을 재사용하되 다른 가설의 main은 실행하지 않는다.
    from weekday_sales_analyze import read, check_unique, KEY, SALES
    facilities = read(raw / '서울시 상권분석서비스(집객시설-행정동).csv')
    check_unique(facilities, KEY)
    counts = [c for c in facilities if c.endswith('_수') and c != '집객시설_수']
    features = []
    for col in ['집객시설_수'] + counts:
        facilities[col] = pd.to_numeric(facilities[col], errors='coerce').where(lambda s: s >= 0)
    for col in counts:
        # 총계와 세부항목 합계가 불일치하므로 합계 100% 구성비로 해석하지 않는다.
        # 빈값은 시설 부재로 단정하지 않으며, 0/결측 분모의 비율은 결측으로 남긴다.
        name = col.removesuffix('_수') + '_총계대비비율'
        facilities[name] = facilities[col] / facilities['집객시설_수'].where(facilities['집객시설_수'] > 0) * 100
        features.append(name)
    files = sorted(raw.glob('서울시 상권분석서비스(추정매출-행정동)*.csv'))
    if not files:
        raise FileNotFoundError('행정동 추정매출 파일이 없습니다.')
    sales = pd.concat([read(p) for p in files], ignore_index=True)
    sales = sales[sales['서비스_업종_코드'].str.startswith('CS1', na=False)].copy()
    if sales.empty:
        raise ValueError('외식업 자료가 없습니다.')
    check_unique(sales, KEY + ['서비스_업종_코드'])
    industries = sales[['서비스_업종_코드', '서비스_업종_코드_명']].drop_duplicates()
    for col in SALES:
        sales[col] = pd.to_numeric(sales[col], errors='coerce').where(lambda s: s >= 0)
    grouped = sales.groupby(KEY)
    totals = grouped[SALES].sum(min_count=1)
    for col in SALES:
        # 관측된 업종 중 일부가 결측이면 부분 합계를 전체 매출로 오인하지 않도록 제외한다.
        totals.loc[grouped[col].count() != grouped.size(), col] = np.nan
    totals = totals.join(grouped['행정동_코드_명'].first()).reset_index()
    data = totals.merge(facilities[KEY + ['집객시설_수'] + counts + features],
                        on=KEY, how='left', validate='one_to_one', indicator=True)
    denominator = data[SALES].sum(axis=1, min_count=2).where(lambda s: s > 0)
    data['평일_매출비중'] = data[SALES[0]] / denominator * 100
    data['주말_매출비중'] = data[SALES[1]] / denominator * 100
    metrics = SALES + ['평일_매출비중', '주말_매출비중']
    rows = []
    for quarter, frame in data.groupby(KEY[0]):
        for feature in features:
            for metric in metrics:
                pair = frame[[feature, metric]].replace([np.inf, -np.inf], np.nan).dropna()
                status = 'ok' if len(pair) >= 20 and (pair.nunique() >= 2).all() else 'insufficient_or_constant'
                rho, p = spearmanr(pair[feature], pair[metric]) if status == 'ok' else (np.nan, np.nan)
                rows.append(dict(quarter=quarter, feature=feature, metric=metric, n=len(pair),
                                 excluded=len(frame)-len(pair), spearman_rho=rho, p_value=p, status=status))
    results = pd.DataFrame(rows)
    # 多重比較のため BH 보정. 지역 간 공간 의존성을 보정한 것은 아니므로 참고값이다.
    ordered = results.p_value.dropna().sort_values()
    results['q_bh'] = np.nan
    if len(ordered):
        q = ordered.to_numpy() * len(ordered) / np.arange(1, len(ordered)+1)
        results.loc[ordered.index, 'q_bh'] = np.minimum.accumulate(q[::-1])[::-1].clip(0, 1)
    valid = results[results.status == 'ok']
    summary = valid.groupby(['feature', 'metric']).agg(
        median_rho=('spearman_rho', 'median'), min_rho=('spearman_rho', 'min'),
        max_rho=('spearman_rho', 'max'), quarters=('quarter', 'nunique'),
        min_n=('n', 'min'), max_n=('n', 'max'), positive_fraction=('spearman_rho', lambda s: (s > 0).mean()),
        negative_fraction=('spearman_rho', lambda s: (s < 0).mean())).reset_index()
    summary['direction_consistency'] = np.where(summary.median_rho >= 0, summary.positive_fraction, summary.negative_fraction)
    # 임의 탐색 기준은 검토 우선순위일 뿐 가설 채택/유의성 판정이 아니다.
    summary['exploratory_candidate'] = (summary.median_rho.abs() >= .3) & (summary.direction_consistency >= .8)
    summary = summary.sort_values('median_rho', key=lambda s: s.abs(), ascending=False)
    quality = pd.DataFrame([dict(feature=f, rows=len(data), missing=int(data[f].isna().sum()),
                                 ratio_over_100=int((data[f] > 100).sum())) for f in features])
    out = output / ('hypothesis4_' + pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True, exist_ok=False)
    for name, frame in [('analysis_data', data), ('correlations_by_quarter', results), ('summary', summary),
                        ('facility_quality', quality), ('selected_industries', industries)]:
        frame.to_csv(out / f'{name}.csv', index=False, encoding='utf-8-sig')
    sns.set_theme(style='whitegrid')
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ['Malgun Gothic', 'NanumGothic', 'Noto Sans CJK KR', 'AppleGothic'] if f in fonts), None)
    if font:
        plt.rcParams['font.family'] = font
    else:
        warnings.warn('한글 폰트 설치가 필요합니다.')
    plt.rcParams['axes.unicode_minus'] = False
    def heatmap(matrix, title, name):
        fig, ax = plt.subplots(figsize=(14, max(7, len(matrix)*.42)))
        sns.heatmap(matrix, annot=True, fmt='.2f', annot_kws={'fontsize': 8},
                    vmin=-1, vmax=1, center=0, cmap='RdBu_r', ax=ax)
        ax.set_title(title); fig.tight_layout(); fig.savefig(out / name, dpi=150); plt.close(fig)
    if not valid.empty:
        heatmap(summary.pivot(index='feature', columns='metric', values='median_rho'),
                '가설 4: 분기별 Spearman 중앙값 (통합 상관계수가 아님)', 'summary_heatmap.png')
        for quarter, frame in valid.groupby('quarter'):
            heatmap(frame.pivot(index='feature', columns='metric', values='spearman_rho'),
                    f'{quarter} 시설 비율과 매출: 행정동 간 Spearman', f'heatmap_{quarter}.png')
    notes = ('외식업·행정동 기준. 시설별 비율=관측 시설 수/원본 집객시설 수×100. '
             '시설 항목 합계와 총계가 불일치하므로 합계 100% 구성비가 아니며 시설 밀도·절대 수와 구분합니다. '
             '시설 빈값은 0으로 채우지 않고 쌍별 제외하므로 시설마다 분석 표본이 다릅니다. '
             '시설이 관측된 지역에 한정된 선택 편향과 공통 분모의 영향을 주의하세요. '
             '평일·주말 비중은 합계 100%로 서로 보완적이므로 두 상관은 독립적인 증거가 아닙니다. '
             '비중이 높다는 것은 절대 매출이 높다는 뜻이 아닙니다. '
             '총매출은 점포 수·면적·입지 영향을 받으며 이를 통제하지 않았습니다. '
             '최소 20쌍·비상수 열에만 분기별 Spearman을 계산합니다. p/q는 공간 의존성 미보정 참고값입니다. '
             '탐색 후보 기준은 |ρ 중앙값|≥0.3, 방향 일치≥80%이며 통계적 가설 채택 기준이 아닙니다. '
             '동일 지역의 시간적 변화나 인과관계를 입증하지 않습니다. 관광명소는 아직 사용하지 않았습니다.')
    report = '<!doctype html><meta charset="utf-8"><title>가설 4 시설비율과 매출</title>'
    report += '<h1>가설 4: 집객시설 비율에 따라 평일·주말 매출이 연관되는가?</h1><p>'+html.escape(notes)+'</p>'
    report += '<h2>주요 관계 요약</h2>'
    for metric in metrics:
        candidates = summary[(summary.metric == metric) & summary.exploratory_candidate]
        report += '<h3>'+html.escape(metric)+'</h3>'
        report += f'<p>탐색 기준에 해당하는 시설 비율 {len(candidates)}개. 아래는 절대 계수 순 최대 5개이며 전체 결과는 하단 표에 있습니다.</p><ul>'
        for row in candidates.head(5).itertuples():
            direction = '높은' if row.median_rho > 0 else '낮은'
            text = f'{row.feature}이 높은 행정동에서 {metric}이 {direction} 경향: ρ 중앙값 {row.median_rho:+.3f}, 범위 {row.min_rho:+.3f}~{row.max_rho:+.3f}, {row.quarters}분기, n={row.min_n}~{row.max_n}.'
            report += '<li>'+html.escape(text)+'</li>'
        report += '</ul>'
    report += '<h2>전체 관계</h2>'+summary.to_html(index=False)+'<h2>결측 및 품질</h2>'+quality.to_html(index=False)
    for p in sorted(out.glob('*.png'), key=lambda p: (p.name != 'summary_heatmap.png', p.name)):
        report += f'<h3>{html.escape(p.stem)}</h3><img src="{p.name}" style="max-width:100%">'
    report += '<p>이번 실행은 4번 가설만 수행했으며 5번은 진행하지 않았습니다.</p>'
    (out / 'report.html').write_text(report, encoding='utf-8')
    print(out)
    print(summary[summary.metric == '평일_매출비중'].head(6).to_string(index=False))
