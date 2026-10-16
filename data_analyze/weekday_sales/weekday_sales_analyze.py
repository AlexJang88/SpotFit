"""가설 1만 분석: 직장인구가 많은 행정동에서 외식업 평일 매출이 높은가?

실행: python data_analyze/weekday_sales/weekday_sales_analyze.py
관광명소/법정동 대응표는 이 가설에 필요하지 않아 결합하지 않는다.
"""
from pathlib import Path
import argparse
import calendar
import html
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')  # GUI 없는 Linux에서도 그림을 파일로 저장한다.
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns

KEY = ['기준_년분기_코드', '행정동_코드']
WORK = '총_직장_인구_수'
SALES = ['주중_매출_금액', '주말_매출_금액']


def read(path):
    """한글 CSV 인코딩을 처리하고 행정동 코드를 문자열로 보존한다."""
    for encoding in ('utf-8-sig', 'cp949'):
        try:
            return pd.read_csv(path, encoding=encoding, dtype={'행정동_코드': str})
        except UnicodeDecodeError:
            continue
    raise ValueError(f'인코딩 확인 필요: {path}')


def check_unique(frame, keys):
    """중복 원본 파일로 매출이 이중 합산되거나 병합 행 수가 늘어나는 것을 막는다."""
    if frame.duplicated(keys).any():
        raise ValueError(f'중복 키 발견: {keys}. 입력 파일의 기간 중복을 확인하세요.')


def day_counts(quarter):
    """분기 달력의 월~금/토~일 일수를 센다. 공휴일을 주말로 재분류하지 않는다.

    원본 주중/주말도 월~금/토~일로 구분된다는 가정의 보조 지표다.
    단순 5:2로 나누지 않아 분기별 요일 수 차이를 반영한다.
    """
    year, q = divmod(int(quarter), 10)
    if q not in (1, 2, 3, 4):
        raise ValueError(f'잘못된 분기: {quarter}')
    first, last = (q - 1) * 3 + 1, q * 3
    days = pd.date_range(f'{year}-{first:02d}-01',
                         f'{year}-{last:02d}-{calendar.monthrange(year, last)[1]}')
    return int((days.dayofweek < 5).sum()), int((days.dayofweek >= 5).sum())


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=root / 'data/raw')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'results')
    args = parser.parse_args()
    workers = read(args.raw / '서울시 상권분석서비스(직장인구-행정동).csv')
    check_unique(workers, KEY)
    files = sorted(args.raw.glob('서울시 상권분석서비스(추정매출-행정동)*.csv'))
    if not files:
        raise FileNotFoundError('행정동 추정매출 CSV가 없습니다.')
    sales = pd.concat([read(p) for p in files], ignore_index=True)
    # CS1 계열은 외식업. 실제 사용 업종을 별도 파일에 기록해 범위를 검토한다.
    sales = sales[sales['서비스_업종_코드'].str.startswith('CS1', na=False)].copy()
    if sales.empty:
        raise ValueError('외식업 자료가 없습니다.')
    check_unique(sales, KEY + ['서비스_업종_코드'])
    industries = sales[['서비스_업종_코드', '서비스_업종_코드_명']].drop_duplicates()
    for col in SALES:
        sales[col] = pd.to_numeric(sales[col], errors='coerce').where(lambda s: s >= 0)
    workers[WORK] = pd.to_numeric(workers[WORK], errors='coerce').where(lambda s: s >= 0)
    grouped = sales.groupby(KEY)
    totals = grouped[SALES].sum(min_count=1)
    # 일부 업종의 매출이 결측인데 나머지만 합산해 낮은 매출로 오인하지 않도록 한다.
    for col in SALES:
        totals.loc[grouped[col].count() != grouped.size(), col] = np.nan
    names = grouped['행정동_코드_명'].first().rename('행정동_코드_명')
    data = totals.join(names).reset_index().merge(workers[KEY + [WORK]], on=KEY,
                                                how='left', validate='one_to_one', indicator=True)
    # 매출이 있는 지역을 보존하고 직장인구 미매칭/결측을 CSV에서 확인할 수 있게 한다.
    counts = data[KEY[0]].map(day_counts)
    data['평일_일수'] = counts.map(lambda v: v[0])
    data['주말_일수'] = counts.map(lambda v: v[1])
    data['평일_하루평균_매출'] = data[SALES[0]] / data['평일_일수']
    data['주말_하루평균_매출'] = data[SALES[1]] / data['주말_일수']
    denominator = data[SALES[0]] + data[SALES[1]]
    data['평일_매출비중'] = data[SALES[0]] / denominator.where(denominator > 0) * 100
    data['평일대비주말_일평균비'] = data['평일_하루평균_매출'] / data['주말_하루평균_매출'].where(data['주말_하루평균_매출'] > 0)
    metrics = [SALES[0], '평일_하루평균_매출', '평일_매출비중', '평일대비주말_일평균비']
    rows = []
    for quarter, frame in data.groupby(KEY[0]):
        for metric in metrics:
            pair = frame[[WORK, metric]].replace([np.inf, -np.inf], np.nan).dropna()
            status = 'ok' if len(pair) >= 20 and (pair.nunique() >= 2).all() else 'insufficient_or_constant'
            rho, p = spearmanr(pair[WORK], pair[metric]) if status == 'ok' else (np.nan, np.nan)
            rows.append(dict(quarter=quarter, metric=metric, n=len(pair), spearman_rho=rho,
                             p_value=p, status=status, excluded=len(frame)-len(pair)))
    result = pd.DataFrame(rows)
    # 全검정 BH 보정. 공간적 독립성은 보장되지 않으므로 p/q는 참고용이다.
    ordered = result.p_value.dropna().sort_values()
    result['q_bh'] = np.nan
    if len(ordered):
        q = ordered.to_numpy() * len(ordered) / np.arange(1, len(ordered)+1)
        result.loc[ordered.index, 'q_bh'] = np.minimum.accumulate(q[::-1])[::-1].clip(0, 1)
    valid = result[result.status == 'ok']
    summary = valid.groupby('metric').agg(median_rho=('spearman_rho', 'median'),
        min_rho=('spearman_rho', 'min'), max_rho=('spearman_rho', 'max'),
        quarters=('quarter', 'nunique'), min_n=('n', 'min'), max_n=('n', 'max')).reset_index()
    out = args.output / ('hypothesis1_' + pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True, exist_ok=False)
    for name, frame in [('analysis_data', data), ('correlations_by_quarter', result),
                        ('summary', summary), ('selected_industries', industries)]:
        frame.to_csv(out / f'{name}.csv', index=False, encoding='utf-8-sig')
    sns.set_theme(style='whitegrid')
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    available = next((f for f in ['Malgun Gothic', 'NanumGothic', 'Noto Sans CJK KR', 'AppleGothic'] if f in fonts), None)
    if available:
        plt.rcParams['font.family'] = available
    else:
        warnings.warn('한글 폰트를 설치하세요: NanumGothic 또는 Noto Sans CJK KR')
    plt.rcParams['axes.unicode_minus'] = False
    if not valid.empty:
        fig, ax = plt.subplots(figsize=(17, 5))
        sns.heatmap(valid.pivot(index='metric', columns='quarter', values='spearman_rho'),
                    annot=True, fmt='.2f', vmin=-1, vmax=1, center=0, cmap='RdBu_r', ax=ax)
        ax.set_title('가설 1: 직장인구와 평일 매출의 분기별 행정동 간 Spearman')
        fig.tight_layout(); fig.savefig(out / 'heatmap.png', dpi=150); plt.close(fig)
        latest = valid.quarter.max()
        frame = data[data[KEY[0]] == latest]
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, metric in zip(axes, [SALES[0], '평일_매출비중']):
            sns.scatterplot(data=frame, x=WORK, y=metric, alpha=.5, ax=ax)
            ax.set_title(f'{latest} / {metric}')
        fig.tight_layout(); fig.savefig(out / 'scatter_latest.png', dpi=150); plt.close(fig)
    notes = ('외식업·행정동 기준, 분기별 지역 간 Spearman 탐색 분석. 매출과 직장인구가 모두 관측된 쌍만 사용. '
             '관광명소·법정동 매핑은 가설 1에 불필요하여 사용하지 않음. '
             '총매출은 점포 수·면적·입지 등 상권 규모의 영향을 받으며 인과관계는 검증하지 않음. '
             '평일 매출비중=주중/(주중+주말)×100. 일평균비=평일 하루평균/주말 하루평균으로 1보다 크면 평일이 높음. '
             '일평균은 분기 매출 및 월~금/토~일 구분이라는 가정 아래 달력 일수로 나누며 공휴일은 재분류하지 않음. '
             '같은 분기에서 일수는 공통 상수이므로 평일 총매출과 하루평균 매출의 순위 상관은 동일함. '
             '매출비중과 일평균비도 같은 분기에서 순위가 같을 수 있어 독립적인 추가 증거로 세지 않음. '
             '시설/점포별 매출이나 동일 지역 내 시간적 증가를 검증한 결과가 아님. p/q값은 공간 의존성 미보정 참고값.')
    conclusions = []
    for metric in [SALES[0], '평일_매출비중']:
        sub = valid[valid.metric == metric]
        if sub.empty:
            conclusions.append(f'{metric}: 유효 표본 부족')
            continue
        median = sub.spearman_rho.median()
        positive = (sub.spearman_rho > 0).mean()
        # 0.3/80%는 설명용 탐색 기준이며 통계적 가설 채택 기준이 아니다.
        verdict = '양의 관계 경향 관찰' if median >= .3 and positive >= .8 else '뚜렷한 양의 관계 근거 부족'
        conclusions.append(f'{metric}: {verdict}. 분기별 ρ 중앙값 {median:+.3f}, 양의 방향 분기 비율 {positive:.0%}.')
    report = '<!doctype html><meta charset="utf-8"><title>가설 1 직장인구와 평일매출</title>'
    report += '<h1>가설 1: 직장인구가 많으면 평일 매출이 높은가?</h1><p>' + html.escape(notes) + '</p>'
    report += '<h2>결과 요약</h2><ul>' + ''.join('<li>'+html.escape(c)+'</li>' for c in conclusions) + '</ul>'
    report += '<p>설명용 기준: ρ 중앙값 ≥0.3 및 양의 방향 분기 ≥80%. 확정적 가설 채택이 아닙니다.</p>'
    report += summary.to_html(index=False) + '<h2>분기별 표본·제외 수</h2>' + result.to_html(index=False)
    report += ''.join(f'<img src="{p.name}" style="max-width:100%">' for p in out.glob('*.png'))
    report += '<p>2~5번 가설은 실행하지 않았습니다.</p>'
    (out / 'report.html').write_text(report, encoding='utf-8')
    print(out); print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
