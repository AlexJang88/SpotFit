"""행정동 집객시설 가설 탐색: 원본을 보존하고 CSV/PNG/HTML 보고서를 생성한다.

실행: python data_analyze/attraction/attraction_analyze.py
전체 업종 비교: python data_analyze/attraction/attraction_analyze.py --scope all
분석은 같은 분기의 행정동 간 Spearman 상관이며 인과관계 검증이 아니다.
"""
from pathlib import Path
import argparse
import html
import warnings

import numpy as np
import pandas as pd
import matplotlib

# 서버/터미널에서도 창 없이 그림 파일을 생성하기 위한 비대화형 백엔드.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
from scipy.stats import spearmanr

TIME = '기준_년분기_코드'
DONG = '행정동_코드'
NAME = '행정동_코드_명'
KEY = [TIME, DONG]


def read_csv(path):
    """서울시 CSV의 인코딩 차이를 처리하고 코드의 앞자리 0을 보존한다."""
    for encoding in ('utf-8-sig', 'cp949'):
        try:
            return pd.read_csv(path, encoding=encoding, dtype={DONG: str})
        except UnicodeDecodeError:
            continue
    raise ValueError(f'지원하지 않는 인코딩: {path}')


def unique(frame, keys, label):
    """병합 시 행이 증식하는 것을 막는다. 중복을 임의로 버리지 않는다."""
    if frame.duplicated(keys).any():
        raise ValueError(f'{label}: 중복 키 발견 {keys}')


def ratio(numerator, denominator):
    """분모가 없거나 0인 비율은 0%가 아니라 계산 불가능한 결측값이다."""
    return numerator.div(denominator.where(denominator > 0)) * 100


def prepare(raw, scope):
    """시설 기준 패널에 행정동×분기 단위 특징을 1:1로 붙인다.

    인구/시설은 전체 지역 특성이고 점포/매출만 외식업 선택을 적용한다.
    데이터가 없는 분기는 보간하거나 전년 값으로 채우지 않는다.
    """
    def load(kind):
        return read_csv(raw / f'서울시 상권분석서비스({kind}-행정동).csv')

    facilities = load('집객시설')
    unique(facilities, KEY, '집객시설')
    types = [c for c in facilities if c.endswith('_수') and c != '집객시설_수']
    data = facilities.copy()
    targets = ['집객시설_수']
    quality = []
    for col in types:
        # 세부항목 합계와 공식 총계가 불일치하므로 '합계 100% 구성비'가 아니다.
        # 관측된 항목 / 원본 총계라는 명확한 정의만 사용하고 결측은 유지한다.
        name = col.removesuffix('_수') + '_총계대비비율'
        data[name] = ratio(data[col], data['집객시설_수'])
        targets.append(name)
        quality.append({'column': col, 'missing': int(data[col].isna().sum()),
                        'rows': len(data), 'ratio_over_100': int((data[name] > 100).sum())})
    # min_count로 모든 항목이 결측인 행을 0으로 잘못 만드는 것을 방지한다.
    data['관측시설항목합'] = data[types].sum(axis=1, min_count=1)
    data['시설항목결측수'] = data[types].isna().sum(axis=1)
    hypotheses = []
    for number, kind, col in [(1, '길단위인구', '총_유동인구_수'),
                              (2, '상주인구', '총_상주인구_수'),
                              (4, '직장인구', '총_직장_인구_수')]:
        other = load(kind)
        unique(other, KEY, kind)
        data = data.merge(other[KEY + [col]], on=KEY, how='left', validate='one_to_one')
        hypotheses.append((number, col, 'positive'))

    def select(frame):
        # 서울시 서비스 업종 코드의 CS1 계열: 한식 등 외식업.
        # 실제 선택된 코드/명은 출력하여 필터 범위를 검토할 수 있게 한다.
        if scope == 'food':
            return frame[frame['서비스_업종_코드'].str.startswith('CS1', na=False)].copy()
        return frame.copy()

    shops = select(load('점포'))
    unique(shops, KEY + ['서비스_업종_코드'], '점포')
    selected = shops[['서비스_업종_코드', '서비스_업종_코드_명']].drop_duplicates()
    # 업종별 비율을 평균하면 소규모 업종에 과도한 가중치가 부여된다.
    # 여기서는 원본 개업률 정의를 단정하지 않고 공통 분모의 자체 지표로 명명한다.
    counts = ['전체_점포_수', '개업_점포_수', '폐업_점포_수']
    shop_sum = shops.groupby(KEY)[counts].sum(min_count=1).reset_index()
    for event in ('개업', '폐업'):
        shop_sum[f'{event}수_점포수대비율'] = ratio(shop_sum[f'{event}_점포_수'], shop_sum['전체_점포_수'])
    data = data.merge(shop_sum, on=KEY, how='left', validate='one_to_one')
    hypotheses += [(5, '전체_점포_수', 'positive'),
                   (6, '개업수_점포수대비율', 'positive'),
                   (7, '폐업수_점포수대비율', 'negative')]
    files = sorted(raw.glob('서울시 상권분석서비스(추정매출-행정동)*.csv'))
    if not files:
        raise FileNotFoundError('행정동 추정매출 CSV가 없습니다.')
    sales = select(pd.concat([read_csv(p) for p in files], ignore_index=True))
    unique(sales, KEY + ['서비스_업종_코드'], '매출: 기간이 겹치는 파일을 확인하세요')
    age_cols = [c for c in sales if c.startswith('연령대_') and c.endswith(('_매출_금액', '_매출_건수'))]
    aggregated = sales.groupby(KEY)[age_cols].sum(min_count=1)
    for unit in ('금액', '건수'):
        columns = [c for c in age_cols if c.endswith(unit)]
        # 나이를 알 수 없는 결제 등이 총매출에 포함될 수 있으므로 연령 확인 매출을 분모로 한다.
        # 일부 연령이 결측이면 완전한 구성비를 계산할 수 없어 해당 행을 제외한다.
        denominator = aggregated[columns].sum(axis=1, min_count=len(columns))
        for col in columns:
            feature = col + '_비율'
            aggregated[feature] = ratio(aggregated[col], denominator)
            hypotheses.append((3, feature, 'unspecified'))
    data = data.merge(aggregated.reset_index(), on=KEY, how='left', validate='one_to_one')
    return data, targets, sorted(hypotheses), pd.DataFrame(quality), selected


def correlate(data, targets, hypotheses):
    """분기별 지역 간 순위 상관을 계산한다. 반복 시점 합산 p값은 만들지 않는다.

    Spearman은 인구/매출처럼 치우친 분포에서 선형 관계만을 가정하지 않는다.
    단, 면적/입지 등 교란요인과 공간 의존성은 제거되지 않은 탐색 분석이다.
    """
    rows = []
    for quarter, frame in data.groupby(TIME):
        for number, feature, expected in hypotheses:
            for target in targets:
                pair = frame[[feature, target]].replace([np.inf, -np.inf], np.nan).dropna()
                n = len(pair)
                reason = 'ok'
                rho = p = np.nan
                if n < 20:
                    reason = 'fewer_than_20_pairs'
                elif (pair.nunique() < 2).any():
                    reason = 'constant_column'
                else:
                    rho, p = spearmanr(pair[feature], pair[target])
                rows.append({'quarter': quarter, 'hypothesis': number, 'feature': feature,
                             'target': target, 'expected': expected if target == '집객시설_수' else 'unspecified',
                             'spearman_rho': rho, 'abs_rho': abs(rho), 'p_value': p, 'n': n, 'status': reason})
    result = pd.DataFrame(rows)
    # 전체 실행에서 수행한 다중검정에 BH 보정을 적용한다.
    # 지역 간 독립성이 보장되지 않으므로 보정 p값도 확정적 증거로 사용하지 않는다.
    valid = result['p_value'].dropna().sort_values()
    result['q_bh'] = np.nan
    if len(valid):
        adjusted = valid.to_numpy() * len(valid) / np.arange(1, len(valid) + 1)
        result.loc[valid.index, 'q_bh'] = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
    return result


def visualize(data, results, output):
    """분기별 히트맵·최신 유효 분기 산점도·기간 추세를 파일로 저장한다."""
    sns.set_theme(style='whitegrid')
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    for name in ('Malgun Gothic', 'NanumGothic', 'Noto Sans CJK KR', 'AppleGothic'):
        if name in fonts:
            plt.rcParams['font.family'] = name
            break
    else:
        warnings.warn('한글 폰트가 없습니다. Linux에서는 Noto Sans CJK 또는 NanumGothic을 설치하세요.')
    plt.rcParams['axes.unicode_minus'] = False
    valid = results[results.status == 'ok']
    for quarter, frame in valid.groupby('quarter'):
        matrix = frame.pivot(index='feature', columns='target', values='spearman_rho')
        fig, ax = plt.subplots(figsize=(19, max(7, len(matrix) * .45)))
        # 색은 관계의 방향과 강도를, 숫자는 실제 상관계수를 보여준다.
        # 소수점 둘째 자리와 작은 글씨로 표시해 칸 안에서 숫자가 겹치지 않게 한다.
        # 결측값(분석 불가)은 기존처럼 빈칸으로 유지한다.
        sns.heatmap(
            matrix, vmin=-1, vmax=1, center=0, cmap='RdBu_r',
            annot=True, fmt='.2f', annot_kws={'fontsize': 8}, ax=ax,
        )
        ax.set_title(f'{quarter} 행정동 간 Spearman (빈칸: 분석 불가, 시설 비율은 원본 총계 대비)')
        fig.tight_layout()
        fig.savefig(output / f'heatmap_{quarter}.png', dpi=150)
        plt.close(fig)
    total = valid[valid.target == '집객시설_수']
    if not total.empty:
        fig, ax = plt.subplots(figsize=(14, 8))
        for feature, frame in total.groupby('feature'):
            frame = frame.sort_values('quarter')
            ax.plot(frame.quarter.astype(str), frame.spearman_rho, marker='.', label=feature)
        ax.axhline(0, color='gray', linewidth=1)
        ax.set_ylim(-1, 1)
        ax.tick_params(axis='x', rotation=60)
        ax.set_title('분기별 행정동 간 집객시설 수 상관계수 (각 선은 동일 지역의 시계열 상관이 아님)')
        ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1))
        fig.tight_layout()
        fig.savefig(output / 'correlation_trends.png', dpi=150)
        plt.close(fig)
    for feature, frame in total.groupby('feature'):
        row = frame.sort_values('quarter').iloc[-1]
        subset = data[data[TIME] == row.quarter][[feature, '집객시설_수']].dropna()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=subset, x=feature, y='집객시설_수', alpha=.5, ax=ax)
        ax.set_title(f'{row.quarter} / rho={row.spearman_rho:.3f}, n={row.n}')
        fig.tight_layout()
        fig.savefig(output / f'scatter_{feature}.png', dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # 파일이 data_analyze/attraction 아래에 있으므로 두 단계 위의 부모가 프로젝트 루트다.
    # parents[0]=attraction, parents[1]=data_analyze, parents[2]=SpotFit.
    # 실행하는 터미널의 현재 폴더가 달라도 입력/출력 기본 경로가 동일하도록 한다.
    root = Path(__file__).resolve().parents[2]
    parser.add_argument('--raw', type=Path, default=root / 'data/raw')
    parser.add_argument('--output', type=Path, default=root / 'data_analyze/attraction/results')
    parser.add_argument('--scope', choices=['food', 'all'], default='food')
    args = parser.parse_args()
    # 실행마다 별도 디렉터리를 만들어 과거 결과와 외식/전체 결과의 덮어쓰기를 방지한다.
    output = args.output / (args.scope + '_' + pd.Timestamp.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    data, targets, hypotheses, quality, selected = prepare(args.raw, args.scope)
    results = correlate(data, targets, hypotheses)
    summary = results[results.status == 'ok'].groupby(['hypothesis', 'feature', 'target']).agg(
        quarters=('quarter', 'nunique'), median_rho=('spearman_rho', 'median'),
        min_rho=('spearman_rho', 'min'), max_rho=('spearman_rho', 'max'),
        min_n=('n', 'min'), max_n=('n', 'max')).reset_index()
    coverage = data.groupby(TIME)[[f for _, f, _ in hypotheses]].count().reset_index()
    for name, frame in [('analysis_data', data), ('correlations_by_quarter', results),
                        ('correlation_summary', summary), ('facility_quality', quality),
                        ('feature_coverage', coverage), ('selected_industries', selected)]:
        frame.to_csv(output / f'{name}.csv', index=False, encoding='utf-8-sig')
    visualize(data, results, output)
    notes = (
        f'분석 범위: {args.scope}. 행정동×분기 기준. 업종 필터는 점포/매출에만 적용. '
        '시설 빈값은 미확인 값으로 유지하며 0으로 채우지 않음. '
        '시설 비율=관측 시설 수/원본 집객시설 수×100이며 합계 100% 구성비가 아님. '
        '연령별 매출 비율=해당 연령 매출/연령 확인 매출 합×100(금액·건수 별도). '
        '개폐업 지표=해당 분기 개폐업 수/해당 분기 전체 점포 수×100으로, 원본 공식 비율과 다를 수 있음. '
        '같은 분기 행정동 간 Spearman 탐색 분석이며 인과관계나 예측 성능을 증명하지 않음. '
        '최소 20쌍 및 비상수 열에만 계산. p/q값은 공간 의존성 미보정 참고값. '
        '시설 비율은 결측 표본 선택 및 공통 분모 영향에 주의. '
        '면적·입지·상업 규모의 교란을 통제하지 않았으므로 가설 채택을 자동 판정하지 않음. '
        '분기별 계수의 중앙값은 통합 상관계수가 아님. 점포는 2025년부터, 매출은 2025년까지 존재하므로 가설별 기간이 다름.'
    )
    images = ''.join(f'<h3>{html.escape(p.stem)}</h3><img src="{p.name}" style="max-width:100%">'
                     for p in sorted(output.glob('*.png')))
    report = '<!doctype html><meta charset="utf-8"><title>행정동 집객시설 분석</title>'
    report += '<h1>행정동 집객시설 가설 탐색</h1><p>' + html.escape(notes) + '</p>'
    # 분기별 결과에서 가설별 방향·표본수·시설 비율의 한계를 함께 요약한다.
    from hypothesis_report import build_hypothesis_summary
    report += build_hypothesis_summary(results)
    report += '<h2>집객시설 수와의 관계 요약</h2>' + summary[summary.target == '집객시설_수'].to_html(index=False)
    report += '<h2>데이터 품질</h2>' + quality.to_html(index=False) + images
    (output / 'report.html').write_text(report, encoding='utf-8')
    print(f'완료: {output}')
    print(summary[summary.target == '집객시설_수'].to_string(index=False))


if __name__ == '__main__':
    main()
