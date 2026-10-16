"""저장된 분기별 결과에서 가설 요약을 만든다. 통계적 가설 채택을 자동 선언하지 않는다."""
from pathlib import Path
import html
import pandas as pd

START = '<!-- hypothesis-summary-start -->'
END = '<!-- hypothesis-summary-end -->'
TITLES = {
    1: '유동인구 ↑ → 집객시설 수 ↑ / 특정 시설 비율 차이',
    2: '상주인구 ↑ → 집객시설 수 ↑ / 특정 시설 비율 차이',
    3: '연령대별 매출비율에 따른 특정 시설 비율 차이',
    4: '직장인구 ↑ → 집객시설 수 ↑ / 특정 시설 비율 차이',
    5: '점포 수 ↑ → 집객시설 수 ↑ / 특정 시설 비율 차이',
    6: '창업률 ↑ → 집객시설 수 ↑ / 특정 시설 비율 차이',
    7: '폐업률 ↓ → 집객시설 수 ↑ / 특정 시설 비율 차이',
}


def build_hypothesis_summary(results):
    """수·비율을 분리하고 분기별 방향 일관성 및 표본 수를 함께 보고한다.

    |rho| 0.3과 방향 일치 80%는 읽기 쉬운 탐색 요약용 기준일 뿐,
    통계적 유의성/인과성/모델 특징 채택 기준이 아니다. 음의 결과도 숨기지 않는다.
    """
    valid = results[results['status'].eq('ok')].copy()
    evidence = []
    for (number, feature, target), group in valid.groupby(['hypothesis', 'feature', 'target']):
        values = group['spearman_rho']
        median = values.median()
        consistency = ((values > 0) if median >= 0 else (values < 0)).mean()
        evidence.append(dict(hypothesis=int(number), feature=feature, target=target,
                             median=float(median), low=float(values.min()), high=float(values.max()),
                             quarters=len(group), start=int(group.quarter.min()), end=int(group.quarter.max()),
                             min_n=int(group.n.min()), max_n=int(group.n.max()), consistency=float(consistency)))
    def stats(row):
        return (f"ρ 중앙값 {row['median']:+.3f} (범위 {row['low']:+.3f}~{row['high']:+.3f}), "
                f"{row['quarters']}분기 [{row['start']}~{row['end']}], "
                f"분기별 n={row['min_n']}~{row['max_n']}, 중앙값과 방향 일치 {row['consistency']:.0%}")

    blocks = [START, '<section><h2>7개 가설 검토 요약</h2>',
              '<p><b>이 결과는 같은 분기에 서로 다른 행정동을 비교한 탐색 결과입니다.</b> '
              '한 행정동의 인구가 증가하면 시설이 늘어난다는 시간적·인과적 설명은 아닙니다. '
              'ρ는 분기별 Spearman 상관계수이고 아래 중앙값은 통합 상관계수가 아닙니다.</p>',
              '<p>“특정 시설 밀집”은 현재 자료로 직접 검증하지 않았습니다. 여기서는 '
              '시설별 수/원본 집객시설 수 비율과의 관계만 비교합니다. 실제 밀도는 면적당 시설 수, '
              '공간적 군집은 좌표 등을 이용한 별도 검증이 필요합니다. 시설 빈값은 0이 아니라 결측으로 '
              '제외했으므로 시설마다 표본이 다르며, 공통 분모의 영향도 있습니다.</p>',
              '<p>요약용 탐색 기준: |ρ 중앙값|≥0.30이고 분기별 방향 일치≥80%이면 '
              '“관계 경향 관찰”로 표시합니다. 기준 미달은 “뚜렷한 근거 부족”이며 '
              '관계가 없다는 증명은 아닙니다. 공간 의존성·교란요인 미보정으로 확정적 가설 채택은 하지 않습니다.</p>']
    for number, title in TITLES.items():
        rows = [r for r in evidence if r['hypothesis'] == number]
        blocks.append(f'<h3>{number}. {html.escape(title)}</h3>')
        totals = [r for r in rows if r['target'] == '집객시설_수']
        if number != 3:
            for row in totals:
                expected_sign = -1 if number == 7 else 1
                if abs(row['median']) < .3 or row['consistency'] < .8:
                    verdict = '예상한 시설 수 관계에 대한 뚜렷한 근거 부족'
                elif row['median'] * expected_sign > 0:
                    verdict = '예상 방향과 일치하는 시설 수 관계 경향 관찰'
                else:
                    verdict = '예상과 반대 방향의 시설 수 관계 경향 관찰'
                blocks.append('<p><b>' + verdict + '</b><br>' + html.escape(stats(row)) + '</p>')
            if not totals:
                blocks.append('<p>시설 수 관계를 계산할 유효 결과가 없습니다.</p>')
        if number in (6, 7):
            blocks.append('<p>주의: 공식 창업률·폐업률이 아니라 해당 분기 '
                          '개업/폐업 점포 수 ÷ 전체 점포 수로 계산한 대리지표입니다. '
                          '7번 가설은 폐업 지표와 집객시설 수의 음의 상관을 예상합니다.</p>')
        # 모든 조합을 표로 제공해 상위 몇 개만 선택해서 보여주는 편향을 줄인다.
        facilities = sorted([r for r in rows if r['target'] != '집객시설_수'],
                            key=lambda r: abs(r['median']), reverse=True)
        notable = [r for r in facilities if abs(r['median']) >= .3 and r['consistency'] >= .8]
        blocks.append(f'<p>시설 비율: 유효 조합 {len(facilities)}개 중 탐색 기준에 해당하는 조합 '
                      f'{len(notable)}개. <b>시설의 실제 밀집 여부는 판단 보류</b>.</p>')
        if notable:
            blocks.append('<p>대표 관계(절대 상관 중앙값 순, 최대 4개):</p><ul>')
            for row in notable[:4]:
                direction = '높은' if row['median'] > 0 else '낮은'
                description = (f"{row['feature']}가 높은 행정동에서 {row['target']}이 {direction} 경향. "
                               + stats(row))
                blocks.append('<li>' + html.escape(description) + '</li>')
            blocks.append('</ul>')
        if facilities:
            table = pd.DataFrame([{'비교 지표': r['feature'], '시설 비율': r['target'],
                                   'ρ 중앙값': round(r['median'], 3), '분기 수': r['quarters'],
                                   'ρ 최솟값': round(r['low'], 3), 'ρ 최댓값': round(r['high'], 3),
                                   '최소 n': r['min_n'], '최대 n': r['max_n'],
                                   '방향 일치': f"{r['consistency']:.0%}"} for r in facilities])
            blocks.append('<details><summary>시설 비율 전체 관계 펼치기 (양·음 모두 포함)</summary>'
                          + table.to_html(index=False, escape=True) + '</details>')
        else:
            blocks.append('<p>시설 비율을 계산할 유효 결과가 없습니다.</p>')
    blocks += ['<p>판단 범위: 결과 폴더의 업종 범위를 따릅니다. food는 점포·매출이 외식업 기준이며 '
               '인구·시설은 행정동 전체 기준입니다. 가설별 기간 차이, 반복 관측, 지역 면적 및 '
               '상업 규모의 영향 때문에 상관계수를 그대로 인과효과나 예측 성능으로 해석하지 마세요.</p>',
               '</section>', END]
    return '\n'.join(blocks)


def update_existing_report(folder):
    """원본 CSV·그래프는 건드리지 않고 HTML에만 요약을 갱신한다. 최초 원본은 보관한다."""
    folder = Path(folder)
    path = folder / 'report.html'
    original = path.read_text(encoding='utf-8')
    results = pd.read_csv(folder / 'correlations_by_quarter.csv')
    section = build_hypothesis_summary(results)
    if START in original:
        before, rest = original.split(START, 1)
        _, after = rest.split(END, 1)
        updated = before + section + after
    else:
        marker = '<h2>집객시설 수와의 관계 요약</h2>'
        if marker not in original:
            raise ValueError('보고서 삽입 위치를 찾을 수 없습니다.')
        updated = original.replace(marker, section + marker, 1)
    backup = folder / 'report.before_hypothesis_summary.html'
    if not backup.exists():
        backup.write_text(original, encoding='utf-8')
    path.write_text(updated, encoding='utf-8')
    print(f'요약 반영: {path}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_folder', type=Path)
    update_existing_report(parser.parse_args().result_folder)
