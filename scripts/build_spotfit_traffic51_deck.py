"""Restyle the supplied traffic-model PPT in the established SpotFit template.

This is a document transformation, not a model training/verification task.
Original reported values are preserved, with source-slide traceability in notes.
"""
from pathlib import Path
import hashlib
import json

from pptx import Presentation
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches

from build_spotfit_project_deck import (
    ROOT, OUT, DARK, GREEN, GREY, TEAL, PALE, MEDIUM, FONT,
    fresh, text, rect, panel, table, chart, walk,
)

SOURCE = Path('C:/Users/M/Desktop/교통사고_51차_2027_2028_직접예측_발표자료.pptx')
STYLE = OUT / 'SPOT_FIT_프로젝트발표_템플릿유지_20261110.pptx'
DEST = OUT / 'SPOT_FIT_교통사고_51차_직접예측_템플릿유지_20261111.pptx'
REVIEW = OUT / 'traffic_51_review'


def banner(s, value):
    rect(s, 1.25, 3.25, 17.5, .85, PALE)
    text(s, value, 1.5, 3.43, 17, .5, 24, DARK, MEDIUM)


def footer(s, value):
    text(s, value, 1.25, 10.12, 17.5, .43, 17, GREY)


def metric(s, title, value, caption, x, y=4.65, w=5.6):
    rect(s, x, y, w, 2.5, 'FAFAFA')
    rect(s, x, y, w, .07, GREEN, GREEN)
    text(s, title, x+.25, y+.25, w-.5, .5, 24, GREEN, MEDIUM)
    text(s, value, x+.25, y+.91, w-.5, .85, 48, GREEN, MEDIUM)
    text(s, caption, x+.25, y+1.98, w-.5, .4, 20, GREY)


def notes(s, index, detail):
    s.notes_slide.notes_text_frame.text = (
        f'원본: {SOURCE.name}\n원본 슬라이드: {index}\n'
        '재작성일: 2026-11-11 (Asia/Seoul)\n'
        '내용·수치는 사용자 제공 PPT 기준. 원모델/학습코드/예측 원자료를 재검증하거나 재학습하지 않았다.\n'
        '디자인: 기존 SpotFit 발표자료의 섹션07, 녹색·회색 계열, 네모고딕 글꼴을 재사용.\n\n'
        + detail
    )


def snapshot(p):
    records = []
    for i, s in enumerate(p.slides, 1):
        record = {'slide': i, 'text': [], 'tables': [], 'charts': []}
        for shape in walk(s.shapes):
            if shape.has_text_frame and shape.text:
                record['text'].append(shape.text)
            if shape.has_table:
                record['tables'].append([[c.text for c in row.cells] for row in shape.table.rows])
            if shape.has_chart:
                record['charts'].append([{'name': series.name, 'values': list(series.values)} for series in shape.chart.series])
        records.append(record)
    return records


def main():
    REVIEW.mkdir(parents=True, exist_ok=True)
    original_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    source = Presentation(SOURCE)
    assert len(source.slides) == 9
    evidence = snapshot(source)
    (REVIEW/'source_content.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    source_tables = lambda i: [s.table for s in source.slides[i-1].shapes if s.has_table]
    rows = lambda tbl: [[c.text for c in r.cells] for r in tbl.rows]
    structures = rows(source_tables(3)[0])
    preprocess = rows(source_tables(6)[0])
    predictions = rows(source_tables(7)[0])
    performance = rows(source_tables(8)[0])
    # Numbers come from the attachment; do not substitute the older repository model.
    perf27, perf28 = performance[2], performance[3]
    assert perf27[1:] == ['56.0%', '87.2%', '0.576']
    assert perf28[1:] == ['55.0%', '87.0%', '0.580']
    assert len(predictions) == 11

    prs = Presentation(STYLE)
    inherited = len(prs.slides)

    # 1. Study scope and headline metrics, explicitly historical validation.
    s = fresh(prs, '교통안전모델 ① · 2027·2028 직접예측', 'Traffic Safety · Model 51 Overview')
    banner(s, '2025년 실제 데이터를 입력해, 2027년과 2028년을 각각 독립적으로 예측')
    metric(s, '2027년 · 입력 기준 2년 후', perf27[1], '정확등급 일치율 · 5개 검증연도 평균', 1.25)
    metric(s, '2028년 · 입력 기준 3년 후', perf28[1], '정확등급 일치율 · 4개 검증연도 평균', 7.2)
    metric(s, '예측 방식', '직접예측', '중간 예측값 재사용 없음', 13.15)
    table(s, [['구분', '2027년 모델', '2028년 모델'],
              ['±1등급 이내 일치율', perf27[2], perf28[2]],
              ['등급 평균오차', perf27[3]+'등급', perf28[3]+'등급']], 1.25, 7.65, 17.5, 1.7, [6.1,5.7,5.7], 22)
    footer(s, '1등급 = 고위험 · 5등급 = 저위험  |  서울 25개 자치구 상대 5분위  |  수치는 원본의 과거 검증 결과')
    notes(s, 1, '56%/55%는 2027/2028년의 실제 관측 결과를 평가한 수치가 아니라 해당 예측기간 모델의 과거 검증 평균이다. 정확등급은 완전일치, ±1등급은 한 단계 이내 오차를 인정하는 별도 보조지표다. 기존 범죄 모델의 A=저위험과 숫자1=고위험 방향을 혼동하지 않는다.')

    # 2. Native-shape flow illustration, preserving both approaches.
    s = fresh(prs, '교통안전모델 ② · 연쇄예측에서 직접예측으로', 'Prediction Strategy · Recursive vs Direct')
    banner(s, '미래 예측값을 다시 입력하는 연쇄 구조 대신, 목표연도별 모델을 별도로 학습')
    rect(s, 1.25, 4.6, 8.5, 4.65, 'FAFAFA')
    rect(s, 10.05, 4.6, 8.7, 4.65, 'FAFAFA')
    text(s, '사용하지 않는 방식 · 연쇄예측', 1.5, 4.88, 8, .55, 25, GREEN, MEDIUM)
    text(s, '채택한 방식 · 독립 직접예측', 10.3, 4.88, 8.2, .55, 25, GREEN, MEDIUM)
    for i, label in enumerate(['2025 실제 입력', '2026 예측', '2027 예측', '2028 예측']):
        y = 5.65 + i*.7
        rect(s, 1.65, y, 3.6, .5, PALE)
        text(s, label, 1.8, y+.055, 3.3, .38, 21)
        if i < 3:
            text(s, '↓', 3.16, y+.49, .5, .25, 12, GREEN)
    text(s, '예측값을 다시 입력\n오차 누적 가능', 5.6, 6.4, 3.7, 1.3, 22, GREY)
    for i, (period, target) in enumerate([('2년 모델', '2027 예측'), ('3년 모델', '2028 예측')]):
        y = 5.95+i*1.25
        rect(s, 10.4, y, 2.6, .65, PALE)
        text(s, '2025 실제 입력', 10.53, y+.13, 2.4, .4, 20)
        text(s, '→', 13.05, y+.12, .5, .4, 22, GREEN)
        rect(s, 13.6, y, 2.0, .65, 'F5F7F6')
        text(s, period, 13.75, y+.13, 1.7, .4, 20)
        text(s, '→', 15.65, y+.12, .5, .4, 22, GREEN)
        rect(s, 16.2, y, 2.2, .65, PALE)
        text(s, target, 16.35, y+.13, 1.9, .4, 20)
    text(s, '학습 예: 2017→2019 / 2017→2020', 10.5, 8.65, 7.8, .4, 21, GREY)
    footer(s, '직접예측은 예측값 재투입 경로를 없애지만, 장기예측의 불확실성 자체를 제거하는 것은 아닙니다.')
    notes(s, 2, '원본의 2025→2026→2027→2028 연쇄 방식과 두 목표연도 직접예측 방식을 도식화했다. 연쇄모델 대비 실측 성능개선율은 원본에 없어 추가하지 않았다.')

    # 3. All source structure table rows retained.
    s = fresh(prs, '교통안전모델 ③ · 두 모델의 학습 구조', 'Training Structure · Separate Forecast Horizons')
    banner(s, '예측 간격에 맞춰 목표연도를 이동하고, 각 기간의 Y를 별도로 생성')
    table(s, structures, 1.25, 4.55, 17.5, 4.45, [3.8,6.85,6.85], 23)
    text(s, 'Y = 목표연도의 종합위험점수와 상대 위험등급', 1.5, 9.4, 17, .55, 27, GREEN, MEDIUM)
    footer(s, '모델 입력은 관측된 실제 자료, 출력은 해당 목표연도의 위험점수·순위·상대등급입니다.')
    notes(s, 3, '원본 학습구조 표를 그대로 옮겼다. 종합위험점수의 세부 산식과 원천13개 Feature 이름은 첨부 PPT에 없으며 임의로 추정해 넣지 않았다. 두 모델의 검증 목표연도 범위가 다르므로 정확도 수치의 단순 우열 비교에 주의한다.')

    # 4/5. Same composition for comparable model explanations.
    for horizon, year, metric_row, ensemble, desc, source_slide in [
        (2, 2027, perf27, '75% : 25%', '점수 수준 예측과 변화량 예측을 결합', 4),
        (3, 2028, perf28, '50% : 50%', '서로 다른 위험점수 예측기를 결합', 5),
    ]:
        number = '④' if horizon == 2 else '⑤'
        s = fresh(prs, f'교통안전모델 {number} · {year}년 {horizon}년 후 예측', f'Direct Forecast {year} · Ensemble Design')
        banner(s, desc)
        metric(s, '정확등급 일치율', metric_row[1], f'{5 if horizon==2 else 4}개 검증연도 평균', 1.25, 4.5)
        metric(s, '±1등급 이내 일치율', metric_row[2], '한 등급 이내 오차를 인정한 보조지표', 7.2, 4.5)
        metric(s, '등급 평균오차', metric_row[3], '단위: 등급 · 낮을수록 좋음', 13.15, 4.5)
        panel(s, '구성 A · Gradient Boosting', f'GBR depth 3\n{horizon}년 후 종합위험점수 예측\n앙상블 반영 비중: {"75%" if horizon==2 else "50%"}', 1.25, 7.45, 6.5, 2.35)
        panel(s, '구성 B · Ridge', ('2년 후 점수 변화량 예측\n현재 대비 변화 학습\n앙상블 반영 비중: 25%' if horizon==2 else '3년 후 종합위험점수 예측\nGBR와 다른 기법으로 수준 학습\n앙상블 반영 비중: 50%'), 8.05, 7.45, 6.5, 2.35)
        rect(s, 14.85, 7.45, 3.9, 2.35, PALE)
        text(s, '최종 앙상블', 15.1, 7.72, 3.4, .5, 23, GREEN, MEDIUM)
        text(s, ensemble, 15.1, 8.45, 3.4, .6, 29, GREEN, MEDIUM)
        text(s, f'{year} 순위·등급 산출', 15.1, 9.18, 3.4, .4, 20)
        footer(s, f'2025년 실제 입력 → {year}년 직접예측  |  성능은 원본 발표자료의 과거 검증 평균입니다.')
        notes(s, source_slide, 'GBR는 Gradient Boosting Regressor, depth3은 원본 기재 내용이다. '+('Ridge는 변화량을 예측하는 구성이다. 원본에는 변화량을 최종 점수로 환산하는 상세 코드·산식이 없으므로 새로운 결합식을 단정해 기재하지 않았다.' if horizon==2 else '두 점수 예측기를 동일 비중으로 결합한다.')+' 모델 원파일과 학습 실행은 이 문서 변환 작업에서 확인하지 않았다.')

    # 6. Target engineering and data hygiene, not a new ETL claim.
    s = fresh(prs, '교통안전모델 ⑥ · 전처리 데이터 변경사항', 'Data Preparation · Horizon-specific Targets')
    banner(s, '1년 후 목표값 대신, 자치구별 2년 후·3년 후 목표값을 생성')
    table(s, preprocess, 1.25, 4.55, 17.5, 4.55, [3.8,13.7], 22)
    text(s, '연도 이동 → 자치구별 목표값 결합 → 시간순 검증 → 2025년 실제 입력으로 예측', 1.25, 9.45, 17.5, .5, 23, GREEN, MEDIUM)
    footer(s, '225행은 2017–2025년의 기준 패널 규모이며, 각 예측기간의 실제 학습쌍 수와 구분해야 합니다.')
    notes(s, 6, '기준패널225행과 h년후정답이붙는학습쌍수는다르다. 원본의 미래누수방지 설명은 검증목표연도보다앞선자료만학습이라고되어있다. 실제운영재현을보장하려면 예측기준시점에 정답 및 Feature가 이용가능했는지도 코드와 공표시점으로 별도확인해야한다. 이작업은그검증을수행하지않았다. 원천13개변수의이름은원본에제시되지않아추가하지않았다.')

    # 7. Correct the misleading source title: 5 per year, 10 rows total.
    s = fresh(prs, '교통안전모델 ⑦ · 연도별 고위험 상위 5개 구', 'Forecast Results · Top Five Districts by Year')
    banner(s, '각 연도에서 상대 위험이 높은 5개 구에 1등급 배정 · 아래는 두 연도 총 10개 예측행')
    text(s, '2027년 · 2년 후 모델', 1.25, 4.48, 8.5, .55, 27, GREEN, MEDIUM)
    text(s, '2028년 · 3년 후 모델', 10.05, 4.48, 8.7, .55, 27, GREEN, MEDIUM)
    for year, x, width in [('2027',1.25,8.5),('2028',10.05,8.7)]:
        r = [['위험순위','자치구','예측점수','등급']] + [line[1:] for line in predictions[1:] if line[0] == year]
        table(s, r, x, 5.2, width, 3.5, [1.5,2.7,2.5,1.3], 23)
    text(s, '순위가 높을수록 상대적 위험이 높음  ·  두 연도는 서로 다른 모델의 결과', 1.25, 9.1, 17.5, .45, 22, GREEN)
    text(s, '원본 운영 제안: 정책 판단 시 49차 피해자율 전망도 함께 확인', 1.25, 9.62, 17.5, .4, 20, GREY)
    footer(s, '표시 점수는 소수 둘째 자리 기준이며 순위는 원본을 유지했습니다. 공식 확정등급이 아닙니다.')
    notes(s, 7, '원본 제목은 고위험상위10개자치구이나 실제표는2027상위5+2028상위5총10행이다. 이를연도별상위5로명확화했다. 동일표시점수(66.52/66.82등)에서순위는원본순위를그대로유지했으며반올림전점수나동률처리규칙은확인하지않았다. 49차피해자율전망은원본에참조되나그모델자료는이첨부에없다. 행정구명/점수/순위/등급을원표에서직접복사했다.')

    # 8. Editable chart and companion table: preserve all source metrics.
    s = fresh(prs, '교통안전모델 ⑧ · 성능 비교와 해석', 'Evaluation · Exact Grade and Adjacent Grade Agreement')
    text(s, '원본에 제시된 예측기간별 검증 성능', 1.25, 3.3, 17.5, .55, 27, GREEN, MEDIUM)
    exact = [float(row[1].rstrip('%')) for row in performance[1:]]
    adjacent = [float(row[2].rstrip('%')) for row in performance[1:]]
    comparison_chart = chart(s, ['기존 1년 후','2027 · 2년 후','2028 · 3년 후'], [('정확등급', exact), ('±1등급', adjacent)], 1.25, 4.2, 9.6, 3.35, 100, 20, '0.0"%"', True)
    comparison_chart.has_legend = False
    rect(s, 3.2, 7.7, .18, .18, GREEN, GREEN)
    text(s, '정확등급', 3.5, 7.63, 2.3, .4, 18)
    rect(s, 6.0, 7.7, .18, .18, TEAL, TEAL)
    text(s, '±1등급', 6.3, 7.63, 2.3, .4, 18)
    panel(s, '평가할 때 함께 볼 내용', '2028 모델은 학습 가능한 연도쌍이 적음\n장기예측의 등급 오차·순위 안정성 확인\n상대 위험 예측이지 공식 확정등급 아님\n새 실제연도가 확보되면 재학습·재검증', 11.2, 4.1, 7.55, 3.8)
    table(s, performance, 1.25, 8.2, 17.5, 1.75, [5.6,4,4,3.9], 18)
    footer(s, '검증연도 범위가 달라 기간별 우열을 단정할 수 없습니다. ±1등급 일치율은 정확등급 일치율과 다릅니다.')
    notes(s, 8, '원본기존1년모델:정확60%,±1등급90.4%,평균오차.496. 2년56/87.2/.576,3년55/87/.580. 기존1년검증연도는첨부에없음;2년2021–2025,3년2022–2025. 따라서차이를예측기간만의효과로인과적으로단정하지않도록원본의장기하락은자연스러운결과라는문장을신중하게재표현했다. ±1등급보조지표를정확도약87%로요약하지않는다. 직접예측채택결론은유지한다.')

    # 9. Operations proposal, not an assertion the new model is already deployed.
    s = fresh(prs, '교통안전모델 ⑨ · 최종 운영안', 'Operation Plan · Separate Years, Actual-data Refresh')
    banner(s, '2027년과 2028년을 분리해 제공하고, 새 실제 데이터로 예측을 갱신')
    panel(s, '2027년 제공값', '2025년 입력 · 2년 후 모델\n위험점수 · 순위 · 5등급\n2026년 현재 기준: 내년 전망', 1.25, 4.55, 5.6, 2.65)
    panel(s, '2028년 제공값', '2025년 입력 · 3년 후 모델\n위험점수 · 순위 · 5등급\n2026년 현재 기준: 2년 뒤 전망', 7.2, 4.55, 5.6, 2.65)
    panel(s, '공통 입력 원칙', '2025년 실제 자치구 자료\n중간연도 예측값을 재사용하지 않음\n입력연도와 목표연도를 함께 표시', 13.15, 4.55, 5.6, 2.65)
    table(s, [['운영 항목', '적용 계획'],
              ['표시 정보', '등급만 표시하지 않고 점수·순위·예측연도·예측간격·불확실성을 함께 안내'],
              ['2026 실제 자료 확보 시', '2027년 결과를 1년 후 모델로 재산출'],
              ['2027 실제 자료 확보 시', '2028년 결과도 1년 후 모델로 재산출']], 1.25, 7.65, 17.5, 2.1, [4.6,12.9], 21)
    footer(s, '원본의 운영 제안입니다. 51차 모델의 현재 백엔드·프론트엔드 적용 여부는 이번 작업에서 확인하지 않았습니다.')
    notes(s, 9, '원본운영안을재정리. 이새교통51차모델을기존recommender/v1교통2차모델과같은파일로간주하지않는다. 이번작업은PPT재작성뿐이며모델복사/학습/API교체/배포를수행하지않았다. 불확실성표시는원본의제안이며검증된확률적예측구간산식이제공된것은아니다. 2026년현재와2025입력기준의예측간격을구분해안내했다.')

    for item in list(prs.slides._sldIdLst)[:inherited]:
        prs.part.drop_rel(item.rId)
        prs.slides._sldIdLst.remove(item)
    assert len(prs.slides) == 9
    prs.save(DEST)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == original_hash
    # Check exact source numeric strings in final table cells/text.
    final = Presentation(DEST)
    body = '\n'.join(s.text for slide in final.slides for s in walk(slide.shapes) if s.has_text_frame)
    cells = '\n'.join(c.text for slide in final.slides for s in slide.shapes if s.has_table for r in s.table.rows for c in r.cells)
    for source_row in performance[1:]:
        for token in source_row[1:]:
            assert token in body+cells, token
    final_tables = [rows(s.table) for slide in final.slides for s in slide.shapes if s.has_table]
    for r in predictions[1:]:
        assert any(r[1:] in t for t in final_tables), r
    audit = {'source': str(SOURCE), 'source_sha256': original_hash, 'output': str(DEST),
             'source_slides': 9, 'output_slides': len(final.slides), 'original_unchanged': True,
             'all_reported_performance_values_preserved': True, 'all_10_prediction_rows_preserved': True,
             'model_training_or_service_changes': False,
             'clarifications': ['Risk table contains 5 districts per year, not 10 per year.',
                                'Metrics describe historical validation, not observed 2027/2028 outcomes.',
                                'Forecast horizons refer to the 2025 input year.',
                                'New traffic model deployment is not verified.']}
    (REVIEW/'content_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
