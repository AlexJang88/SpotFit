from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.chart.data import ChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/presentations/교통사고_94차_개발과정_3개모델비교.pptx"

NAVY = RGBColor(21, 38, 61)
BLUE = RGBColor(37, 99, 235)
SKY = RGBColor(14, 165, 233)
TEAL = RGBColor(13, 148, 136)
GREEN = RGBColor(22, 163, 74)
ORANGE = RGBColor(234, 88, 12)
RED = RGBColor(220, 38, 38)
INK = RGBColor(30, 41, 59)
MID = RGBColor(100, 116, 139)
LIGHT = RGBColor(241, 245, 249)
WHITE = RGBColor(255, 255, 255)
LINE = RGBColor(203, 213, 225)


def box(slide, x, y, w, h, fill=WHITE, line=LINE, radius=True):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h),
    )
    shape.fill.solid(); shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line
    return shape


def text(slide, x, y, w, h, value, size=16, bold=False, color=INK,
         align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP, font="맑은 고딕"):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.clear(); tf.word_wrap = True; tf.vertical_anchor = valign
    p = tf.paragraphs[0]; p.text = value; p.alignment = align
    p.font.name = font; p.font.size = Pt(size); p.font.bold = bold; p.font.color.rgb = color
    return tb


def rich(slide, x, y, w, h, lines, size=15, color=INK, bullet=True):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.clear(); tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line; p.font.name = "맑은 고딕"; p.font.size = Pt(size); p.font.color.rgb = color
        p.space_after = Pt(8)
        if bullet: p.text = "• " + p.text
    return tb


def header(slide, title, subtitle, page):
    text(slide, .65, .35, 11.8, .45, title, 25, True, NAVY)
    text(slide, .67, .88, 11.7, .28, subtitle, 11, False, MID)
    text(slide, 12.2, .42, .45, .28, f"{page:02d}", 10, True, MID, PP_ALIGN.RIGHT)
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(.65), Inches(1.23), Inches(12), Inches(.025))
    line.fill.solid(); line.fill.fore_color.rgb = LINE; line.line.fill.background()


def metric(slide, x, y, title, value, note, color=BLUE, w=2.75):
    box(slide, x, y, w, 1.35, WHITE, LINE)
    text(slide, x+.18, y+.14, w-.36, .25, title, 11, True, MID)
    text(slide, x+.18, y+.43, w-.36, .45, value, 24, True, color)
    text(slide, x+.18, y+.98, w-.36, .22, note, 9, False, MID)


def add_bar_chart(slide, x, y, w, h, categories, series, max_value=1.0, percent=True):
    data = ChartData(); data.categories = categories
    for name, values in series: data.add_series(name, values)
    chart = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), data).chart
    chart.has_legend = True; chart.legend.position = XL_LEGEND_POSITION.BOTTOM
    chart.legend.font.size = Pt(10); chart.has_title = False
    chart.value_axis.minimum_scale = 0; chart.value_axis.maximum_scale = max_value
    chart.value_axis.has_major_gridlines = True
    chart.value_axis.tick_labels.font.size = Pt(9)
    if percent: chart.value_axis.tick_labels.number_format = '0%'
    chart.category_axis.tick_labels.font.size = Pt(10)
    chart.series[0].format.fill.solid(); chart.series[0].format.fill.fore_color.rgb = MID
    if len(chart.series) > 1:
        chart.series[1].format.fill.solid(); chart.series[1].format.fill.fore_color.rgb = BLUE
    return chart


def main():
    prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # 1. Cover
    s = prs.slides.add_slide(blank); s.background.fill.solid(); s.background.fill.fore_color.rgb = NAVY
    text(s, .8, .72, 2.0, .28, "MODEL STORY · v94", 12, True, RGBColor(125, 211, 252))
    text(s, .8, 1.5, 11.7, 1.25, "교통사고 94차 모델은\n어떻게 만들어졌나", 34, True, WHITE)
    text(s, .82, 3.05, 10.8, .72, "안전기준표 1차 → Persistence 기준 91차 → TOPIS 교통량을 더한 94차\n변수·학습 구조·독립검증·95차 미채택까지 한 번에 정리", 17, False, RGBColor(203, 213, 225))
    for i, (label, c) in enumerate([("안전기준표 1차", SKY), ("91차 기준모델", MID), ("94차 최종모델", GREEN)]):
        box(s, .82 + i*3.15, 5.25, 2.75, .68, c, c)
        text(s, .82+i*3.15, 5.42, 2.75, .25, label, 13, True, WHITE, PP_ALIGN.CENTER)
    text(s, .82, 6.78, 11.8, .25, "근거: 저장소 내 모델 파일·model_info·2025 독립검증·Raw→Processed 재현 검증", 9, False, RGBColor(148, 163, 184))

    # 2. Executive answer
    s = prs.slides.add_slide(blank); header(s, "한 장 요약", "91차 대비 94차의 가장 큰 개선은 3년 후 정확등급 +20%p", 2)
    metric(s, .7, 1.55, "2년형 정확등급", "68% → 68%", "91차 → 94차, 변화 없음", BLUE, 2.85)
    metric(s, 3.75, 1.55, "2년형 점수 MAE", "2.991 → 2.897", "3.1% 감소(낮을수록 좋음)", GREEN, 2.85)
    metric(s, 6.8, 1.55, "3년형 정확등급", "48% → 68%", "+20%p · 상대 41.7% 향상", GREEN, 2.85)
    metric(s, 9.85, 1.55, "3년형 등급오차", "0.56 → 0.40", "28.6% 감소(낮을수록 좋음)", GREEN, 2.85)
    box(s, .7, 3.25, 12, 2.7, LIGHT, LIGHT)
    text(s, 1.0, 3.55, 2.15, .42, "결론", 18, True, NAVY)
    rich(s, 1.0, 4.05, 11.0, 1.45, [
        "91차는 미래 지표가 현재와 같다고 보는 Persistence 기준선이고, 94차는 지표별 예측모델과 TOPIS 교통량을 사용합니다.",
        "동일한 2025 독립검증에서 3년형 정확등급이 48%에서 68%로 +20%p 개선됐습니다. 다만 ±1등급은 96%에서 92%로 4%p 낮아졌습니다.",
        "안전기준표 1차는 타깃이 달라 직접 향상률을 계산할 수 없으며, 후속 95차는 독립검증 악화로 미채택했습니다."
    ], 14)

    # 3. Three model comparison
    s = prs.slides.add_slide(blank); header(s, "세 모델의 역할 비교", "안전기준표 1차는 목적·타깃이 달라 성능 수치의 직접 비교가 불가능", 3)
    cols = [(.7, "안전기준표 1차", SKY), (4.45, "91차 기준모델", MID), (8.2, "94차 최종", GREEN)]
    bodies = [
        ["목적: 기준표만으로 가능한 최소 모델", "타깃: 다음 해 전체 교통사고 건수", "입력: 안전기준표 점수 1개(원지표 6개)", "알고리즘: Gradient Boosting", "2025 홀드아웃 MAE 415.8 / WMAPE 30.5%"],
        ["목적: 94차의 동일조건 기준선", "타깃: 2·3년 후 어린이 안전등급", "점수지표: 2년형 21개 / 3년형 18개", "미래지표 예측: 마지막 관측값 유지", "2025 정확등급: 2년 68% / 3년 48%"],
        ["목적: 교통 노출량을 반영한 개선", "타깃: 91차와 동일", "점수지표: 91차와 동일", "추가 입력: TOPIS 교통량 5개", "2025 정확등급: 2년 68% / 3년 68%"]
    ]
    for (x, title_, c), lines in zip(cols, bodies):
        box(s, x, 1.55, 3.45, 4.6, WHITE, c)
        box(s, x, 1.55, 3.45, .7, c, c)
        text(s, x, 1.76, 3.45, .27, title_, 16, True, WHITE, PP_ALIGN.CENTER)
        rich(s, x+.22, 2.52, 3.0, 3.25, lines, 12)
    box(s, .7, 6.42, 11.95, .55, RGBColor(255, 247, 237), RGBColor(253, 186, 116))
    text(s, .9, 6.57, 11.55, .25, "주의: 1차와 94차는 예측대상·기간·평가지표가 다릅니다. 1차는 설명용 기준모델, 91차는 94차의 정량 비교군입니다.", 11, True, ORANGE, PP_ALIGN.CENTER)

    # 4. Timeline
    s = prs.slides.add_slide(blank); header(s, "개선 흐름", "단순 기준표 점수에서 다변량 미래예측·독립검증 구조로 발전", 4)
    y = 3.35
    line = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.1), Inches(y), Inches(10.9), Inches(.08)); line.fill.solid(); line.fill.fore_color.rgb = LINE; line.line.fill.background()
    steps = [
        (1.05, "1차", "안전기준표", "6개 원지표를\n100점 점수로 압축", SKY),
        (3.4, "41~92차", "탐색·정교화", "다변량·등급화·\n장기예측 구조 탐색", MID),
        (5.85, "91차", "Persistence 기준", "미래값을 마지막\n관측값으로 고정", BLUE),
        (8.25, "94차", "교통량 추가", "TOPIS 2017~2025\n미래지표 예측 개선", GREEN),
        (10.65, "95차", "미채택 실험", "도로위험도 추가\n독립검증 악화", RED),
    ]
    for x, ver, name, desc, c in steps:
        shape = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y-.25), Inches(.58), Inches(.58)); shape.fill.solid(); shape.fill.fore_color.rgb=c; shape.line.color.rgb=c
        text(s, x-.25, 1.62, 1.1, .35, ver, 16, True, c, PP_ALIGN.CENTER)
        text(s, x-.55, 2.12, 1.7, .34, name, 13, True, NAVY, PP_ALIGN.CENTER)
        text(s, x-.7, 4.08, 2.0, .8, desc, 11, False, INK, PP_ALIGN.CENTER)
    box(s, 1.0, 5.55, 11.05, .85, LIGHT, LIGHT)
    text(s, 1.25, 5.8, 10.55, .35, "방식 변화의 핵심: 기준표 점수 1개 → 91차의 현재값 유지 기준선 → 지표별 미래예측과 교통량 보조입력 → 5분위 안전등급", 13, True, NAVY, PP_ALIGN.CENTER)

    # 5. Data pipeline
    s = prs.slides.add_slide(blank); header(s, "94차가 만들어지는 데이터 흐름", "Raw 154개 → TOPIS 집계 → Processed 패널 → 2·3년형 모델 → v1 배포", 5)
    stages = [
        ("RAW", "93차 기반 46개\n+ TOPIS 월별 108개", "154 files", MID),
        ("INTERIM", "시간대별 교통량을\n연도×자치구로 집계", "216 rows / 24구", SKY),
        ("PROCESSED", "2017~2025 패널 +\n모델 설정·계수·가중치", "교통사고_94차_데이터.xlsx", BLUE),
        ("MODEL", "2년형 21개\n3년형 18개", "2025 independent test", TEAL),
        ("SERVING", "2027·2028 예측\n연도별 25개 자치구", "recommender/v1", GREEN),
    ]
    for i, (name, desc, tag, c) in enumerate(stages):
        x=.55+i*2.55
        box(s, x, 2.05, 2.15, 2.65, WHITE, c)
        text(s, x+.15, 2.3, 1.85, .3, name, 13, True, c, PP_ALIGN.CENTER)
        text(s, x+.18, 2.93, 1.8, .75, desc, 12, True, INK, PP_ALIGN.CENTER)
        text(s, x+.12, 4.1, 1.9, .3, tag, 9, False, MID, PP_ALIGN.CENTER)
        if i < 4: text(s, x+2.16, 3.02, .38, .35, "→", 22, True, MID, PP_ALIGN.CENTER)
    box(s, .75, 5.35, 11.85, 1.05, LIGHT, LIGHT)
    text(s, 1.0, 5.58, 11.35, .55, "재현 검증: TOPIS 108개(2017~2025, 매년 12개) 재집계값과 Processed의 최대 차이 5.82×10⁻¹¹ · v1 예측은 2027/2028 각각 25개 자치구", 12, True, NAVY, PP_ALIGN.CENTER)

    # 6. Architecture
    s = prs.slides.add_slide(blank); header(s, "94차 모델 구조", "교통량은 최종 점수식의 새 항목이 아니라 ‘미래 지표 예측’을 돕는 보조 변수", 6)
    box(s, .7, 1.65, 2.7, 3.9, RGBColor(239, 246, 255), BLUE)
    text(s, .9, 1.93, 2.3, .3, "입력 시계열", 17, True, BLUE, PP_ALIGN.CENTER)
    rich(s, .98, 2.55, 2.1, 2.2, ["핵심 지표의 현재값", "전년값·증감", "최근 3년 평균", "자치구", "TOPIS 교통량 5개"], 12)
    text(s, 3.52, 3.25, .5, .4, "→", 24, True, MID, PP_ALIGN.CENTER)
    box(s, 4.05, 1.65, 2.7, 3.9, RGBColor(240, 253, 250), TEAL)
    text(s, 4.25, 1.93, 2.3, .3, "지표별 미래예측", 17, True, TEAL, PP_ALIGN.CENTER)
    rich(s, 4.3, 2.55, 2.2, 2.2, ["Persistence / 이동평균", "Ridge", "Random Forest", "Extra Trees", "Gradient Boosting", "후보 혼합(ensemble)"], 12)
    text(s, 6.87, 3.25, .5, .4, "→", 24, True, MID, PP_ALIGN.CENTER)
    box(s, 7.4, 1.65, 2.35, 3.9, RGBColor(255, 247, 237), ORANGE)
    text(s, 7.6, 1.93, 1.95, .3, "종합점수", 17, True, ORANGE, PP_ALIGN.CENTER)
    rich(s, 7.67, 2.55, 1.8, 2.2, ["Poisson 계수", "|계수| 정규화", "지표 방향 반영", "가중합 점수"], 12)
    text(s, 9.86, 3.25, .5, .4, "→", 24, True, MID, PP_ALIGN.CENTER)
    box(s, 10.35, 1.65, 2.25, 3.9, RGBColor(240, 253, 244), GREEN)
    text(s, 10.55, 1.93, 1.85, .3, "안전등급", 17, True, GREEN, PP_ALIGN.CENTER)
    rich(s, 10.65, 2.55, 1.62, 2.2, ["연도 내 5분위", "1 = 가장 안전", "5 = 가장 위험", "2년/3년 별도"], 12)
    box(s, 1.15, 6.0, 11.1, .62, LIGHT, LIGHT)
    text(s, 1.35, 6.17, 10.7, .25, "즉, 94차의 변화는 ‘점수 항목 추가’가 아니라 ‘미래 입력값을 예측하는 방법의 개선’입니다.", 13, True, NAVY, PP_ALIGN.CENTER)

    # 7. Added variables
    s = prs.slides.add_slide(blank); header(s, "94차에서 새로 들어간 교통량 변수", "TOPIS 시간대별 관측을 자치구·연도 단위 5개 노출지표로 변환", 7)
    items = [
        ("평균 일교통량", "관측지점별 하루 총 교통량의 평균", BLUE),
        ("평일 평균 일교통량", "월~금 통행 노출 수준", TEAL),
        ("주말 평균 일교통량", "토·일 통행 패턴 분리", ORANGE),
        ("등교시간 비중", "07~09시 교통량 / 일교통량", SKY),
        ("하교시간 비중", "14~18시 교통량 / 일교통량", GREEN),
    ]
    for i,(name,desc,c) in enumerate(items):
        x=.8+(i%3)*4.15; y=1.65+(i//3)*2.1
        box(s,x,y,3.75,1.62,WHITE,c)
        text(s,x+.22,y+.25,3.3,.3,name,15,True,c)
        text(s,x+.22,y+.78,3.3,.45,desc,11,False,INK)
    box(s, 8.95, 3.75, 3.6, 1.62, RGBColor(255, 247, 237), RGBColor(253, 186, 116))
    text(s, 9.2, 4.02, 3.1, .3, "커버리지 주의", 14, True, ORANGE)
    text(s, 9.2, 4.48, 3.05, .55, "25개 중 24개 자치구 관측\n미관측 구는 학습 파이프라인에서 중앙값 처리", 10, False, INK)
    text(s, .85, 6.08, 11.7, .5, "원천 범위: 2017~2025년 × 12개월 = 108개 파일 · 시간대 0~23시 · 지점별 일자 자료", 13, True, NAVY, PP_ALIGN.CENTER)

    # 8. Core variables
    s = prs.slides.add_slide(blank); header(s, "최종 점수를 구성하는 핵심 변수", "2년형 21개, 3년형 18개 — 14개 공통축 + 예측기간별 특화 변수", 8)
    common = ["어린이1만명당 자동차등록대수", "특수차 등록대수", "고등학교 평균학생수", "연간 사고·중독 경험률", "전체사고다발 부상신고자수", "법규위반다발 부상신고자수", "전체사고다발 발생건수", "보호구역 어린이다발 사망자수", "보호구역 보행어린이 사고율", "보호구역 보행어린이 부상자수", "승용차 전일 평균속도", "자전거다발 경상자수", "자전거다발 지점수", "자동차 1만대당 전체사고건수"]
    only2 = ["유치원 사립수", "도로연장", "보행어린이 사고율", "보행어린이 사망자수", "전체사고다발 지점수", "자동차 등록대수 합계", "운전자 안전벨트 착용률"]
    only3 = ["중학교 사립수", "도로율", "전체사고다발 지점수", "자전거다발 부상신고자수"]
    box(s,.65,1.5,5.85,4.95,WHITE,BLUE); text(s,.9,1.78,5.35,.3,"공통 14개",17,True,BLUE,PP_ALIGN.CENTER)
    rich(s,.9,2.28,5.25,3.8,[f"{i+1}. {v}" for i,v in enumerate(common)],10,bullet=False)
    box(s,6.75,1.5,2.75,4.95,WHITE,TEAL); text(s,6.95,1.78,2.35,.3,"2년형 전용 7개",15,True,TEAL,PP_ALIGN.CENTER)
    rich(s,7.0,2.33,2.25,3.4,only2,10)
    box(s,9.75,1.5,2.9,4.95,WHITE,ORANGE); text(s,9.95,1.78,2.5,.3,"3년형 전용 4개",15,True,ORANGE,PP_ALIGN.CENTER)
    rich(s,10.0,2.33,2.4,3.0,only3,10)
    text(s,.8,6.7,11.8,.25,"※ ‘전체사고다발 지점수’는 두 모델 공통이므로 실제 합계는 2년 21개, 3년 18개입니다.",9,False,MID,PP_ALIGN.CENTER)

    # 9. Performance
    s = prs.slides.add_slide(blank); header(s, "91차 vs 94차 — 동일조건 성능 비교", "2025년 25개 자치구 독립검증 · 같은 타깃·등급체계·예측기간", 9)
    add_bar_chart(s,.65,1.55,5.75,3.45,["2년 정확등급","2년 ±1등급","3년 정확등급","3년 ±1등급"],[
        ("91차", [.68,1.00,.48,.96]), ("94차", [.68,1.00,.68,.92])
    ],1.0,True)
    text(s,1.1,5.08,4.9,.35,"등급 정확도",14,True,NAVY,PP_ALIGN.CENTER)
    add_bar_chart(s,6.75,1.55,5.85,3.45,["2년 등급오차","2년 점수MAE","3년 등급오차","3년 점수MAE"],[
        ("91차", [.32,2.991,.56,3.754]), ("94차", [.32,2.897,.40,3.401])
    ],4.2,False)
    text(s,7.2,5.08,4.95,.35,"오차 지표 (낮을수록 좋음)",14,True,NAVY,PP_ALIGN.CENTER)
    box(s,.9,5.75,11.5,.78,LIGHT,LIGHT)
    text(s,1.1,5.97,11.1,.35,"결론: 3년 정확등급 +20%p(상대 +41.7%) · 3년 등급오차 28.6%↓ · 점수 MAE 9.4%↓ / 단, 3년 ±1등급은 4%p↓",12,True,GREEN,PP_ALIGN.CENTER)

    # 10. Safety baseline details
    s = prs.slides.add_slide(blank); header(s, "안전기준표 1차 모델은 무엇이 달랐나", "공식 기준표 변수만 사용한 설명 가능한 최소 비교모델", 10)
    labels=[("교통사고 사망자/1만명","50.00",RED),("재난약자/1만명","10.00",ORANGE),("도로면적당 CCTV(대체)","20.00",GREEN),("안전벨트 착용률","8.99",TEAL),("음주운전 경험률","8.70",RED),("법규위반 다발지역(대체)","2.31",ORANGE)]
    for i,(name,weight,c) in enumerate(labels):
        x=.75+(i%3)*4.15; y=1.55+(i//3)*1.35
        box(s,x,y,3.7,1.0,WHITE,c); text(s,x+.18,y+.2,2.65,.28,name,11,True,INK); text(s,x+2.85,y+.18,.62,.32,weight,16,True,c,PP_ALIGN.RIGHT)
    box(s,.75,4.55,5.65,1.45,LIGHT,LIGHT)
    text(s,1.0,4.8,5.15,.3,"모델 방식",14,True,NAVY)
    text(s,1.0,5.25,5.15,.48,"자치구별 min-max 정규화 → 위험/안전 방향 변환 → 가중합 100점 → Gradient Boosting",11,False,INK)
    box(s,6.75,4.55,5.85,1.45,RGBColor(255,247,237),RGBColor(253,186,116))
    text(s,7.0,4.8,5.35,.3,"비교 해석",14,True,ORANGE)
    text(s,7.0,5.25,5.35,.52,"2025 전체사고 건수 MAE 415.8 · WMAPE 30.5% · R² 0.016. 94차의 어린이 안전등급과 공통척도가 아니므로 ‘몇 % 향상’ 계산 불가",11,False,INK)
    text(s,.9,6.55,11.6,.3,"정직한 결론: 94차가 목적 적합성과 장기예측 범위는 넓지만, 1차 대비 정확한 향상률은 공통 타깃 재평가 전에는 제시할 수 없음",12,True,NAVY,PP_ALIGN.CENTER)

    # 11. 95 rejection
    s = prs.slides.add_slide(blank); header(s, "왜 95차가 아니라 94차가 최종인가", "개발 성능 상승보다 독립검증 일반화를 우선", 11)
    add_bar_chart(s,.85,1.65,7.1,3.7,["2년 개발","2년 독립","3년 개발","3년 독립"],[
        ("94차", [.78,.68,.76,.68]), ("95차", [.77,.68,.80,.56])
    ],1.0,True)
    box(s,8.35,1.65,4.15,3.7,RGBColor(254,242,242),RGBColor(252,165,165))
    text(s,8.65,1.98,3.55,.35,"95차 실험",17,True,RED,PP_ALIGN.CENTER)
    rich(s,8.72,2.58,3.35,2.2,["도로위험도 7개 추가", "2년형: 개발부터 78→77% 하락", "3년형: 개발 76→80% 상승", "그러나 독립검증 68→56% 하락", "과적합으로 미채택"],12)
    box(s,1.0,5.72,11.3,.72,LIGHT,LIGHT)
    text(s,1.25,5.94,10.8,.3,"채택 원칙: 새 데이터가 개발셋에서 좋아 보여도, 보지 않은 2025년에서 재현되지 않으면 대표모델로 승격하지 않는다.",12,True,NAVY,PP_ALIGN.CENTER)

    # 12. Limits and use
    s = prs.slides.add_slide(blank); header(s, "최종 판단과 사용 시 주의사항", "94차는 현재 최선이지만, 정확도 68%의 의사결정 보조모델", 12)
    blocks=[
        (.75,1.55,"강점",GREEN,["91차와 동일 타깃·검증으로 공정 비교", "3년 정확등급 +20%p", "Raw→Processed→v1 재현 경로 검증", "2년·3년형을 분리해 예측"]),
        (4.65,1.55,"한계",ORANGE,["정확등급 68%로 완벽한 분류 아님", "TOPIS 관측은 24개 자치구", "교통량은 인과효과가 아닌 예측 보조값", "안전등급은 연도별 상대 5분위"]),
        (8.55,1.55,"권장 사용",BLUE,["자치구 간 우선순위 참고", "정확등급과 ±1등급 범위를 함께 제시", "점수·등급 변화 원인을 변수별 검토", "매년 최신 확정 통계로 재학습"]),
    ]
    for x,y,title_,c,lines in blocks:
        box(s,x,y,3.55,3.95,WHITE,c); text(s,x+.2,y+.28,3.15,.3,title_,17,True,c,PP_ALIGN.CENTER); rich(s,x+.3,y+1.0,2.95,2.55,lines,12)
    box(s,1.0,5.95,11.3,.72,NAVY,NAVY)
    text(s,1.25,6.16,10.8,.3,"최종 대표모델: 교통사고 94차 TOPIS 교통량 추가모델 · v1 서비스 산출물 검증 완료",13,True,WHITE,PP_ALIGN.CENTER)

    # 13. Appendix sources
    s = prs.slides.add_slide(blank); header(s, "근거 파일", "발표 내용은 아래 로컬 산출물에서 재확인 가능", 13)
    sources=[
        "models/교통사고/1차/model_info.json · 모델평가_및_2026예측.xlsx",
        "models/교통사고/91차.어린이교통사고수_Persistence모델/comparison_with_v92.json",
        "models/교통사고/94차.11.13.개선/model_info.json",
        "models/교통사고/95차.도로위험도추가모델/model_info.json · 읽어주세요.md",
        "data/processed/교통사고_94차_데이터.xlsx",
        "models/recommender/v1/traffic_accident_v93_v94_pipeline_report.json",
        "scripts/rebuild_traffic_v93_v94_to_v1.py",
    ]
    box(s,.8,1.55,11.7,4.5,LIGHT,LIGHT); rich(s,1.15,1.95,11.0,3.7,sources,13)
    text(s,.85,6.45,11.6,.35,"작성 기준일 2026-11-16 · 수치 반올림으로 합계·개선율에 미세한 차이가 있을 수 있음",10,False,MID,PP_ALIGN.CENTER)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
