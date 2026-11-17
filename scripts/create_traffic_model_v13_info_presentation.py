"""교통사고 안전등급 13차 모델 설명용 PPT를 생성한다."""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "presentations" / "교통사고_13차_모델정보.pptx"

NAVY = RGBColor(19, 37, 58)
TEAL = RGBColor(27, 142, 134)
BLUE = RGBColor(36, 112, 170)
ORANGE = RGBColor(230, 142, 49)
RED = RGBColor(190, 65, 65)
INK = RGBColor(38, 49, 61)
MID = RGBColor(100, 113, 126)
PALE = RGBColor(239, 244, 247)
WHITE = RGBColor(255, 255, 255)


def add_text(slide, x, y, w, h, value, size=18, bold=False, color=INK,
             align=PP_ALIGN.LEFT, font="맑은 고딕"):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear(); frame.word_wrap = True; frame.margin_left = frame.margin_right = Inches(.03)
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = frame.paragraphs[0]; p.text = value; p.alignment = align
    p.font.name = font; p.font.size = Pt(size); p.font.bold = bold; p.font.color.rgb = color
    return box


def rect(slide, x, y, w, h, fill=WHITE, line=PALE, radius=True):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid(); shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line
    return shape


def header(slide, title, subtitle, number):
    add_text(slide, .65, .28, 10.9, .5, title, 26, True, NAVY)
    add_text(slide, .67, .78, 11.5, .3, subtitle, 11, False, MID)
    add_text(slide, 12.15, .3, .5, .4, str(number), 13, True, TEAL, PP_ALIGN.RIGHT)
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(.65), Inches(1.08), Inches(12), Inches(.025))
    line.fill.solid(); line.fill.fore_color.rgb = TEAL; line.line.fill.background()


def footer(slide):
    add_text(slide, .68, 7.18, 7.5, .16, "안전 · 교통사고 · 13차 기준모델", 8, False, MID)


def metric_card(slide, x, y, title, value, note, color=TEAL, w=2.85):
    rect(slide, x, y, w, 1.35, WHITE, RGBColor(215, 225, 232))
    add_text(slide, x+.18, y+.12, w-.36, .28, title, 11, True, MID)
    add_text(slide, x+.18, y+.42, w-.36, .42, value, 24, True, color)
    add_text(slide, x+.18, y+.91, w-.36, .25, note, 10, False, INK)


def table(slide, x, y, w, h, rows, widths=None, font_size=12):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(h))
    tbl = shape.table
    if widths:
        for col, width in zip(tbl.columns, widths): col.width = Inches(width)
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            cell = tbl.cell(r, c); cell.text = str(value)
            cell.margin_left = cell.margin_right = Inches(.08)
            cell.fill.solid(); cell.fill.fore_color.rgb = NAVY if r == 0 else (WHITE if r % 2 else PALE)
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER if c in (1, 2) else PP_ALIGN.LEFT
                p.font.name = "맑은 고딕"; p.font.size = Pt(font_size)
                p.font.bold = r == 0; p.font.color.rgb = WHITE if r == 0 else INK
    return tbl


def bullet_list(slide, x, y, w, h, items, size=16, color=INK):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.clear(); tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = item; p.level = 0; p.font.name = "맑은 고딕"; p.font.size = Pt(size)
        p.font.color.rgb = color; p.space_after = Pt(11); p.text = "•  " + p.text
    return box


def build():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # 1. 표지
    s = prs.slides.add_slide(blank); s.background.fill.solid(); s.background.fill.fore_color.rgb = NAVY
    add_text(s, .82, 1.08, 11.6, .7, "교통사고 안전등급 13차 모델", 34, True, WHITE)
    add_text(s, .84, 1.9, 10.8, .45, "안전기준표의 기본 지표만 사용한 초기 기준모델", 20, False, RGBColor(190, 220, 225))
    rect(s, .84, 3.05, 11.65, 1.45, RGBColor(27, 52, 76), RGBColor(54, 83, 105))
    add_text(s, 1.13, 3.32, 11.05, .38, "목표", 13, True, RGBColor(149, 217, 211))
    add_text(s, 1.13, 3.73, 11.05, .48, "현재연도 지표로 다음연도의 안전등급(1~5등급)을 예측", 23, True, WHITE)
    add_text(s, .84, 6.55, 10.5, .3, "서울 25개 자치구 · 2017~2025년 · RandomForest", 12, False, RGBColor(190, 220, 225))

    # 2. 한눈에 보기
    s = prs.slides.add_slide(blank); header(s, "13차 모델 한눈에 보기", "새 변수를 붙이기 전 성능을 판단하는 기준점", 2)
    metric_card(s, .7, 1.42, "최종 결과", "안전등급 1~5", "다음연도 등급", TEAL, 3.75)
    metric_card(s, 4.78, 1.42, "입력 변수", "4개", "안전기준표 기본 지표", BLUE, 3.75)
    metric_card(s, 8.86, 1.42, "선정 기법", "RandomForest", "등급 정확도 우선", ORANGE, 3.75)
    rect(s, .7, 3.15, 11.91, 2.82, PALE, PALE)
    add_text(s, 1.0, 3.42, 3.1, .35, "이 모델의 역할", 17, True, NAVY)
    bullet_list(s, 1.0, 3.88, 11.0, 1.65, [
        "안전기준표의 기본 지표만으로 만든 출발점입니다.",
        "이후 모델은 13차에 변수를 하나씩 추가하고, 같은 검증 방식으로 성능 향상을 비교합니다.",
        "예측 점수를 계산한 뒤 최종적으로 1~5 안전등급을 제공합니다.",
    ], 16)
    add_text(s, .9, 6.35, 11.5, .4, "핵심: 13차는 ‘최고 성능 모델’보다 ‘비교 기준모델’이라는 의미가 큽니다.", 15, True, TEAL)
    footer(s)

    # 3. 데이터 흐름
    s = prs.slides.add_slide(blank); header(s, "무엇을 학습하고 무엇을 예측하나?", "현재연도 자료를 넣어 다음연도의 안전등급을 예측", 3)
    labels = [
        ("현재연도 입력", "4개 안전지표"),
        ("안전점수 계산", "영역별 가중치 적용"),
        ("모델 예측", "다음연도 안전점수"),
        ("최종 변환", "안전등급 1~5"),
    ]
    for i, (title, desc) in enumerate(labels):
        x = .55 + i*3.2
        rect(s, x, 1.72, 2.55, 1.35, WHITE, RGBColor(205, 221, 229))
        add_text(s, x+.15, 1.92, 2.25, .34, title, 15, True, [BLUE, TEAL, ORANGE, NAVY][i], PP_ALIGN.CENTER)
        add_text(s, x+.15, 2.39, 2.25, .28, desc, 11, False, INK, PP_ALIGN.CENTER)
        if i < 3: add_text(s, x+2.62, 2.08, .48, .35, "→", 22, True, MID, PP_ALIGN.CENTER)
    rect(s, .72, 3.72, 11.87, 2.15, PALE, PALE)
    add_text(s, 1.0, 3.98, 2.6, .35, "데이터 구분", 17, True, NAVY)
    bullet_list(s, 1.0, 4.42, 11.0, 1.25, [
        "학습: 2017~2023년 입력 → 2018~2024년 결과 (25개 자치구, 175행)",
        "최종 검증: 2024년 입력 → 2025년 결과 (25개 자치구, 25행)",
        "운영 예측: 2025년 입력 → 2026년 안전등급",
    ], 15)
    footer(s)

    # 4. 변수와 가중치
    s = prs.slides.add_slide(blank); header(s, "사용 변수와 실제 가중치", "경감지표가 없으므로 50:10:20:20을 나머지 영역에 비례 재배분", 4)
    rows = [
        ["변수", "영역", "가중치", "값이 커질 때"],
        ["인구만명당 교통사고 사망자수", "위해", "62.50%", "더 위험"],
        ["인구만명당 의료보장 사업장수", "취약", "12.50%", "더 위험"],
        ["운전자 안전벨트 착용률", "의식", "12.70%", "더 안전"],
        ["음주운전 경험률", "의식", "12.30%", "더 위험"],
    ]
    table(s, .43, 1.35, 12.45, 3.85, rows, [4.7, 1.45, 1.7, 4.6], 12)
    rect(s, .7, 5.52, 11.9, .95, RGBColor(250, 244, 234), RGBColor(240, 210, 165))
    add_text(s, .95, 5.7, 11.4, .3, "공식 영역 비중  위해 50% · 취약 10% · 경감 20% · 의식 20%", 14, True, ORANGE)
    add_text(s, .95, 6.08, 11.4, .25, "경감자료 미확보 → 위해 62.5% · 취약 12.5% · 경감 0% · 의식 25%", 13, False, INK)
    add_text(s, .75, 6.65, 11.8, .27, "※ 의식영역 25%는 안전기준표 가중치 8.99 : 8.70 비율로 나누었습니다.", 11, False, MID)
    footer(s)

    # 5. 기법 선택
    s = prs.slides.add_slide(blank); header(s, "왜 RandomForest를 선택했나?", "점수 오차보다 최종 안전등급의 정확한 일치를 우선", 5)
    rows = [
        ["후보 기법", "등급 정확도", "점수 MAE", "R²", "판단"],
        ["RandomForest", "43%", "12.36", "0.022", "선정"],
        ["ExtraTrees", "42%", "11.97", "0.081", "보류"],
        ["Ridge", "36%", "11.93", "0.085", "보류"],
        ["GradientBoosting", "27%", "12.83", "-0.086", "보류"],
    ]
    table(s, .68, 1.38, 11.95, 3.5, rows, [3.15, 2.1, 2.0, 1.75, 2.95], 12)
    rect(s, .72, 5.18, 11.9, 1.08, PALE, PALE)
    add_text(s, 1.0, 5.37, 11.3, .3, "선택 기준", 14, True, NAVY)
    add_text(s, 1.0, 5.74, 11.3, .3, "최종 제공값이 안전등급이므로, 후보 중 등급 정확도가 가장 높은 RandomForest를 선택했습니다.", 15, True, TEAL)
    add_text(s, .8, 6.55, 11.7, .3, "참고: ExtraTrees와 Ridge는 점수 오차는 조금 작지만, 정확한 등급 일치율은 더 낮았습니다.", 11, False, MID)
    footer(s)

    # 6. 성능
    s = prs.slides.add_slide(blank); header(s, "13차 모델의 실제 성능", "여러 연도를 반복 검증한 결과와 2025년 최종 검증을 함께 확인", 6)
    add_text(s, .72, 1.31, 5.8, .35, "내부 시계열 교차검증 (2021~2024)", 16, True, NAVY)
    metric_card(s, .72, 1.77, "정확한 등급 일치", "43%", "100건 중 약 43건", TEAL, 2.75)
    metric_card(s, 3.72, 1.77, "±1등급 이내", "84%", "한 등급 차이까지", BLUE, 2.75)
    metric_card(s, 6.72, 1.77, "점수 MAE", "12.36", "평균 절대 오차", ORANGE, 2.75)
    metric_card(s, 9.72, 1.77, "R²", "0.022", "설명력은 낮음", RED, 2.75)
    add_text(s, .72, 3.55, 5.8, .35, "2025년 최종 검증", 16, True, NAVY)
    metric_card(s, .72, 4.02, "정확한 등급 일치", "48%", "25개 구 중 12개", TEAL, 2.75)
    metric_card(s, 3.72, 4.02, "±1등급 이내", "76%", "25개 구 중 19개", BLUE, 2.75)
    metric_card(s, 6.72, 4.02, "점수 MAE", "14.63", "평균 절대 오차", ORANGE, 2.75)
    metric_card(s, 9.72, 4.02, "R²", "-0.119", "평균 예측보다 불리", RED, 2.75)
    add_text(s, .82, 6.05, 11.6, .55, "해석: 등급을 한 단계 안쪽으로 맞히는 비율은 높지만, 안전점수 자체의 예측력은 아직 약합니다.", 15, True, INK)
    footer(s)

    # 7. 주의사항
    s = prs.slides.add_slide(blank); header(s, "어떻게 해석해야 하나?", "13차는 안전기준표 기반의 정직한 출발점이며 완성형 모델은 아님", 7)
    rect(s, .72, 1.42, 5.78, 4.65, PALE, PALE)
    add_text(s, 1.0, 1.72, 5.2, .35, "장점", 18, True, TEAL)
    bullet_list(s, 1.0, 2.2, 5.05, 3.25, [
        "안전기준표 이외의 데이터를 섞지 않았습니다.",
        "영역 가중치 규칙을 명확하게 지켰습니다.",
        "새 변수를 추가했을 때 좋아졌는지 판단할 기준이 됩니다.",
    ], 16)
    rect(s, 6.82, 1.42, 5.78, 4.65, RGBColor(252, 244, 244), RGBColor(239, 211, 211))
    add_text(s, 7.1, 1.72, 5.2, .35, "한계와 주의", 18, True, RED)
    bullet_list(s, 7.1, 2.2, 5.05, 3.4, [
        "경감지표가 없어 공식 전체 지수와 동일한 등급은 아닙니다.",
        "2025년 의료보장 사업장수는 2024년 값을 유지해 사용했습니다.",
        "R²가 낮아 개별 점수는 보수적으로 해석해야 합니다.",
    ], 16)
    add_text(s, .8, 6.42, 11.7, .42, "결론: 13차는 이후 모델이 실제로 개선됐는지 비교하기 위한 핵심 기준모델입니다.", 16, True, NAVY, PP_ALIGN.CENTER)
    footer(s)

    prs.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())
