from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


OUT = Path.home() / "Desktop" / "범죄율_3년예측모델_핵심요약_2장.pptx"
FONT = "맑은 고딕"
NAVY = RGBColor(14, 32, 52)
NAVY2 = RGBColor(25, 53, 79)
TEAL = RGBColor(29, 173, 157)
MINT = RGBColor(222, 246, 241)
BLUE = RGBColor(224, 237, 249)
WHITE = RGBColor(255, 255, 255)
INK = RGBColor(35, 48, 61)
GRAY = RGBColor(102, 117, 132)
LIGHT = RGBColor(246, 249, 252)
ORANGE = RGBColor(244, 161, 72)
RED = RGBColor(207, 76, 76)


def shape(slide, x, y, w, h, fill, rounded=True, line=None):
    item = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h),
    )
    item.fill.solid(); item.fill.fore_color.rgb = fill
    if line:
        item.line.color.rgb = line
    else:
        item.line.fill.background()
    return item


def txt(slide, x, y, w, h, value, size=16, color=INK, bold=False,
        align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP, margin=0.04):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.clear()
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(margin)
    tf.vertical_anchor = valign
    p = tf.paragraphs[0]; p.alignment = align
    run = p.add_run(); run.text = value
    run.font.name = FONT; run.font.size = Pt(size); run.font.bold = bold; run.font.color.rgb = color
    return box


def rich(slide, x, y, w, h, lines, fill=None, rounded=True):
    if fill:
        shape(slide, x, y, w, h, fill, rounded)
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.clear(); tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.16)
    tf.margin_top = tf.margin_bottom = Inches(0.10)
    for idx, (value, size, color, bold, space) in enumerate(lines):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.space_after = Pt(space); p.alignment = PP_ALIGN.LEFT
        r = p.add_run(); r.text = value
        r.font.name = FONT; r.font.size = Pt(size); r.font.color.rgb = color; r.font.bold = bold
    return box


def header(slide, kicker, title, page):
    txt(slide, 0.72, 0.35, 5.5, 0.3, kicker, 10, TEAL, True)
    txt(slide, 0.72, 0.70, 11.5, 0.55, title, 25, NAVY, True)
    txt(slide, 12.0, 0.42, 0.58, 0.3, f"0{page}", 11, TEAL, True, PP_ALIGN.RIGHT)
    shape(slide, 0.72, 1.27, 0.95, 0.055, TEAL, False)


def footer(slide):
    txt(slide, 0.72, 7.18, 7.5, 0.16, "SpotFit · 범죄율 3년 예측모델 핵심 요약", 8, GRAY)
    txt(slide, 11.3, 7.18, 1.3, 0.16, "2026.11.05", 8, GRAY, align=PP_ALIGN.RIGHT)


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # Slide 1: executive summary + performance progression
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "MODEL STORY · EXECUTIVE SUMMARY", "3년 뒤 범죄율, 작은 표본에서도 더 안정적으로", 1)

    # Left definition panel
    shape(s, 0.72, 1.60, 3.05, 4.88, NAVY, True)
    txt(s, 0.98, 1.90, 2.5, 0.3, "예측 문제", 12, TEAL, True)
    txt(s, 0.98, 2.37, 2.5, 1.02, "기준연도 자료로\n3년 후 자치구별\n범죄율 예측", 22, WHITE, True)
    shape(s, 0.98, 3.62, 2.52, 0.02, RGBColor(64, 89, 112), False)
    txt(s, 0.98, 3.90, 2.5, 0.70, "범죄율 = 5대범죄 발생건수\n÷ 등록인구 × 10,000", 12, RGBColor(210, 223, 235), True)
    txt(s, 0.98, 4.85, 2.5, 0.3, "검증 원칙", 12, TEAL, True)
    txt(s, 0.98, 5.27, 2.5, 0.88, "2022~2024 확대창 순차검증\n2025년 25개 구 최종 확인\n512개 조합 비교", 11, WHITE)

    # Hero metric
    shape(s, 4.08, 1.60, 8.55, 1.16, MINT, True)
    txt(s, 4.37, 1.82, 3.4, 0.32, "최종 CV RMSE", 12, NAVY, True)
    txt(s, 7.15, 1.72, 2.15, 0.55, "12.9757", 27, NAVY, True, PP_ALIGN.CENTER)
    txt(s, 9.55, 1.76, 2.72, 0.45, "Persistence 대비 4.60% ↓", 15, TEAL, True, PP_ALIGN.CENTER)
    txt(s, 4.37, 2.28, 6.5, 0.24, "기준선 13.6013  →  최종 Ridge 12.9757", 10, GRAY)

    # Three-step progression
    steps = [
        ("1", "첫 개선", "Ridge α=1", "13.5013", "현재 범죄율 + 경찰관서·CCTV\n+스트레스 인지율", MINT),
        ("2", "Feature 확장", "ElasticNet", "13.2223", "인구·세대·남성비율\n전년 변화율 추가", BLUE),
        ("3", "알고리즘 개선", "Ridge α=1", "12.9757", "동일 11개 변수에서\n작은 표본 안정성 확보", MINT),
    ]
    for i, (num, stage, alg, rmse, desc, fill) in enumerate(steps):
        x = 4.08 + i * 2.87
        shape(s, x, 3.08, 2.58, 2.72, fill, True)
        shape(s, x + 0.18, 3.28, 0.42, 0.42, TEAL, True)
        txt(s, x + 0.18, 3.31, 0.42, 0.30, num, 11, WHITE, True, PP_ALIGN.CENTER)
        txt(s, x + 0.72, 3.30, 1.62, 0.32, stage, 13, NAVY, True)
        txt(s, x + 0.18, 3.92, 2.22, 0.28, alg, 12, GRAY, True)
        txt(s, x + 0.18, 4.33, 2.22, 0.48, rmse, 24, NAVY, True)
        txt(s, x + 0.18, 5.03, 2.22, 0.55, desc, 10, INK)
        if i < 2:
            txt(s, x + 2.60, 4.17, 0.26, 0.45, "→", 19, TEAL, True, PP_ALIGN.CENTER)

    shape(s, 4.08, 6.04, 8.55, 0.45, NAVY2, True)
    txt(s, 4.28, 6.12, 8.15, 0.23, "결론  |  변수 확장만큼 알고리즘 선택이 중요했고, 최종 Ridge가 가장 안정적", 11, WHITE, True, PP_ALIGN.CENTER)
    footer(s)

    # Slide 2: current model + roadmap
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "MODEL DESIGN · NEXT MOVE", "현재 모델은 간결하게, 다음 개선은 선행성과 유형 분리로", 2)

    # Current model panel
    shape(s, 0.72, 1.58, 4.0, 4.95, NAVY, True)
    txt(s, 0.98, 1.84, 2.9, 0.3, "FINAL MODEL", 10, TEAL, True)
    txt(s, 0.98, 2.22, 2.8, 0.48, "Ridge Regression", 23, WHITE, True)
    txt(s, 3.82, 2.25, 0.62, 0.35, "α=1", 12, TEAL, True, PP_ALIGN.CENTER)
    txt(s, 0.98, 2.93, 2.2, 0.28, "선택된 11개 변수", 12, RGBColor(210, 223, 235), True)
    vars_text = (
        "현재 범죄율\n"
        "경찰관서 1만 명당\n"
        "CCTV 1만 명당 · 전년변화율\n"
        "스트레스 인지율\n"
        "인구수 · 세대수 · 세대당 인구\n"
        "남성비율\n"
        "인구수 · 세대수 전년변화율"
    )
    txt(s, 0.98, 3.35, 3.35, 1.92, vars_text, 11, WHITE)
    shape(s, 0.98, 5.55, 3.46, 0.62, TEAL, True)
    txt(s, 1.10, 5.69, 3.22, 0.3, "채택 기준  CV RMSE 최소", 12, WHITE, True, PP_ALIGN.CENTER)

    # Roadmap 1
    shape(s, 5.02, 1.58, 3.67, 4.95, MINT, True)
    txt(s, 5.28, 1.84, 0.40, 0.30, "01", 11, TEAL, True)
    txt(s, 5.28, 2.22, 3.05, 0.46, "실제 선행변수 추가", 20, NAVY, True)
    txt(s, 5.28, 2.83, 3.05, 0.54, "예측 시점에 이미 확정되어 있고\n범죄 변화보다 먼저 움직이는 정보", 11, GRAY)
    items1 = [
        ("신고·치안", "112 신고·민원, 순찰·인력 계획"),
        ("방범환경", "CCTV·보안등 계획, 방범예산"),
        ("생활·상권", "야간 유동인구, 유흥업소 변화"),
        ("주거·사회", "빈집, 재개발 이주, 입주·실업"),
    ]
    for i, (head, body) in enumerate(items1):
        y = 3.64 + i * 0.58
        txt(s, 5.28, y, 0.92, 0.25, head, 10, TEAL, True)
        txt(s, 6.20, y, 2.15, 0.33, body, 10, INK)
    shape(s, 5.28, 5.98, 3.14, 0.31, NAVY2, True)
    txt(s, 5.39, 6.03, 2.92, 0.18, "원칙  실행일 기준 공개 자료만", 9, WHITE, True, PP_ALIGN.CENTER)

    # Roadmap 2
    shape(s, 8.98, 1.58, 3.65, 4.95, BLUE, True)
    txt(s, 9.24, 1.84, 0.40, 0.30, "02", 11, TEAL, True)
    txt(s, 9.24, 2.22, 3.02, 0.46, "범죄유형별 모델 분리", 20, NAVY, True)
    txt(s, 9.24, 2.83, 3.02, 0.54, "5대 범죄 합계 대신 유형별\n영향요인과 변동성을 따로 학습", 11, GRAY)
    items2 = [
        ("절도", "상권·유동인구·지하철·점포"),
        ("폭력", "야간 생활인구·유흥·음주상권"),
        ("성범죄", "야간 유동·신고환경·방범시설"),
        ("살인·강도", "희소 건수 → Poisson·음이항"),
    ]
    for i, (head, body) in enumerate(items2):
        y = 3.64 + i * 0.58
        txt(s, 9.24, y, 0.82, 0.25, head, 10, TEAL, True)
        txt(s, 10.06, y, 2.25, 0.33, body, 10, INK)
    shape(s, 9.24, 5.98, 3.12, 0.31, NAVY2, True)
    txt(s, 9.35, 6.03, 2.90, 0.18, "최종 범죄율 = 유형별 예측률 합", 9, WHITE, True, PP_ALIGN.CENTER)

    shape(s, 0.72, 6.70, 11.91, 0.32, RGBColor(255, 242, 222), True)
    txt(s, 0.90, 6.75, 11.55, 0.18,
        "채택 조건  |  동일 순차검증에서 기존 Ridge보다 RMSE 개선 + 3개 검증연도 중 2개 이상 개선",
        9, RED, True, PP_ALIGN.CENTER)
    footer(s)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
