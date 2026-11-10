from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "presentations" / "아동_물리환경_가중치_점수환산_1장.pptx"

NAVY = RGBColor(27, 52, 67)
GREEN = RGBColor(31, 111, 82)
MINT = RGBColor(223, 240, 232)
GOLD = RGBColor(226, 173, 61)
PALE = RGBColor(246, 248, 245)
GRAY = RGBColor(99, 116, 110)
LINE = RGBColor(218, 226, 221)
WHITE = RGBColor(255, 255, 255)


def box(slide, x, y, w, h, fill=WHITE, line=LINE, radius=True):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h),
    )
    shape.fill.solid(); shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line
    return shape


def text(slide, value, x, y, w, h, size=14, color=NAVY, bold=False,
         align=PP_ALIGN.LEFT, font="Noto Sans KR", valign=MSO_ANCHOR.MIDDLE):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.clear(); tf.word_wrap = True; tf.vertical_anchor = valign
    p = tf.paragraphs[0]; p.alignment = align
    run = p.add_run(); run.text = value
    run.font.name = font; run.font.size = Pt(size); run.font.bold = bold; run.font.color.rgb = color
    return tb


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid(); slide.background.fill.fore_color.rgb = RGBColor(250, 251, 249)

    text(slide, "아동 물리환경 점수: 8개 지표 가중치와 100점 환산", .55, .28, 10.7, .55, 25, NAVY, True)
    text(slide, "서울연구원 AHP 방법을 따르되, 2015–2024 공통기간이 확보된 지표만 재정규화", .57, .82, 11.2, .32, 11, GRAY)
    tag = box(slide, 11.45, .34, 1.28, .48, GREEN, GREEN)
    text(slide, "발표용 1장", 11.45, .34, 1.28, .48, 10, WHITE, True, PP_ALIGN.CENTER)

    # Left: scoring method
    box(slide, .5, 1.35, 3.0, 5.42, WHITE, LINE)
    text(slide, "① 원자료 → 지표값", .75, 1.57, 2.45, .35, 15, GREEN, True)
    text(slide, "시설 수·면적을 대상 인구로 표준화", .75, 1.94, 2.45, .34, 11, NAVY, True)
    text(slide, "예: 3~5세 1,000명당 유치원 수\n인구 10만명당 종합병원급 수", .75, 2.28, 2.35, .67, 10, GRAY)
    text(slide, "↓", 1.78, 2.94, .35, .32, 18, GREEN, True, PP_ALIGN.CENTER)
    text(slide, "② 연도별 25개 구 상대등급", .75, 3.28, 2.45, .35, 15, GREEN, True)
    text(slide, "정규분포: 평균 ± 모집단표준편차\n비정규분포: Jenks Natural Breaks", .75, 3.67, 2.4, .68, 10, GRAY)
    levels = [("1등급", "100점", GREEN), ("2등급", "75점", RGBColor(93, 166, 124)),
              ("3등급", "50점", GOLD), ("4등급", "25점", RGBColor(213, 105, 74))]
    for i, (label, score, color) in enumerate(levels):
        y = 4.48 + i * .42
        box(slide, .76, y, 1.02, .31, color, color)
        text(slide, label, .76, y, 1.02, .31, 9, WHITE, True, PP_ALIGN.CENTER)
        text(slide, score, 1.9, y, .75, .31, 10, NAVY, True)
    text(slide, "논문 원점수 0.25·0.50·0.75·1.00을\n현재 화면에서는 ×100하여 표시", .75, 6.2, 2.4, .43, 9, GRAY)

    # Center: weights
    box(slide, 3.72, 1.35, 5.72, 5.42, WHITE, LINE)
    text(slide, "③ 현재 사용 중인 최종 가중치", 3.98, 1.56, 4.7, .35, 15, GREEN, True)
    rows = [
        ("교육", "3~5세 1,000명당 유치원 수", 47.39, GREEN),
        ("공원", "1인당 공원면적", 7.40, RGBColor(82, 154, 105)),
        ("공원", "0~5세 1,000명당 어린이공원 면적", 7.40, RGBColor(82, 154, 105)),
        ("문화", "0~5세 10,000명당 지역문화복지시설", 13.96, RGBColor(190, 143, 47)),
        ("의료", "1인당 보건소 인력", 5.96, RGBColor(71, 126, 167)),
        ("의료", "인구 10만명당 종합병원급", 5.96, RGBColor(71, 126, 167)),
        ("의료", "인구 1,000명당 병원·의원", 5.96, RGBColor(71, 126, 167)),
        ("의료", "0~5세 1,000명당 소아청소년과", 5.96, RGBColor(71, 126, 167)),
    ]
    max_w = 47.39
    for i, (category, label, weight, color) in enumerate(rows):
        y = 2.02 + i * .49
        box(slide, 3.98, y, .55, .28, color, color)
        text(slide, category, 3.98, y, .55, .28, 8, WHITE, True, PP_ALIGN.CENTER)
        text(slide, label, 4.65, y - .01, 2.63, .31, 9.2, NAVY)
        bar_w = 1.35 * weight / max_w
        box(slide, 7.35, y + .05, bar_w, .16, color, color, False)
        text(slide, f"{weight:.2f}%", 8.72, y - .01, .5, .3, 9, NAVY, True, PP_ALIGN.RIGHT)
    text(slide, "합계", 7.36, 6.1, .6, .3, 10, GRAY, True)
    text(slide, "100.00%", 8.35, 6.1, .87, .3, 11, GREEN, True, PP_ALIGN.RIGHT)
    text(slide, "※ 공개 대분류 가중치 중 확보 지표 몫(0.709)을 1.0으로 재정규화", 3.98, 6.42, 5.1, .24, 8.5, GRAY)

    # Right: formula and interpretation
    box(slide, 9.66, 1.35, 3.17, 5.42, NAVY, NAVY)
    text(slide, "최종 환산식", 9.94, 1.62, 2.55, .35, 16, WHITE, True)
    text(slide, "지표별 등급점수", 9.94, 2.1, 2.5, .26, 10, RGBColor(190, 215, 204), True)
    text(slide, "sᵢ ∈ {25, 50, 75, 100}", 9.94, 2.37, 2.52, .42, 17, WHITE, True, PP_ALIGN.CENTER)
    text(slide, "×", 11.0, 2.82, .42, .3, 18, GOLD, True, PP_ALIGN.CENTER)
    text(slide, "최종 가중치 wᵢ", 9.94, 3.12, 2.5, .26, 10, RGBColor(190, 215, 204), True)
    text(slide, "Σ wᵢ = 1", 9.94, 3.39, 2.52, .4, 17, WHITE, True, PP_ALIGN.CENTER)
    box(slide, 10.0, 3.98, 2.47, .03, GOLD, GOLD, False)
    text(slide, "육아환경 점수(100점)", 9.94, 4.19, 2.5, .3, 11, RGBColor(190, 215, 204), True, PP_ALIGN.CENTER)
    text(slide, "S = Σ(wᵢ × sᵢ)", 9.89, 4.53, 2.62, .52, 20, WHITE, True, PP_ALIGN.CENTER)
    box(slide, 9.94, 5.28, 2.58, .84, RGBColor(44, 75, 89), RGBColor(75, 105, 116))
    text(slide, "해석 범위 25~100점\n높을수록 서울 내 상대적 인프라 수준이 높음", 10.08, 5.4, 2.3, .57, 9.5, WHITE, False, PP_ALIGN.CENTER)
    text(slide, "주의  병원 3개 지표는 설립일부터 계속 운영 가정", 9.94, 6.34, 2.5, .24, 8, RGBColor(225, 198, 126))

    source = "근거: 이주림·구자훈(2015), 『AHP를 이용한 서울시 자치구의 물리적 육아환경 평가』, 서울도시연구 16(1), pp.137–147 · 현재값: data/processed/child_environment_scores/weights.csv"
    text(slide, source, .57, 6.94, 12.1, .26, 7.6, GRAY)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
