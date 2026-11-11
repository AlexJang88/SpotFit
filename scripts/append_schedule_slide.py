"""Append a one-slide schedule in the visual language of the crime-model deck."""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

SOURCE = Path(r"C:\Users\M\Desktop\JUS\SpotFit\SPOT_FIT_범죄모델_개선과정_2년3년_20261111.pptx")
OUTPUT = Path(r"C:\Users\M\Desktop\JUS\SpotFit\SPOT_FIT_범죄모델_개선과정_일정포함_6장_20261111.pptx")

NAVY = RGBColor(0x00, 0x25, 0x35)
DEEP = RGBColor(0x06, 0x39, 0x3A)
TEAL = RGBColor(0x00, 0x86, 0x73)
MINT = RGBColor(0xE5, 0xF5, 0xEF)
PALE = RGBColor(0xF4, 0xF8, 0xF6)
LINE = RGBColor(0xC9, 0xD9, 0xD4)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GRAY = RGBColor(0x55, 0x66, 0x66)
BLACK = RGBColor(0x14, 0x25, 0x26)
FONT = "네모고딕"
FONT_MEDIUM = "네모고딕 Medium"


def box(slide, x, y, w, h, fill, radius=True, line=None):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h),
    )
    shape.fill.solid(); shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line or fill
    return shape


def text(slide, value, x, y, w, h, size, color=BLACK, bold=False,
         align=PP_ALIGN.LEFT, font=FONT, margin=0.02, valign=MSO_ANCHOR.MIDDLE):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.clear(); frame.word_wrap = True
    frame.margin_left = frame.margin_right = Inches(margin)
    frame.margin_top = frame.margin_bottom = Inches(margin)
    frame.vertical_anchor = valign
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    paragraph.space_before = paragraph.space_after = Pt(0)
    run = paragraph.add_run(); run.text = value
    run.font.name = font; run.font.size = Pt(size); run.font.bold = bold
    run.font.color.rgb = color
    return shape


def main() -> None:
    prs = Presentation(SOURCE)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.background.fill; bg.solid(); bg.fore_color.rgb = WHITE

    # Same compact header system used throughout the working deck.
    box(slide, 0.42, 0.34, 0.60, 0.36, MINT)
    text(slide, "08", 0.42, 0.34, 0.60, 0.36, 11, TEAL, True, PP_ALIGN.CENTER, FONT_MEDIUM)
    text(slide, "최종 시연·발표 일정 · 11월 13일 이후", 1.18, 0.26, 12.3, 0.55, 24, DEEP, True)
    text(slide, "모델 개선 마감부터 최종 발표까지 · 담당자별 11/13 업무와 공통 마일스톤", 1.20, 0.80, 12.5, 0.35, 11, GRAY)
    box(slide, 0.42, 1.24, 19.15, 0.04, LINE, False)

    # Key objective strip.
    box(slide, 0.42, 1.48, 19.15, 0.82, NAVY)
    text(slide, "11월 핵심 목표", 0.78, 1.66, 2.2, 0.36, 12, RGBColor(0x42, 0xE3, 0xB4), True, font=FONT_MEDIUM)
    text(slide, "모델 개선  →  최종 시연·리허설  →  프로젝트 수정  →  최종 발표", 3.10, 1.60, 15.6, 0.46, 17, WHITE, True, font=FONT_MEDIUM)

    people = [
        ("장의석", "PM · 어린이 건강 모델", "파이프라인·서비스 구현"),
        ("김홍수", "범죄 모델", "2년·3년 예측모델 최종 개선"),
        ("남궁다인", "아파트 매매·전월세", "가격 예측모델 최종 개선"),
        ("임동현", "교통사고 모델", "교통사고 예측모델 최종 개선"),
    ]
    start_x, card_w, gap = 0.42, 4.55, 0.31
    for index, (name, role, task) in enumerate(people):
        x = start_x + index * (card_w + gap)
        box(slide, x, 2.62, card_w, 3.98, PALE, line=LINE)
        box(slide, x, 2.62, card_w, 0.74, NAVY)
        text(slide, name, x + 0.22, 2.69, 1.42, 0.37, 17, WHITE, True, font=FONT_MEDIUM)
        text(slide, role, x + 1.56, 2.72, card_w - 1.78, 0.30, 9.5, MINT, False, PP_ALIGN.RIGHT)
        text(slide, "11/13", x + 0.22, 3.60, 0.78, 0.36, 11, TEAL, True, font=FONT_MEDIUM)
        text(slide, task, x + 1.12, 3.51, card_w - 1.36, 0.62, 12, BLACK, True)
        box(slide, x + 0.22, 4.32, card_w - 0.44, 0.02, LINE, False)
        text(slide, "공통", x + 0.22, 4.86, 0.78, 0.34, 10.5, TEAL, True, font=FONT_MEDIUM)
        text(slide, "11/16  최종 시연\n11/17  최종 리허설\n11/18–19  프로젝트 수정\n11/20  최종 발표", x + 1.12, 4.77, card_w - 1.36, 1.38, 10.5, GRAY, False, valign=MSO_ANCHOR.TOP)

    text(slide, "※ 오우석: 원본 일정표에 담당 업무가 기재되지 않아 본 담당자 카드에서는 제외", 0.62, 6.78, 10.5, 0.33, 9, GRAY)

    # Bottom milestone rail: only 11/13 and later.
    milestones = [
        ("11/13", "모델·서비스 개선"),
        ("11/16", "최종 시연"),
        ("11/17", "최종 리허설"),
        ("11/18–19", "프로젝트 수정"),
        ("11/20", "최종 발표"),
    ]
    widths = [3.55, 3.15, 3.15, 4.05, 3.55]
    x = 0.42
    for (date, label), width in zip(milestones, widths):
        box(slide, x, 7.34, width, 1.34, NAVY)
        text(slide, date, x + 0.18, 7.58, width - 0.36, 0.28, 10.5, RGBColor(0x42, 0xE3, 0xB4), True, PP_ALIGN.CENTER, FONT_MEDIUM)
        text(slide, label, x + 0.18, 8.00, width - 0.36, 0.38, 12.5, WHITE, True, PP_ALIGN.CENTER, FONT_MEDIUM)
        x += width + 0.18

    text(slide, "SPOT FIT  |  담당자별 실행계획", 0.48, 10.42, 5.2, 0.28, 8.5, GRAY)
    text(slide, "11/13–11/20", 16.65, 10.42, 2.85, 0.28, 8.5, GRAY, False, PP_ALIGN.RIGHT)

    prs.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
