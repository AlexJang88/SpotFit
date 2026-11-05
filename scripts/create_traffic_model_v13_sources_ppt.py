from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


OUT = Path.home() / "Desktop" / "SpotFit_교통사고_13차모델_데이터와출처_20261105.pptx"
NAVY = RGBColor(19, 42, 66)
TEAL = RGBColor(28, 143, 134)
MINT = RGBColor(215, 241, 237)
BLUE = RGBColor(219, 232, 246)
WHITE = RGBColor(255, 255, 255)
DARK = RGBColor(39, 50, 61)
GRAY = RGBColor(102, 115, 128)
LIGHT = RGBColor(245, 248, 251)
RED = RGBColor(192, 67, 67)
FONT = "맑은 고딕"


def rect(slide, x, y, w, h, fill, radius=False, line=None):
    from pptx.enum.shapes import MSO_SHAPE

    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line:
        shape.line.color.rgb = line
    else:
        shape.line.fill.background()
    return shape


def text(slide, x, y, w, h, value, size=18, color=DARK, bold=False,
         align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP, margin=0.06):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.margin_left = Inches(margin)
    frame.margin_right = Inches(margin)
    frame.margin_top = Inches(margin)
    frame.margin_bottom = Inches(margin)
    frame.vertical_anchor = valign
    p = frame.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = value
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return box


def title(slide, heading, number):
    text(slide, 0.65, 0.38, 11.5, 0.55, heading, 25, NAVY, True)
    text(slide, 12.05, 0.40, 0.55, 0.4, f"{number:02d}", 13, TEAL, True, PP_ALIGN.RIGHT)
    rect(slide, 0.67, 1.05, 1.0, 0.06, TEAL)


def footer(slide, label="SpotFit · 안전-교통사고-13차"):
    text(slide, 0.67, 7.18, 8.4, 0.18, label, 8, GRAY)
    text(slide, 11.4, 7.18, 1.2, 0.18, "2026.11.05", 8, GRAY, align=PP_ALIGN.RIGHT)


def table(slide, x, y, w, h, headers, rows, widths=None, font_size=12):
    tbl = slide.shapes.add_table(len(rows) + 1, len(headers), Inches(x), Inches(y), Inches(w), Inches(h)).table
    if widths:
        for i, width in enumerate(widths):
            tbl.columns[i].width = Inches(width)
    for c, header in enumerate(headers):
        cell = tbl.cell(0, c)
        cell.text = header
        cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
        for p in cell.text_frame.paragraphs:
            p.alignment = PP_ALIGN.CENTER
            for r in p.runs:
                r.font.name = FONT; r.font.size = Pt(font_size); r.font.bold = True; r.font.color.rgb = WHITE
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    for r, row in enumerate(rows, 1):
        for c, value in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = str(value)
            cell.fill.solid(); cell.fill.fore_color.rgb = LIGHT if r % 2 else BLUE
            cell.margin_left = Inches(0.06); cell.margin_right = Inches(0.06)
            cell.margin_top = Inches(0.04); cell.margin_bottom = Inches(0.04)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.LEFT if c else PP_ALIGN.CENTER
                for run in p.runs:
                    run.font.name = FONT; run.font.size = Pt(font_size); run.font.color.rgb = DARK
    return tbl


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # 1. Cover
    s = prs.slides.add_slide(blank)
    rect(s, 0, 0, 13.333, 7.5, NAVY)
    rect(s, 0.72, 1.08, 0.12, 4.85, TEAL)
    text(s, 1.15, 1.30, 10.8, 0.42, "SPOTFIT · DATA TRACE", 12, TEAL, True)
    text(s, 1.15, 2.05, 10.7, 1.15, "교통사고 13차 모델\n데이터와 출처", 32, WHITE, True)
    text(s, 1.17, 3.55, 9.8, 0.65, "실제 학습 코드·모델정보·학습 CSV를 대조한 입력자료 기준", 16, RGBColor(203, 216, 228))
    rect(s, 1.15, 4.55, 3.1, 0.72, TEAL, True)
    text(s, 1.30, 4.67, 2.8, 0.35, "입력변수 4개 · 서울 25개 자치구", 13, WHITE, True, PP_ALIGN.CENTER)
    text(s, 1.16, 6.75, 6.4, 0.3, "기준일 2026.11.05", 10, RGBColor(163, 184, 201))

    # 2. Overview
    s = prs.slides.add_slide(blank); title(s, "13차 모델에 실제로 들어간 데이터", 2)
    cards = [
        ("01", "교통사고 사망위험", "인구 1만 명당\n교통사고 사망자수", "위해 · 62.5%", MINT),
        ("02", "의료보장 사업장", "인구 1만 명당\n의료보장 사업장수", "취약 · 12.5%", BLUE),
        ("03", "안전벨트 착용", "운전자안전벨트\n착용률", "의식 · 12.70%", MINT),
        ("04", "음주운전 경험", "음주운전\n경험률", "의식 · 12.30%", BLUE),
    ]
    for i, (num, label, metric, weight, fill) in enumerate(cards):
        x = 0.68 + i * 3.13
        rect(s, x, 1.45, 2.8, 3.55, fill, True)
        text(s, x + 0.18, 1.66, 0.45, 0.35, num, 12, TEAL, True)
        text(s, x + 0.18, 2.15, 2.42, 0.4, label, 15, NAVY, True)
        text(s, x + 0.18, 2.85, 2.42, 0.95, metric, 20, DARK, True, valign=MSO_ANCHOR.MIDDLE)
        text(s, x + 0.18, 4.35, 2.42, 0.35, weight, 12, GRAY, True)
    rect(s, 0.68, 5.40, 12.0, 1.15, NAVY, True)
    text(s, 0.95, 5.66, 11.45, 0.62,
         "예측목표  |  다음연도 안전점수 → 자치구 상대순위로 안전등급 1~5 산출\n선정 알고리즘  |  RandomForest",
         15, WHITE, True, valign=MSO_ANCHOR.MIDDLE)
    footer(s)

    # 3. Sources table
    s = prs.slides.add_slide(blank); title(s, "입력변수별 원자료와 공식 출처", 3)
    rows = [
        ["1", "인구 1만 명당 교통사고 사망자수", "TAAS 사망자수 + 서울시 주민등록인구", "한국도로교통공단 TAAS\n서울 열린데이터광장", "2017~2025"],
        ["2", "인구 1만 명당 의료보장 사업장수", "건강보험 사업장수 + 서울시 주민등록인구", "KOSIS·국민건강보험공단\n서울 열린데이터광장", "2017~2024\n(2025 이월)"],
        ["3", "운전자안전벨트 착용률", "자치구별 지역사회건강조사 원자료", "질병관리청\n지역사회건강조사", "2017~2025"],
        ["4", "음주운전 경험률", "자치구별 지역사회건강조사 음주운전율", "질병관리청\n지역사회건강조사", "2017~2025"],
    ]
    table(s, 0.65, 1.42, 12.05, 3.95,
          ["번호", "모델 입력변수", "산출에 사용한 데이터", "공식 출처", "모델 사용기간"], rows,
          [0.55, 2.85, 3.35, 3.05, 2.25], 12)
    rect(s, 0.68, 5.70, 12.0, 0.86, MINT, True)
    text(s, 0.92, 5.90, 11.5, 0.45,
         "핵심: 1·2번은 원자료를 주민등록인구로 나누어 1만 명당 지표로 만든 파생변수입니다.",
         14, NAVY, True, valign=MSO_ANCHOR.MIDDLE)
    footer(s)

    # 4. Lineage and split
    s = prs.slides.add_slide(blank); title(s, "데이터가 13차 모델로 들어간 과정", 4)
    steps = [
        ("공식 원자료", "TAAS · KOSIS/NHIS\n서울시 인구 · 지역사회건강조사"),
        ("자치구·연도 패널", "서울 25개 자치구\n2017~2025"),
        ("4개 안전지표", "위해 1 · 취약 1 · 의식 2\n경감지표는 제외"),
        ("13차 모델", "RandomForest\n다음연도 점수·등급"),
    ]
    for i, (head, body) in enumerate(steps):
        x = 0.72 + i * 3.15
        rect(s, x, 1.55, 2.55, 1.65, MINT if i % 2 == 0 else BLUE, True)
        text(s, x + 0.15, 1.78, 2.25, 0.36, head, 16, NAVY, True, PP_ALIGN.CENTER)
        text(s, x + 0.15, 2.28, 2.25, 0.65, body, 12, DARK, False, PP_ALIGN.CENTER, MSO_ANCHOR.MIDDLE)
        if i < 3:
            text(s, x + 2.60, 2.08, 0.42, 0.5, "→", 24, TEAL, True, PP_ALIGN.CENTER)
    table(s, 0.88, 4.05, 11.55, 1.72,
          ["구분", "입력연도 → 예측연도", "행 수", "역할"],
          [
              ["학습", "2017~2023 → 2018~2024", "175", "모델 학습"],
              ["최종 검증", "2024 → 2025", "25", "서울 25개 자치구 독립 검증"],
              ["미래 예측", "2025 → 2026", "25", "2026 안전점수·등급 산출"],
          ], [1.7, 3.5, 1.2, 5.15], 12)
    text(s, 0.90, 6.13, 11.4, 0.38, "학습 데이터 파일: data/processed/안전/교통사고/13차/교통사고_안전등급_13차_초기_모델데이터.csv", 10, GRAY)
    footer(s)

    # 5. Exact files and URLs
    s = prs.slides.add_slide(blank); title(s, "근거 파일과 출처 링크", 5)
    left = [
        "TAAS 사고자료\ndata/raw/안전/교통사고/TAAS_서울_고속국도제외_*.xlsx",
        "주민등록인구\ndata/interim/안전/교통사고/자치구별_주민등록인구_2017_2025.csv",
        "의료보장 사업장\ndata/raw/안전/교통사고/KOSIS_시군구별_적용인구현황_2017_2024_한글정상.csv",
        "운전자 안전\ndata/interim/안전/교통사고/자치구별_운전자안전_2차.csv",
    ]
    right = [
        "TAAS\nhttps://taas.koroad.or.kr/",
        "서울 열린데이터광장\nhttps://data.seoul.go.kr/bsp/wgs/gis/gis03.do",
        "KOSIS\nhttps://kosis.kr/  ·  표 ID TX_35001_A008",
        "지역사회건강조사\nhttps://chs.kdca.go.kr/",
    ]
    for i in range(4):
        y = 1.42 + i * 1.25
        rect(s, 0.72, y, 6.0, 0.98, LIGHT, True)
        rect(s, 6.90, y, 5.72, 0.98, MINT if i % 2 == 0 else BLUE, True)
        text(s, 0.92, y + 0.12, 5.62, 0.72, left[i], 11, DARK, i == 0, valign=MSO_ANCHOR.MIDDLE)
        text(s, 7.10, y + 0.12, 5.32, 0.72, right[i], 11, NAVY, True, valign=MSO_ANCHOR.MIDDLE)
    rect(s, 0.72, 6.43, 11.9, 0.48, RGBColor(255, 239, 218), True)
    text(s, 0.90, 6.50, 11.55, 0.28, "주의: 2025년 의료보장 사업장수는 2024년 최근 공표값을 이월했습니다. 13차 등급은 경감지표를 제외한 환산 등급이며 공식 지역안전지수 등급이 아닙니다.", 9, RED, True)
    footer(s)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
