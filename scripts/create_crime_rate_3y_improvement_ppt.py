from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "presentations" / "범죄율_3년예측모델_초기부터_성능개선과정_향후개선포함.pptx"

NAVY = RGBColor(24, 47, 63)
GREEN = RGBColor(28, 117, 86)
BLUE = RGBColor(44, 109, 170)
GOLD = RGBColor(230, 174, 57)
ORANGE = RGBColor(219, 116, 58)
WHITE = RGBColor(255, 255, 255)
BG = RGBColor(248, 250, 249)
GRAY = RGBColor(91, 107, 103)
LINE = RGBColor(214, 224, 219)
MINT = RGBColor(225, 242, 234)
PALE_BLUE = RGBColor(231, 240, 249)
PALE_GOLD = RGBColor(250, 243, 222)
FONT = "맑은 고딕"


def rect(slide, x, y, w, h, fill=WHITE, line=LINE, rounded=True):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h),
    )
    shape.fill.solid(); shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line
    return shape


def text(slide, value, x, y, w, h, size=13, color=NAVY, bold=False,
         align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.MIDDLE):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame; frame.clear(); frame.word_wrap = True; frame.vertical_anchor = valign
    paragraph = frame.paragraphs[0]; paragraph.alignment = align
    run = paragraph.add_run(); run.text = value
    run.font.name = FONT; run.font.size = Pt(size); run.font.bold = bold; run.font.color.rgb = color
    return box


def header(slide, title, subtitle, number):
    slide.background.fill.solid(); slide.background.fill.fore_color.rgb = BG
    text(slide, title, .62, .30, 10.7, .52, 25, NAVY, True)
    text(slide, subtitle, .64, .84, 11.1, .30, 10.5, GRAY)
    rect(slide, 11.75, .34, .90, .46, GREEN, GREEN)
    text(slide, f"0{number}", 11.75, .34, .90, .46, 11, WHITE, True, PP_ALIGN.CENTER)


def metric(slide, x, y, label, value, note, color=GREEN, width=2.25):
    rect(slide, x, y, width, 1.12, WHITE, LINE)
    rect(slide, x, y, .07, 1.12, color, color, False)
    text(slide, label, x + .18, y + .10, width - .32, .22, 9.5, GRAY, True)
    text(slide, value, x + .18, y + .32, width - .32, .39, 19, color, True)
    text(slide, note, x + .18, y + .76, width - .32, .21, 8, GRAY)


def slide_overview(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header(slide, "범죄율 3년 예측모델: 초기부터 최종 개선까지",
           "서울 25개 자치구 · 목표값: 3년 후 인구 1만 명당 5대 범죄 발생건수", 1)
    rect(slide, .72, 1.45, 3.45, 4.90, NAVY, NAVY)
    text(slide, "예측 문제", 1.02, 1.77, 2.75, .34, 17, GOLD, True)
    text(slide, "기준연도 자료", 1.02, 2.34, 2.75, .28, 11, WHITE, True, PP_ALIGN.CENTER)
    text(slide, "↓ 3년 후", 1.02, 2.78, 2.75, .32, 16, GOLD, True, PP_ALIGN.CENTER)
    text(slide, "자치구별 범죄율", 1.02, 3.23, 2.75, .40, 20, WHITE, True, PP_ALIGN.CENTER)
    rect(slide, 1.02, 4.02, 2.75, 1.15, RGBColor(43, 72, 86), RGBColor(72, 101, 113))
    text(slide, "범죄율 = 5대범죄 발생건수\n÷ 등록인구 × 10,000", 1.16, 4.16, 2.47, .78, 13, WHITE, True, PP_ALIGN.CENTER)
    text(slide, "예: 2022년 자료 → 2025년 예측", 1.02, 5.53, 2.75, .30, 9.5, RGBColor(207, 222, 216), False, PP_ALIGN.CENTER)

    text(slide, "검증 설계", 4.55, 1.46, 2.2, .34, 16, GREEN, True)
    items = [
        ("최초 기준", "Persistence — 현재 범죄율을 3년 후에도 유지"),
        ("모델 선택", "2022~2024 확대창 순차검증 평균 RMSE 최소"),
        ("최종 확인", "선택에 사용하지 않은 2025 홀드아웃 25개 구"),
        ("비교 규모", "64개 feature 조합 × 8개 알고리즘 = 512개"),
    ]
    for i, (title, body) in enumerate(items):
        y = 1.93 + i * 1.00
        rect(slide, 4.55, y, 7.55, .78, WHITE, LINE)
        rect(slide, 4.73, y + .17, 1.15, .42, GREEN, GREEN)
        text(slide, title, 4.73, y + .17, 1.15, .42, 9, WHITE, True, PP_ALIGN.CENTER)
        text(slide, body, 6.10, y + .12, 5.65, .52, 12, NAVY, True)
    rect(slide, 4.55, 6.06, 7.55, .49, PALE_GOLD, PALE_GOLD)
    text(slide, "회귀모델은 단일 정답률 대신 MAE·RMSE·R²·MAPE·순위상관으로 평가", 4.77, 6.14, 7.10, .28, 10, ORANGE, True, PP_ALIGN.CENTER)


def slide_baseline(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header(slide, "초기 기준모델 — Persistence",
           "학습 없이 현재 범죄율을 3년 후 예측값으로 그대로 사용", 2)
    rect(slide, .72, 1.42, 4.00, 4.90, NAVY, NAVY)
    text(slide, "최초 예측식", 1.03, 1.77, 3.38, .32, 16, GOLD, True)
    text(slide, "3년 후 예측범죄율", 1.04, 2.49, 3.35, .30, 12, WHITE, True, PP_ALIGN.CENTER)
    text(slide, "=", 1.04, 2.90, 3.35, .36, 22, GOLD, True, PP_ALIGN.CENTER)
    text(slide, "현재 범죄율", 1.04, 3.35, 3.35, .48, 24, WHITE, True, PP_ALIGN.CENTER)
    rect(slide, 1.07, 4.35, 3.28, 1.10, RGBColor(43, 72, 86), RGBColor(72, 101, 113))
    text(slide, "사용 feature 1개\n현재 인구 1만 명당 범죄율", 1.27, 4.51, 2.88, .68, 12, WHITE, True, PP_ALIGN.CENTER)
    text(slide, "지역별 범죄수준이 지속된다는 가정", 1.08, 5.71, 3.28, .26, 9.5, RGBColor(207, 222, 216), False, PP_ALIGN.CENTER)

    text(slide, "순차검증 성능", 5.12, 1.47, 2.2, .34, 16, GREEN, True)
    metric(slide, 5.12, 1.93, "MAE", "8.82", "평균 절대오차")
    metric(slide, 7.58, 1.93, "RMSE", "13.60", "개선 판단 기준")
    metric(slide, 10.04, 1.93, "R²", "0.8646", "분산 설명률")
    metric(slide, 5.12, 3.27, "MAPE", "8.47%", "평균 비율오차", BLUE)
    metric(slide, 7.58, 3.27, "순위 Spearman", "0.9582", "자치구 순위 상관", BLUE)
    metric(slide, 10.04, 3.27, "CV RMSE", "기준 100", "이후 개선 비교", ORANGE)
    rect(slide, 5.12, 4.78, 7.17, 1.17, MINT, MINT)
    text(slide, "왜 강한 기준인가?", 5.40, 4.94, 1.82, .29, 12, GREEN, True)
    text(slide, "자치구별 범죄율 수준과 순위가 단기간에 크게 바뀌지 않아\n복잡한 모델도 이 기준보다 좋아야 실제 개선으로 인정할 수 있습니다.",
         7.12, 4.87, 4.84, .68, 11, NAVY, True)
    text(slide, "선택 지표: 큰 예측 실패에 더 큰 벌점을 주는 RMSE", 5.15, 6.29, 7.1, .30, 9.5, GRAY, False, PP_ALIGN.CENTER)


def stage_card(slide, x, y, step, title, model, features, rmse, gain, fill, accent):
    rect(slide, x, y, 3.84, 4.65, fill, fill)
    rect(slide, x + .22, y + .20, .60, .60, accent, accent)
    text(slide, str(step), x + .22, y + .20, .60, .60, 16, WHITE, True, PP_ALIGN.CENTER)
    text(slide, title, x + .98, y + .18, 2.55, .34, 13, accent, True)
    text(slide, model, x + .98, y + .53, 2.55, .26, 10, GRAY, True)
    text(slide, "추가·선택 feature", x + .28, y + 1.02, 3.25, .28, 10, NAVY, True)
    text(slide, features, x + .28, y + 1.34, 3.24, 1.46, 10, NAVY, False, PP_ALIGN.LEFT, MSO_ANCHOR.TOP)
    rect(slide, x + .27, y + 3.03, 3.29, .03, accent, accent, False)
    text(slide, "CV RMSE", x + .28, y + 3.19, 1.16, .26, 9, GRAY, True)
    text(slide, rmse, x + 1.43, y + 3.12, 1.90, .41, 20, accent, True, PP_ALIGN.RIGHT)
    text(slide, gain, x + .28, y + 3.69, 3.25, .52, 11, accent, True, PP_ALIGN.CENTER)


def slide_improvements(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header(slide, "성능이 갱신된 3단계",
           "기준모델 이후 CV RMSE 최저 기록이 실제로 바뀐 경우만 정리", 3)
    stage_card(slide, .62, 1.47, 1, "첫 성능 개선", "Ridge (α=1)",
               "현재 범죄율\n+ 경찰관서 1만 명당\n+ CCTV 1만 명당\n+ CCTV 전년변화율\n+ 스트레스 인지율",
               "13.5013", "직전 대비 0.73% 개선", MINT, GREEN)
    stage_card(slide, 4.75, 1.47, 2, "feature 확장", "ElasticNet",
               "1단계 feature\n+ 인구수·세대수\n+ 세대당 인구·남성비율\n+ 인구수 전년변화율\n+ 세대수 전년변화율",
               "13.2223", "직전 대비 2.07% 개선", PALE_BLUE, BLUE)
    stage_card(slide, 8.88, 1.47, 3, "최종 알고리즘 개선", "Ridge (α=1)",
               "2단계와 동일한 11개 feature\n\nElasticNet보다 Ridge가\n작은 표본에서 더 안정적으로\n계수를 규제",
               "12.9757", "직전 대비 1.86% 개선", PALE_GOLD, ORANGE)
    text(slide, "Persistence 13.6013 → 최종 Ridge 12.9757  |  전체 CV RMSE 4.60% 개선", .82, 6.46, 11.70, .39, 13, NAVY, True, PP_ALIGN.CENTER)


def slide_final(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header(slide, "최종모델 성능과 feature 선택 결과",
           "모델 선택에 사용하지 않은 2025년 홀드아웃에서 최종 확인", 4)
    text(slide, "2025 홀드아웃", .72, 1.36, 2.2, .34, 16, GREEN, True)
    metric(slide, .72, 1.79, "MAE", "11.60", "기준 15.69 · 26.0%↓", GREEN)
    metric(slide, 3.12, 1.79, "RMSE", "12.85", "기준 17.40 · 26.2%↓", GREEN)
    metric(slide, 5.52, 1.79, "R²", "0.8947", "기준 0.8067", GREEN)
    metric(slide, 7.92, 1.79, "MAPE", "16.11%", "기준 20.36%", BLUE)
    metric(slide, 10.32, 1.79, "순위 상관", "0.9400", "기준 0.9123", BLUE)

    rect(slide, .72, 3.18, 5.75, 2.80, WHITE, LINE)
    text(slide, "최종 선택된 11개 feature", .98, 3.42, 4.95, .31, 15, GREEN, True)
    text(slide,
         "• 현재 범죄율\n• 인구수 · 세대수 · 세대당 인구 · 남성비율\n• 인구수 · 세대수 전년변화율\n• 경찰관서 · CCTV 1만 명당\n• CCTV 전년변화율 · 스트레스 인지율",
         .98, 3.86, 5.10, 1.75, 10.5, NAVY, False, PP_ALIGN.LEFT, MSO_ANCHOR.TOP)

    rect(slide, 6.72, 3.18, 5.39, 2.80, WHITE, LINE)
    text(slide, "다른 feature 조합과 비교", 6.98, 3.42, 4.70, .31, 15, BLUE, True)
    rows = [
        ("현재 범죄율만", "15.6013", "16.83% 개선"),
        ("28개 전체 feature", "15.7645", "17.69% 개선"),
        ("동일 feature ElasticNet", "13.2223", "1.86% 개선"),
    ]
    for i, (label, rmse, gain) in enumerate(rows):
        y = 3.92 + i * .56
        text(slide, label, 7.00, y, 2.35, .28, 10, NAVY, True)
        text(slide, f"RMSE {rmse}", 9.18, y, 1.08, .28, 9, GRAY, False, PP_ALIGN.RIGHT)
        rect(slide, 10.43, y, 1.40, .29, PALE_BLUE, PALE_BLUE)
        text(slide, gain, 10.43, y, 1.40, .29, 8.5, BLUE, True, PP_ALIGN.CENTER)
    text(slide, "결론", 7.00, 5.62, .66, .25, 10, ORANGE, True)
    text(slide, "모든 변수를 넣기보다 잡음이 큰 변수를 제외한\n11개 조합이 3년 후 예측에 가장 안정적", 7.72, 5.47, 4.05, .55, 10.5, NAVY, True)

    rect(slide, .72, 6.30, 11.39, .51, NAVY, NAVY)
    text(slide, "최종 저장: Ridge(α=1) 1개  |  2021~2025 목표연도 전체로 재학습  |  2026년 범죄율 예측 생성",
         .92, 6.40, 10.95, .28, 10.5, WHITE, True, PP_ALIGN.CENTER)
    text(slide, "※ CCTV는 철거 이력이 없어 설치연도 기반 누적 추정치이며, 결과는 예측 관계이지 인과관계를 의미하지 않습니다.",
         .76, 6.94, 11.7, .23, 8.2, GRAY)


def slide_next_improvements(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header(slide, "향후 모델 개선 방향",
           "3년 뒤 변화를 먼저 보여주는 정보와 범죄 유형별 특성을 분리해 반영", 5)

    rect(slide, .72, 1.42, 5.78, 4.95, MINT, MINT)
    rect(slide, 1.00, 1.72, .72, .72, GREEN, GREEN)
    text(slide, "1", 1.00, 1.72, .72, .72, 19, WHITE, True, PP_ALIGN.CENTER)
    text(slide, "실제 선행변수 추가", 1.92, 1.72, 3.92, .40, 19, GREEN, True)
    text(slide, "예측 시점에 이미 확보할 수 있으면서\n향후 범죄 변화보다 먼저 움직이는 자료", 1.92, 2.13, 3.98, .62, 10.5, NAVY)

    lead_rows = [
        ("신고·치안", "112 신고·민원 증감, 경찰 순찰·인력 계획"),
        ("방범환경", "CCTV·보안등 설치계획, 방범시설 예산"),
        ("생활·상권", "야간 유동인구, 유흥업소 인허가·폐업"),
        ("주거·사회", "빈집률, 재개발 이주, 입주 예정, 실업·취약계층"),
    ]
    for i, (tag, body) in enumerate(lead_rows):
        y = 2.97 + i * .66
        rect(slide, 1.00, y, 1.05, .36, GREEN, GREEN)
        text(slide, tag, 1.00, y, 1.05, .36, 8.5, WHITE, True, PP_ALIGN.CENTER)
        text(slide, body, 2.20, y - .03, 3.84, .43, 9.7, NAVY, True)
    rect(slide, 1.00, 5.77, 5.20, .39, WHITE, WHITE)
    text(slide, "원칙  예측 실행일에 확정·공개된 자료만 사용", 1.18, 5.83, 4.84, .25, 9.2, GREEN, True, PP_ALIGN.CENTER)

    rect(slide, 6.78, 1.42, 5.33, 4.95, PALE_BLUE, PALE_BLUE)
    rect(slide, 7.06, 1.72, .72, .72, BLUE, BLUE)
    text(slide, "2", 7.06, 1.72, .72, .72, 19, WHITE, True, PP_ALIGN.CENTER)
    text(slide, "범죄 유형별 모델 분리", 7.98, 1.72, 3.65, .40, 19, BLUE, True)
    text(slide, "5대 범죄 합계를 한 번에 예측하지 않고\n유형별 영향요인과 변동성을 따로 학습", 7.98, 2.13, 3.65, .62, 10.5, NAVY)

    type_rows = [
        ("절도", "상권·유동인구·지하철·점포 수"),
        ("폭력", "야간 생활인구·유흥업소·음주상권"),
        ("성범죄", "야간 유동인구·신고환경·방범시설"),
        ("살인·강도", "희소 건수 → Poisson·음이항 모델 검토"),
    ]
    for i, (tag, body) in enumerate(type_rows):
        y = 2.97 + i * .66
        rect(slide, 7.06, y, 1.05, .36, BLUE, BLUE)
        text(slide, tag, 7.06, y, 1.05, .36, 8.5, WHITE, True, PP_ALIGN.CENTER)
        text(slide, body, 8.27, y - .03, 3.50, .43, 9.7, NAVY, True)
    rect(slide, 7.06, 5.77, 4.75, .39, WHITE, WHITE)
    text(slide, "최종 범죄율 = 유형별 예측 발생률의 합", 7.24, 5.83, 4.38, .25, 9.2, BLUE, True, PP_ALIGN.CENTER)

    rect(slide, .72, 6.56, 11.39, .46, NAVY, NAVY)
    text(slide, "검증 기준  기존 Ridge와 동일한 연도 순차검증에서 RMSE 개선 + 3개 연도 중 2개 이상 개선 시 채택",
         .93, 6.64, 10.96, .28, 10, WHITE, True, PP_ALIGN.CENTER)


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
    slide_overview(prs)
    slide_baseline(prs)
    slide_improvements(prs)
    slide_final(prs)
    slide_next_improvements(prs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
