from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "presentations" / "범죄_아동환경_초기모델_성능과_개선방향_선행변수상세.pptx"

NAVY = RGBColor(24, 47, 63)
GREEN = RGBColor(30, 117, 87)
MINT = RGBColor(225, 242, 234)
BLUE = RGBColor(46, 111, 171)
PALE_BLUE = RGBColor(231, 240, 249)
GOLD = RGBColor(230, 174, 57)
WHITE = RGBColor(255, 255, 255)
BG = RGBColor(248, 250, 249)
GRAY = RGBColor(93, 108, 104)
LINE = RGBColor(216, 225, 220)
FONT = "맑은 고딕"


def rect(slide, x, y, w, h, fill=WHITE, line=LINE, rounded=True):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line
    return shape


def add_text(slide, value, x, y, w, h, size=14, color=NAVY, bold=False,
             align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.MIDDLE):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.vertical_anchor = valign
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = value
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return box


def header(slide, title, subtitle, number, accent):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = BG
    add_text(slide, title, .62, .30, 10.8, .52, 25, NAVY, True)
    add_text(slide, subtitle, .64, .84, 11.1, .30, 10.5, GRAY)
    rect(slide, 11.75, .34, .90, .46, accent, accent)
    add_text(slide, f"0{number}", 11.75, .34, .90, .46, 11, WHITE, True, PP_ALIGN.CENTER)


def metric_card(slide, x, y, label, value, note, accent):
    rect(slide, x, y, 2.15, 1.25, WHITE, LINE)
    rect(slide, x, y, .08, 1.25, accent, accent, False)
    add_text(slide, label, x + .20, y + .13, 1.72, .24, 10, GRAY, True)
    add_text(slide, value, x + .20, y + .38, 1.72, .42, 20, accent, True)
    add_text(slide, note, x + .20, y + .86, 1.72, .22, 8.2, GRAY)


def improvement_card(slide, x, y, number, title, body, accent, pale):
    rect(slide, x, y, 5.45, 1.62, pale, pale)
    rect(slide, x + .20, y + .25, .60, .60, accent, accent)
    add_text(slide, str(number), x + .20, y + .25, .60, .60, 16, WHITE, True, PP_ALIGN.CENTER)
    add_text(slide, title, x + 1.02, y + .17, 4.05, .34, 15, accent, True)
    add_text(slide, body, x + 1.02, y + .50, 4.05, .94, 8.8, NAVY)


def crime_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header(slide, "범죄 초기모델 성능과 개선 방향",
           "2023년 홀드아웃 · 서울 25개 자치구 · 표준모델: 전년도 점수 유지(Persistence)", 1, GREEN)

    add_text(slide, "현재 성능", .72, 1.38, 2.2, .34, 16, GREEN, True)
    metric_card(slide, .72, 1.83, "MAE", "6.59점", "평균 절대오차", GREEN)
    metric_card(slide, 3.03, 1.83, "RMSE", "8.53점", "큰 오차에 더 민감", GREEN)
    metric_card(slide, 5.34, 1.83, "R²", "0.9733", "분산 설명률 97.33%", GREEN)
    metric_card(slide, 7.65, 1.83, "순위 Spearman", "0.9862", "순위 상관계수", GREEN)
    metric_card(slide, 9.96, 1.83, "등급 정확도", "92%", "A등급 재현율 80%", GREEN)

    rect(slide, .72, 3.32, 11.39, .73, NAVY, NAVY)
    add_text(slide, "핵심 해석", .95, 3.47, 1.25, .28, 11, GOLD, True)
    add_text(slide, "전년도 유지값이 현재 최우수 기준선이며, 개선 모델은 ‘변화분’을 더 정확히 설명해야 합니다.",
             2.15, 3.42, 9.55, .38, 13, WHITE, True)

    add_text(slide, "개선 방향", .72, 4.34, 2.2, .34, 16, GREEN, True)
    improvement_card(slide, .72, 4.69, 1, "전년도 유지값의 잔차를 예측",
                     "다음 해 점수 자체가 아니라 실제값 − 전년도 유지값을 학습해, 기준선에서 달라지는 폭만 보정합니다.", GREEN, MINT)
    improvement_card(slide, 6.36, 4.69, 2, "선행변수 추가",
                     "• 112 신고·민원 건수와 유형별 증감률\n• 야간 생활인구·유동인구·상권 매출 변화\n• 유흥업소 인허가, 빈집률, 실업·취약계층 변화\n• CCTV·보안등 설치계획, 경찰력·순찰 투입계획", GREEN, MINT)

    add_text(slide, "※ R²는 적중률이 아니라 자치구 간 점수 차이를 설명한 비율입니다. 검증 표본이 25개이므로 추가 연도 검증이 필요합니다.",
             .75, 6.65, 11.7, .28, 8.5, GRAY)


def child_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header(slide, "아동환경 초기모델 성능과 개선 방향",
           "2024년 홀드아웃 · 서울 25개 자치구 · 3년 후 예측모델: Extra Trees", 2, BLUE)

    add_text(slide, "현재 성능", .72, 1.31, 2.2, .34, 16, BLUE, True)

    metric_card(slide, .72, 1.78, "MAE", "4.52점", "Persistence 4.33점", BLUE)
    metric_card(slide, 3.03, 1.78, "RMSE", "5.72점", "Persistence 6.06점", BLUE)
    metric_card(slide, 5.34, 1.78, "R²", "0.8662", "Persistence 0.8496", BLUE)
    metric_card(slide, 7.65, 1.78, "등급 정확도", "68%", "정확한 A~E 등급", BLUE)
    metric_card(slide, 9.96, 1.78, "90% 예측구간", "±12.60점", "CV 잔차 기반 범위", BLUE)

    rect(slide, .72, 3.26, 11.39, .78, NAVY, NAVY)
    add_text(slide, "핵심 해석", .96, 3.44, 1.28, .28, 11, GOLD, True)
    add_text(slide, "MAE는 Persistence가 0.19점 낮지만, 현재 모델은 RMSE를 0.35점 줄이고 R²를 높였습니다.",
             2.22, 3.38, 9.45, .42, 13, WHITE, True)

    add_text(slide, "개선 방향", .72, 4.38, 2.2, .34, 16, BLUE, True)
    improvement_card(slide, .72, 4.74, 1, "미래 선행변수 추가",
                     "• 유치원·어린이집 개원 및 폐원 예정\n• 공원 조성·정비계획과 문화복지시설 예산\n• 병원·의원·소아청소년과 인허가 및 폐업 예정\n• 보건소 인력계획, 아동인구 전망, 공동주택 입주계획", BLUE, PALE_BLUE)
    improvement_card(slide, 6.36, 4.74, 2, "Persistence 잔차 학습",
                     "전년도 점수를 기준값으로 두고 실제 변화분만 학습해, 안정적인 기준선과 머신러닝의 변화 포착을 결합합니다.", BLUE, PALE_BLUE)

    rect(slide, .72, 6.42, 11.39, .42, PALE_BLUE, PALE_BLUE)
    add_text(slide, "선행변수 사용 원칙: 예측 실행일에 확정·공개된 자료만 사용  |  목표연도 실적은 제외",
             .93, 6.49, 10.95, .25, 9.5, BLUE, True, PP_ALIGN.CENTER)
    add_text(slide, "※ 성능지표는 2024년 홀드아웃 25개 자치구 기준이며, R²는 적중률이 아니라 분산 설명률입니다.",
             .75, 6.94, 11.7, .25, 8.5, GRAY)


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    crime_slide(prs)
    child_slide(prs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
