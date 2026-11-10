from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "presentations" / "아동지역범죄노출_초기_1년후_3년예측_모델개선과정.pptx"

NAVY = RGBColor(22, 42, 58)
GREEN = RGBColor(27, 126, 96)
BLUE = RGBColor(49, 111, 183)
ORANGE = RGBColor(224, 126, 55)
GOLD = RGBColor(226, 176, 55)
RED = RGBColor(192, 74, 74)
WHITE = RGBColor(255, 255, 255)
BG = RGBColor(247, 250, 249)
GRAY = RGBColor(91, 105, 110)
LINE = RGBColor(215, 224, 222)
MINT = RGBColor(225, 243, 236)
PALE_BLUE = RGBColor(232, 241, 250)
PALE_ORANGE = RGBColor(252, 239, 227)
PALE_GRAY = RGBColor(238, 242, 242)
FONT = "맑은 고딕"


def shape(slide, x, y, w, h, fill=WHITE, line=LINE, rounded=True):
    s = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h),
    )
    s.fill.solid(); s.fill.fore_color.rgb = fill
    s.line.color.rgb = line
    return s


def text(slide, value, x, y, w, h, size=12, color=NAVY, bold=False,
         align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.MIDDLE):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame; frame.clear(); frame.word_wrap = True; frame.vertical_anchor = valign
    p = frame.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = value
    r.font.name = FONT; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color
    return box


def header(slide, title, subtitle, number):
    slide.background.fill.solid(); slide.background.fill.fore_color.rgb = BG
    text(slide, title, .62, .27, 10.9, .50, 24, NAVY, True)
    text(slide, subtitle, .64, .80, 11.35, .30, 10, GRAY)
    shape(slide, 11.85, .30, .78, .43, GREEN, GREEN)
    text(slide, f"{number:02d}", 11.85, .30, .78, .43, 10, WHITE, True, PP_ALIGN.CENTER)


def chip(slide, label, x, y, w, fill=GREEN):
    shape(slide, x, y, w, .34, fill, fill)
    text(slide, label, x, y, w, .34, 8.5, WHITE, True, PP_ALIGN.CENTER)


def metric(slide, x, y, label, before, after, note, color=GREEN, w=2.75):
    shape(slide, x, y, w, 1.13, WHITE, LINE)
    text(slide, label, x + .18, y + .10, w - .36, .22, 9, GRAY, True)
    text(slide, before, x + .18, y + .36, .88, .34, 13, GRAY, True)
    text(slide, "→", x + 1.03, y + .36, .35, .34, 13, color, True, PP_ALIGN.CENTER)
    text(slide, after, x + 1.39, y + .31, w - 1.57, .44, 18, color, True, PP_ALIGN.RIGHT)
    text(slide, note, x + .18, y + .80, w - .36, .20, 7.8, color, True)


def footer(slide, value):
    shape(slide, .62, 6.88, 12.00, .31, NAVY, NAVY, False)
    text(slide, value, .83, 6.91, 11.58, .22, 8.5, WHITE, False, PP_ALIGN.CENTER)


def slide_title(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid(); s.background.fill.fore_color.rgb = NAVY
    chip(s, "MODEL JOURNEY", .72, .65, 1.52, GOLD)
    text(s, "아동 지역범죄 노출 예측모델", .72, 1.34, 11.50, .66, 31, WHITE, True)
    text(s, "초기모델 → 2027년 1년 후 전망 → 2028년 중기 전망", .74, 2.05, 11.30, .45, 18, GOLD, True)
    shape(s, .74, 2.88, 11.86, 2.28, RGBColor(34, 62, 78), RGBColor(55, 84, 98))
    text(s, "예측 목표", 1.04, 3.17, 1.42, .30, 12, GOLD, True)
    text(s, "서울 25개 자치구의 5대 범죄건수와 아동인구를 이용해\n아동 1만 명당 지역범죄 노출 수준과 상대순위를 전망", 2.60, 3.02, 8.95, .72, 17, WHITE, True)
    text(s, "Y = log(1 + 5대 범죄 발생건수 ÷ 0~17세 인구 × 10,000)", 1.04, 4.18, 10.95, .43, 16, WHITE, True, PP_ALIGN.CENTER)
    text(s, "2025년까지 확정된 데이터를 기준으로 2026년 현재의 미래 전망 제공", .74, 5.72, 11.60, .35, 12, RGBColor(205, 220, 222), False, PP_ALIGN.CENTER)
    text(s, "SpotFit · 1차 발표용 모델 요약", .74, 6.72, 11.60, .28, 9, RGBColor(158, 184, 187), False, PP_ALIGN.RIGHT)


def slide_terms(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "예측연도와 모델 명칭을 먼저 구분", "데이터의 마지막 확정연도는 2025년, 서비스 현재연도는 2026년", 2)
    cards = [
        ("초기모델", "데이터 기준 3년", "초기 비교·기준선", "1년 변화를 3회 반복", PALE_GRAY, GRAY),
        ("1년 후 전망", "2025 → 2027", "2026년 기준 1년 후", "데이터 기준 2년 모델", MINT, GREEN),
        ("중기 전망", "2025 → 2028", "2026년 기준 2년 후", "데이터 기준 3년 모델", PALE_BLUE, BLUE),
    ]
    for i, (title, years, service, tech, fill, accent) in enumerate(cards):
        x = .72 + i * 4.10
        shape(s, x, 1.47, 3.72, 3.38, fill, fill)
        chip(s, f"0{i+1}", x + .24, 1.72, .56, accent)
        text(s, title, x + .95, 1.68, 2.40, .39, 18, accent, True)
        text(s, years, x + .28, 2.40, 3.16, .49, 22, NAVY, True, PP_ALIGN.CENTER)
        text(s, service, x + .28, 3.12, 3.16, .34, 12, accent, True, PP_ALIGN.CENTER)
        shape(s, x + .36, 3.70, 3.00, .64, WHITE, WHITE)
        text(s, tech, x + .48, 3.82, 2.76, .36, 10.5, NAVY, True, PP_ALIGN.CENTER)
    shape(s, .72, 5.24, 11.92, 1.12, WHITE, LINE)
    text(s, "핵심", .99, 5.48, .74, .30, 11, ORANGE, True)
    text(s, "2027년과 2028년은 서로 다른 목표연도이므로 한 모델만 고르는 것이 아니라 두 모델을 함께 운영한다.", 1.82, 5.39, 10.35, .46, 15, NAVY, True)
    footer(s, "발표·프론트엔드에는 ‘2027년 1년 후 / 2028년 2년 후’, 모델 파일에는 ‘2년 horizon / 3년 horizon’을 유지")


def slide_initial(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "초기모델: 1년 변화를 세 번 순차 적용", "초기 3년 예측 기준선 · Recursive Ridge(α=1)", 3)
    shape(s, .72, 1.42, 4.05, 4.93, NAVY, NAVY)
    text(s, "초기 로직", 1.02, 1.74, 1.42, .31, 15, GOLD, True)
    text(s, "t년 상태", 1.05, 2.32, 1.04, .39, 14, WHITE, True, PP_ALIGN.CENTER)
    text(s, "→", 2.14, 2.32, .42, .39, 18, GOLD, True, PP_ALIGN.CENTER)
    text(s, "t+1 예측", 2.63, 2.32, 1.25, .39, 14, WHITE, True, PP_ALIGN.CENTER)
    text(s, "동일한 1년 변화모델을 3회 반복", 1.05, 2.92, 3.37, .38, 12, GOLD, True, PP_ALIGN.CENTER)
    shape(s, 1.04, 3.58, 3.42, 1.57, RGBColor(39, 69, 84), RGBColor(62, 92, 105))
    text(s, "Y·범죄건수·아동인구 상태 갱신\n→ 다음 연도 Feature로 재사용\n→ 3년 뒤 Y 계산", 1.28, 3.81, 2.94, 1.08, 13, WHITE, True, PP_ALIGN.CENTER)
    text(s, "장점: 구조가 단순함\n한계: 반복할수록 오차 누적", 1.06, 5.49, 3.37, .53, 10.5, RGBColor(206, 220, 222), True, PP_ALIGN.CENTER)

    text(s, "초기 Feature 조합", 5.12, 1.46, 2.44, .34, 16, GREEN, True)
    groups = [
        ("기본 상태", "현재 Y · log 범죄건수 · log 아동인구"),
        ("유흥시설", "유흥주점·단란주점 수 / 인구 1만 명당 수"),
        ("사업체 밀도", "사업체 수·밀도 · 종사자 밀도 · 사업체당 종사자"),
        ("인구이동", "구간·구내·시도 간 전입·전출률 · 총전입·총전출률"),
    ]
    for i, (tag, body) in enumerate(groups):
        y = 1.94 + i * .86
        chip(s, tag, 5.13, y + .10, 1.22, GREEN)
        text(s, body, 6.55, y, 5.60, .56, 11, NAVY, True)
    shape(s, 5.12, 5.45, 6.99, 1.13, PALE_BLUE, PALE_BLUE)
    text(s, "초기 CV 성능", 5.35, 5.61, 1.28, .27, 10, BLUE, True)
    text(s, "Y RMSE 0.1020", 6.67, 5.57, 1.65, .36, 14, NAVY, True, PP_ALIGN.CENTER)
    text(s, "Spearman 0.9685", 8.36, 5.57, 1.80, .36, 14, NAVY, True, PP_ALIGN.CENTER)
    text(s, "순위 MAE 1.28위 · 5분위 73.3%", 10.18, 5.55, 1.69, .43, 9.5, BLUE, True, PP_ALIGN.CENTER)
    footer(s, "초기모델은 Feature 확대보다 ‘반복예측 오차’와 ‘범죄/아동인구 오차 상쇄’를 해결하는 것이 핵심 과제")


def slide_journey(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "모델 개선과정: Feature 확대보다 목적함수와 로직 변경", "각 단계에서 검증 성능이 실제로 좋아진 조합만 다음 단계로 전달", 4)
    stages = [
        ("1", "초기 재귀", "Y를 1년씩\n3회 반복", "유흥시설\n사업체·이동", PALE_GRAY, GRAY),
        ("2", "구성요소 분리", "범죄건수와\n아동인구 별도 예측", "가구·산업\n출생·추세", MINT, GREEN),
        ("3", "하위모델 개선", "범죄유형 합산\n코호트 기반 인구", "학생 수·학급\n연령 코호트", PALE_BLUE, BLUE),
        ("4", "순위 목적 전환", "Spearman·순위MAE\n상위20%로 선택", "성능 없는\n스트레스 제외", PALE_ORANGE, ORANGE),
        ("5", "통합 보정", "기존 92% +\n직접 Y 8%", "학생 전입·전출\nY 변화 추세", MINT, GREEN),
    ]
    for i, (num, title, logic, feats, fill, accent) in enumerate(stages):
        x = .52 + i * 2.55
        shape(s, x, 1.56, 2.27, 4.52, fill, fill)
        chip(s, num, x + .18, 1.76, .50, accent)
        text(s, title, x + .78, 1.72, 1.30, .39, 13, accent, True)
        text(s, "로직", x + .20, 2.44, .55, .26, 9, GRAY, True)
        text(s, logic, x + .20, 2.72, 1.87, .88, 11.5, NAVY, True, PP_ALIGN.CENTER)
        shape(s, x + .20, 3.76, 1.87, .03, accent, accent, False)
        text(s, "Feature 변화", x + .20, 4.00, 1.87, .27, 9, GRAY, True)
        text(s, feats, x + .20, 4.36, 1.87, .90, 10.5, NAVY, True, PP_ALIGN.CENTER)
        if i < len(stages) - 1:
            text(s, "→", x + 2.27, 3.26, .28, .40, 17, accent, True, PP_ALIGN.CENTER)
    shape(s, .72, 6.30, 11.91, .42, NAVY, NAVY)
    text(s, "중요: 아동인구 ExtraTrees는 MAPE를 개선했지만 최종 순위는 하락 → 하위모델 정확도와 최종 순위 목적을 분리", .92, 6.36, 11.50, .28, 10, WHITE, True, PP_ALIGN.CENTER)


def slide_2027(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "2027년 1년 후 전망: 2025 완결 데이터 기반 2년 모델", "서비스 기준 1년 후 · 기술 기준 2년 horizon · Ridge(α=10) 직접 Y 예측", 5)
    shape(s, .72, 1.42, 4.00, 4.98, MINT, MINT)
    chip(s, "2025 → 2027", 1.03, 1.72, 1.56, GREEN)
    text(s, "학습표본 225개", 2.76, 1.71, 1.48, .36, 14, GREEN, True, PP_ALIGN.RIGHT)
    text(s, "2015~2023 입력\n→ 2017~2025 정답\n→ 서울 25개 자치구", 1.03, 2.30, 3.37, 1.02, 16, NAVY, True, PP_ALIGN.CENTER)
    shape(s, 1.03, 3.59, 3.38, 1.38, WHITE, WHITE)
    text(s, "최종 12개 Feature", 1.24, 3.78, 2.96, .30, 13, GREEN, True, PP_ALIGN.CENTER)
    text(s, "현재 Y·범죄·아동인구\n각 변수의 lag1·lag2·최근 변화량", 1.24, 4.15, 2.96, .59, 11.5, NAVY, True, PP_ALIGN.CENTER)
    text(s, "2024년까지만 있는 자료는 최종 조합에서 제외", 1.02, 5.46, 3.40, .42, 10, ORANGE, True, PP_ALIGN.CENTER)

    text(s, "2025년 잠금검증", 5.10, 1.47, 2.48, .32, 16, GREEN, True)
    metric(s, 5.10, 1.91, "순위 종합점수", "89.40", "90.34", "Persistence 대비 개선", GREEN, 3.39)
    metric(s, 8.73, 1.91, "5분위 일치율", "68%", "76%", "+8%p", GREEN, 3.39)
    metric(s, 5.10, 3.26, "상위 20% 적중", "80%", "80%", "유지", BLUE, 3.39)
    metric(s, 8.73, 3.26, "Y RMSE", "0.0928", "0.1015", "점수 오차는 악화", RED, 3.39)
    shape(s, 5.10, 4.76, 7.02, 1.20, PALE_ORANGE, PALE_ORANGE)
    text(s, "사용 해석", 5.37, 4.97, 1.05, .28, 11, ORANGE, True)
    text(s, "정확한 점수보다 2027년 예상 등급·상위 위험지역 전망에 사용\n점수는 ‘예측치’로 표시하고 순위구간과 함께 제공", 6.55, 4.88, 5.20, .67, 12, NAVY, True)
    footer(s, "엄격형과 lag1 허용형을 모두 비교했으나 최종 선택 Feature가 동일해 결과도 동일")


def slide_2028(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "2028년 중기 전망: 구성요소 예측 + 최종 Y 통합보정", "서비스 기준 2년 후 · 기술 기준 3년 horizon · 기존모델 92% + 직접 Y 8%", 6)
    shape(s, .72, 1.42, 4.00, 4.98, PALE_BLUE, PALE_BLUE)
    chip(s, "2025 → 2028", 1.03, 1.72, 1.56, BLUE)
    text(s, "학습표본 200개", 2.76, 1.71, 1.48, .36, 14, BLUE, True, PP_ALIGN.RIGHT)
    text(s, "범죄건수 예측", 1.03, 2.36, 1.45, .34, 12, BLUE, True)
    text(s, "기존 전체모델 + 5대 범죄 유형별 합산\n유흥시설·가구·산업·학생 수", 2.52, 2.23, 1.72, .75, 10.5, NAVY, True)
    text(s, "아동인구 예측", 1.03, 3.28, 1.45, .34, 12, GREEN, True)
    text(s, "인구구조·사업체·가구·이동·출생사망\n과거 Y·범죄·아동인구 추세", 2.52, 3.15, 1.72, .75, 10.5, NAVY, True)
    shape(s, 1.03, 4.31, 3.38, .03, BLUE, BLUE, False)
    text(s, "최종 보정", 1.03, 4.58, 1.04, .30, 12, ORANGE, True)
    text(s, "기존 순위모델 92%\n+ 학생 전입·전출 기반 직접 Y 모델 8%", 2.05, 4.43, 2.19, .74, 10.5, NAVY, True, PP_ALIGN.CENTER)
    text(s, "스트레스 인지율: 검증성능 악화로 제외", 1.02, 5.55, 3.40, .35, 10, RED, True, PP_ALIGN.CENTER)

    text(s, "2025년 잠금검증", 5.10, 1.47, 2.48, .32, 16, BLUE, True)
    metric(s, 5.10, 1.91, "순위 종합점수", "91.71", "91.73", "소폭 개선", BLUE, 3.39)
    metric(s, 8.73, 1.91, "Spearman", "0.9723", "0.9731", "개선", BLUE, 3.39)
    metric(s, 5.10, 3.26, "5분위 일치율", "84%", "84%", "유지", GREEN, 3.39)
    metric(s, 8.73, 3.26, "Y RMSE", "0.1851", "0.1764", "약 4.7% 개선", GREEN, 3.39)
    shape(s, 5.10, 4.76, 7.02, 1.20, MINT, MINT)
    text(s, "사용 해석", 5.37, 4.97, 1.05, .28, 11, GREEN, True)
    text(s, "2028년 중기 등급·순위 전망의 기본모델\n2027년 모델보다 잠금검증 순위 안정성이 높음", 6.55, 4.88, 5.20, .67, 12, NAVY, True)
    footer(s, "개선된 아동인구·범죄건수 모델을 동시에 쓰면 Y 오차는 감소하지만 순위가 하락해 92:8 통합보정을 채택")


def slide_compare(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "세 모델의 차이점 한눈에 보기", "예측 대상과 평가목적이 달라 하나를 폐기하는 관계가 아니라 단계별·연도별 역할 분담", 7)
    cols = [(.72, 2.25), (3.00, 3.06), (6.10, 3.06), (9.20, 3.08)]
    headers = ["구분", "초기모델", "2027년 1년 후", "2028년 중기"]
    fills = [NAVY, GRAY, GREEN, BLUE]
    for (x, w), h, f in zip(cols, headers, fills):
        shape(s, x, 1.43, w, .55, f, f, False)
        text(s, h, x, 1.43, w, .55, 11, WHITE, True, PP_ALIGN.CENTER)
    rows = [
        ("기술 horizon", "3년 재귀", "2년", "3년"),
        ("예측연도", "초기 비교", "2027", "2028"),
        ("핵심 로직", "1년 × 3회", "Y 직접 회귀", "구성요소 + 통합보정"),
        ("주요 Feature", "시설·밀도·이동", "현재값·lag·추세", "가구·산업·학생·이동"),
        ("잠금검증 강점", "기준선", "점수 오차", "순위·등급 안정성"),
        ("서비스 역할", "개선 비교용", "단기 전망", "중기 전망"),
    ]
    for i, row in enumerate(rows):
        y = 2.04 + i * .68
        bg = WHITE if i % 2 == 0 else PALE_GRAY
        for (x, w), value in zip(cols, row):
            shape(s, x, y, w, .62, bg, LINE, False)
            text(s, value, x + .08, y + .06, w - .16, .50, 10 if x == .72 else 10.5, NAVY, x == .72, PP_ALIGN.CENTER)
    shape(s, .72, 6.31, 11.56, .43, PALE_ORANGE, PALE_ORANGE)
    text(s, "주의  2027 점수와 2028 점수의 차이를 그대로 증감률로 해석하지 않고, 연도별 등급·순위 중심으로 제공", .94, 6.37, 11.10, .27, 10, ORANGE, True, PP_ALIGN.CENTER)


def slide_recommendation(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    header(s, "최종 운영 권장안", "두 모델을 함께 사용하되 예측연도·모델 버전·데이터 기준연도를 명시", 8)
    shape(s, .72, 1.43, 5.70, 3.80, MINT, MINT)
    chip(s, "기본", 1.02, 1.73, .76, GREEN)
    text(s, "2027년 전망", 1.98, 1.68, 2.20, .42, 20, GREEN, True)
    text(s, "현재 기준 1년 후", 4.37, 1.74, 1.54, .30, 10, GRAY, True, PP_ALIGN.RIGHT)
    text(s, "• 2년 horizon 엄격형 모델\n• 예상 등급·상위 위험지역 중심\n• 2025년까지 실제 값이 있는 Feature만 사용", 1.06, 2.42, 4.97, 1.30, 14, NAVY, True)
    shape(s, 1.04, 4.11, 5.05, .67, WHITE, WHITE)
    text(s, "operational_exposure_2y_2027_strict.joblib", 1.20, 4.24, 4.72, .34, 9.5, GREEN, True, PP_ALIGN.CENTER)

    shape(s, 6.72, 1.43, 5.56, 3.80, PALE_BLUE, PALE_BLUE)
    chip(s, "추가", 7.02, 1.73, .76, BLUE)
    text(s, "2028년 전망", 7.98, 1.68, 2.20, .42, 20, BLUE, True)
    text(s, "현재 기준 2년 후", 10.31, 1.74, 1.45, .30, 10, GRAY, True, PP_ALIGN.RIGHT)
    text(s, "• 3년 horizon 통합모델\n• 순위·등급 안정성 중심\n• 기존 순위모델 92% + 직접 Y 보정 8%", 7.06, 2.42, 4.87, 1.30, 14, NAVY, True)
    shape(s, 7.04, 4.11, 4.91, .67, WHITE, WHITE)
    text(s, "operational_exposure_integrated_blend_3y.joblib", 7.18, 4.24, 4.64, .34, 9, BLUE, True, PP_ALIGN.CENTER)

    shape(s, .72, 5.57, 11.56, .86, NAVY, NAVY)
    text(s, "화면 표기", .99, 5.81, 1.08, .30, 11, GOLD, True)
    text(s, "예측 기준 2026년  |  2027년 1년 후 전망  |  2028년 2년 후 전망  |  점수는 예측치, 등급·순위는 상대지표", 2.19, 5.72, 9.64, .48, 12, WHITE, True, PP_ALIGN.CENTER)
    footer(s, "현재 두 모델은 experiments에 저장 · 서비스용 recommender/v1 교체는 백엔드 추론 규격 확인 후 진행")


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
    slide_title(prs)
    slide_terms(prs)
    slide_initial(prs)
    slide_journey(prs)
    slide_2027(prs)
    slide_2028(prs)
    slide_compare(prs)
    slide_recommendation(prs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
