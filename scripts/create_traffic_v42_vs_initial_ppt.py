from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches

from create_crime_forecast_compact_ppt import (
    BLUE, GRAY, INK, LIGHT, MINT, NAVY, NAVY2, RED, TEAL, WHITE,
    footer, header, shape, txt,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "presentations" / "교통사고_42차_vs_아동초기모델_비교.pptx"


def metric_card(slide, x, title, before, after, improvement, fill):
    shape(slide, x, 1.72, 3.73, 2.25, fill, True)
    txt(slide, x + .28, 2.00, 3.17, .26, title, 12, NAVY, True)
    txt(slide, x + .28, 2.43, 3.17, .55, improvement, 28, TEAL, True)
    txt(slide, x + .28, 3.20, 3.17, .30, f"{before}  →  {after}", 14, NAVY, True)
    txt(slide, x + .28, 3.57, 3.17, .20, "MAE · 낮을수록 좋음", 9, GRAY)


def bar(slide, y, label, value, color):
    txt(slide, .98, y, 2.22, .26, label, 10, INK, True)
    shape(slide, 3.18, y + .02, 3.50, .24, LIGHT, True)
    shape(slide, 3.18, y + .02, 3.50 * value / 6, .24, color, True)
    txt(slide, 6.82, y - .01, .68, .28, f"{value:.3f}", 10, color, True)


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "SPOTFIT · TRAFFIC MODEL 42", "38개 후보에서 여러 연도에 안정적인 42차 조합을 탐색", 1)
    shape(s, .72, 1.72, 3.45, 4.78, NAVY, True)
    txt(s, 1.00, 2.02, 2.85, .28, "예측 대상", 11, TEAL, True)
    txt(s, 1.00, 2.43, 2.85, 1.18, "다음 해 어린이\n1만 명당\n교통사고 피해자 수", 23, WHITE, True)
    txt(s, 1.00, 4.00, 2.85, .82, "Ridge alpha 0.1\n최종 17개 변수\n1년 뒤 예측", 13, RGBColor(218, 228, 236), True)
    txt(s, 1.00, 5.36, 2.85, .56, "후보 조합 51,794개\n알고리즘 8종 비교", 12, TEAL, True)
    metric_card(s, 4.50, "2024년 검증", "3.623", "2.166", "40.2% 개선", MINT)
    metric_card(s, 8.78, "2025년 독립검증", "5.609", "5.337", "4.8% 개선", BLUE)
    shape(s, 4.50, 4.38, 8.01, 1.45, NAVY2, True)
    txt(s, 4.80, 4.68, 1.52, .25, "핵심 결과", 11, TEAL, True)
    txt(s, 6.20, 4.60, 5.95, .68,
        "초기 고정값 모델보다 두 연도 모두 MAE가 낮았다.\n다만 2025년 기준 41차 MAE 5.081보다는 높다.", 14, WHITE, True)
    shape(s, 4.50, 6.04, 8.01, .46, RGBColor(255, 244, 226), True)
    txt(s, 4.72, 6.14, 7.58, .20, "판정  |  42차는 유효 후보지만 현재 champion은 41차", 11, RED, True, PP_ALIGN.CENTER)
    footer(s)

    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "SEARCH & VALIDATION", "단일 연도 최적화 대신 2021~2024년 순차검증으로 조합을 선정", 2)
    steps = [
        ("01", "후보 구성", "7개 안정 조합 + 35개 실험 조합\n중복 제거 후 총 38개"),
        ("02", "부분집합 탐색", "전진·부동·확률 교체 탐색\n고유 조합 51,794개"),
        ("03", "알고리즘 비교", "상위 30개 × 8개 알고리즘\n동일 순차검증 조건"),
        ("04", "독립검증", "2025년은 선택에 미사용\n최종 후보에 한 번만 적용"),
    ]
    for i, (num, title, body) in enumerate(steps):
        x = .72 + i * 3.00
        shape(s, x, 1.65, 2.74, 2.12, MINT if i % 2 == 0 else BLUE, True)
        shape(s, x + .22, 1.91, .48, .42, TEAL, True)
        txt(s, x + .22, 1.98, .48, .22, num, 10, WHITE, True, PP_ALIGN.CENTER)
        txt(s, x + .82, 1.94, 1.62, .30, title, 13, NAVY, True)
        txt(s, x + .22, 2.63, 2.30, .74, body, 11, INK, False, PP_ALIGN.CENTER)
    shape(s, .72, 4.08, 5.75, 2.20, NAVY, True)
    txt(s, 1.00, 4.38, 5.18, .28, "선정 모델", 11, TEAL, True)
    txt(s, 1.00, 4.84, 5.18, .45, "Ridge · alpha 0.1 · 17개 변수", 21, WHITE, True)
    txt(s, 1.00, 5.52, 5.18, .45, "2021~2024 평균 MAE 3.207\n평균 WMAPE 17.30%", 12, RGBColor(217, 228, 236))
    shape(s, 6.74, 4.08, 5.89, 2.20, RGBColor(255, 244, 226), True)
    txt(s, 7.04, 4.38, 5.28, .28, "검증 원칙", 11, RED, True)
    txt(s, 7.04, 4.83, 5.20, .92,
        "• 미래 연도의 정보를 이전 학습에 사용하지 않음\n• MAE를 주 기준으로 RMSE·R²·WMAPE 동시 확인\n• 2025년 결과로 변수 조합을 다시 고르지 않음", 12, INK)
    footer(s)

    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "FINAL 17 FEATURES", "사고이력·교통노출·도로·학교·행태 정보를 함께 사용", 3)
    left = [
        "어린이 피해자율 최근 3년 평균",
        "보행어린이 사고율",
        "어린이 교통사고 부상자 수",
        "어린이 부상률",
        "특수차 등록대수",
        "도로연장",
        "초등학교 사립 수",
        "중학교 학급 수",
        "중학교 공립 수",
    ]
    right = [
        "음주운전 차량 동승 경험률",
        "전체 사고다발 발생건수",
        "전체 사고다발 경상자 수",
        "전체 사고다발 부상신고자 수",
        "보호구역 어린이다발 경상자 수",
        "자전거 사고다발 지점 수",
        "자전거 사고다발 사망자 수",
        "자전거 사고다발 경상자 수",
    ]
    for x, title, items, fill in [
        (.72, "사고·인구·교통·학교", left, MINT),
        (6.74, "행태·사고다발 세부지표", right, BLUE),
    ]:
        shape(s, x, 1.58, 5.89, 4.88, fill, True)
        txt(s, x + .30, 1.88, 5.28, .30, title, 15, NAVY, True)
        for i, item in enumerate(items):
            y = 2.42 + i * .42
            shape(s, x + .31, y + .03, .20, .20, TEAL, True)
            txt(s, x + .64, y, 4.90, .28, item, 10, INK)
    footer(s)

    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "PERFORMANCE & DECISION", "42차는 초기모델을 개선했지만 41차를 교체할 정도는 아니었다", 4)
    shape(s, .72, 1.52, 7.10, 4.98, WHITE, True, RGBColor(220, 228, 224))
    txt(s, 1.00, 1.84, 6.45, .28, "2025년 독립검증 MAE", 14, NAVY, True)
    bar(s, 2.48, "아동 초기모델", 5.609, GRAY)
    bar(s, 3.06, "42차 모델", 5.337, TEAL)
    bar(s, 3.64, "41차 모델", 5.081, NAVY)
    txt(s, 1.00, 4.38, 6.35, .64,
        "42차는 초기모델 대비 4.8% 개선했지만,\n41차보다 MAE가 0.256 높았다.", 14, INK, True)
    shape(s, 1.00, 5.35, 6.38, .74, NAVY2, True)
    txt(s, 1.20, 5.54, 5.98, .30, "champion 유지: 41차 · challenger 보존: 42차", 12, WHITE, True, PP_ALIGN.CENTER)
    shape(s, 8.10, 1.52, 4.53, 2.18, MINT, True)
    txt(s, 8.40, 1.84, 3.93, .28, "42차 2025년 지표", 12, NAVY, True)
    for i, (k, v) in enumerate([("MAE", "5.337"), ("R²", "0.276"), ("WMAPE", "23.54%")]):
        x = 8.40 + i * 1.25
        txt(s, x, 2.38, 1.06, .22, k, 9, GRAY, True, PP_ALIGN.CENTER)
        txt(s, x, 2.74, 1.06, .40, v, 16, NAVY, True, PP_ALIGN.CENTER)
    shape(s, 8.10, 4.02, 4.53, 2.48, RGBColor(255, 244, 226), True)
    txt(s, 8.40, 4.34, 3.93, .28, "최종 판단", 12, RED, True)
    txt(s, 8.40, 4.80, 3.88, 1.06,
        "42차는 여러 연도에서 안정적인 조합을 찾은 실험이다.\n그러나 배포 승격 조건은 41차 독립검증 성능 초과이므로 현재는 보류한다.", 12, INK)
    footer(s)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
