from __future__ import annotations

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
OUT = ROOT / "artifacts" / "presentations" / "교통사고_41차_vs_아동초기모델_비교.pptx"


def bar(slide, x, y, width, value, maximum, color, label, value_text):
    txt(slide, x, y, 2.25, 0.26, label, 10, INK, True)
    shape(slide, x + 2.35, y + 0.02, width, 0.24, LIGHT, True)
    shape(slide, x + 2.35, y + 0.02, width * value / maximum, 0.24, color, True)
    txt(slide, x + 2.35 + width + 0.12, y - 0.01, 0.76, 0.28, value_text, 10, color, True)


def main() -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "SPOTFIT · CHILD TRAFFIC SAFETY", "41차 모델은 아동 초기모델보다 다음 해 피해자율 예측오차를 줄였다", 1)
    shape(s, 0.72, 1.55, 3.45, 4.95, NAVY, True)
    txt(s, 1.00, 1.86, 2.85, 0.28, "예측 대상 Y", 11, TEAL, True)
    txt(s, 1.00, 2.28, 2.85, 1.18, "다음 해 어린이\n1만 명당\n교통사고 피해자 수", 23, WHITE, True)
    shape(s, 1.00, 3.72, 2.88, 0.02, RGBColor(66, 91, 114), False)
    txt(s, 1.00, 4.04, 2.85, 0.76, "피해자 = 사망자 + 부상자\n대상 = 서울 25개 자치구", 12, RGBColor(215, 226, 235))
    txt(s, 1.00, 5.22, 2.85, 0.65, "2017~2025년 확보자료만 사용", 11, TEAL, True)

    shape(s, 4.50, 1.55, 3.82, 2.22, MINT, True)
    txt(s, 4.80, 1.84, 3.20, 0.26, "2024년 MAE 개선", 11, NAVY, True)
    txt(s, 4.80, 2.25, 2.12, 0.62, "20.1%", 31, TEAL, True)
    txt(s, 4.80, 3.02, 3.10, 0.32, "3.623  →  2.896", 14, NAVY, True)
    shape(s, 8.58, 1.55, 4.05, 2.22, BLUE, True)
    txt(s, 8.88, 1.84, 3.35, 0.26, "2025년 독립검증 MAE 개선", 11, NAVY, True)
    txt(s, 8.88, 2.25, 2.12, 0.62, "9.4%", 31, TEAL, True)
    txt(s, 8.88, 3.02, 3.30, 0.32, "5.609  →  5.081", 14, NAVY, True)

    shape(s, 4.50, 4.08, 8.13, 1.55, NAVY2, True)
    txt(s, 4.82, 4.36, 2.70, 0.28, "핵심 판단", 11, TEAL, True)
    txt(s, 4.82, 4.78, 7.48, 0.56,
        "2023년 값이 그대로 유지된다는 초기 가정보다\n7개 요인을 학습한 41차 모델이 두 평가연도 모두 MAE가 낮았다.",
        15, WHITE, True)
    shape(s, 4.50, 5.91, 8.13, 0.58, RGBColor(255, 244, 226), True)
    txt(s, 4.72, 6.04, 7.70, 0.25, "결론  |  41차는 초기모델을 대체할 우선 후보 · 추가 연도 검증 필요", 11, RED, True, PP_ALIGN.CENTER)
    footer(s)

    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "COMPARISON DESIGN", "같은 Y와 같은 25개 자치구에서 초기 가정과 41차 모델을 비교", 2)
    shape(s, 0.72, 1.55, 3.62, 4.85, BLUE, True)
    txt(s, 1.00, 1.85, 2.95, 0.30, "아동 초기모델", 18, NAVY, True)
    txt(s, 1.00, 2.36, 2.95, 0.86, "2024년 예측 = 2023년 실제값\n2025년 예측 = 2023년 실제값", 14, INK, True)
    shape(s, 1.00, 3.52, 2.98, 0.02, RGBColor(194, 207, 218), False)
    txt(s, 1.00, 3.84, 2.95, 0.28, "장점", 11, TEAL, True)
    txt(s, 1.00, 4.22, 2.95, 0.62, "단순하고 설명이 쉬움\n학습 없이 즉시 계산", 11, INK)
    txt(s, 1.00, 5.15, 2.95, 0.28, "한계", 11, RED, True)
    txt(s, 1.00, 5.52, 2.95, 0.55, "지역 환경 변화와 사고 추세를\n반영하지 못함", 11, INK)
    txt(s, 4.47, 3.58, 0.55, 0.55, "VS", 19, TEAL, True, PP_ALIGN.CENTER)

    shape(s, 5.10, 1.55, 4.10, 4.85, MINT, True)
    txt(s, 5.40, 1.85, 3.45, 0.30, "41차 Ridge 모델", 18, NAVY, True)
    txt(s, 5.40, 2.32, 3.45, 0.66, "19개 변수 전체 부분집합 탐색\n52만 조합 × alpha 4종", 13, INK, True)
    txt(s, 5.40, 3.22, 3.45, 0.28, "선정 결과", 11, TEAL, True)
    txt(s, 5.40, 3.60, 3.45, 0.84, "최고 7개 변수\nRidge alpha = 0.1\n총 2,097,148회 평가", 12, INK)
    txt(s, 5.40, 4.80, 3.45, 0.28, "검증 구분", 11, TEAL, True)
    txt(s, 5.40, 5.18, 3.45, 0.74, "2024년: 변수 선택 검증\n2025년: 독립검증\n2025년은 변수 선택에 미사용", 11, INK)

    shape(s, 9.48, 1.55, 3.15, 4.85, NAVY, True)
    txt(s, 9.76, 1.85, 2.58, 0.28, "평가 기준", 12, TEAL, True)
    for i, (head, body) in enumerate([
        ("MAE", "평균적으로 몇 명 차이 나는가"),
        ("RMSE", "큰 오차에 더 큰 벌점"),
        ("R²", "자치구 간 차이 설명력"),
        ("WMAPE", "전체 실제값 대비 오차 비율"),
    ]):
        y = 2.38 + i * 0.82
        txt(s, 9.76, y, 0.72, 0.28, head, 11, WHITE, True)
        txt(s, 10.50, y, 1.82, 0.48, body, 10, RGBColor(218, 228, 236))
    shape(s, 9.76, 5.77, 2.58, 0.34, TEAL, True)
    txt(s, 9.86, 5.83, 2.38, 0.18, "주 선정 기준 = MAE", 10, WHITE, True, PP_ALIGN.CENTER)
    footer(s)

    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "PERFORMANCE", "41차 모델은 2024년과 2025년 모두 평균 절대오차를 개선", 3)
    shape(s, 0.72, 1.55, 7.18, 4.95, WHITE, True, RGBColor(220, 228, 224))
    txt(s, 1.02, 1.84, 6.55, 0.28, "MAE 비교 · 낮을수록 좋음", 14, NAVY, True)
    txt(s, 1.02, 2.36, 1.30, 0.28, "2024년", 12, TEAL, True)
    bar(s, 1.02, 2.82, 3.65, 3.623, 6.0, GRAY, "아동 초기모델", "3.623")
    bar(s, 1.02, 3.30, 3.65, 2.896, 6.0, TEAL, "41차 모델", "2.896")
    txt(s, 6.55, 3.02, 0.90, 0.40, "-20.1%", 14, TEAL, True, PP_ALIGN.CENTER)
    shape(s, 1.02, 3.88, 6.50, 0.02, RGBColor(225, 231, 228), False)
    txt(s, 1.02, 4.20, 1.30, 0.28, "2025년", 12, TEAL, True)
    bar(s, 1.02, 4.66, 3.65, 5.609, 6.0, GRAY, "아동 초기모델", "5.609")
    bar(s, 1.02, 5.14, 3.65, 5.081, 6.0, TEAL, "41차 모델", "5.081")
    txt(s, 6.55, 4.86, 0.90, 0.40, "-9.4%", 14, TEAL, True, PP_ALIGN.CENTER)
    txt(s, 1.02, 5.88, 6.45, 0.28, "단위: 어린이 1만 명당 피해자 수", 9, GRAY)

    shape(s, 8.18, 1.55, 4.45, 2.25, MINT, True)
    txt(s, 8.48, 1.84, 3.82, 0.28, "2025년 독립검증", 13, NAVY, True)
    cards = [("MAE", "5.081"), ("R²", "0.363"), ("WMAPE", "22.41%")]
    for i, (head, value) in enumerate(cards):
        x = 8.48 + i * 1.25
        txt(s, x, 2.42, 1.08, 0.22, head, 9, GRAY, True, PP_ALIGN.CENTER)
        txt(s, x, 2.76, 1.08, 0.42, value, 17, NAVY, True, PP_ALIGN.CENTER)
    txt(s, 8.48, 3.36, 3.76, 0.22, "2025년을 변수 선택에 사용하지 않은 결과", 9, TEAL, True, PP_ALIGN.CENTER)
    shape(s, 8.18, 4.08, 4.45, 2.42, NAVY2, True)
    txt(s, 8.48, 4.38, 3.82, 0.28, "해석", 12, TEAL, True)
    txt(s, 8.48, 4.82, 3.76, 0.96,
        "• 평균 오차는 초기모델보다 감소\n• 2025년은 2024년보다 예측 난도가 높음\n• 일부 자치구의 큰 오차는 여전히 존재",
        12, WHITE)
    txt(s, 8.48, 5.98, 3.76, 0.24, "다음 단계: 이상오차 자치구 원인 점검", 10, TEAL, True)
    footer(s)

    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "MODEL INPUTS · DECISION", "과거 위험 수준에 보행환경·도로·CCTV 정보를 더한 7개 변수", 4)
    groups = [
        ("과거 위험", "현재 어린이 피해자율\n최근 3년 평균 피해자율\n최근 3년 피해자율 변동성", NAVY),
        ("보행 위험", "보행어린이 사고율\n보행어린이 사고다발지역 수", TEAL),
        ("지역 환경", "도로율\n도로면적당 전체 CCTV", RGBColor(80, 126, 157)),
    ]
    for i, (head, body, color) in enumerate(groups):
        x = 0.72 + i * 3.15
        shape(s, x, 1.55, 2.88, 3.12, MINT if i != 1 else BLUE, True)
        shape(s, x + 0.22, 1.82, 2.44, 0.42, color, True)
        txt(s, x + 0.30, 1.89, 2.28, 0.23, head, 11, WHITE, True, PP_ALIGN.CENTER)
        txt(s, x + 0.30, 2.58, 2.28, 1.52, body, 13, NAVY, True, PP_ALIGN.CENTER)
    shape(s, 10.17, 1.55, 2.46, 3.12, NAVY, True)
    txt(s, 10.43, 1.88, 1.94, 0.28, "모델 설정", 12, TEAL, True, PP_ALIGN.CENTER)
    txt(s, 10.43, 2.48, 1.94, 1.20, "Ridge\nalpha 0.1\n7개 변수\n1년 뒤 예측", 16, WHITE, True, PP_ALIGN.CENTER)
    txt(s, 10.43, 4.02, 1.94, 0.26, "수동 가중치 없음", 10, RGBColor(213, 225, 234), True, PP_ALIGN.CENTER)

    shape(s, 0.72, 5.00, 7.65, 1.48, NAVY2, True)
    txt(s, 1.02, 5.28, 1.65, 0.28, "현재 결론", 12, TEAL, True)
    txt(s, 2.72, 5.24, 5.30, 0.62,
        "41차는 초기 고정값 모델보다 실용적인 예측 후보이며,\n2026년 예측과 후속 검증에 사용할 수 있다.", 15, WHITE, True)
    shape(s, 8.65, 5.00, 3.98, 1.48, RGBColor(255, 244, 226), True)
    txt(s, 8.95, 5.28, 3.38, 0.28, "주의사항", 12, RED, True)
    txt(s, 8.95, 5.70, 3.38, 0.52, "2024년 변수 선택 편향 가능성\n추가 연도·다른 알고리즘 검증 필요", 11, INK)
    footer(s)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
