from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches

from create_crime_forecast_compact_ppt import (
    BLUE, INK, GRAY, LIGHT, MINT, NAVY, NAVY2, RED, TEAL, WHITE,
    footer, header, shape, txt,
)


OUT = Path.home() / "Desktop" / "교통사고_13차_기본모델_핵심요약_2장.pptx"


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # 1. 모델 개요와 성능
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "BASE MODEL · EXECUTIVE SUMMARY", "교통사고 13차 기본 모델 — 다음연도 안전등급 예측", 1)

    shape(s, 0.72, 1.60, 3.05, 4.88, NAVY, True)
    txt(s, 0.98, 1.90, 2.5, 0.3, "예측 문제", 12, TEAL, True)
    txt(s, 0.98, 2.37, 2.5, 1.02, "현재연도 안전지표로\n다음연도 자치구별\n안전점수·등급 예측", 21, WHITE, True)
    shape(s, 0.98, 3.62, 2.52, 0.02, RGBColor(64, 89, 112), False)
    txt(s, 0.98, 3.90, 2.5, 0.68, "대상  서울 25개 자치구\n출력  안전등급 1~5", 12, RGBColor(210, 223, 235), True)
    txt(s, 0.98, 4.85, 2.5, 0.3, "검증 설계", 12, TEAL, True)
    txt(s, 0.98, 5.27, 2.5, 0.88, "2021~2024 확대창 순차검증\n2025년 25개 구 최종검증\n2026년 예측", 11, WHITE)

    shape(s, 4.08, 1.60, 8.55, 1.16, MINT, True)
    txt(s, 4.37, 1.82, 2.65, 0.32, "선정 알고리즘", 12, NAVY, True)
    txt(s, 6.62, 1.72, 2.95, 0.55, "RandomForest", 25, NAVY, True, PP_ALIGN.CENTER)
    txt(s, 9.75, 1.76, 2.28, 0.45, "입력변수 4개", 15, TEAL, True, PP_ALIGN.CENTER)
    txt(s, 4.37, 2.28, 7.6, 0.24, "안전기준표 기반 · 경감지표 제외 후 가중치 재배분", 10, GRAY)

    cards = [
        ("01", "교통사고 사망위험", "인구 1만 명당\n사망자수", "62.5%", MINT),
        ("02", "의료보장 사업장", "인구 1만 명당\n사업장수", "12.5%", BLUE),
        ("03", "안전벨트 착용", "운전자안전벨트\n착용률", "12.70%", MINT),
        ("04", "음주운전 경험", "음주운전\n경험률", "12.30%", BLUE),
    ]
    for i, (num, label, metric, weight, fill) in enumerate(cards):
        x = 4.08 + i * 2.14
        shape(s, x, 3.07, 1.92, 2.24, fill, True)
        txt(s, x + 0.14, 3.24, 0.34, 0.26, num, 10, TEAL, True)
        txt(s, x + 0.14, 3.65, 1.64, 0.40, label, 11, NAVY, True)
        txt(s, x + 0.14, 4.15, 1.64, 0.55, metric, 13, INK, True, valign=MSO_ANCHOR.MIDDLE)
        txt(s, x + 0.14, 4.88, 1.64, 0.24, weight, 11, GRAY, True)

    shape(s, 4.08, 5.62, 8.55, 0.87, NAVY2, True)
    txt(s, 4.28, 5.77, 2.55, 0.22, "2025 최종검증", 10, TEAL, True)
    txt(s, 4.28, 6.04, 2.55, 0.24, "등급 정확도 48%", 13, WHITE, True)
    txt(s, 7.10, 5.77, 2.45, 0.22, "허용오차 ±1등급", 10, TEAL, True)
    txt(s, 7.10, 6.04, 2.45, 0.24, "정확도 76%", 13, WHITE, True)
    txt(s, 9.82, 5.77, 2.40, 0.22, "점수 오차", 10, TEAL, True)
    txt(s, 9.82, 6.04, 2.40, 0.24, "MAE 14.63점", 13, WHITE, True)
    footer(s)

    # 2. 데이터 출처와 해석
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "DATA TRACE · INTERPRETATION", "4개 입력지표의 출처와 모델 해석 기준", 2)

    # source rows
    rows = [
        ("01", "교통사고 사망자수", "한국도로교통공단 TAAS", "TAAS 사망자수 ÷ 서울시 주민등록인구 × 10,000", MINT),
        ("02", "의료보장 사업장수", "KOSIS · 국민건강보험공단", "사업장수 ÷ 서울시 주민등록인구 × 10,000", BLUE),
        ("03", "안전벨트 착용률", "질병관리청 지역사회건강조사", "자치구별 운전자 안전벨트 착용률 원자료", MINT),
        ("04", "음주운전 경험률", "질병관리청 지역사회건강조사", "자치구별 음주운전율을 경험률로 대응", BLUE),
    ]
    for i, (num, metric, source, calc, fill) in enumerate(rows):
        y = 1.55 + i * 1.05
        shape(s, 0.72, y, 7.72, 0.82, fill, True)
        shape(s, 0.90, y + 0.19, 0.42, 0.42, TEAL, True)
        txt(s, 0.90, y + 0.22, 0.42, 0.28, num, 10, WHITE, True, PP_ALIGN.CENTER)
        txt(s, 1.49, y + 0.13, 2.02, 0.28, metric, 12, NAVY, True)
        txt(s, 3.55, y + 0.13, 2.37, 0.28, source, 10, NAVY, True)
        txt(s, 3.55, y + 0.44, 4.55, 0.23, calc, 9, GRAY)

    # right panel: split and caution
    shape(s, 8.72, 1.55, 3.90, 2.22, NAVY, True)
    txt(s, 8.98, 1.80, 3.28, 0.28, "데이터 구성", 12, TEAL, True)
    txt(s, 8.98, 2.22, 3.28, 0.30, "학습 175행", 17, WHITE, True)
    txt(s, 8.98, 2.60, 3.28, 0.25, "2017~2023 → 2018~2024", 10, RGBColor(210, 223, 235))
    txt(s, 8.98, 3.02, 3.28, 0.30, "최종검증 25행", 17, WHITE, True)
    txt(s, 8.98, 3.39, 3.28, 0.22, "2024 → 2025", 10, RGBColor(210, 223, 235))

    shape(s, 8.72, 4.00, 3.90, 1.75, LIGHT, True)
    txt(s, 8.98, 4.25, 3.28, 0.28, "반드시 함께 표기", 12, RED, True)
    txt(s, 8.98, 4.67, 3.28, 0.78,
        "• 2025 의료보장 사업장수는\n   2024년 최근 공표값 이월\n• 공식 지역안전지수 등급이 아님", 10, INK)

    shape(s, 0.72, 5.98, 11.90, 0.70, NAVY2, True)
    txt(s, 0.95, 6.10, 2.45, 0.22, "등급 산출", 10, TEAL, True)
    txt(s, 2.05, 6.10, 4.30, 0.25, "예측 안전점수를 자치구 상대순위로 1~5등급화", 11, WHITE, True)
    txt(s, 6.70, 6.10, 1.55, 0.22, "가중치 처리", 10, TEAL, True)
    txt(s, 8.05, 6.10, 4.20, 0.25, "경감 20% 제외 → 위해 62.5 · 취약 12.5 · 의식 25", 11, WHITE, True)
    footer(s)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
