from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches

from create_crime_forecast_compact_ppt import (
    BLUE, GRAY, INK, LIGHT, MINT, NAVY, NAVY2, RED, TEAL, WHITE,
    footer, header, shape, txt,
)


OUT = Path.home() / "Desktop" / "교통사고_안전등급모델_초기부터_성능개선과정_핵심요약_2장.pptx"


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # 1. 예측 문제 + 성능 개선 과정
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "MODEL STORY · EXECUTIVE SUMMARY", "다음연도 교통안전등급, 기준모델에서 최우선 모델까지", 1)

    shape(s, 0.72, 1.60, 3.05, 4.88, NAVY, True)
    txt(s, 0.98, 1.90, 2.5, 0.3, "예측 문제", 12, TEAL, True)
    txt(s, 0.98, 2.37, 2.5, 1.02, "현재연도 자료로\n다음연도 자치구별\n안전점수·등급 예측", 21, WHITE, True)
    shape(s, 0.98, 3.62, 2.52, 0.02, RGBColor(64, 89, 112), False)
    txt(s, 0.98, 3.90, 2.5, 0.68, "출력  안전등급 1~5\n대상  서울 25개 자치구", 12, RGBColor(210, 223, 235), True)
    txt(s, 0.98, 4.85, 2.5, 0.3, "검증 설계", 12, TEAL, True)
    txt(s, 0.98, 5.27, 2.5, 0.88, "2021~2024 확대창 순차검증\n2025년 25개 구 최종 확인\n등급 정확도 우선 선택", 11, WHITE)

    shape(s, 4.08, 1.60, 8.55, 1.16, MINT, True)
    txt(s, 4.37, 1.82, 3.2, 0.32, "최종 내부 CV 등급 정확도", 12, NAVY, True)
    txt(s, 7.15, 1.72, 1.55, 0.55, "56%", 28, NAVY, True, PP_ALIGN.CENTER)
    txt(s, 9.18, 1.76, 3.12, 0.45, "13차 대비 +13%p", 15, TEAL, True, PP_ALIGN.CENTER)
    txt(s, 4.37, 2.28, 7.6, 0.24, "13차 RandomForest 43%  →  19차 ExtraTrees 56%", 10, GRAY)

    steps = [
        ("1", "기준모델", "13차 · RandomForest", "43%", "안전기준표에서 확보된\n기본지표 4개", MINT),
        ("2", "사고이력 확장", "18차 · RandomForest", "53%", "현재 사고건수 +\n최근 3년 사고평균", BLUE),
        ("3", "최우선 모델", "19차 · ExtraTrees", "56%", "현재 사고건수 +\n자동차 등록 대수", MINT),
    ]
    for i, (num, stage, alg, acc, desc, fill) in enumerate(steps):
        x = 4.08 + i * 2.87
        shape(s, x, 3.08, 2.58, 2.72, fill, True)
        shape(s, x + 0.18, 3.28, 0.42, 0.42, TEAL, True)
        txt(s, x + 0.18, 3.31, 0.42, 0.30, num, 11, WHITE, True, PP_ALIGN.CENTER)
        txt(s, x + 0.72, 3.30, 1.62, 0.32, stage, 13, NAVY, True)
        txt(s, x + 0.18, 3.92, 2.22, 0.28, alg, 11, GRAY, True)
        txt(s, x + 0.18, 4.33, 2.22, 0.48, acc, 25, NAVY, True)
        txt(s, x + 0.18, 5.03, 2.22, 0.55, desc, 10, INK)
        if i < 2:
            txt(s, x + 2.60, 4.17, 0.26, 0.45, "→", 19, TEAL, True, PP_ALIGN.CENTER)

    shape(s, 4.08, 6.04, 8.55, 0.45, NAVY2, True)
    txt(s, 4.28, 6.12, 8.15, 0.23,
        "결론  |  13차는 설명 기준, 19차는 여러 연도에서 가장 안정적인 예측 후보",
        11, WHITE, True, PP_ALIGN.CENTER)
    footer(s)

    # 2. 최종 모델 + 향후 개선 방향
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "MODEL DESIGN · NEXT MOVE", "현재 모델은 6개 변수, 다음 개선은 공식 경감지표와 사고유형 분리", 2)

    shape(s, 0.72, 1.58, 4.0, 4.95, NAVY, True)
    txt(s, 0.98, 1.84, 2.9, 0.3, "CURRENT PRIORITY MODEL", 10, TEAL, True)
    txt(s, 0.98, 2.22, 2.60, 0.48, "19차 · ExtraTrees", 22, WHITE, True)
    txt(s, 3.62, 2.25, 0.82, 0.35, "6개 변수", 11, TEAL, True, PP_ALIGN.CENTER)
    txt(s, 0.98, 2.93, 2.2, 0.28, "입력변수", 12, RGBColor(210, 223, 235), True)
    vars_text = (
        "인구 1만 명당 교통사고 사망자수\n"
        "인구 1만 명당 의료보장 사업장수\n"
        "운전자 안전벨트 착용률\n"
        "음주운전 경험률\n"
        "현재연도 교통사고 건수\n"
        "자동차 등록 대수"
    )
    txt(s, 0.98, 3.35, 3.35, 1.70, vars_text, 11, WHITE)
    shape(s, 0.98, 5.38, 3.46, 0.75, TEAL, True)
    txt(s, 1.10, 5.48, 3.22, 0.22, "CV 정확도 56% · ±1등급 92%", 11, WHITE, True, PP_ALIGN.CENTER)
    txt(s, 1.10, 5.76, 3.22, 0.20, "안전점수 MAE 6.69점", 10, WHITE, True, PP_ALIGN.CENTER)

    shape(s, 5.02, 1.58, 3.67, 4.95, MINT, True)
    txt(s, 5.28, 1.84, 0.40, 0.30, "01", 11, TEAL, True)
    txt(s, 5.28, 2.22, 3.05, 0.46, "공식 경감지표 확보", 20, NAVY, True)
    txt(s, 5.28, 2.83, 3.05, 0.54, "현재 빠진 경감영역을 실제 연도별\n자료로 채워 환산등급 한계 해소", 11, GRAY)
    items1 = [
        ("단속", "교통법규 위반·단속 실적"),
        ("시설", "과거 CCTV·안전시설 스냅샷"),
        ("도로", "도로 구조·교차로·보행환경"),
        ("취약", "공식 재난약자·고령운전자"),
    ]
    for i, (head, body) in enumerate(items1):
        y = 3.64 + i * 0.58
        txt(s, 5.28, y, 0.72, 0.25, head, 10, TEAL, True)
        txt(s, 6.05, y, 2.27, 0.33, body, 10, INK)
    shape(s, 5.28, 5.98, 3.14, 0.31, NAVY2, True)
    txt(s, 5.39, 6.03, 2.92, 0.18, "원칙  당시 시점에 존재한 자료만", 9, WHITE, True, PP_ALIGN.CENTER)

    shape(s, 8.98, 1.58, 3.65, 4.95, BLUE, True)
    txt(s, 9.24, 1.84, 0.40, 0.30, "02", 11, TEAL, True)
    txt(s, 9.24, 2.22, 3.02, 0.46, "사고유형별 모델 분리", 20, NAVY, True)
    txt(s, 9.24, 2.83, 3.02, 0.54, "전체 사고를 한 번에 예측하지 않고\n유형별 위험요인과 희소성을 반영", 11, GRAY)
    items2 = [
        ("보행자", "유동인구·횡단보도·보행환경"),
        ("어린이", "스쿨존·학교·보호시설"),
        ("고령자", "고령인구·의료·보행 취약성"),
        ("이륜차", "배달·상권·시간대·도로특성"),
    ]
    for i, (head, body) in enumerate(items2):
        y = 3.64 + i * 0.58
        txt(s, 9.24, y, 0.82, 0.25, head, 10, TEAL, True)
        txt(s, 10.06, y, 2.25, 0.33, body, 10, INK)
    shape(s, 9.24, 5.98, 3.12, 0.31, NAVY2, True)
    txt(s, 9.35, 6.03, 2.90, 0.18, "최종 등급 = 유형별 위험 종합", 9, WHITE, True, PP_ALIGN.CENTER)

    shape(s, 0.72, 6.70, 11.91, 0.32, RGBColor(255, 242, 222), True)
    txt(s, 0.90, 6.75, 11.55, 0.18,
        "채택 조건  |  동일 순차검증에서 19차 CV 정확도 56% 초과 + ±1등급·독립연도 성능 동시 확인",
        9, RED, True, PP_ALIGN.CENTER)
    footer(s)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
