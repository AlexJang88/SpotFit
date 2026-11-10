from __future__ import annotations

import json
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
OUT = ROOT / "artifacts" / "presentations" / "아동초기모델_41차_42차_43차_변수및성능비교.pptx"


def load_info(version: int) -> dict:
    return json.loads(
        (ROOT / "models" / "교통사고" / f"{version}차" / "model_info.json").read_text(encoding="utf-8")
    )


def feature_panel(slide, x, y, w, h, title, subtitle, features, fill):
    shape(slide, x, y, w, h, fill, True)
    txt(slide, x + .25, y + .24, w - .5, .27, title, 14, NAVY, True)
    txt(slide, x + .25, y + .59, w - .5, .22, subtitle, 9, TEAL, True)
    start = y + .98
    gap = min(.40, (h - 1.20) / max(len(features), 1))
    for i, feature in enumerate(features):
        yy = start + i * gap
        shape(slide, x + .27, yy + .04, .18, .18, TEAL, True)
        txt(slide, x + .55, yy, w - .82, .27, feature, 9, INK)


def performance_bar(slide, y, name, value, color, tag=""):
    txt(slide, .98, y, 2.35, .28, name, 11, INK, True)
    shape(slide, 3.34, y + .02, 4.40, .26, LIGHT, True)
    shape(slide, 3.34, y + .02, 4.40 * value / 6.2, .26, color, True)
    txt(slide, 7.90, y - .01, .78, .30, f"{value:.3f}", 11, color, True)
    if tag:
        shape(slide, 8.80, y - .02, 1.30, .33, color, True)
        txt(slide, 8.88, y + .04, 1.14, .18, tag, 9, WHITE, True, PP_ALIGN.CENTER)


def main():
    info41, info42, info43 = load_info(41), load_info(42), load_info(43)
    f41 = info41["최종변수"]
    f42 = info42["최종변수"]
    f43 = info43["최종변수"]

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # 1. Executive summary
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "SPOTFIT · CHILD TRAFFIC MODEL", "초기모델부터 43차까지, 변수 구성과 독립검증 성능 비교", 1)
    cards = [
        ("아동 초기모델", "1개 기준값", "2023년 피해자율 고정", "5.609", BLUE),
        ("41차", "7개 변수", "Ridge · alpha 0.1", "5.081", MINT),
        ("42차", "17개 변수", "Ridge · alpha 0.1", "5.337", BLUE),
        ("43차", "14개 변수", "42차 전용변수 실험", "5.703", MINT),
    ]
    for i, (name, count, model, mae, fill) in enumerate(cards):
        x = .72 + i * 3.00
        shape(s, x, 1.62, 2.72, 3.35, fill, True)
        txt(s, x + .24, 1.91, 2.24, .32, name, 16, NAVY, True, PP_ALIGN.CENTER)
        txt(s, x + .24, 2.47, 2.24, .28, count, 11, TEAL, True, PP_ALIGN.CENTER)
        txt(s, x + .24, 2.92, 2.24, .25, model, 9, GRAY, True, PP_ALIGN.CENTER)
        txt(s, x + .24, 3.45, 2.24, .22, "2025 독립검증 MAE", 9, GRAY, True, PP_ALIGN.CENTER)
        txt(s, x + .24, 3.82, 2.24, .52, mae, 27, NAVY, True, PP_ALIGN.CENTER)
        if name == "41차":
            shape(s, x + .62, 4.51, 1.48, .31, TEAL, True)
            txt(s, x + .70, 4.57, 1.32, .17, "현재 CHAMPION", 9, WHITE, True, PP_ALIGN.CENTER)
    shape(s, .72, 5.30, 11.91, 1.14, NAVY2, True)
    txt(s, 1.02, 5.57, 2.05, .28, "핵심 결론", 12, TEAL, True)
    txt(s, 3.00, 5.50, 9.10, .58,
        "변수를 많이 넣을수록 독립연도 성능이 좋아지지는 않았다.\n2025년 기준 가장 단순하면서 정확한 모델은 41차 7개 변수 조합이다.",
        15, WHITE, True)
    footer(s)

    # 2. Initial + 41
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "BASELINE vs MODEL 41", "고정값 기준에서 7개 핵심 요인을 학습하는 모델로 전환", 2)
    feature_panel(
        s, .72, 1.52, 3.45, 4.95, "아동 초기모델", "학습 없음 · 2023년 값을 2024·2025년에 고정",
        ["2023년 어린이 1만 명당\n교통사고 피해자 수"], BLUE,
    )
    txt(s, 4.38, 3.48, .55, .48, "→", 24, TEAL, True, PP_ALIGN.CENTER)
    feature_panel(
        s, 5.10, 1.52, 7.53, 4.95, "41차 최종 7개 변수", "2024년 변수선택 · 2025년 독립검증",
        [
            "현재연도 어린이 교통사고 피해자율",
            "어린이 피해자율 최근 3년 평균",
            "어린이 피해자율 최근 3년 변동성",
            "보행어린이 교통사고율",
            "보행어린이 사고다발지역 수",
            "도로율",
            "도로면적당 전체 CCTV",
        ], MINT,
    )
    shape(s, .98, 5.62, 2.92, .53, NAVY, True)
    txt(s, 1.14, 5.75, 2.60, .23, "2025 MAE  5.609", 12, WHITE, True, PP_ALIGN.CENTER)
    shape(s, 9.16, 5.62, 3.16, .53, TEAL, True)
    txt(s, 9.34, 5.75, 2.80, .23, "2025 MAE  5.081 · 9.4% 개선", 12, WHITE, True, PP_ALIGN.CENTER)
    footer(s)

    # 3. 42 features
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "MODEL 42 · 17 FEATURES", "38개 후보를 여러 연도에서 탐색한 challenger 모델", 3)
    left42 = [
        "어린이 피해자율 최근 3년 평균",
        "보행어린이 교통사고율",
        "어린이 교통사고 부상자 수",
        "어린이 부상률",
        "특수차 등록대수",
        "도로연장",
        "사립초등학교 수",
        "중학교 학급 수",
        "공립중학교 수",
    ]
    right42 = [
        "음주운전 차량 동승 경험률",
        "전체 사고다발 발생건수",
        "전체 사고다발 경상자 수",
        "전체 사고다발 부상신고자 수",
        "보호구역 어린이다발 경상자 수",
        "자전거 사고다발 지점 수",
        "자전거 사고다발 사망자 수",
        "자전거 사고다발 경상자 수",
    ]
    feature_panel(s, .72, 1.50, 5.77, 4.95, "사고·인구·도로·학교", "9개 변수", left42, MINT)
    feature_panel(s, 6.78, 1.50, 5.85, 4.95, "행태·사고다발 세부지표", "8개 변수", right42, BLUE)
    shape(s, 4.32, 5.78, 4.70, .49, NAVY2, True)
    txt(s, 4.52, 5.90, 4.30, .22, "2025 MAE 5.337 · 초기모델 대비 개선, 41차보다 열세", 11, WHITE, True, PP_ALIGN.CENTER)
    footer(s)

    # 4. 43 features
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "MODEL 43 · 14 FEATURES", "42차 전용 15개 컬럼의 모든 부분집합 32,767개 전수검사", 4)
    left43 = [
        "어린이 교통사고 부상자 수",
        "어린이 부상률",
        "특수차 등록대수",
        "도로연장",
        "사립초등학교 수",
        "중학교 학급 수",
        "공립중학교 수",
    ]
    right43 = [
        "음주운전 차량 동승 경험률",
        "전체 사고다발 발생건수",
        "전체 사고다발 부상신고자 수",
        "보호구역 어린이다발 경상자 수",
        "자전거 사고다발 지점 수",
        "자전거 사고다발 사망자 수",
        "자전거 사고다발 경상자 수",
    ]
    feature_panel(s, .72, 1.50, 5.77, 4.58, "사고·교통·도로·학교", "7개 변수", left43, MINT)
    feature_panel(s, 6.78, 1.50, 5.85, 4.58, "행태·사고다발 세부지표", "7개 변수", right43, BLUE)
    shape(s, .72, 6.27, 11.91, .22, LIGHT, True)
    txt(s, 1.00, 6.12, 11.35, .28,
        "42차 전용 15개 중 제외된 변수: 전체사고다발 경상자 수 합계  |  2025 MAE 5.703 · 초기모델보다 1.7% 악화",
        10, RED, True, PP_ALIGN.CENTER)
    footer(s)

    # 5. Performance
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "2025 INDEPENDENT HOLDOUT", "동일한 25개 자치구의 2025년 실제값으로 최종 비교", 5)
    shape(s, .72, 1.52, 9.62, 4.90, WHITE, True, RGBColor(220, 228, 224))
    txt(s, 1.00, 1.84, 8.95, .28, "MAE 비교 · 낮을수록 좋음", 14, NAVY, True)
    performance_bar(s, 2.48, "아동 초기모델", 5.609, GRAY, "기준")
    performance_bar(s, 3.20, "41차 · 7개 변수", 5.081, TEAL, "CHAMPION")
    performance_bar(s, 3.92, "42차 · 17개 변수", 5.337, RGBColor(75, 122, 156), "CHALLENGER")
    performance_bar(s, 4.64, "43차 · 14개 변수", 5.703, RED, "탈락")
    txt(s, 1.00, 5.52, 8.88, .54,
        "41차는 초기모델보다 MAE 9.4% 개선 · 42차는 4.8% 개선 · 43차는 1.7% 악화",
        12, NAVY, True, PP_ALIGN.CENTER)
    shape(s, 10.62, 1.52, 2.01, 4.90, NAVY, True)
    txt(s, 10.87, 1.86, 1.51, .28, "최종 판정", 12, TEAL, True, PP_ALIGN.CENTER)
    txt(s, 10.87, 2.48, 1.51, .82, "41차\n유지", 24, WHITE, True, PP_ALIGN.CENTER)
    txt(s, 10.87, 3.65, 1.51, 1.62,
        "변수 수가 적고\n독립검증 MAE가\n가장 낮음\n\n42·43차는\n실험 후보 보존",
        11, RGBColor(216, 227, 235), True, PP_ALIGN.CENTER)
    footer(s)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
