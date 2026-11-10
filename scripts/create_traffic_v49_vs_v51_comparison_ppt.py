from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches

from create_crime_forecast_compact_ppt import (
    BLUE, GRAY, INK, LIGHT, MINT, NAVY, NAVY2, RED, TEAL, WHITE,
    footer, header, shape, txt,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "presentations" / "교통사고_49차_피해자율_vs_51차_안전등급_비교.pptx"
DESKTOP = Path.home() / "Desktop" / OUT.name
MODEL_ROOT = ROOT / "models" / "교통사고"


def panel(slide, x, y, w, h, title, body, fill=BLUE, body_size=11):
    shape(slide, x, y, w, h, fill, True)
    txt(slide, x + .25, y + .24, w - .50, .30, title, 14, NAVY, True)
    txt(slide, x + .25, y + .76, w - .50, h - 1.00, body, body_size, INK)


def pill(slide, x, y, w, text, fill=TEAL):
    shape(slide, x, y, w, .38, fill, True)
    txt(slide, x + .08, y + .09, w - .16, .19, text, 9.5, WHITE, True, PP_ALIGN.CENTER)


def feature_box(slide, x, y, w, h, title, items, fill=MINT, columns=1, font=8.5):
    shape(slide, x, y, w, h, fill, True)
    txt(slide, x + .22, y + .22, w - .44, .27, title, 13, NAVY, True)
    per_col = (len(items) + columns - 1) // columns
    col_w = (w - .44) / columns
    gap = min(.36, (h - .86) / max(per_col, 1))
    for i, item in enumerate(items):
        col, row = i // per_col, i % per_col
        txt(slide, x + .22 + col * col_w, y + .67 + row * gap,
            col_w - .10, .25, f"• {item}", font, INK)


def main():
    info49 = json.loads((MODEL_ROOT / "41차~49차" / "49차" / "model_info.json").read_text(encoding="utf-8"))
    info51 = json.loads((MODEL_ROOT / "51차" / "model_info.json").read_text(encoding="utf-8"))
    weights = pd.read_excel(ROOT / "data" / "processed" / "교통사고_51차_최종모델데이터.xlsx", sheet_name="변수가중치")
    features = info49["최종변수"]

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]

    # 1. Executive summary
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "SPOTFIT · MODEL 49 vs MODEL 51", "피해자율 예측모델과 자치구 5등급 예측모델의 역할 비교", 1)
    panel(s, .72, 1.50, 5.78, 4.72, "49차 · 피해자율 예측",
          "질문\n내년 어린이 교통사고 피해자율은 얼마인가?\n\n출력\n어린이 인구 1만 명당 예상 사망·부상자 수\n\n모델\nRidge α=0.1 · 원본변수 13개\n\n2025 MAE\n4.609명/1만 명", BLUE, 12)
    panel(s, 6.82, 1.50, 5.80, 4.72, "51차 · 점수·5등급 예측",
          "질문\n내년 우리 자치구는 상대적으로 몇 등급인가?\n\n출력\n100점 위험점수 + 1~5등급\n\n모델\nRidge 점수예측 + 선형 SVM 등급분류\n\n정확한 등급 정확도\n56.0%", MINT, 12)
    pill(s, 2.00, 6.55, 3.22, "49차 = 절대적인 사고위험 크기", NAVY)
    pill(s, 8.08, 6.55, 3.22, "51차 = 서울 내 상대적인 위험등급", TEAL)
    footer(s)

    # 2. Relationship
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "HOW THEY CONNECT", "51차는 49차를 버린 새 모델이 아니라, 49차를 등급 체계로 확장한 모델", 2)
    blocks = [
        ("49차 원본 13개 변수", "차량·속도·학교·사고다발·보호구역·자전거", BLUE),
        ("49차 Ridge 계수", "계수 절댓값을 합계 100% 가중치로 변환", MINT),
        ("자치구별 5분위", "각 변수마다 100·80·60·40·20점", BLUE),
        ("51차 종합점수", "가중치 × 분위점수를 모두 합산", MINT),
        ("51차 위험등급", "25개 자치구를 5개씩 1~5등급 배정", RGBColor(205, 235, 220)),
    ]
    for i, (title_, body, fill) in enumerate(blocks):
        x = .38 + i * 2.58
        shape(s, x, 2.12, 2.28, 2.65, fill, True)
        txt(s, x + .18, 2.43, 1.92, .48, title_, 13, NAVY, True, PP_ALIGN.CENTER)
        txt(s, x + .18, 3.35, 1.92, .92, body, 10, INK, True, PP_ALIGN.CENTER)
        if i < 4:
            txt(s, x + 2.28, 3.18, .30, .30, "→", 17, TEAL, True, PP_ALIGN.CENTER)
    shape(s, 1.30, 5.50, 10.73, .82, NAVY2, True)
    txt(s, 1.62, 5.73, 10.09, .34,
        "같은 13개 원천변수를 사용하지만, Y와 학습 목적이 서로 다르다.", 15, WHITE, True, PP_ALIGN.CENTER)
    footer(s)

    # 3. Target comparison
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "TARGET · Y COMPARISON", "두 모델이 맞히려는 정답 자체가 다름", 3)
    panel(s, .72, 1.50, 5.80, 4.95, "49차 Y · 연속형 숫자",
          "다음 연도 어린이 교통사고 피해자율\n\n(다음 연도 어린이 사망자 + 부상자)\n──────────────────────── × 10,000\n다음 연도 0~12세 인구\n\n예시\n22.4명/1만 명, 35.7명/1만 명\n\n평가\nMAE · RMSE · R² · WMAPE", BLUE, 12)
    panel(s, 6.82, 1.50, 5.80, 4.95, "51차 Y · 점수와 범주",
          "다음 연도 자치구 종합위험점수와 위험등급\n\n1등급  상대적으로 가장 위험한 상위 20%\n2등급  다음 20%\n3등급  중간 20%\n4등급  다음 20%\n5등급  상대적으로 가장 낮은 하위 20%\n\n평가\n점수 MAE · 정확한 등급 · ±1등급 정확도", MINT, 11.5)
    footer(s)

    # 4. Inputs
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "INPUT FEATURES", "원천 13개는 같고, 51차는 점수·시간 파생정보를 추가", 4)
    short_features = [
        "어린이1만명당 자동차등록대수", "특수차 등록대수", "고등학교 평균학생수",
        "연간 사고중독 경험률", "전체사고 부상신고자수", "법규위반 부상신고자수",
        "전체사고 발생건수", "보호구역 어린이 사망자수", "보호구역 보행어린이 사고율",
        "보호구역 보행어린이 부상자수", "승용차 전일 평균속도",
        "자전거 사고다발 경상자수", "자전거 사고다발 지점수",
    ]
    feature_box(s, .72, 1.47, 5.72, 4.96, "공통 원천변수 · 13개", short_features, BLUE, 1, 8.7)
    derived = [
        "13개 변수별 5분위 점수", "현재연도 종합위험점수", "현재연도 위험등급",
        "최근 2년 종합점수 평균", "최근 3년 종합점수 평균", "최근 3년 점수 변동성",
        "전년 대비 점수 증감", "전년도 위험등급",
    ]
    feature_box(s, 6.72, 1.47, 5.90, 3.56, "51차에만 추가되는 파생입력", derived, MINT, 1, 10)
    shape(s, 6.72, 5.30, 5.90, 1.13, NAVY2, True)
    txt(s, 7.00, 5.54, 5.34, .55,
        "새 외부데이터를 추가한 것이 아니라\n49차 변수에서 점수와 시간정보를 파생", 12, WHITE, True, PP_ALIGN.CENTER)
    footer(s)

    # 5. Weights
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "MODEL 51 · WEIGHTS", "49차 표준화 계수 절댓값을 합계 100%로 환산", 5)
    for i, row in weights.sort_values("가중치_pct", ascending=False).reset_index(drop=True).iterrows():
        y = 1.43 + i * .40
        name = str(row["변수"]).replace("_합계", "").replace("_어린이1만명당", "")
        name = name[:29]
        value = float(row["가중치_pct"])
        color = TEAL if row["방향"] == "정방향" else RGBColor(75, 122, 156)
        txt(s, .72, y, 3.90, .20, name, 8.5, INK)
        shape(s, 4.75, y + .01, 5.82, .20, LIGHT, True)
        shape(s, 4.75, y + .01, 5.82 * value / 22, .20, color, True)
        txt(s, 10.72, y, .70, .20, f"{value:.2f}%", 8.5, color, True, PP_ALIGN.RIGHT)
        pill(s, 11.60, y - .05, .92, str(row["방향"]), color)
    txt(s, .82, 6.82, 11.68, .23,
        "정방향: 값이 높을수록 고위험 점수  |  역방향: 값이 낮을수록 고위험 점수  |  인과관계 의미는 아님",
        9.5, GRAY, True, PP_ALIGN.CENTER)
    footer(s)

    # 6. Scoring example
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "MODEL 51 · SCORING EXAMPLE", "변수별 상대순위가 최종 점수에 반영되는 방식", 6)
    panel(s, .72, 1.48, 3.48, 4.65, "① 변수 가중치",
          "어린이1만명당\n자동차등록대수\n\n가중치\n21.02%", BLUE, 16)
    txt(s, 4.28, 3.32, .45, .35, "×", 22, TEAL, True, PP_ALIGN.CENTER)
    panel(s, 4.82, 1.48, 3.48, 4.65, "② 자치구 5분위",
          "상위 20%라면\n\n분위점수\n100점\n\n다음 20%는 80점", MINT, 15)
    txt(s, 8.38, 3.32, .45, .35, "=", 22, TEAL, True, PP_ALIGN.CENTER)
    panel(s, 8.92, 1.48, 3.70, 4.65, "③ 가중점수",
          "0.2102 × 100\n\n21.02점\n\n13개 변수의 가중점수를\n모두 더해 종합점수 계산", RGBColor(205, 235, 220), 14)
    pill(s, 3.62, 6.45, 6.12, "최종 종합점수 순위로 자치구를 1~5등급에 5개씩 배정", NAVY)
    footer(s)

    # 7. Model mechanics
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "ALGORITHM COMPARISON", "같은 기반데이터라도 예측문제에 맞춰 알고리즘이 달라짐", 7)
    panel(s, .72, 1.48, 5.78, 4.90, "49차 · Ridge 회귀",
          "입력\n13개 원천변수\n\n알고리즘\nRidge Regression α=0.1\n\n출력\n연속형 피해자율 숫자\n\n장점\n구조가 단순하고 계수 해석 가능", BLUE, 12)
    panel(s, 6.82, 1.48, 5.80, 4.90, "51차 · 2단계 예측",
          "점수 예측\nRidge Regression α=10\n\n등급 예측\nLinear SVM C=10\n\n후처리\n25개 자치구 위험순위 보정\n\n출력\n100점 위험점수 + 정확한 5등급", MINT, 12)
    footer(s)

    # 8. Performance
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "PERFORMANCE", "지표의 단위가 다르므로 숫자를 직접 대소 비교하면 안 됨", 8)
    cards = [
        (.72, "49차 피해자율 MAE", "4.609", "명 / 어린이 1만 명", BLUE),
        (3.78, "49차 WMAPE", "20.32%", "전체 실제값 대비 오차", BLUE),
        (6.84, "51차 정확한 등급", "56.0%", "5등급 완전 일치", MINT),
        (9.90, "51차 ±1등급", "92.8%", "한 단계 이내 일치", MINT),
    ]
    for x, title_, value, sub, fill in cards:
        shape(s, x, 1.72, 2.72, 3.42, fill, True)
        txt(s, x + .20, 2.06, 2.32, .48, title_, 12, NAVY, True, PP_ALIGN.CENTER)
        txt(s, x + .20, 2.92, 2.32, .62, value, 27, TEAL, True, PP_ALIGN.CENTER)
        txt(s, x + .20, 4.16, 2.32, .45, sub, 9.5, GRAY, True, PP_ALIGN.CENTER)
    shape(s, 1.18, 5.58, 10.98, .86, NAVY2, True)
    txt(s, 1.48, 5.78, 10.38, .44,
        "49차는 수치 오차를 평가하고, 51차는 등급 일치율을 평가한다.\n두 모델은 경쟁관계가 아니라 서로 다른 의사결정 질문에 답한다.",
        12.5, WHITE, True, PP_ALIGN.CENTER)
    footer(s)

    # 9. Usage
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "WHEN TO USE WHICH MODEL", "보고서와 서비스 화면에서 두 결과를 함께 사용하는 것이 가장 유용", 9)
    panel(s, .72, 1.48, 5.78, 4.82, "49차가 필요한 질문",
          "• 내년 피해자율이 실제로 얼마나 될까?\n\n• 전년보다 피해규모가 증가할까?\n\n• 정책 시행 전후 수치가 얼마나 변했나?\n\n• 자치구별 예상 피해자 수치를 제공해야 하나?", BLUE, 12)
    panel(s, 6.82, 1.48, 5.80, 4.82, "51차가 필요한 질문",
          "• 서울에서 상대적으로 위험한 자치구는 어디인가?\n\n• 예산과 관리 우선순위를 어떻게 정할까?\n\n• 지도에 1~5등급을 어떻게 표시할까?\n\n• 비전문가에게 결과를 쉽게 설명해야 하나?", MINT, 12)
    pill(s, 2.48, 6.55, 8.40, "권장 화면: 예상 피해자율  +  종합위험점수  +  5등급을 함께 표시", TEAL)
    footer(s)

    # 10. Final takeaway
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "FINAL TAKEAWAY", "49차와 51차의 관계를 한 문장으로 정리", 10)
    shape(s, .78, 1.60, 11.76, 1.22, NAVY2, True)
    txt(s, 1.15, 1.92, 11.02, .56,
        "49차는 ‘얼마나 발생할지’를 예측하고, 51차는 ‘서울에서 어느 위험등급인지’를 예측한다.",
        18, WHITE, True, PP_ALIGN.CENTER)
    panel(s, .78, 3.25, 3.62, 2.70, "49차",
          "절대 위험수치\n피해자율 예측\nRidge 회귀", BLUE, 14)
    panel(s, 4.86, 3.25, 3.62, 2.70, "연결고리",
          "13개 공통변수\nRidge 계수 가중치\n자치구별 5분위", WHITE, 14)
    panel(s, 8.92, 3.25, 3.62, 2.70, "51차",
          "상대 위험등급\n점수·등급 예측\nRidge + SVM", MINT, 14)
    pill(s, 3.18, 6.48, 6.98, "최종 권고 · 두 모델을 함께 제공하면 해석력과 활용성이 가장 높음", TEAL)
    footer(s)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    try:
        shutil.copy2(OUT, DESKTOP)
        print(DESKTOP)
    except PermissionError:
        print(f"DESKTOP_COPY_FAILED:{DESKTOP}")
    print(OUT)


if __name__ == "__main__":
    main()
