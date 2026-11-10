from __future__ import annotations

import json
import shutil
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
OUT = ROOT / "artifacts" / "presentations" / "교통사고_41차부터_49차_모델종합비교.pptx"
DESKTOP = Path.home() / "Desktop" / OUT.name
VERSIONS = list(range(41, 50))


def load(v):
    return json.loads((ROOT / "models" / "교통사고" / "41차~49차" / f"{v}차" / "model_info.json").read_text(encoding="utf-8"))


def features(info):
    return info.get("최종변수", [])


def count(info):
    return info.get("최종변수수", len(features(info)))


def metric(info, key):
    return info["2025_독립검증"][key]


def pill(slide, x, y, w, text, fill=TEAL, color=WHITE, size=9):
    shape(slide, x, y, w, .34, fill, True)
    txt(slide, x + .08, y + .07, w - .16, .18, text, size, color, True, PP_ALIGN.CENTER)


def panel(slide, x, y, w, h, title, body, fill=BLUE, title_color=NAVY, body_size=11):
    shape(slide, x, y, w, h, fill, True)
    txt(slide, x + .25, y + .24, w - .5, .30, title, 14, title_color, True)
    txt(slide, x + .25, y + .75, w - .5, h - .98, body, body_size, INK)


def table(slide, x, y, widths, headers, rows, row_h=.43, font=9, highlight=None):
    xx = x
    for w, text_ in zip(widths, headers):
        shape(slide, xx, y, w, row_h, NAVY, False)
        txt(slide, xx + .04, y + .10, w - .08, .20, text_, font, WHITE, True, PP_ALIGN.CENTER)
        xx += w
    for r, values in enumerate(rows):
        yy = y + (r + 1) * row_h
        fill = MINT if highlight is not None and r == highlight else (WHITE if r % 2 == 0 else LIGHT)
        xx = x
        for w, value in zip(widths, values):
            shape(slide, xx, yy, w, row_h, fill, False, RGBColor(220, 228, 224))
            txt(slide, xx + .04, yy + .10, w - .08, .20, str(value), font, NAVY if fill == MINT else INK,
                fill == MINT, PP_ALIGN.CENTER)
            xx += w


def feature_box(slide, x, y, w, h, title, items, fill=MINT, columns=1, font=8.5):
    shape(slide, x, y, w, h, fill, True)
    txt(slide, x + .22, y + .22, w - .44, .28, title, 13, NAVY, True)
    per_col = (len(items) + columns - 1) // columns
    cw = (w - .44) / columns
    gap = min(.36, (h - .86) / max(per_col, 1))
    for i, item in enumerate(items):
        col, row = i // per_col, i % per_col
        txt(slide, x + .22 + col * cw, y + .68 + row * gap, cw - .10, .25, f"• {item}", font, INK)


def main():
    infos = {v: load(v) for v in VERSIONS}
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]

    # 1. Cover / conclusion
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "SPOTFIT · CHILD TRAFFIC MODEL", "41차부터 49차까지: 확장, 과적합 점검, 경량화, 최종 선정", 1)
    shape(s, .78, 1.55, 7.25, 4.78, NAVY2, True)
    txt(s, 1.12, 1.98, 6.55, .38, "결론부터 보면", 16, TEAL, True)
    txt(s, 1.12, 2.65, 6.55, 1.05, "49차 · 13개 변수", 31, WHITE, True)
    txt(s, 1.12, 3.83, 6.55, 1.28,
        "2025 MAE  4.609\n46차 대비 변수 59% 축소 · MAE 5.22% 개선",
        18, RGBColor(218, 232, 236), True)
    txt(s, 1.12, 5.48, 6.55, .38, "성능과 경량성을 동시에 잡은 현재 최종안", 12, TEAL, True)
    cards = [
        ("41차", "7개", "MAE 5.081", BLUE),
        ("46차", "32개", "MAE 4.863", MINT),
        ("49차", "13개", "MAE 4.609", RGBColor(204, 235, 220)),
    ]
    for i, (name, n, mae, fill) in enumerate(cards):
        y = 1.55 + i * 1.62
        shape(s, 8.42, y, 4.10, 1.34, fill, True)
        txt(s, 8.68, y + .23, 1.10, .28, name, 15, NAVY, True)
        txt(s, 9.86, y + .23, 2.35, .28, n, 12, TEAL, True, PP_ALIGN.RIGHT)
        txt(s, 8.68, y + .75, 3.53, .26, mae, 13, INK, True)
    footer(s)

    # 2. Problem definition
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "WHAT ARE WE PREDICTING?", "모든 차수는 같은 Y와 같은 2025 평가 기준으로 비교", 2)
    panel(s, .72, 1.48, 5.80, 2.10, "Y · 다음 연도 어린이 교통사고 피해자율",
          "(다음 연도 어린이 교통사고 사망자 수 + 부상자 수)\n÷ 다음 연도 0~12세 인구 × 10,000", MINT, body_size=15)
    panel(s, 6.82, 1.48, 5.80, 2.10, "데이터 단위",
          "서울 25개 자치구 × 2017~2025년\nX는 현재연도, Y는 다음 연도 값", BLUE, body_size=15)
    steps = [("①", "과거연도 학습"), ("②", "2021~2024 순차검증"), ("③", "2025 성능 비교"), ("④", "2026 예측")]
    for i, (num, label) in enumerate(steps):
        x = .84 + i * 3.08
        shape(s, x, 4.25, 2.72, 1.45, WHITE, True, RGBColor(215, 225, 222))
        txt(s, x + .18, 4.52, .50, .32, num, 17, TEAL, True, PP_ALIGN.CENTER)
        txt(s, x + .75, 4.54, 1.72, .28, label, 12, NAVY, True, PP_ALIGN.CENTER)
        if i < 3:
            txt(s, x + 2.72, 4.72, .36, .25, "→", 16, GRAY, True, PP_ALIGN.CENTER)
    shape(s, .84, 6.08, 11.80, .56, NAVY2, True)
    txt(s, 1.10, 6.23, 11.28, .25, "주 평가지표 MAE: 자치구 예측이 실제 피해자율에서 평균적으로 얼마나 벗어났는지", 12, WHITE, True, PP_ALIGN.CENTER)
    footer(s)

    # 3. Journey timeline
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "MODEL JOURNEY", "좋은 변수를 찾기 위해 넓혔다가, 중복과 과적합을 줄이며 다시 압축", 3)
    labels = {
        41: "핵심 7개", 42: "38개 후보 탐색", 43: "42 전용 전수검사", 44: "대표 4개 추가",
        45: "교체 탐색", 46: "전진 추가", 47: "중복 제거", 48: "추가 경량화", 49: "전수 후진제거",
    }
    for i, v in enumerate(VERSIONS):
        x = .55 + i * 1.40
        y = 2.02 if i % 2 == 0 else 4.02
        fill = RGBColor(202, 236, 220) if v == 49 else (MINT if v >= 46 else BLUE)
        shape(s, x, y, 1.22, 1.22, fill, True)
        txt(s, x + .10, y + .18, 1.02, .28, f"{v}차", 13, NAVY, True, PP_ALIGN.CENTER)
        txt(s, x + .10, y + .58, 1.02, .22, f"{count(infos[v])}개", 11, TEAL, True, PP_ALIGN.CENTER)
        txt(s, x - .10, y + 1.38, 1.42, .54, labels[v], 8.5, INK, True, PP_ALIGN.CENTER)
        if i < 8:
            yy = 3.47
            txt(s, x + 1.20, yy, .22, .22, "→", 13, GRAY, True, PP_ALIGN.CENTER)
    pill(s, .78, 6.43, 3.30, "41~45 · 핵심 조합 탐색", NAVY)
    pill(s, 4.35, 6.43, 3.30, "46 · 최대 확장", RGBColor(75, 122, 156))
    pill(s, 7.92, 6.43, 4.55, "47~49 · 중복 제거 및 경량화", TEAL)
    footer(s)

    # 4. Performance table
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "ALL MODELS · 2025 RESULT", "동일한 2025년 25개 자치구 기준 · MAE/RMSE/WMAPE는 낮을수록, R²는 높을수록 좋음", 4)
    rows = []
    for v in VERSIONS:
        rows.append([f"{v}차", count(infos[v]), f"{metric(infos[v], 'MAE'):.3f}", f"{metric(infos[v], 'RMSE'):.3f}",
                     f"{metric(infos[v], 'R2'):.3f}", f"{metric(infos[v], 'WMAPE_pct'):.2f}%"])
    table(s, 1.15, 1.48, [1.35, 1.35, 1.70, 1.70, 1.70, 1.90],
          ["모델", "변수 수", "MAE ↓", "RMSE ↓", "R² ↑", "WMAPE ↓"], rows, .49, 10, 8)
    shape(s, 1.15, 6.67, 10.55, .38, NAVY2, True)
    txt(s, 1.32, 6.76, 10.21, .19, "49차는 네 가지 성능지표 모두 41~49차 중 1위", 10, WHITE, True, PP_ALIGN.CENTER)
    footer(s)

    # 5. MAE and feature bars
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "PERFORMANCE vs COMPLEXITY", "변수 수가 많다고 성능이 자동으로 좋아지지는 않음", 5)
    txt(s, .82, 1.44, 5.70, .28, "2025 MAE · 짧을수록 좋음", 13, NAVY, True)
    txt(s, 7.05, 1.44, 5.45, .28, "변수 수 · 모델 복잡도", 13, NAVY, True)
    for i, v in enumerate(VERSIONS):
        y = 1.96 + i * .52
        mae = metric(infos[v], "MAE")
        n = count(infos[v])
        color = TEAL if v == 49 else (RGBColor(75, 122, 156) if v >= 46 else GRAY)
        txt(s, .82, y, .62, .21, f"{v}차", 9, INK, True)
        shape(s, 1.48, y + .02, 4.30, .22, LIGHT, True)
        shape(s, 1.48, y + .02, 4.30 * mae / 6.0, .22, color, True)
        txt(s, 5.88, y, .58, .22, f"{mae:.3f}", 9, color, True, PP_ALIGN.RIGHT)
        txt(s, 7.05, y, .62, .21, f"{v}차", 9, INK, True)
        shape(s, 7.71, y + .02, 4.30, .22, LIGHT, True)
        shape(s, 7.71, y + .02, 4.30 * n / 32, .22, color, True)
        txt(s, 12.10, y, .42, .22, str(n), 9, color, True, PP_ALIGN.RIGHT)
    txt(s, .90, 6.74, 11.55, .28, "46차는 32개로 확장했지만 49차는 13개만으로 더 정확해짐 → 변수 선택이 핵심", 12, NAVY, True, PP_ALIGN.CENTER)
    footer(s)

    # 6. Models 41-43
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "PHASE 1 · 41–43", "핵심모델과 대규모 부분집합 실험: 더 많은 변수가 항상 답은 아니었다", 6)
    panel(s, .65, 1.45, 3.85, 4.90, "41차 · 7개 핵심변수",
          "현재 피해자율·최근 3년 평균/변동성\n보행어린이 사고율·사고다발 지점\n도로율·도로면적당 CCTV\n\nRidge α=0.1\n2025 MAE 5.081", MINT, body_size=11)
    panel(s, 4.74, 1.45, 3.85, 4.90, "42차 · 17개",
          "38개 후보에서 51,794개 고유 조합 탐색\n학교·차량·사고다발·자전거 지표 확대\n\n2025 MAE 5.337\n41차보다 성능 하락", BLUE, body_size=11)
    panel(s, 8.83, 1.45, 3.85, 4.90, "43차 · 14개",
          "42차 전용 15개 변수의\n부분집합 32,767개 전수검사\n\n2025 MAE 5.703\n복잡한 조합도 일반화 실패", BLUE, body_size=11)
    pill(s, 3.86, 6.58, 5.60, "교훈 · 변수 개수보다 독립연도 일반화 성능이 중요", NAVY)
    footer(s)

    # 7. Models 44-45
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "PHASE 2 · 44–45", "41차 핵심을 유지하면서 대표 변수와 교체 조합을 정교하게 탐색", 7)
    feature_box(s, .72, 1.48, 5.70, 4.78, "44차 · 핵심 7 + 대표 4 = 11개", features(infos[44]), BLUE, 1, 9)
    f44, f45 = set(features(infos[44])), set(features(infos[45]))
    body = (
        "제거\n• " + "\n".join(sorted(f44 - f45)) +
        "\n\n추가\n• " + "\n".join(sorted(f45 - f44)) +
        "\n\n341개 추가·제거·교체 조합 평가\nRidge α=100"
    )
    panel(s, 6.74, 1.48, 5.88, 4.78, "45차 · 같은 11개, 더 좋은 구성", body, MINT, body_size=11)
    pill(s, 2.03, 6.48, 3.10, "44차 MAE 5.047", GRAY)
    pill(s, 8.06, 6.48, 3.10, "45차 MAE 4.924", TEAL)
    footer(s)

    # 8. Model 46 expansion
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "PHASE 3 · MODEL 46", "45차에서 검증 MAE가 좋아지는 변수를 하나씩 추가", 8)
    added46 = infos[46].get("단계별추가", [])
    feature_box(s, .72, 1.46, 3.68, 4.95, "출발 · 45차 11개", features(infos[45]), BLUE, 1, 8.3)
    txt(s, 4.48, 3.35, .55, .42, "+21", 20, TEAL, True, PP_ALIGN.CENTER)
    feature_box(s, 5.12, 1.46, 7.50, 4.95, "검증 개선으로 추가된 21개", added46, MINT, 2, 8.1)
    pill(s, 1.23, 6.62, 2.66, "변수 11 → 32", NAVY)
    pill(s, 5.66, 6.62, 2.66, "MAE 4.924 → 4.863", TEAL)
    pill(s, 9.64, 6.62, 2.66, "다중공선성 증가", RED)
    footer(s)

    # 9. Models 47-48
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "PHASE 4 · 47–48", "46차의 중복 변수를 우선 제거하고 성능을 다시 확인", 9)
    removed47 = infos[47].get("제거변수", [])
    removed48 = infos[48].get("제거변수", [])
    panel(s, .72, 1.48, 5.72, 4.78, "47차 · 32 → 29개",
          "제거한 변수\n• " + "\n• ".join(removed47) +
          "\n\n결과\nMAE 4.856 · RMSE 8.284\nR² 0.3725 · WMAPE 21.413%", MINT, body_size=11)
    panel(s, 6.72, 1.48, 5.90, 4.78, "48차 · 29 → 28개",
          "추가 제거\n• " + "\n• ".join(removed48) +
          "\n\n결과\nMAE 4.858 · RMSE 8.283\nR² 0.3726 · WMAPE 21.424%\n\n성능은 사실상 47차와 동일", BLUE, body_size=11)
    pill(s, 3.92, 6.50, 5.50, "중복 제거로 변수는 줄고, 성능은 유지 또는 소폭 개선", NAVY)
    footer(s)

    # 10. Model 49 elimination
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "PHASE 5 · MODEL 49", "48차의 모든 변수를 대상으로 매 단계 단일 제거 조합을 전수 비교", 10)
    removed49 = infos[49].get("제거변수", [])
    feature_box(s, .72, 1.46, 7.65, 4.96, "제거된 15개 변수", removed49, BLUE, 2, 8.3)
    panel(s, 8.65, 1.46, 3.98, 4.96, "선택 규칙",
          "1. 남은 모든 변수에서\n   하나씩 제거해 전수 평가\n\n2. 2021~2024 순차검증\n   MAE 기준으로 최선 선택\n\n3. 48차 CV MAE 대비\n   1% 범위 내에서 경량화\n\n4. 다음 제거가 범위를\n   넘으면 즉시 종료", MINT, body_size=11)
    pill(s, 1.40, 6.62, 2.28, "329개 조합", NAVY)
    pill(s, 5.45, 6.62, 2.28, "28 → 13개", TEAL)
    pill(s, 9.52, 6.62, 2.28, "MAE 5.13% 개선", TEAL)
    footer(s)

    # 11. Final 13 features
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "MODEL 49 · FINAL 13 FEATURES", "최종 모델에 남은 변수와 의미", 11)
    cats = [
        ("차량·속도", ["어린이1만명당 자동차등록대수", "특수차 등록대수", "승용차 전일 평균속도"]),
        ("학교·생활", ["고등학교 평균학생수", "연간 사고중독 경험률"]),
        ("전체·법규위반 사고다발", ["전체사고다발 부상신고자수", "법규위반다발 부상신고자수", "전체사고다발 발생건수"]),
        ("어린이 보호구역·보행", ["보호구역 어린이다발 사망자수", "보호구역 보행어린이 사고율", "보호구역 내 보행어린이 부상자수"]),
        ("자전거", ["자전거다발 경상자수", "자전거다발 지점수"]),
    ]
    positions = [(.72, 1.48, 3.78, 2.10), (4.78, 1.48, 3.78, 2.10), (8.84, 1.48, 3.78, 2.10),
                 (1.80, 4.03, 4.78, 2.15), (6.86, 4.03, 4.78, 2.15)]
    for idx, ((title_, items), (x, y, w, h)) in enumerate(zip(cats, positions)):
        feature_box(s, x, y, w, h, title_, items, MINT if idx in (0, 3) else BLUE, 1, 9.2)
    pill(s, 4.25, 6.56, 4.82, "Ridge α=0.1 · 표준화 후 학습", NAVY)
    footer(s)

    # 12. Feature survival matrix for final features
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "FEATURE SURVIVAL", "49차 최종 13개가 어느 차수부터 등장하고 살아남았는지", 12)
    final = features(infos[49])
    short = {
        "어린이1만명당_자동차등록대수": "어린이1만명당 자동차",
        "특수차 등록대수": "특수차",
        "고등학교평균학생수": "고교 평균학생수",
        "연간사고중독경험률(%)": "사고중독 경험률",
        "전체사고다발_부상신고자수합계": "전체사고 부상신고",
        "법규위반다발_부상신고자수합계": "법규위반 부상신고",
        "전체사고다발_발생건수합계": "전체사고 발생건수",
        "보호구역어린이다발_사망자수합계": "보호구역 사망자",
        "보호구역보행어린이교통사고율_어린이1만명당": "보호구역 보행사고율",
        "어린이 보호구역 내 보행어린이 교통사고 부상자 수": "보호구역 보행부상",
        "승용차_전일_평균속도(km/h)": "승용차 전일속도",
        "자전거다발_경상자수합계": "자전거 경상자",
        "자전거다발_지점수": "자전거 지점수",
    }
    headers = ["최종 변수"] + [f"{v}" for v in VERSIONS]
    widths = [3.25] + [.82] * 9
    rows = []
    for f in final:
        rows.append([short.get(f, f)] + [("●" if f in features(infos[v]) else "–") for v in VERSIONS])
    table(s, 1.26, 1.34, widths, headers, rows, .39, 8.2)
    txt(s, 1.30, 6.98, 10.70, .20, "● 포함  |  – 미포함   · 49차는 46차 확장 후보 중 다음 연도 예측에 기여한 변수만 압축", 9, GRAY, True, PP_ALIGN.CENTER)
    footer(s)

    # 13. Final decision and caveats
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "FINAL DECISION", "49차를 현재 고정 모델로 선택하되, 운영 시 지켜야 할 기준", 13)
    panel(s, .72, 1.48, 5.72, 4.92, "왜 49차인가?",
          "• 41~49차 중 MAE 최저: 4.609\n\n• RMSE 최저: 8.060\n\n• R² 최고: 0.406\n\n• WMAPE 최저: 20.324%\n\n• 46차 32개 대비 13개로 축소\n\n• 해석·관리·재학습 부담 감소", MINT, body_size=12)
    panel(s, 6.72, 1.48, 5.90, 4.92, "주의할 점",
          "• 2025년을 여러 차례 확인했으므로\n  완전한 미사용 독립검증으로 보긴 어려움\n\n• 자전거 경상자수와 지점수는\n  상관성이 높지만 제거 시 CV 성능 저하\n\n• 신규 2026 실제값이 확보되면\n  반드시 재평가 후 최종 운영 확정\n\n• 개별 계수는 인과관계가 아니라\n  예측모델 내부의 관계로 해석", BLUE, body_size=11.5)
    shape(s, 2.60, 6.63, 8.15, .43, TEAL, True)
    txt(s, 2.84, 6.74, 7.67, .22, "현재 픽스: 49차 · 13개 변수 · Ridge α=0.1", 12, WHITE, True, PP_ALIGN.CENTER)
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
