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
OUT = ROOT / "artifacts" / "presentations" / "교통사고_45차_vs_46차_성능비교.pptx"
DESKTOP = Path.home() / "Desktop" / OUT.name


def load(version: int) -> dict:
    path = ROOT / "models" / "교통사고" / "41차~49차" / f"{version}차" / "model_info.json"
    return json.loads(path.read_text(encoding="utf-8"))


def metric_card(slide, x, title, v45, v46, lower=True):
    better46 = v46 < v45 if lower else v46 > v45
    color = TEAL if better46 else NAVY
    shape(slide, x, 2.05, 2.68, 3.30, MINT if better46 else BLUE, True)
    txt(slide, x + .20, 2.33, 2.28, .30, title, 14, NAVY, True, PP_ALIGN.CENTER)
    txt(slide, x + .25, 3.00, .92, .24, "45차", 10, GRAY, True, PP_ALIGN.CENTER)
    txt(slide, x + 1.50, 3.00, .92, .24, "46차", 10, GRAY, True, PP_ALIGN.CENTER)
    txt(slide, x + .18, 3.43, 1.06, .48, f"{v45:.3f}", 20, NAVY, True, PP_ALIGN.CENTER)
    txt(slide, x + 1.43, 3.43, 1.06, .48, f"{v46:.3f}", 20, color, True, PP_ALIGN.CENTER)
    winner = "46차 우수" if better46 else "45차 우수"
    shape(slide, x + .63, 4.45, 1.42, .38, color, True)
    txt(slide, x + .72, 4.54, 1.24, .18, winner, 9, WHITE, True, PP_ALIGN.CENTER)


def feature_list(slide, x, y, w, h, title, features, fill, columns=1):
    shape(slide, x, y, w, h, fill, True)
    txt(slide, x + .25, y + .24, w - .5, .28, title, 14, NAVY, True)
    per_col = (len(features) + columns - 1) // columns
    col_w = (w - .5) / columns
    gap = min(.36, (h - .95) / max(per_col, 1))
    for i, feature in enumerate(features):
        col, row = i // per_col, i % per_col
        xx, yy = x + .25 + col * col_w, y + .78 + row * gap
        txt(slide, xx, yy, col_w - .12, .25, f"• {feature}", 8.3, INK)


def main():
    i45, i46 = load(45), load(46)
    f45, f46 = i45["최종변수"], i46["최종변수"]
    added = [f for f in f46 if f not in f45]
    cv45, cv46 = i45["2021_2024_순차검증"], i46["2021_2024_순차검증"]
    t45, t46 = i45["2025_독립검증"], i46["2025_독립검증"]

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]

    # 1. Summary
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "SPOTFIT · MODEL 45 vs 46", "어린이 교통사고 피해자율 예측모델 성능 및 변수 비교", 1)
    for x, name, count, alpha, mae, fill in [
        (.82, "45차", 11, i45["alpha"], t45["MAE"], BLUE),
        (6.88, "46차", 32, i46["alpha"], t46["MAE"], MINT),
    ]:
        shape(s, x, 1.62, 5.62, 3.72, fill, True)
        txt(s, x + .30, 1.94, 5.02, .38, name, 20, NAVY, True, PP_ALIGN.CENTER)
        txt(s, x + .30, 2.52, 5.02, .28, f"Ridge α={alpha:g} · 변수 {count}개", 11, GRAY, True, PP_ALIGN.CENTER)
        txt(s, x + .30, 3.16, 5.02, .24, "2025 MAE", 10, GRAY, True, PP_ALIGN.CENTER)
        txt(s, x + .30, 3.55, 5.02, .58, f"{mae:.3f}", 29, TEAL if name == "46차" else NAVY, True, PP_ALIGN.CENTER)
        txt(s, x + .30, 4.48, 5.02, .26,
            "단순성·큰 오차 안정성" if name == "45차" else "평균오차·비율오차 우수",
            11, NAVY, True, PP_ALIGN.CENTER)
    shape(s, .82, 5.67, 11.68, .77, NAVY2, True)
    txt(s, 1.10, 5.89, 11.12, .31,
        "MAE 기준 46차가 1.25% 개선 · 단, RMSE와 R²는 45차가 소폭 우수",
        15, WHITE, True, PP_ALIGN.CENTER)
    footer(s)

    # 2. Design
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "MODEL DESIGN", "45차의 핵심 조합에서 검증 성능이 좋아지는 변수만 순차 추가", 2)
    feature_list(s, .72, 1.50, 5.25, 4.95, "45차 · 핵심 11개 변수", f45, BLUE)
    txt(s, 6.08, 3.30, .82, .56, "+21", 23, TEAL, True, PP_ALIGN.CENTER)
    feature_list(s, 7.05, 1.50, 5.58, 4.95, "46차 · 총 32개 변수", f46, MINT, 2)
    txt(s, 1.00, 6.17, 11.33, .25,
        "탐색: 2,278개 조합 · 2021~2024 순차검증 · 개선이 멈추는 시점에서 종료",
        10, NAVY, True, PP_ALIGN.CENTER)
    footer(s)

    # 3. Added variables
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "WHAT MODEL 46 ADDED", "45차 공통 11개 + 새로 선택된 21개 변수", 3)
    groups = [
        ("교통·속도·차량", added[:4]),
        ("사고다발·법규위반", added[4:10]),
        ("보호구역·보행", added[10:15]),
        ("자전거·피해결과", added[15:]),
    ]
    for idx, (title, items) in enumerate(groups):
        x = .72 + (idx % 2) * 6.05
        y = 1.52 + (idx // 2) * 2.55
        feature_list(s, x, y, 5.75, 2.23, title, items, MINT if idx % 2 else BLUE)
    txt(s, .90, 6.72, 11.55, .24,
        "주의: 추가 변수는 과거연도 검증 MAE를 낮췄지만 변수 수 증가로 과적합 가능성도 함께 커짐",
        10, RED, True, PP_ALIGN.CENTER)
    footer(s)

    # 4. Rolling validation
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "2021–2024 ROLLING VALIDATION", "여러 연도를 순차적으로 예측한 평균 성능", 4)
    metric_card(s, .78, "MAE ↓", cv45["평균_MAE"], cv46["평균_MAE"])
    metric_card(s, 3.78, "RMSE ↓", cv45["평균_RMSE"], cv46["평균_RMSE"])
    metric_card(s, 6.78, "R² ↑", cv45["평균_R2"], cv46["평균_R2"], False)
    metric_card(s, 9.78, "WMAPE ↓", cv45["평균_WMAPE_pct"], cv46["평균_WMAPE_pct"])
    txt(s, .90, 5.82, 11.55, .48,
        "과거 순차검증에서는 46차가 네 지표 모두 개선 · MAE 17.1% 감소",
        15, NAVY, True, PP_ALIGN.CENTER)
    footer(s)

    # 5. Holdout
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "2025 HOLDOUT RESULT", "동일한 25개 자치구의 다음 연도 피해자율 예측 비교", 5)
    metric_card(s, .78, "MAE ↓", t45["MAE"], t46["MAE"])
    metric_card(s, 3.78, "RMSE ↓", t45["RMSE"], t46["RMSE"])
    metric_card(s, 6.78, "R² ↑", t45["R2"], t46["R2"], False)
    metric_card(s, 9.78, "WMAPE ↓", t45["WMAPE_pct"], t46["WMAPE_pct"])
    txt(s, .90, 5.82, 11.55, .54,
        "46차: 평균적인 오차는 감소  |  45차: 일부 큰 오차에 대한 안정성과 설명력은 소폭 우수",
        13, NAVY, True, PP_ALIGN.CENTER)
    footer(s)

    # 6. Recommendation
    s = prs.slides.add_slide(blank)
    shape(s, 0, 0, 13.333, 7.5, WHITE, False)
    header(s, "FINAL RECOMMENDATION", "목적에 따라 두 모델의 역할을 분리", 6)
    shape(s, .82, 1.58, 5.55, 4.62, BLUE, True)
    txt(s, 1.12, 1.92, 4.95, .34, "45차 · 운영 안정형", 18, NAVY, True, PP_ALIGN.CENTER)
    txt(s, 1.25, 2.65, 4.70, 2.35,
        "• 변수 11개로 관리가 쉬움\n\n• 2025 RMSE·R² 우수\n\n• 해석과 재현성이 상대적으로 높음",
        13, INK, True)
    shape(s, 6.96, 1.58, 5.55, 4.62, MINT, True)
    txt(s, 7.26, 1.92, 4.95, .34, "46차 · MAE 최적형", 18, NAVY, True, PP_ALIGN.CENTER)
    txt(s, 7.39, 2.65, 4.70, 2.35,
        "• 현재 MAE·WMAPE 1위\n\n• 평균 자치구 예측오차 최소\n\n• 변수 32개로 과적합 점검 필요",
        13, INK, True)
    shape(s, 3.25, 6.47, 6.83, .45, TEAL, True)
    txt(s, 3.48, 6.58, 6.37, .22, "현재 공식 결론: MAE 기준 46차를 잠정 최고모델로 채택", 11, WHITE, True, PP_ALIGN.CENTER)
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
