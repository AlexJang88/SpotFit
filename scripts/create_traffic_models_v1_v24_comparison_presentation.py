"""교통사고 1~12차 사고건수 모델과 13~24차 안전등급 모델 비교 PPT를 생성한다."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib.pyplot as plt
import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches

from create_traffic_models_v1_v5_presentation import (
    BLUE, DARK, MID, NAVY, ORANGE, RED, TEAL, WHITE,
    add_table, bullets, card, header, text,
)


MODEL_ROOT = ROOT / "models" / "안전" / "교통사고"
OUT_DIR = ROOT / "artifacts" / "reports" / "traffic_models_v1_v24_comparison"
OUTPUT = OUT_DIR / "교통사고_1차~12차_vs_13차~24차_모델비교.pptx"
FOCUS_OUTPUT = OUT_DIR / "교통사고_13차_vs_19차_핵심모델비교.pptx"

ADDITIONS = [
    "초기 기준 4개", "현재연도 사고건수", "사고이력 파생변수", "자동차 등록대수",
    "생활인구", "현재사고+3년평균", "현재사고+자동차", "현재사고+생활인구 변동",
    "7차+3년평균", "7차+생활인구 변동", "7차+두 변수", "현재사고+사고심각도",
]


def load_metadata(version: int) -> dict:
    return json.loads((MODEL_ROOT / f"{version}차" / "모델정보.json").read_text(encoding="utf-8"))


def replace_footer_range(prs: Presentation) -> None:
    for slide in prs.slides:
        text(slide, .67, 7.22, 8.5, .18, "안전 · 교통사고 · 1~24차 체계 비교", 9, False, MID)


def make_charts(old_models: list[dict], new_models: list[dict]) -> tuple[Path, Path]:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    old_path = OUT_DIR / "old_performance.png"
    new_path = OUT_DIR / "new_performance.png"

    old_mae = [model["최종검증지표"]["MAE"] for model in old_models]
    fig, ax = plt.subplots(figsize=(11.7, 4.0))
    colors = ["#1d8e86" if i == 1 else "#9eb3c7" for i in range(12)]
    ax.bar([f"{i}차" for i in range(1, 13)], old_mae, color=colors)
    for i, value in enumerate(old_mae):
        ax.text(i, value + 7, f"{value:.1f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_title("1~12차: 2025 사고건수 MAE — 낮을수록 우수", loc="left", fontweight="bold")
    ax.set_ylabel("MAE(건)"); ax.grid(axis="y", alpha=.2); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(old_path, dpi=180, bbox_inches="tight"); plt.close(fig)

    cv_accuracy = [model["CV성능"]["cv_grade_accuracy"] * 100 for model in new_models]
    final_accuracy = [model["2025최종검증"]["grade_accuracy"] * 100 for model in new_models]
    x = np.arange(12); width = .38
    fig, ax = plt.subplots(figsize=(11.7, 4.0))
    ax.bar(x - width/2, cv_accuracy, width, color="#1d8e86", label="내부 시간순 CV")
    ax.bar(x + width/2, final_accuracy, width, color="#2470aa", label="2025 최종검증")
    ax.set_xticks(x, [f"{i}차" for i in range(13, 25)])
    ax.set_ylim(0, 75); ax.set_ylabel("등급 정확도(%)")
    ax.set_title("13~24차: 안전등급 정확도", loc="left", fontweight="bold")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(new_path, dpi=180, bbox_inches="tight"); plt.close(fig)
    return old_path, new_path


def add_pair_table(slide, start: int, old_models: list[dict], new_models: list[dict]) -> None:
    rows = [["기존→신규", "추가 변수군", "기존 기법", "사고 MAE", "신규 기법", "CV 등급", "2025 등급"]]
    for index in range(start, start + 6):
        old = old_models[index]
        new = new_models[index]
        rows.append([
            f"{index+1}→{index+13}차", ADDITIONS[index], old["선정모델"],
            f"{old['최종검증지표']['MAE']:.1f}", new["선정모델"],
            f"{new['CV성능']['cv_grade_accuracy']*100:.0f}%",
            f"{new['2025최종검증']['grade_accuracy']*100:.0f}%",
        ])
    add_table(slide, .35, 1.25, 12.65, 5.75, rows, [1.2, 2.5, 1.7, 1.3, 1.7, 1.2, 1.35])


def build() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    old_models = [load_metadata(version) for version in range(1, 13)]
    new_models = [load_metadata(version) for version in range(13, 25)]
    old_chart, new_chart = make_charts(old_models, new_models)

    prs = Presentation()
    prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    slide = prs.slides.add_slide(blank)
    slide.background.fill.solid(); slide.background.fill.fore_color.rgb = NAVY
    text(slide, .8, 1.15, 11.8, .65, "교통사고 모델 체계 비교", 36, True, WHITE)
    text(slide, .83, 2.0, 11.6, .4, "1~12차 사고건수 예측  vs  13~24차 안전등급 예측", 20, False, RGBColor(190, 220, 225))
    text(slide, .83, 3.05, 11.4, 1.25,
         "같은 변수 추가 실험을 서로 다른 최종 목표로 재구성\n예측 성능·영역 분류·가중치·안전등급을 함께 비교", 26, True, WHITE)
    text(slide, .83, 6.7, 8, .25, "서울 25개 자치구 · 2017~2025 · 1년 예측", 12, False, RGBColor(190, 220, 225))

    slide = prs.slides.add_slide(blank); header(slide, "두 모델 체계의 차이", "수치의 목적과 평가 단위가 다르므로 체계 안에서 비교", 2)
    card(slide, .7, 1.4, 5.75, 1.5, "기존 1~12차", "다음연도 사고건수", "MAE·RMSE·R²·WMAPE", BLUE)
    card(slide, 6.9, 1.4, 5.75, 1.5, "신규 13~24차", "다음연도 안전등급", "정확도·±1등급·Macro-F1·Kappa", TEAL)
    bullets(slide, .85, 3.45, 5.55, 2.45, [
        "예측모델 계수가 변수 영향력을 자동 학습",
        "최종 출력은 사고건수",
        "안전등급은 예측값과 직접 연결되지 않았음",
    ], 17)
    bullets(slide, 7.05, 3.45, 5.4, 2.45, [
        "위해·취약·경감·의식 영역을 명시",
        "영역 가중치 한도를 적용",
        "최종 출력은 안전점수와 1~5등급",
    ], 17)
    text(slide, .85, 6.45, 11.7, .35, "사고건수 MAE와 등급 정확도는 서로 다른 지표이므로 직접 우열 비교하지 않음", 14, True, RED)

    slide = prs.slides.add_slide(blank); header(slide, "차수 대응 관계", "기존 변수 조합을 새 안전등급 체계에서 동일하게 재실험", 3)
    rows = [["기존", "신규", "변수 실험"],
            *[[f"{i+1}차", f"{i+13}차", ADDITIONS[i]] for i in range(12)]]
    add_table(slide, .8, 1.15, 11.75, 5.95, rows, [1.6, 1.6, 8.55])

    slide = prs.slides.add_slide(blank); header(slide, "기존 1~12차 성능", "2025년 다음연도 사고건수 최종검증", 4)
    slide.shapes.add_picture(str(old_chart), Inches(.65), Inches(1.35), width=Inches(12.0))
    text(slide, .8, 6.45, 11.7, .35, "내부 시간순 CV 기준 최우선: 2차 Ridge · CV MAE 88.84건", 15, True, TEAL)

    slide = prs.slides.add_slide(blank); header(slide, "신규 13~24차 성능", "다음연도 안전등급 정확도 · 2025는 모델 선택 이후 최종검증", 5)
    slide.shapes.add_picture(str(new_chart), Inches(.65), Inches(1.35), width=Inches(12.0))
    text(slide, .8, 6.45, 11.7, .35, "내부 CV 최우선: 19차 56% · 2025 최고: 20차 64% · 선택 기준은 내부 CV", 15, True, TEAL)

    slide = prs.slides.add_slide(blank); header(slide, "대응 모델 비교: 1~6차 ↔ 13~18차", "사고 MAE는 기존 체계, 등급 정확도는 신규 체계", 6)
    add_pair_table(slide, 0, old_models, new_models)

    slide = prs.slides.add_slide(blank); header(slide, "대응 모델 비교: 7~12차 ↔ 19~24차", "동일한 추가 변수도 최종 목표가 바뀌면 선정 알고리즘이 달라질 수 있음", 7)
    add_pair_table(slide, 6, old_models, new_models)

    slide = prs.slides.add_slide(blank); header(slide, "가중치 체계 비교", "모델 영향 비율과 안전지수 가중치를 분리", 8)
    card(slide, .7, 1.35, 5.75, 1.4, "1~12차", "학습된 모델 영향도", "Ridge 계수 또는 트리 중요도", BLUE)
    card(slide, 6.9, 1.35, 5.75, 1.4, "13~24차", "영역 한도 고정", "위해·취약·경감·의식", TEAL)
    rows = [
        ["구분", "공식 기준", "현재 적용", "처리"],
        ["위해", "50%", "62.5%", "경감 미확보 비례 환산"],
        ["취약", "10%", "12.5%", "경감 미확보 비례 환산"],
        ["경감", "20%", "0%", "완전한 시계열 없음"],
        ["의식", "20%", "25.0%", "8.99:8.70 비율 유지"],
    ]
    add_table(slide, .8, 3.2, 11.7, 2.9, rows, [2.1, 2.0, 2.0, 5.6])
    text(slide, .85, 6.45, 11.6, .35, "경감지표 확보 시 즉시 50·10·20·20 원래 체계로 복원", 14, True, ORANGE)

    slide = prs.slides.add_slide(blank); header(slide, "신규 변수의 영역 분류", "13~24차에서 적용한 의미 기반 분류", 9)
    rows = [
        ["영역", "포함 변수"],
        ["위해", "사망자수 · 현재 사고건수 · 사고 이력 · 사고당 부상자수 · 어린이 사고비율"],
        ["취약", "의료보장 사업장수 · 자동차 등록대수 · 생활인구"],
        ["경감", "현재 완전한 시계열 없음"],
        ["의식", "안전벨트 착용률 · 음주운전 경험률"],
    ]
    add_table(slide, .7, 1.35, 11.95, 4.4, rows, [2.0, 9.95])
    bullets(slide, .85, 6.0, 11.5, .85, [
        "같은 영역의 새 변수는 영역 총가중치 안에서 균등 배분",
        "의식지표는 안전기준표의 8.99:8.70 상대비율 적용",
    ], 14)

    slide = prs.slides.add_slide(blank); header(slide, "핵심 모델 비교", "각 체계의 내부검증 기준 우선 모델", 10)
    card(slide, .7, 1.35, 3.75, 1.45, "사고건수 최우선", "2차 Ridge", "CV MAE 88.84건", BLUE)
    card(slide, 4.8, 1.35, 3.75, 1.45, "안전등급 최우선", "19차 ExtraTrees", "CV 정확도 56% · ±1 92%", TEAL)
    card(slide, 8.9, 1.35, 3.75, 1.45, "2025 등급 최고", "20차 Ridge", "정확도 64% · ±1 88%", ORANGE)
    bullets(slide, .9, 3.35, 11.6, 2.7, [
        "2차는 사고건수 예측용 최우선이며 안전등급 모델과 역할이 다름",
        "19차는 현재사고와 자동차 등록대수를 포함한 신규 체계의 CV 최우선",
        "20차는 2025 성능이 가장 높지만 테스트 결과만으로 최우선 모델을 변경하지 않음",
        "13~24차는 경감지표 제외 환산 등급이므로 공식 전체지표 등급과 구분",
    ], 17)

    slide = prs.slides.add_slide(blank); header(slide, "결론과 다음 단계", "기존 실험은 보존하고 안전등급 체계를 중심으로 발전", 11)
    bullets(slide, .9, 1.45, 11.6, 4.8, [
        "1~12차: 사고건수 예측 실험과 변수 효과 확인 자료로 보존",
        "13~24차: 최종 안전등급을 제공하는 신규 운영 후보군",
        "현재 내부검증 기준 최우선은 19차, 초기 기준선은 13차",
        "경감지표가 확보되면 모든 신규 모델을 50·10·20·20 기준으로 재산출",
        "추가 연도 독립검증 전에는 Registry champion 또는 서비스 배포를 보류",
        "향후 새 변수는 영역을 먼저 결정한 뒤 해당 영역 한도 안에서만 가중치 배분",
    ], 19)
    text(slide, .9, 6.45, 11.5, .35, "최종 제공값: 자치구별 다음연도 예측 안전점수 + 안전등급(1~5)", 15, True, TEAL)

    replace_footer_range(prs)
    prs.save(OUTPUT)
    return OUTPUT


def build_focus_13_19() -> Path:
    """13차 기준모델과 19차 최우선 모델을 쉬운 한국어로 비교한다."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model13 = load_metadata(13)
    model19 = load_metadata(19)
    candidate13 = __import__("pandas").read_csv(MODEL_ROOT / "13차" / "후보모델_비교.csv")
    candidate19 = __import__("pandas").read_csv(MODEL_ROOT / "19차" / "후보모델_비교.csv")

    prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    slide = prs.slides.add_slide(blank); slide.background.fill.solid(); slide.background.fill.fore_color.rgb = NAVY
    text(slide, .8, 1.15, 11.8, .65, "교통사고 안전등급 핵심 모델 비교", 34, True, WHITE)
    text(slide, .83, 2.0, 11.4, .42, "13차 기준모델  vs  19차 최고성능 모델", 21, False, RGBColor(190, 220, 225))
    text(slide, .83, 3.05, 11.4, 1.25,
         "무엇을 넣었고, 왜 넣었고,\n왜 이 모델을 선택했는지 결정 순서대로 설명", 27, True, WHITE)
    text(slide, .83, 6.7, 8, .25, "최종 결과: 다음연도 안전점수와 안전등급(1~5)", 12, False, RGBColor(190, 220, 225))

    slide = prs.slides.add_slide(blank); header(slide, "먼저 결론", "두 모델의 역할은 서로 다름", 2)
    card(slide, .75, 1.45, 5.7, 1.55, "13차", "기준모델", "안전기준표의 확보된 기본변수 4개", BLUE)
    card(slide, 6.9, 1.45, 5.7, 1.55, "19차", "현재 최우선 모델", "기본변수 + 현재 사고건수 + 자동차 등록대수", TEAL)
    bullets(slide, .9, 3.55, 11.6, 2.15, [
        "13차는 새 안전등급 체계가 제대로 작동하는지 확인하는 출발점",
        "19차는 같은 규칙을 지키면서 예측에 도움이 되는 변수 2개를 추가",
        "새 모델은 항상 13차와 비교하고, 현재 최고인 19차도 함께 비교",
    ], 19)
    text(slide, .9, 6.25, 11.5, .4, "기준을 설명할 때는 13차, 실제 예측 후보를 설명할 때는 19차가 핵심", 15, True, ORANGE)

    slide = prs.slides.add_slide(blank); header(slide, "공통으로 지킨 계산 규칙", "모델이 마음대로 영역 가중치를 바꾸지 않음", 3)
    rows = [
        ["영역", "원래 기준", "현재 적용", "이유"],
        ["위해", "50%", "62.5%", "경감자료가 없어 비례 환산"],
        ["취약", "10%", "12.5%", "경감자료가 없어 비례 환산"],
        ["경감", "20%", "0%", "완전한 연도별 자료 없음"],
        ["의식", "20%", "25.0%", "경감자료가 없어 비례 환산"],
    ]
    add_table(slide, .75, 1.25, 11.85, 3.65, rows, [2.0, 2.0, 2.0, 5.85])
    bullets(slide, .9, 5.3, 11.5, 1.15, [
        "경감지표가 하나라도 충분히 확보되면 원래 50·10·20·20으로 복원",
        "일부 지역만 자료가 없다고 그 지역에만 가중치를 더 주지 않음",
    ], 15)
    text(slide, .9, 6.65, 11.5, .3, "현재 결과는 ‘경감지표 제외 환산 안전등급’", 13, True, RED)

    slide = prs.slides.add_slide(blank); header(slide, "13차: 왜 이 변수들을 선택했나?", "안전기준표에서 실제 확보된 기본항목만 사용", 4)
    rows = [
        ["변수", "영역", "가중치", "선택 이유"],
        ["인구 1만명당 사고 사망자수", "위해", "62.50%", "실제 피해 결과를 나타내는 핵심 지표"],
        ["인구 1만명당 의료보장 사업장수", "취약", "12.50%", "안전기준표의 취약 항목"],
        ["안전벨트 착용률", "의식", "12.70%", "안전 행동을 나타내며 높을수록 안전"],
        ["음주운전 경험률", "의식", "12.30%", "위험 행동을 나타내며 높을수록 불리"],
    ]
    add_table(slide, .45, 1.2, 12.45, 4.65, rows, [3.35, 1.25, 1.45, 6.4])
    text(slide, .75, 6.15, 11.8, .65,
         "왜 13차가 기준인가?  새로운 보조변수를 넣기 전에 안전기준표 기본정보만으로 만든 출발점이기 때문", 15, True, TEAL)

    slide = prs.slides.add_slide(blank); header(slide, "13차: 왜 RandomForest인가?", "같은 데이터와 같은 시간순 검증에서 4개 기법 비교", 5)
    rows = [["후보 기법", "등급 정확도", "±1등급", "판단"]]
    for row in candidate13.itertuples(index=False):
        judgment = "선택" if row.알고리즘 == "RandomForest" else "보류"
        rows.append([row.알고리즘, f"{row.cv_grade_accuracy*100:.0f}%", f"{row.cv_grade_within1_accuracy*100:.0f}%", judgment])
    add_table(slide, .75, 1.25, 11.85, 3.7, rows, [3.0, 2.3, 2.3, 4.25])
    bullets(slide, .9, 5.25, 11.5, 1.15, [
        "RandomForest의 정확한 등급 일치율이 43%로 가장 높았음",
        "Ridge는 ±1등급 비율은 높지만 정확한 등급 일치가 36%로 낮았음",
    ], 15)
    text(slide, .9, 6.62, 11.5, .3, "선택 기준: 최종 목표가 안전등급이므로 ‘정확한 등급 일치율’을 먼저 봄", 13, True, ORANGE)

    slide = prs.slides.add_slide(blank); header(slide, "19차: 무엇을 더 넣었나?", "13차에 현재 사고건수와 자동차 등록대수를 추가", 6)
    card(slide, .7, 1.35, 3.75, 1.45, "추가 변수 1", "현재연도 사고건수", "위해지표 · 사고가 반복되는 수준", RED)
    card(slide, 4.8, 1.35, 3.75, 1.45, "추가 변수 2", "자동차 등록대수", "취약지표 · 교통 노출 규모", ORANGE)
    card(slide, 8.9, 1.35, 3.75, 1.45, "총 입력", "6개 변수", "13차 4개 + 신규 2개", TEAL)
    rows = [
        ["영역", "영역 안의 변수", "영역 가중치", "변수별 배분"],
        ["위해", "사망자수 + 현재 사고건수", "62.5%", "31.25%씩"],
        ["취약", "의료보장 + 자동차 등록", "12.5%", "6.25%씩"],
        ["의식", "안전벨트 + 음주운전", "25.0%", "12.70% + 12.30%"],
    ]
    add_table(slide, .75, 3.25, 11.85, 2.65, rows, [1.8, 4.25, 2.15, 3.65])
    text(slide, .9, 6.28, 11.5, .45, "새 변수가 들어와도 자기 영역의 한도를 넘지 않도록 같은 영역 안에서만 나눔", 14, True, TEAL)

    slide = prs.slides.add_slide(blank); header(slide, "19차: 왜 ExtraTrees인가?", "점수 오차보다 최종 등급 정확도를 우선", 7)
    rows = [["후보 기법", "등급 정확도", "±1등급", "점수 MAE", "판단"]]
    for row in candidate19.itertuples(index=False):
        judgment = "선택" if row.알고리즘 == "ExtraTrees" else "보류"
        rows.append([row.알고리즘, f"{row.cv_grade_accuracy*100:.0f}%", f"{row.cv_grade_within1_accuracy*100:.0f}%", f"{row.cv_score_mae:.2f}", judgment])
    add_table(slide, .55, 1.25, 12.25, 3.85, rows, [2.7, 2.1, 2.1, 2.0, 3.35])
    bullets(slide, .85, 5.35, 11.7, 1.0, [
        "ExtraTrees의 등급 정확도가 56%로 가장 높았음",
        "Ridge의 점수 MAE는 6.68로 아주 조금 낮지만 등급 정확도는 54%였음",
    ], 15)
    text(slide, .85, 6.55, 11.7, .35, "최종 제공값이 ‘등급’이므로 2%p 더 잘 맞힌 ExtraTrees를 선택", 14, True, ORANGE)

    slide = prs.slides.add_slide(blank); header(slide, "13차와 19차 성능 비교", "2021~2024 시간순 내부검증을 우선 사용", 8)
    rows = [
        ["지표", "13차", "19차", "변화"],
        ["정확한 등급 일치", "43%", "56%", "+13%p"],
        ["±1등급 이내", "84%", "92%", "+8%p"],
        ["안전점수 MAE", "12.36점", "6.69점", "5.67점 감소"],
        ["안전점수 R²", "0.0216", "0.5200", "+0.4984"],
        ["2025 등급 정확도", "48%", "48%", "동일"],
        ["2025 ±1등급", "76%", "88%", "+12%p"],
    ]
    add_table(slide, .75, 1.15, 11.85, 5.15, rows, [3.4, 2.3, 2.3, 3.85])
    text(slide, .9, 6.55, 11.5, .35, "결론: 여러 연도에서 반복 검증한 내부 성능이 좋아졌으므로 19차를 최우선으로 선택", 14, True, TEAL)

    slide = prs.slides.add_slide(blank); header(slide, "선택 과정을 한눈에 보기", "각 단계마다 기준을 먼저 정하고 결과를 확인", 9)
    steps = [
        ("1", "최종 목표 결정", "사고건수가 아니라 안전등급 1~5"),
        ("2", "기준모델 결정", "안전기준표 기본변수만 사용한 13차"),
        ("3", "가중치 규칙 결정", "영역 한도 고정 · 경감 미확보 시 동일 환산"),
        ("4", "새 변수 분류", "현재 사고건수=위해 · 자동차 등록=취약"),
        ("5", "후보 기법 비교", "같은 시간순 검증에서 4개 알고리즘 비교"),
        ("6", "최우선 결정", "등급 정확도가 가장 높은 19차 ExtraTrees"),
    ]
    for idx, (number, title_value, note) in enumerate(steps):
        y = 1.25 + idx * .9
        card(slide, .75, y, 1.0, .65, "단계", number, "", TEAL)
        text(slide, 2.0, y+.02, 3.0, .3, title_value, 15, True, NAVY)
        text(slide, 5.0, y+.02, 7.3, .4, note, 14, False, DARK)

    slide = prs.slides.add_slide(blank); header(slide, "최종 정리", "무엇을 기억하면 되는가", 10)
    card(slide, .75, 1.35, 5.7, 1.5, "13차", "설명의 기준", "기본변수 4개 · RandomForest", BLUE)
    card(slide, 6.9, 1.35, 5.7, 1.5, "19차", "현재 예측 최우선", "변수 6개 · ExtraTrees", TEAL)
    bullets(slide, .9, 3.4, 11.6, 2.35, [
        "새 변수를 시험할 때 13차와 19차를 모두 비교",
        "19차 내부 등급 정확도 56%, ±1등급 정확도 92%",
        "경감지표가 확보되면 원래 50·10·20·20 가중치로 다시 계산",
        "추가 연도 독립검증 전에는 공식 전체지표 등급이나 확정 배포 모델로 표현하지 않음",
    ], 17)
    text(slide, .9, 6.35, 11.5, .45, "한 줄 결론: 13차는 출발점, 19차는 현재 가장 믿을 만한 예측 후보", 16, True, ORANGE)

    replace_footer_range(prs)
    prs.save(FOCUS_OUTPUT)
    return FOCUS_OUTPUT


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--focus-13-19", action="store_true", help="13차와 19차만 쉬운 한국어로 비교")
    args = parser.parse_args()
    print(build_focus_13_19() if args.focus_13_19 else build())
