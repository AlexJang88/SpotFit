from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib.pyplot as plt
import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


MODEL_ROOT = ROOT / "models" / "안전" / "교통사고"
REPORT_DIR = ROOT / "artifacts" / "reports" / "traffic_models_v1_v8"
OUTPUT = REPORT_DIR / "교통사고_1차~8차_모델비교.pptx"
FONT = "맑은 고딕"
NAVY = RGBColor(20, 45, 78)
BLUE = RGBColor(36, 105, 170)
TEAL = RGBColor(29, 142, 134)
ORANGE = RGBColor(232, 126, 61)
RED = RGBColor(196, 65, 65)
LIGHT = RGBColor(242, 246, 250)
MID = RGBColor(105, 120, 136)
DARK = RGBColor(34, 42, 50)
WHITE = RGBColor(255, 255, 255)


def load_models():
    return [json.loads((MODEL_ROOT / f"{v}차" / "모델정보.json").read_text(encoding="utf-8")) for v in range(1, 9)]


def style_run(run, size=18, bold=False, color=DARK):
    run.font.name = FONT; run.font.size = Pt(size); run.font.bold = bold; run.font.color.rgb = color


def text(slide, x, y, w, h, value, size=18, bold=False, color=DARK, align=PP_ALIGN.LEFT,
         valign=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.clear(); tf.word_wrap = True; tf.vertical_anchor = valign
    p = tf.paragraphs[0]; p.alignment = align
    r = p.add_run(); r.text = value; style_run(r, size, bold, color)
    return box


def bullets(slide, x, y, w, h, values, size=17, color=DARK):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.clear(); tf.word_wrap = True
    for i, value in enumerate(values):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = value; p.space_after = Pt(10); p.font.name = FONT; p.font.size = Pt(size); p.font.color.rgb = color
    return box


def header(slide, title, subtitle, number):
    slide.background.fill.solid(); slide.background.fill.fore_color.rgb = WHITE
    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(.14))
    band.fill.solid(); band.fill.fore_color.rgb = TEAL; band.line.fill.background()
    text(slide, .65, .35, 11.5, .5, title, 27, True, NAVY)
    text(slide, .67, .9, 11.6, .3, subtitle, 11, False, MID)
    text(slide, 12.35, .4, .45, .3, f"{number:02d}", 12, True, TEAL, PP_ALIGN.RIGHT)
    text(slide, .67, 7.22, 8, .18, "안전 · 교통사고 · 1차~8차 모델 비교", 9, False, MID)


def card(slide, x, y, w, h, label, value, note="", accent=TEAL):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid(); shape.fill.fore_color.rgb = LIGHT; shape.line.color.rgb = RGBColor(220, 229, 238)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(.07), Inches(h))
    bar.fill.solid(); bar.fill.fore_color.rgb = accent; bar.line.fill.background()
    text(slide, x+.22, y+.15, w-.35, .25, label, 11, True, MID)
    text(slide, x+.22, y+.48, w-.35, .48, value, 23, True, NAVY)
    if note: text(slide, x+.22, y+h-.35, w-.35, .2, note, 9, False, MID)


def add_table(slide, x, y, w, h, rows, widths=None):
    table = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(h)).table
    if widths:
        for i, width in enumerate(widths): table.columns[i].width = Inches(width)
    for ri, row in enumerate(rows):
        for ci, value in enumerate(row):
            cell = table.cell(ri, ci); cell.text = str(value); cell.margin_left = Inches(.08); cell.margin_right = Inches(.08)
            cell.fill.solid(); cell.fill.fore_color.rgb = NAVY if ri == 0 else (RGBColor(234, 246, 244) if ri == 2 else WHITE)
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER; p.font.name = FONT; p.font.size = Pt(11 if ri else 12)
                p.font.bold = ri == 0 or ci == 0; p.font.color.rgb = WHITE if ri == 0 else DARK
    return table


def make_chart(models, out):
    labels = [f"{i}차" for i in range(1, 9)]
    mae = [m["최종검증지표"]["MAE"] for m in models]
    wmape = [m["최종검증지표"]["WMAPE(%)"] for m in models]
    r2 = [m["최종검증지표"]["R2"] for m in models]
    plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = ["#9eb3c7", "#1d8e86", "#58a88a", "#2470aa", "#e4a85f", "#516f95", "#ad7a99", "#8a9a5b"]
    axes[0].bar(labels, mae, color=colors)
    for i, v in enumerate(mae): axes[0].text(i, v+8, f"{v:.1f}", ha="center", fontweight="bold", fontsize=10)
    axes[0].set_title("2025 최종검증 MAE — 낮을수록 우수", loc="left", fontweight="bold")
    axes[0].set_ylabel("평균 절대오차(건)"); axes[0].grid(axis="y", alpha=.2)
    axes[1].plot(labels, wmape, "o-", lw=2.5, color="#c44141", label="WMAPE(%)")
    axes[1].plot(labels, np.array(r2)*100, "s-", lw=2.5, color="#2470aa", label="R²×100")
    for i, v in enumerate(wmape): axes[1].text(i, v+2, f"{v:.1f}%", ha="center", fontsize=9, color="#c44141")
    axes[1].set_title("오차율과 분산설명력", loc="left", fontweight="bold"); axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=.2); axes[1].set_ylim(0, 105)
    for ax in axes: ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig)


def build():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    models = load_models(); chart = REPORT_DIR / "performance.png"; make_chart(models, chart)
    prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5); blank = prs.slide_layouts[6]

    slide = prs.slides.add_slide(blank); slide.background.fill.solid(); slide.background.fill.fore_color.rgb = NAVY
    text(slide, .8, 1.25, 11.7, .55, "교통사고 1차~8차 모델 비교", 34, True, WHITE)
    text(slide, .83, 2.0, 11.2, .35, "서울 25개 자치구 · 다음 연도 교통사고 건수 예측", 17, False, RGBColor(190, 220, 225))
    text(slide, .83, 3.15, 11.3, 1.1, "1차 독립 확장(2~5차)과 최우수 2차 재확장(6~8차)의\n예측 성능과 해석 가능성을 비교", 25, True, WHITE)
    text(slide, .83, 6.75, 6, .25, "최종검증: 2024년 X → 2025년 y", 12, False, RGBColor(190, 220, 225))

    slide = prs.slides.add_slide(blank); header(slide, "비교 설계", "모든 차수는 1차 변수를 유지한 독립 실험", 2)
    card(slide, .7, 1.45, 2.75, 1.25, "예측 목표", "t+1년 사고 건수", "1년 예측", BLUE)
    card(slide, 3.65, 1.45, 2.75, 1.25, "분석 단위", "서울 25개 구", "연도×자치구", TEAL)
    card(slide, 6.6, 1.45, 2.75, 1.25, "모델 선택", "시간순 CV MAE", "2021~2024 확대창", ORANGE)
    card(slide, 9.55, 1.45, 2.75, 1.25, "최종 검증", "2025년", "선정 이후 1회", RED)
    bullets(slide, .8, 3.15, 5.8, 2.9, ["1차 공통 변수 4개를 모든 모델에서 유지", "추가 변수군은 차수별로 독립 비교", "Ridge·RandomForest·ExtraTrees·GradientBoosting 비교", "추가 변수는 예측용이며 안전지수 가중치는 변경하지 않음"], 17)
    bullets(slide, 6.85, 3.15, 5.45, 2.9, ["위해: 인구만명당 교통사고 사망자수", "취약: 인구만명당 의료보장 사업장수", "의식: 운전자 안전벨트 착용률", "의식: 음주운전 경험률", "경감자료 미확보 → 62.5 : 12.5 : 25 재배분"], 17)

    slide = prs.slides.add_slide(blank); header(slide, "모델 구성과 선정 결과", "각 차수에서 가장 낮은 내부검증 MAE 모델", 3)
    rows = [["차수", "기준", "추가 변수", "선정 기법", "변수 수", "학습행"]]
    additions = ["없음", "현재연도 사고 건수", "사고 이력 파생 4개", "자동차 등록 대수", "생활인구 2개",
                 "최근 3년 사고 평균", "자동차 등록 대수", "생활인구 월변동성"]
    bases = ["-", "1차", "1차", "1차", "1차", "2차", "2차", "2차"]
    for i, m in enumerate(models, 1):
        rows.append([f"{i}차", bases[i-1], additions[i-1], m["선정모델"], len(m["입력변수"]), m.get("학습행", 175)])
    add_table(slide, .55, 1.3, 12.25, 5.55, rows, [.8, .8, 3.35, 2.15, 1.25, 1.3])
    text(slide, .75, 6.92, 11.8, .25, "3·6차는 3년 이력 계산으로 학습 표본이 125행", 10, True, ORANGE)

    slide = prs.slides.add_slide(blank); header(slide, "성능 비교", "2025년 서울 25개 자치구 최종검증", 4)
    slide.shapes.add_picture(str(chart), Inches(.55), Inches(1.35), width=Inches(12.2))
    text(slide, .75, 6.45, 11.8, .35, "전체 최우선은 2차 Ridge — MAE 99.69건·WMAPE 7.30%·R² 0.9626", 15, True, TEAL)

    slide = prs.slides.add_slide(blank); header(slide, "수치로 보는 1차~5차", "MAE·RMSE·WMAPE는 낮을수록, R²는 높을수록 우수", 5)
    rows = [["차수", "선정 기법", "MAE", "RMSE", "R²", "WMAPE"]]
    for i, m in enumerate(models, 1):
        met = m["최종검증지표"]
        rows.append([f"{i}차", m["선정모델"], f"{met['MAE']:.2f}", f"{met['RMSE']:.2f}", f"{met['R2']:.4f}", f"{met['WMAPE(%)']:.2f}%"])
    add_table(slide, .65, 1.25, 12.05, 5.85, rows, [1.0, 2.5, 1.7, 1.7, 1.7, 1.7])
    text(slide, .75, 7.12, 11.8, .2, "100−WMAPE는 등급 적중률이 아닌 오차 기반 참고값", 9, False, MID)

    slide = prs.slides.add_slide(blank); header(slide, "모델별 핵심 해석", "무엇을 추가했을 때 성능이 좋아졌는가", 6)
    items = [
        ("1차", "안전기준표 O 항목만 사용", "초기 기준선 · 설명력 제한", MID),
        ("2차", "현재연도 사고 건수 추가", "사고 수준의 강한 지속성을 포착 · 최고 성능", TEAL),
        ("3차", "1년전·3년평균·증감·변동성", "직접 현재값 없이도 이력 패턴으로 높은 성능", BLUE),
        ("4차", "자동차 등록 대수 추가", "교통 노출 규모를 반영", ORANGE),
        ("5차", "생활인구 평균·최대 추가", "실제 활동인구 노출을 반영", RED),
    ]
    for idx, (name, desc, note, color) in enumerate(items):
        y = 1.35 + idx*1.05
        card(slide, .75, y, 1.25, .82, name, "", "", color)
        text(slide, 2.25, y+.06, 4.25, .3, desc, 15, True, NAVY)
        text(slide, 6.6, y+.06, 5.7, .45, note, 14, False, DARK)

    slide = prs.slides.add_slide(blank); header(slide, "2차 기반 재확장: 6~8차", "내부 시간순 CV로 2차 대비 개선 여부 판정", 7)
    rows = [
        ["차수", "추가 변수", "선정 기법", "2차기준 CV MAE", "확장 CV MAE", "판정", "2025 MAE"],
        ["6차", "최근 3년 사고 평균", "Ridge", "125.80", "107.81", "개선", "111.18"],
        ["7차", "자동차 등록 대수", "Ridge", "88.84", "97.34", "개선 없음", "99.19"],
        ["8차", "생활인구 월변동성", "Ridge", "88.84", "92.61", "개선 없음", "102.90"],
    ]
    add_table(slide, .45, 1.45, 12.45, 3.55, rows, [.8, 2.4, 1.5, 1.75, 1.65, 1.45, 1.45])
    bullets(slide, .75, 5.35, 11.8, 1.35, ["6차는 같은 125행 표본의 2차 기준보다 CV 개선", "7차는 2025 MAE가 근소하게 낮지만 CV가 악화되어 채택하지 않음", "8차도 CV 개선이 없어 2차 우선순위를 유지"], 15)

    slide = prs.slides.add_slide(blank); header(slide, "결론과 권고", "예측 성능과 활용 목적을 함께 고려", 8)
    card(slide, .75, 1.35, 3.7, 1.35, "최고 성능", "2차 Ridge", "MAE 99.69 · R² 0.9626", TEAL)
    card(slide, 4.8, 1.35, 3.7, 1.35, "조건부 개선", "6차 Ridge", "축소 표본 CV 개선 · MAE 111.18", BLUE)
    card(slide, 8.85, 1.35, 3.7, 1.35, "보류", "7차·8차", "2차 대비 CV 개선 없음", ORANGE)
    bullets(slide, .85, 3.2, 11.7, 2.8, ["다음 연도 사고 건수 예측의 현재 최우선은 2차 모델", "6차는 사고 이력 3년이 확보되는 경우의 보조 후보", "7·8차는 최종 테스트값으로 판정을 뒤집지 않고 보류", "추가 변수는 예측용이며 안전지수 가중치는 기존 산식 유지", "단일 연도 2025 최종검증이므로 추가 연도에서 재검증 필요"], 18)
    text(slide, .85, 6.55, 11.6, .3, "주의: R²는 정확도 자체가 아니며, 모델 간 비교는 동일한 2025 최종검증 기준", 12, True, RED)

    prs.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())
